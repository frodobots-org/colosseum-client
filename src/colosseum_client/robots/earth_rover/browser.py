"""The Earth Rovers SDK's browser transport, embedded in the client.

A headless Chrome page (``vendor/rover.html``) joins the ride's Agora RTC channel
for the cameras and logs into Agora RTM 1.x for commands and telemetry, exactly
as the SDK's ``browser_service.py`` does. Playwright runs on its own thread with
an asyncio loop; the link's threads call into it synchronously.
"""
from __future__ import annotations

import asyncio
import base64
from collections import deque
import io
import json
import os
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ...navigation import FRONT_CAMERA, REAR_CAMERA
from .telemetry import TelemetryStore, parse_report
from .video import FrameStore

VENDOR = Path(__file__).parent / 'vendor'
PAGE_ORIGIN = 'http://localhost'  # a secure context for the Agora web SDK; served from memory via page.route
CAMERA_UIDS = {1000: FRONT_CAMERA, 1001: REAR_CAMERA}
LAUNCH_ARGS = ['--ignore-certificate-errors', '--no-sandbox', '--autoplay-policy=no-user-gesture-required',
               '--use-fake-ui-for-media-stream', '--disable-application-cache', '--disk-cache-size=0']
CONTENT_TYPES = {'.html': 'text/html; charset=utf-8', '.js': 'application/javascript'}


def data_url_to_rgb(data_url):
    """A canvas data URL (JPEG/PNG) → RGB uint8 (height, width, 3)."""
    header, _, payload = data_url.partition(',')
    if not header.startswith('data:image/') or not payload:
        raise ValueError('not an image data URL')
    with Image.open(io.BytesIO(base64.b64decode(payload))) as image:
        return np.asarray(image.convert('RGB'), dtype=np.uint8)


class BrowserTransport:
    def __init__(self, *, cameras, chrome_executable=None, codec='vp8', frame_hz=10, quality=0.8):
        self.cameras = set(cameras)
        self.chrome_executable = chrome_executable or os.environ.get('CHROME_EXECUTABLE_PATH') or None
        self.codec = codec
        self.frame_hz = frame_hz
        self.quality = quality
        self.frames = FrameStore()
        self.telemetry = TelemetryStore()
        self.lamp = 0  # headlamp bit carried by every command; the rover keeps it until told otherwise
        self.errors = deque(maxlen=50)  # recent page, capture and stop errors, shown by the operator page
        self.browser_name = ''
        self.loop = None
        self._thread = None
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._capture_task = None

    # --- loop thread --------------------------------------------------------
    def start(self):
        if self.loop is not None:
            return
        ready = threading.Event()
        failure = []

        def run():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.loop = loop
            try:
                loop.run_until_complete(self._launch())
            except Exception as exc:
                failure.append(exc)
                ready.set()
                return
            ready.set()
            loop.run_forever()

        self._thread = threading.Thread(target=run, name='earth-rover-browser', daemon=True)
        self._thread.start()
        ready.wait(60)
        if failure:
            self.loop = None
            raise RuntimeError(f'Browser launch failed: {failure[0]}') from failure[0]

    def _call(self, coroutine, timeout):
        if self.loop is None:
            coroutine.close()
            raise RuntimeError('Browser transport is not started')
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result(timeout)

    async def _launch(self):
        try:
            from playwright.async_api import Error as PlaywrightError, async_playwright
        except ImportError as exc:
            raise RuntimeError("The Earth Rover link needs Playwright: uv pip install playwright==1.60.0") from exc
        self._playwright = await async_playwright().start()
        options = dict(headless=True, args=LAUNCH_ARGS)
        if self.chrome_executable:
            self._browser = await self._playwright.chromium.launch(executable_path=self.chrome_executable, **options)
            self.browser_name = 'CHROME_EXECUTABLE_PATH'
            return
        try:  # Google Chrome ships the H.264 decoder the Mini streams need; bundled Chromium decodes them as 0x0.
            self._browser = await self._playwright.chromium.launch(channel='chrome', **options)
            self.browser_name = 'Google Chrome'
        except PlaywrightError:
            if not os.path.exists(self._playwright.chromium.executable_path):
                raise RuntimeError('No usable browser: install Google Chrome, run "playwright install chromium", '
                                   'or set CHROME_EXECUTABLE_PATH')
            self._browser = await self._playwright.chromium.launch(**options)
            self.browser_name = 'bundled Chromium (no H.264)'

    # --- page lifecycle -----------------------------------------------------
    def open(self, credentials, *, first_frame_timeout=15.0):
        """Join the ride's channel; returns once RTM is connected and the front camera decodes a frame from it."""
        self.start()
        opened = time.monotonic()
        self._call(self._open(credentials), timeout=first_frame_timeout + 30)
        deadline = time.monotonic() + first_frame_timeout
        while time.monotonic() < deadline:
            latest = self.frames.latest().get(FRONT_CAMERA)
            if latest is not None and latest[1] >= opened:  # from this channel, not the page that was just closed
                return
            time.sleep(0.1)
        health = self.health()
        raise RuntimeError(f'No front camera frame within {first_frame_timeout:.0f}s ({self.browser_name}; '
                           f'publishers {health.get("publishers")}, rtc {health.get("rtc")}, error {health.get("rtc_error")})')

    async def _serve(self, route):
        name = Path(route.request.url.split('?', 1)[0]).name
        path = VENDOR / name
        if name and path.is_file():
            await route.fulfill(status=200, body=path.read_bytes(), content_type=CONTENT_TYPES.get(path.suffix, 'application/octet-stream'))
        else:
            await route.fulfill(status=404, body=b'not found')

    async def _open(self, credentials):
        await self._close_page()
        self.frames.clear()  # the ride is another channel: nothing from the previous page may count as fresh
        self.telemetry.clear()
        config = {'appid': credentials.app_id, 'channel': credentials.channel, 'uid': int(credentials.user_id),
                  'rtcToken': credentials.rtc_token, 'rtmToken': credentials.rtm_token, 'botUid': credentials.bot_uid,
                  'codec': self.codec}
        self._context = await self._browser.new_context(viewport={'width': 1280, 'height': 720})
        self._page = await self._context.new_page()
        self._page.on('console', lambda message: self.errors.append(message.text[:200]) if message.type == 'error' else None)
        await self._page.expose_function('roverReport', self._on_report)
        await self._page.route(PAGE_ORIGIN + '/*', self._serve)
        await self._page.add_init_script(f'window.ROVER_CONFIG = {json.dumps(config)};')
        await self._page.goto(PAGE_ORIGIN + '/rover.html', wait_until='domcontentloaded')
        await self._page.wait_for_function('() => window.roverPageReady === true', timeout=15000)
        await self._page.evaluate('async () => { await window.joinRtm(); await window.joinRtc(); }')
        await self._page.wait_for_function('() => window.rtmReady === true', timeout=15000)
        self._capture_task = self.loop.create_task(self._capture_loop(self._page))

    def _on_report(self, text):
        try:
            self.telemetry.update(parse_report(text))
        except (ValueError, UnicodeDecodeError) as exc:
            self.errors.append(f'report: {type(exc).__name__}')

    async def _capture_loop(self, page):
        period = 1.0 / self.frame_hz
        while True:
            started = time.monotonic()
            for uid, camera in CAMERA_UIDS.items():
                if camera not in self.cameras:
                    continue
                try:
                    packet = await page.evaluate('(uid) => window.getFramePacket(uid, "jpeg", %s)' % self.quality, uid)
                except Exception as exc:  # page gone: the capture task ends with it
                    self.errors.append(f'capture: {type(exc).__name__}')
                    return
                if isinstance(packet, dict) and packet.get('data_url'):
                    try:
                        self.frames.put(camera, data_url_to_rgb(packet['data_url']))
                    except Exception as exc:
                        self.errors.append(f'decode: {type(exc).__name__}')
                elif isinstance(packet, dict) and packet.get('error') and self.frames.counts.get(camera, 0) == 0:
                    self.errors.append(str(packet['error'])[:200])  # only before the first decoded frame
            await asyncio.sleep(max(0.0, period - (time.monotonic() - started)))

    async def _close_page(self):
        if self._capture_task is not None:
            self._capture_task.cancel()
            self._capture_task = None
        page, self._page = self._page, None
        context, self._context = self._context, None
        if page is not None:
            try:
                await asyncio.wait_for(page.evaluate('async () => await window.leaveAll()'), 5)
            except Exception:
                pass
        if context is not None:
            try:
                await context.close()
            except Exception:
                pass

    def close_page(self):
        if self.loop is not None:
            self._call(self._close_page(), timeout=20)

    def shutdown(self):
        loop, self.loop = self.loop, None
        if loop is None:
            return

        async def stop():
            await self._close_page()
            for closer in (self._browser.close if self._browser else None, self._playwright.stop if self._playwright else None):
                if closer is not None:
                    try:
                        await closer()
                    except Exception:
                        pass
        try:
            asyncio.run_coroutine_threadsafe(stop(), loop).result(30)
        finally:
            loop.call_soon_threadsafe(loop.stop)
            self._thread.join(timeout=10)

    # --- the rover API ------------------------------------------------------
    def _payload(self, linear, angular):
        clip = lambda value: round(max(-1.0, min(1.0, float(value))), 3)
        return {'linear': clip(linear), 'angular': clip(angular), 'lamp': int(self.lamp)}

    def send(self, linear, angular):
        self._call(self._page.evaluate('(m) => window.sendMessage(m)', self._payload(linear, angular)), timeout=3)

    def send_confirmed(self, linear, angular):
        """True only when the rover acknowledged receipt (RTM hasPeerReceived), like the SDK's watchdog stop."""
        try:
            return bool(self._call(self._page.evaluate('async (m) => await window.sendMessageAwait(m)',
                                                       self._payload(linear, angular)), timeout=8))
        except Exception as exc:
            self.errors.append(f'stop: {type(exc).__name__}: {str(exc)[:120]}')
            return False

    def health(self):
        if self._page is None or self.loop is None:
            return {}
        try:
            return self._call(self._page.evaluate('() => window.rtmHealth ? window.rtmHealth() : null'), timeout=3) or {}
        except Exception:
            return {}
