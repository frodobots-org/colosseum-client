"""Local operator page for Earth Rover trials.

It runs the same evaluation flow as ``colosseum-robot`` in a background thread;
every question the terminal would ask is shown as a form on the page, with the
rover's live camera, telemetry, a map, manual drive controls for positioning the
rover between trials, an abort button and the submitted results. Serves on
localhost only, like the Earth Rovers SDK dashboard.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import queue
import sys
import threading
import time
import webbrowser
from collections import deque
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx

from . import evaluation
from .navigation import FRONT_CAMERA, ROVER_TYPE
from .robot_config import RobotClientConfig
from .robots.earth_rover import make_link, use_shared_link

STOP_SENTINEL = object()
MANUAL_HZ = 10
MANUAL_TTL_S = 0.5  # the page streams a held key at 10 Hz; silence means the page is gone
HALT_WAIT_S = 15.0
STATIC = Path(__file__).parent / 'rover_ui_static'
VENDOR = Path(__file__).parent / 'robots' / 'earth_rover' / 'vendor'  # the SDK's Agora web client, shared with the link
STATIC_ROOTS = {'vendor': VENDOR}
STATIC_TYPES = {'.js': 'application/javascript', '.css': 'text/css', '.png': 'image/png'}


class ManualDrive(threading.Thread):
    """The page's d-pad, with the SDK dashboard's rules: re-send the held command at 10 Hz,
    exactly one zero on release, and a zero when the page goes silent.

    The only caller of ``link.send``/``link.stop`` while no trial adapter holds the link.
    """

    def __init__(self, link, ttl_s):
        super().__init__(name='colosseum-manual-drive', daemon=True)
        self.link = link
        self.ttl_s = ttl_s
        self.lock = threading.Lock()
        self.command = (0.0, 0.0)
        self.stamp = time.monotonic()
        self.pending = False
        self.wake = threading.Event()
        self.stop_event = threading.Event()
        self.error = ''
        self.expiries = 0
        self._halt = None

    def set(self, linear, angular):
        with self.lock:
            self.command = (float(linear), float(angular))
            self.stamp = time.monotonic()
            self.pending = True
        self.wake.set()

    def halt(self, timeout=HALT_WAIT_S):
        """A confirmed stop (the rover's receipt), like the SDK's STOP button."""
        done, outcome = threading.Event(), []
        with self.lock:
            self.command = (0.0, 0.0)
            self.pending = False
            self._halt = (done, outcome)
        self.wake.set()
        return bool(done.wait(timeout) and outcome and outcome[0])

    def run(self):
        while not self.stop_event.is_set():
            with self.lock:
                moving = self.command != (0.0, 0.0)
                if moving and time.monotonic() - self.stamp > self.ttl_s:
                    self.command, self.pending, moving = (0.0, 0.0), True, False
                    self.expiries += 1
                send, self.pending = self.pending or moving, False
                command = self.command
                halt, self._halt = self._halt, None
            try:
                if halt is not None:
                    done, outcome = halt
                    try:
                        outcome.append(bool(self.link.stop(confirm=True)))
                    finally:
                        done.set()
                elif send:
                    self.link.send(*command)
                self.error = ''
            except Exception as exc:
                self.error = f'{type(exc).__name__}: {exc}'[:160]
            self.wake.wait(1 / MANUAL_HZ)
            self.wake.clear()

    def close(self):
        """End the thread; a rover that may still be moving gets one more zero after the join."""
        with self.lock:
            moving = self.command != (0.0, 0.0)
            self.command, self.pending, self._halt = (0.0, 0.0), False, None
        self.stop_event.set()
        self.wake.set()
        self.join(timeout=5)
        if moving:
            try:
                self.link.send(0.0, 0.0)
            except Exception as exc:
                self.error = f'{type(exc).__name__}: {exc}'[:160]


class Operator:
    """Bridges the evaluation flow's prompts and output to the page."""

    def __init__(self, config):
        self.config = config
        self.lock = threading.Lock()
        self.pending = None
        self.answers = queue.Queue()
        self.log = deque(maxlen=500)
        self.thread = None
        self.error = ''
        self.phase = 'idle'
        self.counter = 0
        self.link = None
        self.link_phase = 'disconnected'
        self.link_error = ''
        self.bot = config.adapter_config.get('bot_slug', '')
        self.manual = None
        self.claimed = False  # a trial adapter holds the shared link
        self.lamp = 0
        self.last_drive = (0.0, 0.0)
        self._router = (0.0, None)
        self.events = deque(maxlen=200)  # timestamped page-level events: connect, ride, trial, failures
        self._diag = (0.0, None)

    def event(self, text):
        stamp = time.strftime('%H:%M:%S')
        self.events.append(f'{stamp} {text}')
        print(f'[{stamp}] {text}', flush=True)

    def diagnostics(self):
        """The link's stream health, refreshed at most every 2 s (the browser transport asks its headless page)."""
        link = self.link if self.link is not None else self._live_link()
        if link is None or not hasattr(link, 'diagnostics'):
            return None
        checked, cached = self._diag
        if cached is not None and time.monotonic() - checked < 2.0:
            return cached
        try:
            cached = link.diagnostics()
        except Exception as exc:
            cached = {'error': f'{type(exc).__name__}: {exc}'[:200]}
        self._diag = (time.monotonic(), cached)
        return cached

    def _diag_summary(self):
        self._diag = (0.0, None)
        info = self.diagnostics() or {}
        front = (info.get('frames') or {}).get(FRONT_CAMERA) or {}
        rtm = info.get('rtm') or {}
        parts = [f"channel {info.get('channel')}", f"streaming {info.get('started')}",
                 f"front frames {front.get('count')} (newest {front.get('age_s')}s)",
                 f"telemetry {(info.get('telemetry') or {}).get('age_s')}s", f"publishers {rtm.get('publishers')}",
                 f"commands {rtm.get('delivered')}/{rtm.get('dispatched')} delivered, {rtm.get('failed')} failed"]
        if info.get('errors'):
            parts.append(f"page errors {info['errors'][-3:]}")
        return ', '.join(parts)

    def diagnostics_dump(self):
        """Everything worth pasting into a bug report (the page's Copy diagnostics button)."""
        self._diag = (0.0, None)
        link = self._live_link()
        spectator = getattr(link, 'spectator', None)
        video = spectator() if spectator is not None else None
        return {'when': time.strftime('%Y-%m-%d %H:%M:%S'), 'bot': self.bot, 'phase': self.phase, 'error': self.error,
                'link_phase': self.link_phase, 'link_error': self.link_error, 'trial_holds_link': self.claimed,
                'video_channel': video.get('channel') if video else None, 'diagnostics': self.diagnostics(),
                'events': list(self.events), 'log': list(self.log)[-40:]}

    # --- prompts and output -------------------------------------------------
    def ask(self, prompt, options=None):
        with self.lock:
            self.counter += 1
            self.pending = {'id': self.counter, 'prompt': prompt.strip(), 'options': None if not options else [
                {'value': option[0], 'label': option[1]} if isinstance(option, tuple) else {'value': option, 'label': option}
                for option in options]}
        try:
            answer = self.answers.get()
        finally:
            with self.lock:
                self.pending = None
        if answer is STOP_SENTINEL:
            raise evaluation.SessionStopped('session stopped by the operator')
        return answer

    def answer(self, value):
        with self.lock:
            if self.pending is None:
                return False
        self.answers.put(str(value))
        return True

    def write(self, text):
        for line in text.splitlines():
            if line.strip():
                self.log.append(line.rstrip())
        sys.__stdout__.write(text)

    def flush(self):
        sys.__stdout__.flush()

    # --- rover connection (the SDK's connect-first step) --------------------
    def connect(self, bot):
        if self.link is not None or self.link_phase == 'connecting':
            raise RuntimeError('Already connected; disconnect first')
        if self.running():
            raise RuntimeError('Stop the session before connecting another rover')
        from .robots.earth_rover_mini_plus import resolve_secret
        settings = {**self.config.adapter_config, 'bot_slug': bot}
        if settings.get('link', 'frodobots') == 'frodobots':  # the simulator never talks to FrodoBots
            settings['sdk_token'] = resolve_secret(settings.get('sdk_token'))
        settings.pop('mission_slug', None)
        self.bot = bot
        self.link_phase = 'connecting'
        self.link_error = ''

        def worker():
            try:
                link = make_link(settings, cameras=list(self.config.cameras),
                                 size=(self.config.image_width, self.config.image_height))
                link.start()  # token-only preview: live camera and telemetry, no ride yet
                use_shared_link(link, claim=self._claimed, release=self._released)
                self.link = link
                self.link_phase = 'connected'
                self.event(f'Connected to {bot} (preview, no ride)')
            except Exception as exc:
                self.link_error = f'{type(exc).__name__}: {exc}'[:300]
                self.link_phase = 'error'
                self.event(f'Connect to {bot} failed: {self.link_error}')
        threading.Thread(target=worker, name='colosseum-rover-connect', daemon=True).start()

    def disconnect(self):
        if self.running():
            raise RuntimeError('Stop the session first')
        self.release_manual()
        link, self.link = self.link, None
        use_shared_link(None)
        self.link_phase = 'disconnected'
        if link is not None:
            link.close()
            self.event('Disconnected from the rover')

    def router_ok(self):
        checked, status = self._router
        if time.monotonic() - checked > 10:
            try:
                status = httpx.get(evaluation.api_url(self.config).rstrip('/') + '/api/health', timeout=3).is_success
            except Exception:
                status = False
            self._router = (time.monotonic(), status)
        return status

    # --- manual drive (the SDK dashboard's d-pad) ---------------------------
    def _claimed(self):
        """A trial adapter takes the shared link: manual driving ends with a zero first."""
        with self.lock:
            self.claimed = True
        self.release_manual()
        self.event('Trial took the rover link (manual drive off until it ends)')

    def _released(self):
        with self.lock:
            was_claimed, self.claimed = self.claimed, False
        if was_claimed:
            self.event('Trial released the rover link')

    def _manual_link(self):
        link = self.link
        if link is None or not getattr(link, 'started', False):
            raise RuntimeError('Not connected to a rover')
        if self.claimed:
            raise RuntimeError('A trial is driving the rover; Abort trial stops it')
        return link

    def manual_allowed(self):
        try:
            self._manual_link()
        except RuntimeError:
            return False
        return True

    def _manual(self, link):
        with self.lock:
            if self.manual is None or not self.manual.is_alive():
                self.manual = ManualDrive(link, MANUAL_TTL_S)
                self.manual.start()
            return self.manual

    def drive(self, linear, angular):
        values = []
        for value in (linear, angular):
            try:
                number = float(value)
            except (TypeError, ValueError):
                number = math.nan
            if not math.isfinite(number) or not -1 <= number <= 1:
                raise ValueError('linear and angular must be numbers within [-1, 1]')
            values.append(number)
        manual = self._manual(self._manual_link())
        manual.set(*values)
        self.last_drive = tuple(values)

    def halt(self):
        """The page's STOP: a confirmed stop, or the trial abort while a trial drives the rover."""
        if self.claimed:
            self.abort_trial()
            return {'ok': True, 'aborting': True}
        manual = self._manual(self._manual_link())
        self.last_drive = (0.0, 0.0)
        return {'ok': manual.halt(), 'aborting': False}

    def set_lamp(self, on):
        link = self._manual_link()
        setter = getattr(link, 'set_lamp', None)
        if setter is None:
            raise RuntimeError('This rover link has no lamp control')
        setter(bool(on))
        self.lamp = 1 if on else 0
        self.drive(*self.last_drive)  # one command carries the new lamp bit, like the SDK
        return self.lamp

    def release_manual(self):
        with self.lock:
            manual, self.manual = self.manual, None
        self.last_drive = (0.0, 0.0)
        if manual is not None:
            manual.close()

    # --- session lifecycle --------------------------------------------------
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self, *, bot='', mission='', checkpoint='', track='', execute=True):
        if self.running():
            raise RuntimeError('A session is already running')
        if self.link is not None and bot and bot != self.bot:
            raise RuntimeError(f'Connected to {self.bot}; disconnect before using another rover')
        self._router = (0.0, None)
        if not self.router_ok():
            raise RuntimeError(f'Router {evaluation.api_url(self.config)} is not reachable: start it '
                               '(start-router.sh) or fix router_url / api_url in the config')
        settings = dict(self.config.adapter_config)
        bot = bot or self.bot
        if bot:
            settings['bot_slug'] = bot
        if mission:
            settings['mission_slug'] = mission
        else:
            settings.pop('mission_slug', None)
        if checkpoint:
            settings['checkpoint'] = int(checkpoint)
        else:
            settings.pop('checkpoint', None)
        config = replace(self.config, adapter_config=settings)
        self.error = ''
        self.log.clear()
        self.phase = 'running'
        evaluation.INTERRUPT.clear()
        evaluation.STOP.clear()
        evaluation.ask = self.ask
        self.event(f"Session started: bot {bot or '?'}, mission {mission or 'auto'}, checkpoint {checkpoint or 'last'}, "
                   f"track {track or self.config.track}, {'execute actions' if execute else 'observe only'}")

        def worker():
            try:
                with contextlib.redirect_stdout(self):
                    evaluation.run_evaluation(config, track=track or None, execute_action=execute)
                self.phase = 'finished'
                self.event('Session finished')
            except evaluation.SessionStopped:
                self.phase = 'stopped'
                self.event('Session stopped')
            except BaseException as exc:  # The page must show why the session ended.
                self.error = f'{type(exc).__name__}: {exc}'[:400]
                self.phase = 'failed'
                self.event(f'Session failed: {self.error}')
                self.event(f'Streams at failure: {self._diag_summary()}')
            finally:
                self._released()  # every trial adapter is closed by now

        self.thread = threading.Thread(target=worker, name='colosseum-evaluation', daemon=True)
        self.thread.start()

    def abort_trial(self):
        self.event('Abort trial requested')
        evaluation.INTERRUPT.set()

    def stop_session(self):
        evaluation.STOP.set()
        with self.lock:
            pending = self.pending is not None
        if pending:
            self.answers.put(STOP_SENTINEL)

    # --- what the page shows ------------------------------------------------
    def _active_robot(self):
        from .robots.earth_rover_mini_plus import EarthRoverMiniPlusRobot
        robot = EarthRoverMiniPlusRobot.active
        return robot if robot is not None and robot.started and not robot.closed else None

    def _live_link(self):
        """The connection to show: the page's preview link, or the trial adapter's own link."""
        if self.link is not None and getattr(self.link, 'started', False):
            return self.link
        robot = self._active_robot()
        return robot.link if robot is not None else None

    def frame_jpeg(self):
        link = self._live_link()
        if link is None:
            return None
        frames = link.latest_frames()
        if FRONT_CAMERA not in frames:
            return None
        from PIL import Image
        output = io.BytesIO()
        Image.fromarray(frames[FRONT_CAMERA][0]).save(output, format='JPEG', quality=80)
        return output.getvalue()

    def results(self):
        root = Path(self.config.evaluation_dir)
        rows = []
        manifests = sorted(root.glob('*/manifest.json'), key=lambda path: path.stat().st_mtime, reverse=True)
        for manifest in manifests[:20]:
            try:
                data = json.loads(manifest.read_text())
            except (ValueError, OSError):
                continue
            assignment = data.get('assignment') or {}
            rows.append({'id': assignment.get('id'), 'track': assignment.get('track'),
                         'task': (assignment.get('task') or {}).get('instruction'),
                         'goal': (assignment.get('task') or {}).get('goal'),
                         'outcomes': data.get('outcomes') or {}, 'submitted': bool(data.get('submitted')),
                         'uploaded': sorted(data.get('datasets_uploaded') or {}),
                         'when': time.strftime('%Y-%m-%d %H:%M', time.localtime(manifest.stat().st_mtime))})
        return rows

    def state(self):
        link = self._live_link()
        telemetry = None
        mission = None
        if link is not None:
            latest = link.latest_telemetry()
            if latest:
                telemetry = {name: round(value, 6) for name, value in latest[0].items()}
                telemetry['age_s'] = round(time.monotonic() - latest[1], 1)
            mission = getattr(link, 'mission_slug', None)
        robot = self._active_robot()
        spectator = getattr(link, 'spectator', None)
        with self.lock:
            pending = dict(self.pending) if self.pending else None
            manual = self.manual
        return {'phase': self.phase, 'running': self.running(), 'error': self.error, 'pending': pending,
                # The page plays the cameras itself with the ride's spectator identity, like the SDK dashboard.
                'video': spectator() if spectator is not None else None,
                'log': list(self.log)[-80:], 'connected': link is not None, 'link_phase': self.link_phase,
                'link_error': self.link_error, 'router_ok': self.router_ok(), 'telemetry': telemetry, 'mission': mission,
                'results': self.results(), 'api_url': evaluation.api_url(self.config), 'bot': self.bot,
                'track': self.config.track, 'robot_type': self.config.robot_type,
                'manual': self.manual_allowed(), 'trial': self.claimed, 'lamp': self.lamp,
                'drive': list(self.last_drive), 'drive_error': manual.error if manual is not None else '',
                'goal': robot.goal if robot is not None else None,
                'link_live': link is not None, 'diagnostics': self.diagnostics(), 'events': list(self.events)[-30:]}

    def missions(self, bot):
        settings = {**self.config.adapter_config, 'bot_slug': bot}
        session = getattr(self.link, 'session', None) if self.link is not None and bot == self.bot else None
        owned = session is None
        if owned:
            session = evaluation.rover_session(replace(self.config, adapter_config=settings))
        if session is None:
            return {'missions': [], 'note': 'Simulated or test mode: no FrodoBots missions.'}
        try:
            missions = session.missions()
            for mission in missions:
                try:
                    mission['checkpoints'] = session.checkpoints(mission['slug'])
                except Exception as exc:
                    mission['checkpoints'] = []
                    mission['error'] = type(exc).__name__
            return {'missions': missions}
        finally:
            if owned:
                session.close()


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>RoboColosseum Rover Operator</title>
<link rel="stylesheet" href="/static/leaflet/leaflet.css">
<style>
:root{--bg:#f4f6f3;--card:#fff;--ink:#17201b;--muted:#5d6a63;--line:#d9e0db;--accent:#2a5fb0;--ok:#1f6f44;--bad:#a8251d;--warn:#8a5300;--pad:#eef1ee}
*{box-sizing:border-box}[hidden]{display:none !important}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,"Segoe UI",Inter,system-ui,sans-serif}
header{padding:18px 24px;border-bottom:1px solid var(--line);background:var(--card);display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap}
header h1{font-size:18px;margin:0}header .phase{font-family:ui-monospace,Menlo,monospace;font-size:13px;color:var(--muted)}
main{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:16px;padding:16px 24px;max-width:1400px;margin:0 auto}
@media(max-width:900px){main{grid-template-columns:minmax(0,1fr)}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;display:grid;gap:10px;min-width:0;align-content:start}
.card h2{font-size:14px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);margin:0}
.card .head{display:flex;justify-content:space-between;align-items:center;gap:12px;flex-wrap:wrap}
label{display:grid;gap:4px;font-size:13px;color:var(--muted)}input,select{font:inherit;padding:8px 10px;border:1px solid var(--line);border-radius:6px;background:#fff;color:var(--ink)}
label.inline{display:flex;align-items:center;gap:6px}label.inline input{padding:0}
.row{display:flex;gap:10px;flex-wrap:wrap;align-items:end}.row>label{flex:1 1 180px}
button{font:inherit;padding:9px 14px;border-radius:6px;border:1px solid var(--line);background:#fff;color:var(--ink);cursor:pointer}
button.primary{background:var(--accent);border-color:var(--accent);color:#fff}button.danger{background:var(--bad);border-color:var(--bad);color:#fff}
button:disabled{opacity:.5;cursor:default}button.on{border-color:var(--warn);color:var(--warn);background:#fff6e5}
.prompt{border:2px solid var(--accent);background:#eef3fb;grid-column:1/-1}.prompt .q{font-weight:600}.choices{display:flex;gap:8px;flex-wrap:wrap}
.stage{position:relative}.player{position:relative;background:#1a211d;border-radius:8px;overflow:hidden;aspect-ratio:16/9}.player>div{width:100%;height:100%}
.placeholder{position:absolute;inset:0;z-index:5;display:flex;align-items:center;justify-content:center;background:#1a211d;color:#cbd6df;text-align:center;padding:0 24px}
.pip{position:absolute;right:12px;bottom:12px;width:30%;border:2px solid #fff;z-index:6;box-shadow:0 2px 8px rgba(0,0,0,.5)}
img.cam{display:block;width:100%;aspect-ratio:16/9;background:#1a211d;border-radius:8px;object-fit:contain}
table{border-collapse:collapse;width:100%;font-size:13px}td,th{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--muted);font-weight:500}
pre.log{margin:0;max-height:280px;overflow:auto;font:12px/1.5 ui-monospace,Menlo,monospace;background:#f7f8f6;border:1px solid var(--line);border-radius:6px;padding:10px;white-space:pre-wrap}
.err{color:var(--bad)}.ok{color:var(--ok)}.muted{color:var(--muted)}.chip{display:inline-block;padding:2px 8px;border-radius:999px;font-size:12px;background:#e6ebe8}
.kbd{font-family:ui-monospace,Menlo,monospace;font-size:12px;text-transform:none;letter-spacing:0}
#map{height:380px;border-radius:8px;background:#e9ede8;z-index:0}
.rover-icon{filter:drop-shadow(0 1px 3px rgba(0,0,0,.5))}.rover-icon svg{transition:transform .2s ease-out}
.cp-marker{display:flex;height:24px;border-radius:7px;overflow:hidden;border:1px solid rgba(255,255,255,.6);box-shadow:0 1px 4px rgba(0,0,0,.4);font:700 11px ui-monospace,Menlo,monospace;width:max-content}
.cp-seg{display:flex;align-items:center;justify-content:center;min-width:25px;padding:0 5px;background:#46535f;color:#dbe4ec}.cp-seg+.cp-seg{border-left:1px solid rgba(0,0,0,.35)}
.cp-seg.next{background:#ffb224;color:#10151b}.cp-marker.has-next{animation:cp-pulse 2s ease-out infinite}
@keyframes cp-pulse{0%{box-shadow:0 0 0 0 rgba(255,178,36,.7)}70%{box-shadow:0 0 0 12px rgba(255,178,36,0)}100%{box-shadow:0 0 0 0 rgba(255,178,36,0)}}
.goal-marker{width:22px;height:22px;border-radius:50%;border:3px solid var(--bad);background:rgba(168,37,29,.15);box-shadow:0 0 0 2px #fff}
.drive{display:flex;gap:20px;flex-wrap:wrap;align-items:start}
.dpad{display:grid;grid-template-columns:repeat(3,54px);grid-template-rows:repeat(3,54px);gap:6px;grid-template-areas:". up ." "left stop right" ". down ."}
.pad{border:1px solid var(--line);border-radius:10px;background:var(--pad);font-size:18px;touch-action:none;user-select:none;-webkit-user-select:none;padding:0}
.pad[data-dir=forward]{grid-area:up}.pad[data-dir=left]{grid-area:left}.pad[data-dir=right]{grid-area:right}.pad[data-dir=back]{grid-area:down}
.pad.active{background:#dbe6f8;border-color:var(--accent);color:var(--accent)}
.pad-stop{grid-area:stop;font-size:11px;font-weight:700;letter-spacing:.08em;color:var(--bad);border-color:#e3b3af}.pad-stop:not(:disabled):hover{background:#fbe9e7}
.driveSide{display:grid;gap:8px;flex:1 1 200px;min-width:0}.driveSide input[type=range]{padding:0;width:100%}
#driveStatus{font-size:13px;min-height:20px}
pre.log.small{max-height:150px}.diag td:first-child{color:var(--muted);white-space:nowrap;width:1%}.bad{color:var(--bad);font-weight:600}
</style></head><body>
<header><div><h1>RoboColosseum · Earth Rover operator</h1><div class="phase" id="phase">idle</div></div>
<div class="row"><button id="abort" class="danger" disabled>Abort trial</button><button id="stop" disabled>Stop session</button></div></header>
<main>
<section class="card" id="connect"><h2>1 · Robot and mission</h2>
<div class="row"><label>Bot slug<input id="bot" placeholder="link-node-inert"></label><button id="connectBtn" class="primary">Connect</button><button id="disconnectBtn" hidden>Disconnect</button><button id="list">List missions</button></div>
<div id="status" class="muted"></div>
<div class="row"><label>Mission<select id="mission"><option value="">— choose after listing —</option></select></label>
<label>Destination checkpoint<select id="checkpoint"><option value="">last</option></select></label></div>
<div class="row"><label>Track<select id="track"><option value="fine-tuning">Fine-tuning (one policy per trial)</option><option value="open">Open (A/B pair)</option></select></label>
<label>Rover motion<select id="execute"><option value="1">Execute actions</option><option value="0">Observe only (no motion)</option></select></label>
<button id="start" class="primary">Start session</button></div>
<div id="missionInfo" class="muted"></div></section>

<section class="card"><div class="head"><h2>Live rover</h2><div class="row"><span class="muted" id="videoState"></span><button id="audio" disabled>Unmute audio</button></div></div>
<div class="stage"><div class="player" id="front-player"><div class="placeholder" id="front-placeholder">Not connected.</div></div><div class="player pip" id="rear-player" hidden></div>
<img class="cam" id="cam" alt="front camera (snapshot fallback)" hidden></div>
<div id="telemetry" class="muted">Not connected.</div></section>

<section class="card"><div class="head"><h2>Map</h2><div class="row"><label class="inline"><input type="checkbox" id="follow" checked> Follow rover</label><button id="clearTrail">Clear trail</button></div></div>
<div id="map"></div><div id="mapInfo" class="muted">Waiting for a GPS fix…</div></section>

<section class="card"><div class="head"><h2>Drive</h2><span class="muted kbd">WASD / arrows · Space = STOP</span></div>
<div class="drive"><div class="dpad">
<button type="button" class="pad" data-dir="forward" aria-label="Forward" disabled>▲</button>
<button type="button" class="pad" data-dir="left" aria-label="Left" disabled>◀</button>
<button type="button" class="pad pad-stop" id="halt" aria-label="Stop" disabled>STOP</button>
<button type="button" class="pad" data-dir="right" aria-label="Right" disabled>▶</button>
<button type="button" class="pad" data-dir="back" aria-label="Back" disabled>▼</button></div>
<div class="driveSide"><label>Speed <b id="speedValue">30%</b><input type="range" id="speed" min="10" max="100" value="30" step="5"></label>
<button id="lamp" disabled>Lamp on</button><div id="driveStatus" class="muted"></div></div></div></section>

<section class="card"><div class="head"><h2>Diagnostics</h2><button id="copyDiag">Copy diagnostics</button></div>
<div id="diag" class="muted">Connect to a rover to see stream health.</div>
<pre class="log small" id="events"></pre></section>

<section class="card prompt" id="promptCard" hidden><h2>Waiting for you</h2><div class="q" id="promptText"></div>
<div class="choices" id="choices"></div><div class="row" id="freeText" hidden><label>Answer<input id="answer"></label><button id="send" class="primary">Send</button></div></section>

<section class="card"><h2>Results</h2><div id="results" class="muted">No results yet.</div><div class="muted">Router: <a id="admin" href="#" target="_blank">admin overview</a> · <a id="review" href="#" target="_blank">fine-tuning review API</a></div></section>

<section class="card" style="grid-column:1/-1"><h2>Session log</h2><pre class="log" id="log"></pre><div class="err" id="error"></div></section>
</main>
<script src="/static/leaflet/leaflet.js"></script>
<script src="/static/vendor/AgoraRTC_N-4.24.2.js"></script>
<script src="/static/operator.js"></script>
</body></html>
"""


class Handler(BaseHTTPRequestHandler):
    operator: Operator = None  # set by serve()

    def log_message(self, format, *args):  # keep the terminal for the evaluation log
        pass

    def _json(self, payload, status=200):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _bytes(self, body, content_type):
        self.send_response(200)
        self.send_header('Content-Type', content_type)
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _static(self, name):
        first, _, rest = name.partition('/')
        root = STATIC_ROOTS.get(first)
        if root is not None:
            name = rest
        else:
            root = STATIC
        path = (root / name).resolve()
        if not name or '\x00' in name or root.resolve() not in path.parents or not path.is_file():
            return self._json({'error': 'not found'}, 404)
        self._bytes(path.read_bytes(), STATIC_TYPES.get(path.suffix, 'application/octet-stream'))

    def do_GET(self):
        parts = urlsplit(self.path)
        if parts.path == '/':
            self._bytes(PAGE.encode(), 'text/html; charset=utf-8')
        elif parts.path.startswith('/static/'):
            self._static(parts.path[len('/static/'):])
        elif parts.path == '/api/state':
            self._json(self.operator.state())
        elif parts.path == '/api/missions':
            bot = parse_qs(parts.query).get('bot', [''])[0].strip()
            if not bot:
                return self._json({'missions': [], 'error': 'Enter a bot slug first.'})
            try:
                self._json(self.operator.missions(bot))
            except Exception as exc:
                self._json({'missions': [], 'error': f'{type(exc).__name__}: {exc}'[:300]})
        elif parts.path == '/api/diagnostics':
            self._json(self.operator.diagnostics_dump())
        elif parts.path == '/frame.jpg':
            frame = self.operator.frame_jpeg()
            if frame is None:
                return self._json({'error': 'no frame'}, 404)
            self._bytes(frame, 'image/jpeg')
        else:
            self._json({'error': 'not found'}, 404)

    def do_POST(self):
        length = int(self.headers.get('Content-Length') or 0)
        try:
            payload = json.loads(self.rfile.read(length) or b'{}')
        except ValueError:
            return self._json({'error': 'invalid JSON'}, 400)
        try:
            if self.path == '/api/start':
                self.operator.start(bot=str(payload.get('bot', '')), mission=str(payload.get('mission', '')),
                                    checkpoint=str(payload.get('checkpoint', '')), track=str(payload.get('track', '')),
                                    execute=bool(payload.get('execute', True)))
                self._json({'ok': True})
            elif self.path == '/api/answer':
                self._json({'ok': self.operator.answer(payload.get('value', ''))})
            elif self.path == '/api/connect':
                bot = str(payload.get('bot', '')).strip()
                if not bot:
                    raise ValueError('Enter a bot slug first')
                self.operator.connect(bot)
                self._json({'ok': True})
            elif self.path == '/api/disconnect':
                self.operator.disconnect()
                self._json({'ok': True})
            elif self.path == '/api/drive':
                self.operator.drive(payload.get('linear', 0), payload.get('angular', 0))
                self._json({'ok': True})
            elif self.path == '/api/halt':
                self._json(self.operator.halt())
            elif self.path == '/api/lamp':
                self._json({'ok': True, 'lamp': self.operator.set_lamp(payload.get('on', False))})
            elif self.path == '/api/abort':
                self.operator.abort_trial()
                self._json({'ok': True})
            elif self.path == '/api/stop':
                self.operator.stop_session()
                self._json({'ok': True})
            else:
                self._json({'error': 'not found'}, 404)
        except Exception as exc:
            self._json({'ok': False, 'error': f'{type(exc).__name__}: {exc}'[:300]}, 409)


def serve(config, *, host='127.0.0.1', port=8770):
    operator = Operator(config)
    handler = type('OperatorHandler', (Handler,), {'operator': operator})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server, operator


def main():
    parser = argparse.ArgumentParser(description='Local operator page for Earth Rover evaluations')
    parser.add_argument('config', nargs='?', default='configs/robot.yaml')
    parser.add_argument('--port', type=int, default=8770)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    config = RobotClientConfig.from_yaml(args.config)
    if config.robot_type != ROVER_TYPE:
        print(f'Note: {config.robot_type} prompts are shown on the page too, but the live view is for the Earth Rover.', flush=True)
    server, _ = serve(config, port=args.port)
    url = f'http://127.0.0.1:{server.server_address[1]}/'
    print(f'Operator page: {url}  (Ctrl-C to quit)', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
