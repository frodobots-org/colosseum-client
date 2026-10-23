"""Telemetry receivers. The rover reports over RTM 1.x peer messages to the ride user.

Which receiver reaches a real rover is settled by ``scripts/earth_rover_link_check.py``
(Phase 0): ``RtmTelemetry`` uses Agora's native Python RTM client, ``ShadowTelemetry``
polls the backend's device shadow (GPS only; spike use, not valid for a trial).
"""
from __future__ import annotations

import json
import math
import threading
import time

from ...navigation import NAV_STATE_NAMES


def parse_report(message) -> dict[str, float]:
    """Flatten one rover JSON report into the navigation state fields; raise if any is missing."""
    text = message if isinstance(message, str) else bytes(message).decode('utf-8')
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError('rover report is not a JSON object')
    rpms = data.get('rpms')
    latest = rpms[-1] if isinstance(rpms, list) and rpms and isinstance(rpms[-1], (list, tuple)) else ()
    values = {'latitude': data.get('latitude'), 'longitude': data.get('longitude'),
              'heading_deg': data.get('orientation'), 'speed': data.get('speed'), 'battery': data.get('battery')}
    for index in range(1, 5):
        values[f'rpm_{index}'] = latest[index - 1] if len(latest) >= index else data.get(f'rpm_{index}')
    result = {}
    for name in NAV_STATE_NAMES:
        value = values.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'rover report lacks a finite {name}')
        result[name] = float(value)
    return result


class TelemetryStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._latest = None
        self.received = 0

    def update(self, values):
        with self._lock:
            self._latest = (dict(values), time.monotonic())
            self.received += 1

    def latest(self):
        with self._lock:
            return None if self._latest is None else (dict(self._latest[0]), self._latest[1])

    def clear(self):
        with self._lock:
            self._latest = None
            self.received = 0


class RtmTelemetry:
    """Receive the rover's peer messages with Agora's native RTM client, logged in as the ride user."""

    def __init__(self, credentials, *, log_path='agora_rtm.log'):
        self.credentials = credentials
        self.store = TelemetryStore()
        self.client = None
        self.log_path = log_path
        self.login_code = None
        self.errors = []

    def start(self):
        try:
            from agora.rtm.rtm_base import RtmConfig, RtmLogConfig
            from agora.rtm.rtm_client import create_rtm_client
            from agora.rtm.rtm_event_handler import IRtmEventHandler
        except ImportError as exc:
            raise RuntimeError("The Earth Rover link needs the 'earth-rover' extra: uv sync --extra earth-rover") from exc
        receiver = self

        class Handler(IRtmEventHandler):
            def on_login_result(self, request_id, error_code):
                receiver.login_code = error_code

            def on_message_event(self, event):
                if str(event.publisher) != receiver.credentials.bot_uid:
                    return
                try:
                    receiver.store.update(parse_report(event.message))
                except (ValueError, UnicodeDecodeError) as exc:
                    receiver.errors.append(type(exc).__name__)

        config = RtmConfig(app_id=self.credentials.app_id, user_id=str(self.credentials.user_id))
        config.log_config = RtmLogConfig(file_path=self.log_path, file_size_kb=1024)
        self._handler = Handler()
        config.event_handler = self._handler
        self.client = create_rtm_client(config)
        result = self.client.login(self.credentials.rtm_token)
        code = result[0] if isinstance(result, tuple) else result
        if code:
            raise RuntimeError(f'Agora RTM login request failed ({code})')

    def latest(self):
        return self.store.latest()

    def close(self):
        client, self.client = self.client, None
        if client is None:
            return
        for call in (client.logout, client.release):
            try:
                call()
            except Exception:
                pass


class ShadowTelemetry:
    """Poll the backend's device shadow for GPS. Spike-only: no heading, speed or RPM."""

    def __init__(self, session, *, period_s=1.0):
        self.session = session
        self.period_s = period_s
        self.store = TelemetryStore()
        self.errors = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name='earth-rover-shadow', daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                position = self.session.shadow_gps()
                values = {name: math.nan for name in NAV_STATE_NAMES}
                values.update(position)
                self.store.update(values)
            except Exception as exc:
                self.errors.append(type(exc).__name__)
            self._stop.wait(self.period_s)

    def latest(self):
        return self.store.latest()

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2)
