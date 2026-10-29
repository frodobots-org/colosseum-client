"""Transport contract for the Earth Rover Mini+ and an in-memory simulator for tests and dry runs."""
from __future__ import annotations

import math
import threading
import time
from typing import Protocol

import numpy as np

from ...navigation import FRONT_CAMERA, NAV_STATE_NAMES, ROVER_CAMERAS


class RoverLink(Protocol):
    """What the adapter needs from a rover connection.

    ``latest_frames`` and ``latest_telemetry`` are lock-protected snapshots and
    may be called from any thread. ``send`` and ``stop`` are called only from the
    adapter's command sender thread, or after that thread has been joined.
    ``start`` and ``close`` run on the robot worker thread.
    """

    def start(self, mission_slug: str | None = None) -> None:
        """Connect to the rover; with a mission the session is a recorded ride on it."""

    def latest_frames(self) -> dict[str, tuple[np.ndarray, float]]:
        """Camera name → (RGB uint8 HxWx3, monotonic time the frame was received)."""

    def latest_telemetry(self) -> tuple[dict[str, float], float] | None:
        """(values keyed by NAV_STATE_NAMES, monotonic receive time), or None before the first report."""

    def send(self, linear: float, angular: float) -> None: ...

    def stop(self, confirm: bool = True) -> bool:
        """Send a stop; with ``confirm`` return True only once the rover acknowledged it."""

    def close(self) -> None: ...


class SimulatedRoverLink:
    """Dead-reckoning rover: synthetic frames, telemetry integrated from the commands it receives."""
    shared = False

    def __init__(self, cameras=(FRONT_CAMERA,), size=(512, 288), *, start=(37.0, -122.0),
                 stale=False, stop_ok=True, telemetry=True):
        self.cameras = [camera for camera in cameras if camera in ROVER_CAMERAS]
        self.width, self.height = size
        self.stale = stale
        self.stop_ok = stop_ok
        self.telemetry_enabled = telemetry
        self.state = {name: 0.0 for name in NAV_STATE_NAMES}
        self.state.update(latitude=float(start[0]), longitude=float(start[1]), battery=100.0)
        self.sent = []  # (linear, angular, sending thread, monotonic time)
        self.stops = 0
        self.lamp = 0
        self.started = False
        self.closed = False
        self._lock = threading.Lock()
        self._updated = time.monotonic()
        self._frame = 0

    def start(self, mission_slug=None):
        if self.started and not mission_slug:
            return  # the preview connection serves a trial without a mission
        self.started = True
        self.mission_slug = mission_slug
        self._updated = time.monotonic()

    def set_lamp(self, on):
        self.lamp = 1 if on else 0

    def end_trial(self):
        """Shared connection: stop, leave the mission, keep the preview open."""
        confirmed = self.stop(confirm=True)
        self.mission_slug = None
        if not confirmed:
            raise RuntimeError('Rover stop was not confirmed')

    def _integrate(self, linear, angular):
        now = time.monotonic()
        elapsed = max(0.0, now - self._updated)
        self._updated = now
        # Compass heading (clockwise from north); +angular turns left, like the rover.
        heading = (self.state['heading_deg'] - angular * elapsed * 90.0) % 360.0
        radians = math.radians(heading)
        self.state['latitude'] += linear * elapsed * 1e-5 * math.cos(radians)
        self.state['longitude'] += linear * elapsed * 1e-5 * math.sin(radians)
        self.state.update(heading_deg=heading, speed=linear * 1.1)
        for index in range(1, 5):
            self.state[f'rpm_{index}'] = linear * 100.0

    def _stamp(self):
        return time.monotonic() - (10.0 if self.stale else 0.0)

    def latest_frames(self):
        if not self.started:
            return {}
        with self._lock:
            self._frame += 1
            frames = {}
            for camera in self.cameras:
                image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                image[:] = (96, 140, 72) if camera == FRONT_CAMERA else (72, 96, 140)
                image[:, self._frame % self.width, :] = 255
                frames[camera] = (image, self._stamp())
            return frames

    def latest_telemetry(self):
        if not self.telemetry_enabled or not self.started:
            return None
        with self._lock:
            return dict(self.state), self._stamp()

    def send(self, linear, angular):
        with self._lock:
            self.sent.append((float(linear), float(angular), threading.get_ident(), time.monotonic()))
            self._integrate(float(linear), float(angular))

    def stop(self, confirm=True):
        with self._lock:
            self.stops += 1
            self.sent.append((0.0, 0.0, threading.get_ident(), time.monotonic()))
            self._integrate(0.0, 0.0)
        return self.stop_ok if confirm else True

    def diagnostics(self):
        started = self.started
        with self._lock:
            frame, sent = self._frame, len(self.sent)
        age = 10.0 if self.stale else 0.0
        return {'transport': 'simulated', 'started': started, 'mission': getattr(self, 'mission_slug', None),
                'channel': 'simulated' if started else None,
                'frames': {camera: {'count': frame, 'age_s': age} for camera in self.cameras},
                'telemetry': {'count': frame, 'age_s': age} if started and self.telemetry_enabled else None,
                'rtm': {'ready': started, 'dispatched': sent, 'delivered': sent, 'failed': 0,
                        'publishers': [1000, 1001] if started else []},
                'errors': []}

    def close(self):
        self.closed = True
