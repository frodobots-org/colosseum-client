"""Earth Rover Mini+: the standard Robot contract over a RoverLink (FrodoBots platform or simulator).

Safety model: the rover keeps executing its last command until a new one
arrives, so a background sender re-sends the latest command at 10 Hz and
replaces it with a stop once no new action has arrived within
``command_ttl_s``. ``close()`` sends a stop and requires the rover's receipt.
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import os
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

from ..navigation import (ACTION_SPACE, FRONT_CAMERA, GOAL_GPS_KEY, GOAL_IMAGE, NAV_STATE_KEY,
                          NAV_STATE_NAMES, ROVER_CAMERAS)
from ..robot_interface import Robot, RobotObservation
from .earth_rover import make_link, release_link

FRONT_MAX_AGE_S = 2.0
TELEMETRY_MAX_AGE_S = 3.0
SEND_HZ = 10
STOP_ATTEMPTS = 3
STOP_WAIT_S = 15.0
# A single slow REST call must not end a trial; a sustained outage must.
MAX_CONSECUTIVE_SEND_FAILURES = 5
FAULT_AFTER_S = 2.0
SETTINGS = {'bot_slug', 'mission_slug', 'checkpoint', 'frodobots_api_url', 'sdk_token', 'goal_dir', 'goal_gps', 'goal_image',
            'command_ttl_s', 'max_linear', 'max_angular', 'link', 'telemetry', 'timeout_s', 'stopped_s',
            'first_frame_timeout_s', 'transport', 'chrome_executable_path', 'codec',
            'front_max_age_s', 'telemetry_max_age_s'}


def resolve_secret(value):
    """``env:NAME`` reads the process environment, like the LLM credential convention."""
    if isinstance(value, str) and value.startswith('env:'):
        name = value[4:]
        if not name or name not in os.environ:
            raise ValueError(f'Environment variable {name or "(empty)"} is not set')
        return os.environ[name]
    return value


class CommandSender(threading.Thread):
    """The only caller of ``link.send``/``link.stop`` while the adapter is open."""

    def __init__(self, link, ttl_s, on_fault):
        super().__init__(name='earth-rover-commands', daemon=True)
        self.link = link
        self.ttl_s = ttl_s
        self.on_fault = on_fault
        self.lock = threading.Lock()
        self.command = (0.0, 0.0)
        self.stamp = time.monotonic()
        self.wake = threading.Event()
        self.stop_event = threading.Event()
        self.expiries = 0
        self.failures = 0
        self.last_success = time.monotonic()
        self._stop_request = None

    def set(self, linear, angular):
        with self.lock:
            self.command = (float(linear), float(angular))
            self.stamp = time.monotonic()
        self.wake.set()

    def confirmed_stop(self, timeout=STOP_WAIT_S):
        """Ask the sender thread for a confirmed stop and wait for the rover's receipt."""
        done = threading.Event()
        outcome = []
        with self.lock:
            self.command = (0.0, 0.0)
            self.stamp = time.monotonic()
            self._stop_request = (done, outcome)
        self.wake.set()
        return bool(done.wait(timeout) and outcome and outcome[0])

    def run(self):
        while not self.stop_event.is_set():
            expired = False
            with self.lock:
                if self.command != (0.0, 0.0) and time.monotonic() - self.stamp > self.ttl_s:
                    self.command = (0.0, 0.0)
                    expired = True
                command = self.command
                request, self._stop_request = self._stop_request, None
            if expired:
                self.expiries += 1
                print(f'Earth Rover watchdog: no action within {self.ttl_s:.1f}s; sending stop.', flush=True)
            try:
                if request is not None:
                    done, outcome = request
                    try:
                        outcome.append(bool(self.link.stop(confirm=True)))
                    finally:
                        done.set()
                else:
                    self.link.send(*command)
                self.failures = 0
                self.last_success = time.monotonic()
            except Exception as exc:
                self.failures += 1
                if self.failures >= MAX_CONSECUTIVE_SEND_FAILURES or time.monotonic() - self.last_success > FAULT_AFTER_S:
                    self.on_fault(exc)
            self.wake.wait(1 / SEND_HZ)
            self.wake.clear()


class EarthRoverMiniPlusRobot(Robot):
    robot_name = 'Earth Rover Mini+'
    active = None  # The connected adapter, for the local operator page's camera and telemetry.
    joint_count = 0
    has_gripper = False
    action_dim = 2
    action_space_name = ACTION_SPACE
    state_names = NAV_STATE_NAMES

    def __init__(self, config):
        settings = dict(config.adapter_config)
        unknown = sorted(set(settings) - SETTINGS)
        if unknown:
            raise ValueError(f'Unsupported Earth Rover adapter_config keys: {", ".join(unknown)}')
        self.cameras = list(config.cameras)
        if FRONT_CAMERA not in self.cameras or any(camera not in ROVER_CAMERAS for camera in self.cameras):
            raise ValueError('Earth Rover cameras are front_image, optionally with rear_image')
        if not 1 <= config.control_hz <= SEND_HZ:
            raise ValueError('Earth Rover control_hz must be between 1 and 10')
        self.ttl_s = float(settings.get('command_ttl_s', 1.0))
        if not 0 < self.ttl_s <= 5:
            raise ValueError('command_ttl_s must be between 0 and 5 seconds')
        self.limits = (float(settings.get('max_linear', 1.0)), float(settings.get('max_angular', 1.0)))
        if any(not 0 < limit <= 1 for limit in self.limits):
            raise ValueError('max_linear and max_angular must be in (0, 1]')
        # How old the newest front frame / telemetry may be before a trial ends: a 4G hiccup should not fail a
        # trial, but a policy must not act on a view that is seconds behind the rover.
        self.front_max_age_s = float(settings.get('front_max_age_s', FRONT_MAX_AGE_S))
        self.telemetry_max_age_s = float(settings.get('telemetry_max_age_s', TELEMETRY_MAX_AGE_S))
        if not 0 < self.front_max_age_s <= 30 or not 0 < self.telemetry_max_age_s <= 30:
            raise ValueError('front_max_age_s and telemetry_max_age_s must be between 0 and 30 seconds')
        self.size = (config.image_width, config.image_height)
        self.goal_dir = Path(settings.get('goal_dir', 'goals'))
        self.goal_image_path = settings.get('goal_image')
        self.stopped_s = 2.0
        self._goal_gps = None
        self._goal_image = None
        self._fault = None
        self._clipped = 0
        self.closed = False
        self.sender = None
        settings['sdk_token'] = resolve_secret(settings.get('sdk_token'))
        self.mission_slug = settings.get('mission_slug') or ''
        self.link = make_link(settings, cameras=self.cameras, size=self.size)
        self.started = False

    def _ensure_started(self, mission_slug=None):
        """Connect once: on the trial's mission when the task names one, else the configured or token-only session."""
        if self.started:
            return
        self.link.start(mission_slug or self.mission_slug or None)
        self.started = True
        self.sender = CommandSender(self.link, self.ttl_s, self._report_fault)
        self.sender.start()
        type(self).active = self

    def _report_fault(self, exc):
        self._fault = exc

    @property
    def goal(self):
        """The GPS destination of the current task, for the operator page's map."""
        if self._goal_gps is None:
            return None
        return {'latitude': float(self._goal_gps[0]), 'longitude': float(self._goal_gps[1])}

    def _stream_hint(self):
        """What the link saw when a stream went stale, for the session log."""
        diagnostics = getattr(self.link, 'diagnostics', None)
        if diagnostics is None:
            return ''
        try:
            info = diagnostics()
        except Exception:
            return ''
        rtm = info.get('rtm') or {}
        front = (info.get('frames') or {}).get(FRONT_CAMERA) or {}
        parts = [f"publishers {rtm.get('publishers')}", f"front frames {front.get('count')}", f"channel {info.get('channel')}"]
        if info.get('errors'):
            parts.append(f"last page error: {info['errors'][-1]}")
        return '; ' + ', '.join(parts)

    def _fit(self, pixels):
        pixels = np.asarray(pixels, dtype=np.uint8)
        if pixels.shape[:2] == (self.size[1], self.size[0]):
            return pixels
        return np.asarray(Image.fromarray(pixels).resize(self.size), dtype=np.uint8)

    def get_observation(self):
        if self.closed:
            raise RuntimeError('Earth Rover adapter is closed')
        if self._fault is not None:
            raise RuntimeError(f'Earth Rover command sender failed ({type(self._fault).__name__})')
        self._ensure_started()
        now = time.monotonic()
        frames = self.link.latest_frames()
        images = {}
        for camera in self.cameras:
            if camera not in frames:
                raise RuntimeError(f'{camera} has not been received from the rover')
            pixels, received = frames[camera]
            if camera == FRONT_CAMERA and now - received > self.front_max_age_s:
                raise RuntimeError(f'front_image is stale ({now - received:.1f}s old{self._stream_hint()})')
            images[camera] = self._fit(pixels)
        telemetry = self.link.latest_telemetry()
        if telemetry is None:
            raise RuntimeError('No rover telemetry received')
        values, received = telemetry
        if now - received > self.telemetry_max_age_s:
            raise RuntimeError(f'rover telemetry is stale ({now - received:.1f}s old{self._stream_hint()})')
        try:
            vector = np.array([float(values[name]) for name in NAV_STATE_NAMES], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError('rover telemetry lacks a navigation field') from exc
        if not np.isfinite(vector).all():
            raise RuntimeError('rover telemetry contains non-finite values')
        extra = {NAV_STATE_KEY: vector}
        if self._goal_gps is not None:
            extra[GOAL_GPS_KEY] = self._goal_gps.copy()
        if self._goal_image is not None:
            images[GOAL_IMAGE] = self._goal_image
        empty = np.zeros(0, dtype=np.float32)
        return RobotObservation(images, empty, empty.copy(), empty.copy(), extra)

    def execute(self, action):
        value = np.asarray(action, dtype=np.float64)
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError('Earth Rover actions are two finite values: [linear, angular]')
        clipped = np.clip(value, -1.0, 1.0)
        if not np.array_equal(clipped, value):
            self._clipped += 1
            if self._clipped == 1:
                print('Earth Rover action outside [-1, 1] was clipped.', flush=True)
        linear, angular = clipped * np.asarray(self.limits, dtype=np.float64)
        self._ensure_started()
        self.sender.set(float(linear), float(angular))

    def prepare_for_task(self, task):
        goal = task.get('goal') or {}
        navigation = task.get('navigation') or {}
        self.stopped_s = float(navigation.get('stopped_s', 2))
        kind = goal.get('type')
        if kind == 'gps':
            latitude, longitude = float(goal['latitude']), float(goal['longitude'])
            if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
                raise ValueError('Task goal coordinates are out of range')
            self._goal_gps = np.array([latitude, longitude], dtype=np.float64)
            self._goal_image = None
        elif kind == 'image':
            name = str(goal.get('file', ''))
            if not name or Path(name).name != name:
                raise ValueError('Task goal photo must be a plain file name')
            candidates = []
            if self.goal_image_path and Path(self.goal_image_path).name == name:
                candidates.append(Path(self.goal_image_path))
            candidates.append(self.goal_dir / name)
            path = next((candidate for candidate in candidates if candidate.is_file()), None)
            if path is None:
                raise FileNotFoundError(f'Goal photo {name} not found in {self.goal_dir}')
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != str(goal.get('sha256', '')).lower():
                raise ValueError(f'Goal photo {name} does not match the task checksum')
            with Image.open(io.BytesIO(data)) as photo:
                self._goal_image = np.asarray(photo.convert('RGB').resize(self.size), dtype=np.uint8)
            self._goal_gps = None
        else:
            print('Warning: this task has no navigation goal; the policy receives none.', flush=True)
        # The ride starts here, on the mission the task came from, like the SDK's start-mission.
        self._ensure_started(goal.get('mission_slug'))

    def _speed(self):
        telemetry = self.link.latest_telemetry()
        return None if telemetry is None else telemetry[0].get('speed')

    async def finish_trial(self, robot_call):
        """Confirmed stop, then the hold the rubric asks for; the operator sees whether the rover stayed still."""
        if self.sender is None:
            return
        confirmed = await robot_call(self.sender.confirmed_stop)
        await asyncio.sleep(self.stopped_s)
        speed = await robot_call(self._speed)
        print(f'Rover stop confirmed: {confirmed}; speed after the {self.stopped_s:.0f}s hold: {speed}', flush=True)
        if not confirmed:
            raise RuntimeError('Rover stop was not confirmed at the end of the trial')

    def close(self):
        if self.closed:
            return
        self.closed = True
        if type(self).active is self:
            type(self).active = None
        if self.sender is not None:
            self.sender.stop_event.set()
            self.sender.wake.set()
            self.sender.join(timeout=5)
        confirmed = not self.started  # Nothing was ever sent to a rover that was never connected.
        try:
            for _ in range(STOP_ATTEMPTS if self.started else 0):
                if self.link.stop(confirm=True):
                    confirmed = True
                    break
        except Exception as exc:
            print(f'Earth Rover stop failed ({type(exc).__name__})', flush=True)
        try:
            if getattr(self.link, 'shared', False):
                if self.started:
                    self.link.end_trial()  # the operator page keeps its preview connection
            else:
                self.link.close()
        except Exception as exc:
            if confirmed:
                raise
            print(f'Earth Rover link close failed ({type(exc).__name__})', flush=True)
        finally:
            if getattr(self.link, 'shared', False):
                release_link(self.link)
        if not confirmed:
            raise RuntimeError('Rover stop was not confirmed; check the rover before leaving it')
