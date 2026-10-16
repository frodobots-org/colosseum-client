import asyncio
import base64
from dataclasses import dataclass
import io
import math
import time

import httpx
import numpy as np
from PIL import Image, ImageDraw

from . import colosseum_pb2 as pb
from .tensors import tensor_from_numpy, tensor_to_numpy

PUBLIC_TELEMETRY = ('orientation', 'speed', 'battery', 'rpm_1', 'rpm_2', 'rpm_3', 'rpm_4',
                    'accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z')


@dataclass
class Sample:
    telemetry: dict
    images: dict
    captured: dict
    received: float
    telemetry_age_s: float

    def fresh(self):
        age = time.monotonic() - self.received
        return (0 <= age <= 1 and 0 <= self.telemetry_age_s + age <= 1
                and 'front_image' in self.images
                and 0 <= time.time() - self.captured['front_image'] <= 2)


def observation(sample, job, step):
    goal = job['goal']
    fields = list(PUBLIC_TELEMETRY)
    if goal['navigation']['goal_type'] == 'gps':
        fields += ['latitude', 'longitude']
    state = {}
    for field in fields:
        value = sample.telemetry.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            state[field] = tensor_from_numpy(np.asarray([value], dtype=np.float64))
    if goal['navigation']['goal_type'] == 'gps':
        state['goal_gps'] = tensor_from_numpy(np.asarray([goal['latitude'], goal['longitude']], dtype=np.float64))
    sensors = []
    for name, data in sample.images.items():
        if name not in {'front_image', 'rear_image'}:
            continue
        with Image.open(io.BytesIO(data)) as photo:
            width, height = photo.size
        sensors.append(pb.Image(sensor_id=name, encoding=pb.JPEG, capture_time_ns=int(sample.captured[name] * 1e9),
                                width=width, height=height, data=data))
    if goal['navigation']['goal_type'] == 'image':
        raw = base64.b64decode(goal['image_base64'])
        with Image.open(io.BytesIO(raw)) as photo:
            width, height = photo.size
        sensors.append(pb.Image(sensor_id='goal_image', encoding=pb.JPEG, width=width, height=height, data=raw))
    return pb.Observation(robot_time_ns=time.time_ns(), control_step=step,
                          instruction=goal['instruction'], state=state, sensors=sensors)


class ActionBuffer:
    def __init__(self, linear_limit, angular_limit):
        if not (0 < linear_limit <= 1 and 0 < angular_limit <= 1):
            raise ValueError('Qualified command limits must be in (0, 1]')
        self.limits = np.asarray([linear_limit, angular_limit])
        self.actions = np.zeros((0, 2))
        self.received = 0.0
        self.done = False
        self.expiries = 0
        self._expired = True

    def accept(self, plan, now):
        actions = tensor_to_numpy(plan.actions)
        if (actions.ndim != 2 or actions.shape[1] != 2 or not 1 <= len(actions) <= 10
                or not np.all(np.isfinite(actions)) or plan.control_hz != 10
                or (plan.done and np.any(actions))):
            raise ValueError('Invalid rover action plan')
        self.actions = np.clip(actions, -self.limits, self.limits)
        self.received = now
        self.done = plan.done
        self._expired = False

    def current(self, now):
        age = now - self.received
        index = int(max(0, age) * 10)
        if self.done or age >= 1 or index >= len(self.actions):
            if not self._expired and not self.done:
                self.expiries += 1
                self._expired = True
            return (0.0, 0.0)
        return tuple(float(value) for value in self.actions[index])


class RoverSDK:
    def __init__(self, url, bot_slug):
        self.http = httpx.AsyncClient(base_url=url.rstrip('/'), timeout=2)
        self.bot_slug = bot_slug
        self.identity = {}
        self.version = ''
        self.rear_camera = False
        self.simulated = False
        self.acquired = False

    async def start(self):
        status = (await self.http.get('/status')).json()
        if status.get('configured_bot_slug') != self.bot_slug:
            raise ValueError('SDK bot does not match the explicitly configured rover')
        if not status.get('configured_mission_slug'):
            raise ValueError('Configure an SDK-enabled evaluation mission first')
        if not 0 < status.get('control_watchdog_s', 0) <= 1:
            raise ValueError('SDK CONTROL_WATCHDOG_S must be between 0 and 1 second')
        response = await self.http.post('/start-mission', params={'evaluation': 'true'}, timeout=60)
        response.raise_for_status()
        self.acquired = True
        self.identity = response.json().get('identity') or {}
        if self.identity.get('bot_slug') != self.bot_slug or not all(self.identity.get(key) for key in ('bot_id', 'ride_id', 'device_ref_id')):
            raise ValueError('Backend/SDK must report exact bot and ride identity')
        if self.identity.get('sdk_user_admin') or not self.identity.get('recording_enabled') or not self.identity.get('recording_started'):
            raise ValueError('Use a non-admin SDK account and a recording-enabled ride')
        self.version = status.get('sdk_version', '')
        if not await self.stop():
            raise ValueError('Preflight stop was not confirmed')

    async def sample(self):
        status, data, front, rear = await asyncio.gather(
            self.http.get('/status'), self.http.get('/data'), self.http.get('/v2/front'), self.http.get('/v2/rear'))
        for response in (status, data, front):
            response.raise_for_status()
        telemetry = data.json()
        if not isinstance(telemetry, dict):
            raise ValueError('Invalid SDK telemetry')
        images, captured = {}, {}
        for name, response in [('front', front), ('rear', rear)]:
            if response.is_success:
                packet = response.json()
                images[name + '_image'] = base64.b64decode(packet[name + '_frame'].split(',')[-1], validate=True)
                captured[name + '_image'] = float(packet['timestamp'])
        self.rear_camera = 'rear_image' in images
        age = status.json().get('telemetry_age_s')
        return Sample(telemetry, images, captured, time.monotonic(), float(age) if age is not None else math.inf)

    async def control(self, linear, angular):
        response = await self.http.post('/control', json={'command': {'linear': linear, 'angular': angular}}, timeout=0.5)
        response.raise_for_status()

    async def stop(self):
        response = await self.http.post('/stop', timeout=15)
        response.raise_for_status()
        return response.json().get('stop_confirmed') is True

    async def end(self):
        response = await self.http.post('/end-mission', timeout=20)
        response.raise_for_status()
        self.acquired = False


class SimulatedRover:
    def __init__(self, bot_slug='simulated-mini'):
        self.bot_slug = bot_slug
        self.identity = dict(bot_slug=bot_slug, bot_id=1, device_ref_id='frodobot_demo01', ride_id=1,
                             recording_enabled=True, recording_started=True, sdk_user_admin=False)
        self.version = 'simulator-1'
        self.simulated = True
        self.rear_camera = False
        self.linear = self.angular = self.heading = 0.0
        canvas = Image.new('RGB', (640, 360), '#dce3d0')
        drawing = ImageDraw.Draw(canvas)
        drawing.polygon([(220, 160), (420, 160), (600, 360), (40, 360)], fill='#7b8872')
        drawing.rectangle((270, 100, 370, 160), fill='#446235')
        drawing.text((20, 20), 'SIMULATED ROVER - SOFTWARE DEMO', fill='#172017')
        output = io.BytesIO()
        canvas.save(output, format='JPEG')
        self.frame = output.getvalue()

    async def start(self):
        return None

    async def sample(self):
        self.heading = (self.heading - self.angular * 5) % 360
        return Sample(dict(orientation=self.heading, speed=self.linear, battery=90,
                           latitude=0.0, longitude=0.0, rpm_1=self.linear * 50, rpm_2=self.linear * 50,
                           rpm_3=self.linear * 50, rpm_4=self.linear * 50),
                      {'front_image': self.frame}, {'front_image': time.time()}, time.monotonic(), 0)

    async def control(self, linear, angular):
        self.linear, self.angular = linear, angular

    async def stop(self):
        self.linear = self.angular = 0.0
        return True

    async def end(self):
        await self.stop()
