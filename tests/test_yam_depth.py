"""YAM streams aligned RealSense depth to the Policy Server only when CaP-X asks for it."""
import asyncio
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from colosseum_client import RobotClientConfig, make_robot
from colosseum_client.local_policy import LocalPolicyClient
from colosseum_client.robot_runner import protobuf_observation
from colosseum_client.robots import yam


def config(**adapter):
    return RobotClientConfig(url='ws://localhost', token='test', robot_type='yam', control_hz=30,
        cameras=dict(zip(yam.CAMERAS, ('top', 'left', 'right'))),
        adapter_config=dict(left_channel='can_left', right_channel='can_right',
                            left_gripper_limits=[0, 1], right_gripper_limits=[0, 1],
                            joint_low=[-3]*12, joint_high=[3]*12, joint_max_step=[.1]*12, **adapter))


@pytest.fixture
def hardware(monkeypatch):
    opened = []

    class Camera:
        def __init__(self, serial, timeout, depth=False):
            self.serial, self.depth, self.closed, self.reads = serial, depth, 0, 0
            opened.append(self)

        def read(self):
            self.reads += 1
            return np.full((480, 640, 3), self.reads, np.uint8)

        def read_depth(self):
            assert self.depth
            return np.full((480, 640), .001 * self.reads, np.float32), np.eye(3, dtype=np.float32)

        def close(self):
            self.closed += 1

    class Arm:
        def __init__(self, channel, limits):
            self.closed = 0
            self.position = np.r_[np.zeros(6), .5]

        def get_joint_pos(self):
            return self.position.copy()

        def close(self):
            self.closed += 1

    monkeypatch.setattr(yam, '_RealSenseCamera', Camera)
    monkeypatch.setattr(yam, '_open_arm', Arm)
    return opened


def test_depth_is_off_unless_the_policy_server_asks(hardware):
    robot = make_robot(config())
    obs = robot.get_observation()

    assert obs.extra_state == {}
    assert [camera.depth for camera in hardware] == [False, False, False]
    robot.close()


@pytest.mark.parametrize('role', ['head_image', 'right_image'])
def test_depth_camera_streams_aligned_depth_with_its_color_frame(hardware, role):
    robot = make_robot(config(**({} if role == 'head_image' else {'depth_camera': role})),
                       depth=True)
    obs = robot.get_observation()

    assert [camera.depth for camera in hardware] == [name == role for name in yam.CAMERAS]
    assert set(obs.extra_state) == {f'{role}_depth_m', f'{role}_intrinsics'}
    depth = obs.extra_state[f'{role}_depth_m']
    assert depth.shape == obs.images[role].shape[:2] and depth.dtype == np.float32
    # The depth belongs to the same read as the color image of this observation.
    np.testing.assert_allclose(depth, .001 * obs.images[role][0, 0, 0])
    message = protobuf_observation(obs, instruction='hand over', control_step=0)
    assert message.state[f'{role}_depth_m'].shape == [480, 640]
    assert message.state[f'{role}_intrinsics'].shape == [3, 3]
    robot.close()


def test_invalid_depth_camera_opens_no_hardware(hardware):
    with pytest.raises(ValueError, match='depth_camera must be one of'):
        make_robot(config(depth_camera='wrist'), depth=True)
    assert hardware == []


def test_realsense_depth_is_aligned_scaled_and_kept_with_its_frameset(monkeypatch):
    events = []

    class Frame:
        def __init__(self, data):
            self.data = data

        def __bool__(self):
            return self.data is not None

        def get_data(self):
            return self.data

    class Frames:
        def __init__(self, aligned):
            self.aligned = aligned

        def get_color_frame(self):
            return Frame(np.full((480, 640, 3), 7, np.uint8))

        def get_depth_frame(self):
            return Frame(np.full((480, 640), 2000 if self.aligned else 1, np.uint16))

    class Pipeline:
        def start(self, settings):
            events.append(('start', tuple(settings.streams)))
            intrinsics = SimpleNamespace(fx=600.0, fy=601.0, ppx=320.5, ppy=240.5)
            stream = SimpleNamespace(as_video_stream_profile=lambda: SimpleNamespace(
                get_intrinsics=lambda: intrinsics))
            sensor = SimpleNamespace(get_depth_scale=lambda: .001)
            return SimpleNamespace(get_device=lambda: SimpleNamespace(first_depth_sensor=lambda: sensor),
                                   get_stream=lambda kind: stream)

        def wait_for_frames(self, timeout):
            return Frames(aligned=False)

        def stop(self):
            events.append('stop')

    class Config:
        def __init__(self):
            self.streams = []

        def enable_device(self, serial):
            pass

        def enable_stream(self, kind, *args):
            self.streams.append(kind)

    class Align:
        def __init__(self, to):
            events.append(('align', to))

        def process(self, frames):
            return Frames(aligned=True)

    rs = SimpleNamespace(pipeline=Pipeline, config=Config, align=Align,
                         stream=SimpleNamespace(color='color', depth='depth'),
                         format=SimpleNamespace(rgb8='rgb8', z16='z16'))
    monkeypatch.setitem(sys.modules, 'pyrealsense2', rs)

    plain = yam._RealSenseCamera('top', 1000)
    camera = yam._RealSenseCamera('top', 1000, depth=True)
    with pytest.raises(RuntimeError, match='before the first camera read'):
        camera.read_depth()
    image = camera.read()
    depth, intrinsics = camera.read_depth()

    assert events[:3] == [('start', ('color',)), ('start', ('color', 'depth')), ('align', 'color')]
    assert image.shape == (480, 640, 3) and plain.read().shape == (480, 640, 3)
    np.testing.assert_allclose(depth, 2.0)  # aligned frame, z16 * depth scale
    np.testing.assert_allclose(intrinsics, [[600, 0, 320.5], [0, 601, 240.5], [0, 0, 1]])


async def test_yam_capx_ready_message_enables_depth(monkeypatch):
    from colosseum_client import local_policy
    from colosseum_client.local_protocol import encode_control

    cfg = RobotClientConfig(url='ws://router', token='token', cameras={}, robot_type='yam',
        policy_server_url='ws://127.0.0.1:8000', control_hz=30,
        llm_api_keys={'anthropic': 'secret'})
    c = LocalPolicyClient(cfg, 'http://router', robot_type='yam', joint_count=12,
        has_gripper=True, control_hz=30, action_spaces={'joint_position': 14})
    model = dict(model_type='llm', name='claude-opus-5-5-capx', url='https://api.yhlxj.ai',
        revision='rig-v1', action_space='joint_position', action_dim=14, control_hz=30,
        max_horizon=150)
    preparation = dict(type='prepare', protocol_version=1, run_id='run-1',
        preparation_id='prep', model=model, task={'instruction': 'Hand over the cube'}, test=False)
    stop = asyncio.Event()

    class Router:
        async def recv(self):
            if not hasattr(self, 'sent_prepare'):
                self.sent_prepare = True
                return encode_control(preparation)
            await stop.wait()

        async def send(self, value):
            pass

    class Policy:
        async def recv(self):
            if not hasattr(self, 'ready_sent'):
                self.ready_sent = True
                return encode_control({**preparation, 'type': 'ready', 'loaded': True,
                    'control_mode': 'joint_pos', 'internal_action_dim': 14,
                    'message': json.dumps({'observation_extras': ['depth']})})
            await stop.wait()

        async def send(self, value):
            pass

    connections = iter([Router(), Policy()])

    async def connect(*args, **kwargs):
        return next(connections)

    monkeypatch.setattr(local_policy, 'connect', connect)
    try:
        await c.connect(evaluation_run='run-1')
        assert c.observation_extras == ('depth',)
        assert c.native_control_mode == 'joint_pos' and c.native_action_dim == 14
    finally:
        tasks = [task for task in (c.receiver_task, c.reporting_task, c.heartbeat_task) if task]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
