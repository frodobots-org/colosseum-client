"""Remote (Router-relayed) trials: synthetic robots use the real contract, JPEG and keepalive options."""
import asyncio
import io
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from colosseum_client import colosseum_pb2 as pb
from colosseum_client import evaluation as e
from colosseum_client.client import ColosseumClient
from colosseum_client.robot_config import RobotClientConfig
from colosseum_client.robot_runner import protobuf_observation
from colosseum_client.robot_interface import RobotObservation
from colosseum_client.tensors import tensor_from_numpy, tensor_to_numpy

YAM_CAMERAS = {'head_image': 'top', 'left_image': 'left', 'right_image': 'right'}


class FakeClient:
    def __init__(self, action_dim):
        self.action_dim, self.observations = action_dim, []

    async def connect(self, **kwargs):
        return SimpleNamespace(action_spaces=['joint_position'])

    async def infer(self, observation, **kwargs):
        self.observations.append(observation)
        step = observation.control_step
        actions = np.full((4, self.action_dim), 0.5, np.float32)
        return pb.ActionPlan(start_step=step, valid_until_step=step + 3, actions=tensor_from_numpy(actions))

    async def close(self):
        pass


def run_remote_test_trial(tmp_path, monkeypatch, **overrides):
    created = {}
    client = FakeClient(overrides.pop('action_dim', 14))

    def make_client(*args, **kwargs):
        created.update(kwargs)
        return client

    monkeypatch.setattr(e, 'ColosseumClient', make_client)
    config = RobotClientConfig(**{'url': 'ws://localhost', 'token': 't', 'cameras': YAM_CAMERAS, 'robot_type': 'yam',
                                  'test': True, 'control_hz': 30, 'recording': False, 'skip_upload': True,
                                  'evaluation_dir': str(tmp_path), 'image_width': 64, 'image_height': 48,
                                  **overrides})
    api = SimpleNamespace(request=lambda *args: None)
    task = {'instruction': 'pick up the cup', 'max_steps': 8}
    asyncio.run(e.run_trial(config, {'task': task}, {'id': 'run-1'}, tmp_path / 'run-1', api))
    return created, client


def test_synthetic_yam_announces_the_real_contract_in_remote_trials(tmp_path, monkeypatch):
    created, client = run_remote_test_trial(tmp_path, monkeypatch)
    assert (created['robot_type'], created['joint_count'], created['action_spaces'], created['control_hz']) == (
        'yam', 12, {'joint_position': 14}, 30)
    first = client.observations[0]
    assert tensor_to_numpy(first.state['joint_position']).shape == (12,)
    assert tensor_to_numpy(first.state['gripper_position']).shape == (2,)
    assert {s.sensor_id for s in first.sensors} == set(YAM_CAMERAS)
    assert all(s.encoding == pb.RAW_RGB for s in first.sensors)  # default stays RAW


def test_synthetic_yam_requires_the_real_control_rate(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='control_hz: 30'):
        run_remote_test_trial(tmp_path, monkeypatch, control_hz=15)


def test_remote_trials_send_jpeg_and_forward_keepalive(tmp_path, monkeypatch):
    created, client = run_remote_test_trial(tmp_path, monkeypatch, image_encoding='jpeg', jpeg_quality=80,
                                            policy_ping_interval=30, policy_ping_timeout=120)
    assert (created['ping_interval'], created['ping_timeout']) == (30, 120)
    sensor = client.observations[0].sensors[0]
    assert sensor.encoding == pb.JPEG
    assert Image.open(io.BytesIO(sensor.data)).size == (sensor.width, sensor.height) == (64, 48)


def test_jpeg_observation_is_much_smaller_than_raw():
    rng = np.random.default_rng(0)
    smooth = np.clip(np.linspace(0, 255, 640)[None, :, None] + rng.normal(0, 4, (480, 640, 3)), 0, 255).astype(np.uint8)
    current = RobotObservation({'head_image': smooth}, np.zeros(12, np.float32), np.zeros(2, np.float32),
                               np.zeros(6, np.float32))
    raw = protobuf_observation(current, instruction='x', control_step=0)
    jpeg = protobuf_observation(current, instruction='x', control_step=0, image_encoding='jpeg')
    assert jpeg.ByteSize() * 5 < raw.ByteSize()
    decoded = np.asarray(Image.open(io.BytesIO(jpeg.sensors[0].data)))
    assert np.abs(decoded.astype(int) - smooth).mean() < 10


@pytest.mark.parametrize('values, message', [
    ({'image_encoding': 'webp'}, 'image_encoding must be raw or jpeg'),
    ({'jpeg_quality': 0}, 'jpeg_quality must be an integer'),
])
def test_image_options_are_validated(tmp_path, values, message):
    path = tmp_path / 'robot.yaml'
    import yaml
    path.write_text(yaml.safe_dump({'router_url': 'ws://localhost', 'token': 't', 'robot_type': 'yam', 'test': True,
                                    'cameras': YAM_CAMERAS, **values}))
    with pytest.raises(ValueError, match=message):
        RobotClientConfig.from_yaml(path)


def test_client_keepalive_defaults_tolerate_slow_uplinks():
    client = ColosseumClient('ws://localhost', 't', '')
    assert (client.ping_interval, client.ping_timeout) == (20, 60)
