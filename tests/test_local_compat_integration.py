import asyncio
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from colosseum_client import evaluation as evaluation_module, local_policy
from colosseum_client.client import ProtocolError
from colosseum_client.robot_config import RobotClientConfig
from colosseum_client.timeouts import timeout


@pytest.mark.parametrize('action_space,action_dim', [
    ('joint_position', 8), ('joint_velocity', 8), ('cartesian_position', 7),
])
def test_local_trial_constructs_robot_after_preparation_and_confirmation(
    tmp_path, monkeypatch, action_space, action_dim,
):
    events = []
    model = {
        'url': 'https://example.invalid/model', 'revision': 'a' * 40,
        'action_space': action_space, 'action_dim': action_dim,
        'control_hz': 15, 'max_horizon': 1,
    }
    dimension = action_dim

    class Client:
        def __init__(self, *args, **kwargs):
            self.model = model

        async def connect(self, **kwargs):
            events.append(('prepared', threading.get_ident()))
            assert kwargs['before_start'](model, {'max_steps': 1}) is True
            return SimpleNamespace(action_spaces=[action_space])

        async def infer(self, *args, **kwargs):
            return object()

        async def before_action(self, step):
            return None

        async def finish(self):
            events.append(('finished', threading.get_ident()))

        async def close(self):
            return None

    class Robot:
        joint_count = 7
        has_gripper = True
        action_space_name = action_space
        action_dim = dimension

        def __init__(self, config, *, action_space):
            assert action_space == model['action_space']
            assert [event[0] for event in events] == ['prepared', 'confirmed']
            events.append(('created', threading.get_ident()))

        def get_observation(self):
            events.append(('read', threading.get_ident()))
            return object()

        def execute(self, action):
            assert action.shape == (action_dim,)
            events.append(('executed', threading.get_ident()))

        def close(self):
            events.append(('closed', threading.get_ident()))

    def confirm(*args):
        events.append(('confirmed', threading.get_ident()))
        return True

    monkeypatch.setattr(local_policy, 'LocalPolicyClient', Client)
    monkeypatch.setattr(evaluation_module, 'protobuf_observation', lambda *args, **kwargs: object())
    monkeypatch.setattr(evaluation_module, 'action_chunk', lambda *args, **kwargs: np.zeros((1, action_dim)))
    config = RobotClientConfig(
        url='ws://unused', token='unused', cameras={}, policy_server_url='ws://unused', recording=False,
    )
    assignment = {'inference_mode': 'local', 'task': {'instruction': 'test', 'cameras': [], 'max_steps': 1}}
    asyncio.run(evaluation_module.run_trial(
        config, assignment, {'id': 'run'}, tmp_path, None, robot_factory=Robot, confirm_live=confirm,
    ))
    worker_events = [thread_id for name, thread_id in events if name in {'created', 'read', 'executed', 'closed'}]
    assert len(set(worker_events)) == 1
    assert worker_events[0] != threading.get_ident()


@pytest.mark.parametrize('failure', ['prepare', 'declined', 'missing_confirmation'])
def test_local_trial_never_constructs_robot_before_prepare_and_confirmation(tmp_path, monkeypatch, failure):
    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def connect(self, **kwargs):
            if failure == 'prepare':
                raise ProtocolError('prepare failed')
            if kwargs['before_start']({}, {}) is not True:
                raise ProtocolError('confirmation declined')
            pytest.fail('unexpected readiness')

        async def close(self):
            return None

    def forbidden(*args, **kwargs):
        pytest.fail('RobotEnv must not be constructed')

    monkeypatch.setattr(local_policy, 'LocalPolicyClient', Client)
    config = RobotClientConfig(
        url='ws://unused', token='unused', cameras={}, policy_server_url='ws://unused', recording=False,
    )
    assignment = {'inference_mode': 'local', 'task': {'instruction': 'test', 'cameras': [], 'max_steps': 1}}
    callback = None if failure == 'missing_confirmation' else lambda *args: False
    with pytest.raises(ProtocolError):
        asyncio.run(evaluation_module.run_trial(
            config, assignment, {'id': 'run'}, tmp_path, None, robot_factory=forbidden, confirm_live=callback,
        ))


async def test_timeout_compatibility_context_is_available():
    async with timeout(1):
        await asyncio.sleep(0)


def test_local_policy_rejects_unsupported_hardware_before_adapter_construction(tmp_path):
    config = RobotClientConfig(
        url='ws://unused', token='unused', cameras={}, robot_type='custom',
        policy_server_url='ws://unused', recording=False,
    )
    assignment = {'inference_mode': 'local', 'task': {'instruction': 'test', 'cameras': [], 'max_steps': 1}}
    with pytest.raises(ValueError, match='franka, yam or so101'):
        asyncio.run(evaluation_module.run_trial(
            config, assignment, {'id': 'run'}, tmp_path, None,
            robot_factory=lambda *args, **kwargs: pytest.fail('adapter must not be constructed'),
            confirm_live=lambda *args: True,
        ))
