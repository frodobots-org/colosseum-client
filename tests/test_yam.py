from dataclasses import replace

import numpy as np
import pytest

from colosseum_client import RobotClientConfig, make_robot
from colosseum_client.robots import yam


def config():
    return RobotClientConfig(url='ws://localhost', token='test', robot_type='yam', control_hz=30,
        cameras=dict(zip(yam.CAMERAS, ('top', 'left', 'right'))),
        adapter_config=dict(left_channel='can_left', right_channel='can_right',
                            left_gripper_limits=[0, 1], right_gripper_limits=[0, 1],
                            joint_low=[-3]*12, joint_high=[3]*12, joint_max_step=[.1]*12))


@pytest.fixture
def hardware(monkeypatch):
    opened = []

    class Camera:
        def __init__(self, serial, timeout):
            self.serial, self.closed = serial, 0
            opened.append(self)
        def read(self):
            return np.full((2, 3, 3), ('top', 'left', 'right').index(self.serial), np.uint8)
        def close(self):
            self.closed += 1

    class Arm:
        def __init__(self, channel, limits):
            self.channel, self.closed, self.commands = channel, 0, []
            self.position = np.r_[np.arange(6)*.01, .2 if channel == 'can_left' else .8]
            opened.append(self)
        def get_joint_pos(self):
            return self.position.copy()
        def command_joint_pos(self, value):
            self.commands.append(value.copy())
        def close(self):
            self.closed += 1

    monkeypatch.setattr(yam, '_RealSenseCamera', Camera)
    monkeypatch.setattr(yam, '_open_arm', Arm)
    return opened


def test_state_and_action_order_and_close(hardware):
    robot = make_robot(config())
    obs = robot.get_observation()
    np.testing.assert_allclose(obs.joints, np.tile(np.arange(6)*.01, 2))
    np.testing.assert_allclose(obs.gripper, [.2, .8])
    assert obs.cartesian_position.size == 0
    assert obs.images['right_image'][0, 0, 0] == 2
    target = np.r_[obs.joints[:6]+.02, .3, obs.joints[6:]-.03, .7]
    robot.execute(target)
    np.testing.assert_allclose(robot.arms[0].commands[0], target[:7])
    np.testing.assert_allclose(robot.arms[1].commands[0], target[7:])
    robot.close()
    robot.close()
    assert all(item.closed == 1 for item in hardware)
    with pytest.raises(RuntimeError, match='closed'):
        robot.get_observation()


@pytest.mark.parametrize('index,value', [(0, 4), (7, .3), (6, 1.1), (13, -.1), (1, float('nan'))])
def test_invalid_action_never_moves_either_arm(hardware, index, value):
    robot = make_robot(config())
    action = np.r_[robot.arms[0].position, robot.arms[1].position]
    action[index] = value
    with pytest.raises(ValueError):
        robot.execute(action)
    assert all(not arm.commands for arm in robot.arms)
    robot.close()


def test_invalid_settings_open_nothing(hardware):
    with pytest.raises(ValueError, match='joint_low'):
        make_robot(replace(config(), adapter_config={}))
    assert hardware == []


def test_second_arm_failure_closes_first_arm_and_cameras(hardware, monkeypatch):
    factory = yam._open_arm
    def fail(channel, limits):
        if channel == 'can_right':
            raise RuntimeError('CAN unavailable')
        return factory(channel, limits)
    monkeypatch.setattr(yam, '_open_arm', fail)
    with pytest.raises(RuntimeError, match='CAN unavailable'):
        make_robot(config())
    assert len(hardware) == 4 and all(item.closed == 1 for item in hardware)


def test_command_failure_closes_both_arms(hardware):
    robot = make_robot(config())
    def fail(_):
        raise RuntimeError('CAN lost')
    robot.arms[1].command_joint_pos = fail
    with pytest.raises(RuntimeError, match='CAN lost'):
        robot.execute(np.r_[robot.arms[0].position, robot.arms[1].position])
    assert all(item.closed == 1 for item in hardware)


def test_malformed_sdk_state_is_not_padded(hardware):
    robot = make_robot(config())
    robot.arms[0].position = np.zeros(6)
    with pytest.raises(ValueError, match='7 finite'):
        robot.get_observation()
    robot.close()


def test_synthetic_yam_uses_same_split_without_loading_hardware(monkeypatch):
    def fail(*_):
        raise AssertionError('Hardware must not load in test mode')
    monkeypatch.setattr(yam, '_open_arm', fail)
    robot = make_robot(replace(config(), test=True))
    robot.configure_model({'action_dim': 14, 'action_space': 'joint_position'})
    action = np.r_[np.arange(6), .25, np.arange(6,12), .75]
    robot.execute(action)
    obs = robot.get_observation()
    np.testing.assert_array_equal(obs.joints, np.arange(12))
    np.testing.assert_array_equal(obs.gripper, [.25, .75])


@pytest.mark.asyncio
@pytest.mark.parametrize('execute_action', [False, True])
async def test_local_trial_yam_contract_and_hardware_open_after_confirmation(tmp_path, monkeypatch, hardware, execute_action):
    from types import SimpleNamespace
    from colosseum_client import evaluation, local_policy
    from colosseum_client import colosseum_pb2 as pb
    from colosseum_client.tensors import tensor_from_numpy
    events = []
    class Client:
        def __init__(self, config, *args, **kwargs):
            assert kwargs['joint_count'] == 12
            assert kwargs['action_spaces'] == {'joint_position': 14}
            self.model = {'action_space': 'joint_position', 'action_dim': 14}
        async def connect(self, **kwargs):
            assert not hardware
            assert kwargs['before_start'](self.model, {}) is True
            return SimpleNamespace(action_spaces=['joint_position'])
        async def infer(self, obs, **kwargs):
            assert len(hardware) == 5
            assert list(obs.state['joint_position'].shape) == [12]
            assert list(obs.state['gripper_position'].shape) == [2]
            target = np.r_[hardware[3].position, hardware[4].position].astype(np.float32)[None]
            return pb.ActionPlan(start_step=0, valid_until_step=0, control_hz=30,
                                 actions=tensor_from_numpy(target))
        async def before_action(self, step):
            pass
        async def finish(self):
            events.append('finish')
        async def close(self):
            events.append('close')
    monkeypatch.setattr(local_policy, 'LocalPolicyClient', Client)
    cfg = replace(config(), recording=False)
    def confirm(*_):
        events.append('confirm')
        return True
    await evaluation.run_trial(cfg, {'inference_mode': 'local', 'task': {
        'instruction': 'pick', 'cameras': list(yam.CAMERAS), 'max_steps': 1}},
        {'id': 'yam-run'}, tmp_path, SimpleNamespace(), execute_action=execute_action, confirm_live=confirm)
    assert events == ['confirm', 'finish', 'close']
    assert all(resource.closed == 1 for resource in hardware)
    assert len(hardware[3].commands) == int(execute_action)
