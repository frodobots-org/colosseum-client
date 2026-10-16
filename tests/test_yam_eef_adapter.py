"""Exercise the Client/Inspect EE boundary without opening hardware."""

from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip('inspect_robots')

from colosseum_client.adapters import TestRobot as SyntheticRobot
from colosseum_client.robot_config import RobotClientConfig
from colosseum_client.robots.yam_eef import YAMEEFRobot


def _config():
    from inspect_robots_yam.config import DEFAULT_EEF_LOW, DEFAULT_EEF_HIGH
    return RobotClientConfig(
        url='wss://router.example', token='test', robot_type='yam',
        control_hz=30, cameras={'head_image': 'top', 'left_image': 'left', 'right_image': 'right'},
        adapter_config={
            'left_channel': 'can_left', 'right_channel': 'can_right',
            'joint_low': [-2.] * 12, 'joint_high': [2.] * 12,
            'joint_max_step': [.1] * 12,
            'joint_step_mode': 'clip_commanded', 'joint_limit_mode': 'clip',
            'inspect_eef': {'eef_low': DEFAULT_EEF_LOW, 'eef_high': DEFAULT_EEF_HIGH},
        },
    )


def test_adapter_preserves_joint_and_eef_action_semantics():
    from inspect_robots.types import Observation, StepResult

    sent = []

    class Camera:
        def read(self):
            return np.zeros((480, 640, 3), dtype=np.uint8)

    class Hardware:
        def __init__(self, config):
            assert 'inspect_eef' not in config.adapter_config
            assert config.adapter_config['joint_step_mode'] == 'reject'
            assert config.adapter_config['joint_limit_mode'] == 'reject'
            self.cameras = {name: Camera() for name in config.cameras}
            self.closed = False

        def _read_positions(self):
            return np.zeros(7), np.zeros(7)

        def _send_action(self, action):
            sent.append(np.asarray(action).copy())

        def close(self):
            self.closed = True

    class Embodiment:
        def __init__(self, *, driver_factory, camera_reader, **kwargs):
            assert kwargs['control_interface'] == 'eef_pos'
            assert kwargs['unattended'] is True
            self.driver = driver_factory(None)
            self.read_cameras = camera_reader

        def reset(self, scene):
            assert scene.id == 'run-1'
            self.driver.command_joint_pos(np.zeros(14))
            return self._observation()

        def _observation(self):
            return Observation(images=self.read_cameras(None),
                               state={'joint_pos': np.zeros(14), 'eef_state': np.ones(14)})

        def step(self, action):
            joint = np.r_[np.full(6, .05), .2, np.full(6, -.05), .8]
            self.driver.command_joint_pos(joint)
            return StepResult(self._observation(), info={'yam_joint_command': joint.tolist()})

        def close(self):
            self.driver.close()

    robot = YAMEEFRobot(_config(), hardware_factory=Hardware, embodiment_factory=Embodiment)
    robot.prepare_for_trial('Pick the block', 'run-1')
    obs = robot.get_observation()
    assert obs.joints.shape == (12,) and obs.cartesian_position.shape == (14,)
    result = robot.execute(np.zeros(14))
    np.testing.assert_allclose(result, sent[-1])
    robot.close()
    assert robot.closed and robot.hardware.closed


def test_synthetic_yam_eef_keeps_raw_pose_separate_from_joint_command():
    config = SimpleNamespace(robot_type='yam', cameras={'head_image': 'top'},
                             image_width=2, image_height=2)
    robot = SyntheticRobot(config)
    robot.configure_model({'action_space': 'eef_abs_pose', 'action_dim': 14})
    target = np.arange(14, dtype=np.float32) * .01
    sent = robot.execute(target)
    np.testing.assert_allclose(robot.get_observation().cartesian_position, target)
    assert sent.shape == (14,)
    assert not np.array_equal(sent, target)


def test_real_inspect_eef_plugin_runs_ik_through_client_driver():
    from inspect_robots_yam_eef import YamEEFEmbodiment

    class Camera:
        def read(self):
            return np.zeros((480, 640, 3), dtype=np.uint8)

    class Hardware:
        def __init__(self, config):
            self.cameras = {name: Camera() for name in config.cameras}
            self.position = np.zeros(14)
            self.commands = []
            self.closed = False

        def _read_positions(self):
            return self.position[:7].copy(), self.position[7:].copy()

        def _send_action(self, action):
            self.position = np.asarray(action, dtype=float).copy()
            self.commands.append(self.position.copy())

        def close(self):
            self.closed = True

    class Kinematics:
        def __init__(self, offset):
            self.offset = offset
            self.ranges = np.array([[-2., 2.]] * 6 + [[0., .04], [0., .04]])

        def get_joint_ranges(self):
            return self.ranges.copy()

        def set_joint_ranges(self, ranges):
            self.ranges = ranges.copy()

        def fk(self, joints):
            pose = np.eye(4)
            pose[:3, 3] = [.3, 0., .2]
            return pose

        def ik(self, target, init_q, max_iters):
            result = init_q.copy()
            result[:6] += self.offset
            return True, result

    cfg = _config()
    cfg = replace(cfg, adapter_config={
        **cfg.adapter_config,
        'inspect_eef': {**cfg.adapter_config['inspect_eef'], 'motor_temp_limit': None},
    })

    def embodiment_factory(**kwargs):
        return YamEEFEmbodiment(
            **kwargs,
            kinematics_factory=lambda _cfg: (Kinematics(.05), Kinematics(-.05)),
            sleep_fn=lambda _seconds: None,
            rest_secs=.1,
        )

    robot = YAMEEFRobot(cfg, hardware_factory=Hardware, embodiment_factory=embodiment_factory)
    try:
        robot.prepare_for_trial('Move the left gripper', 'router-run-id')
        observed = robot.get_observation()
        target = observed.cartesian_position.copy()
        target[[6, 13]] = [.2, .8]
        sent = robot.execute(target)
        np.testing.assert_allclose(sent, robot.hardware.commands[-1])
        np.testing.assert_allclose(sent[[6, 13]], [.2, .8])
        assert not np.array_equal(sent, target)
    finally:
        robot.close()
    assert robot.hardware.closed
