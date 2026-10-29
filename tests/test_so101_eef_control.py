"""EEF limits and accepted motor-command recording without opening a robot."""
from types import SimpleNamespace

import numpy as np
import pytest

from colosseum_client.robots.so101 import MOTORS
from colosseum_client.robots.so101_eef import SO101EEFController
Rotation = pytest.importorskip("lerobot.utils.rotation").Rotation  # the so101-eef extra; not installed in CI


class Kinematics:
    def forward_kinematics(self, joints):
        pose = np.eye(4)
        pose[0, 3] = joints[0] * .001
        return pose

    def inverse_kinematics(self, measured, desired, **kwargs):
        result = measured.copy()
        result[0] = desired[0, 3] * 1000
        return result


def controller(sent=None):
    commands = []
    def send(target):
        commands.append(target.copy())
        actual = target.copy() if sent is None else np.asarray(sent)
        return {f'{motor}.pos': value for motor, value in zip(MOTORS, actual)}
    value = object.__new__(SO101EEFController)
    value.robot = SimpleNamespace(low=np.full(5, -100), high=np.full(5, 100), _send=send)
    value.kinematics = Kinematics()
    value.rotation = Rotation
    value.expected_pose = value.expected_gripper = value.measured = None
    value._warned_ik_limit = False
    return value, commands


def test_eef_step_uses_ik_and_reports_driver_accepted_joints():
    eef, commands = controller(sent=[.5, 0, 0, 0, 0, 40])
    state = eef.observe(np.array([0, 0, 0, 0, 0, 40]))
    assert state['eef_state'].shape == (7,)
    np.testing.assert_allclose(state['eef_quat_wxyz'], [1, 0, 0, 0])
    eef.begin_chunk()
    accepted = eef.execute([.001, 0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(commands[0], [1, 0, 0, 0, 0, 40])
    np.testing.assert_allclose(accepted, [.5, 0, 0, 0, 0, 40])


def test_eef_motion_has_no_extra_displacement_or_gripper_step_cap():
    eef, commands = controller()
    eef.observe(np.array([0, 0, 0, 0, 0, 40]))
    eef.execute([.01, 0, 0, 0, 0, 0, 20])
    np.testing.assert_allclose(commands[0], [10, 0, 0, 0, 0, 60])


def test_invalid_gripper_target_does_not_send_a_motor_command():
    eef, commands = controller()
    eef.observe(np.array([0, 0, 0, 0, 0, 40]))
    with pytest.raises(ValueError, match='gripper target'):
        eef.execute([0, 0, 0, 0, 0, 0, 61])
    assert commands == []


def test_float32_gripper_chunk_reaches_exact_endpoint_without_false_overflow():
    eef, commands = controller()
    start = np.float32(1.0087424516677856)
    delta = np.float32((100 - start) / 35)
    eef.observe(np.array([0, 0, 0, 0, 0, start]))
    for _ in range(35):
        eef.execute(np.array([0, 0, 0, 0, 0, 0, delta], dtype=np.float32))
    assert len(commands) == 35
    assert commands[-1][5] == 100


def test_ik_roundoff_at_joint_limit_is_clamped():
    eef, commands = controller()
    eef.observe(np.array([0, 0, 0, 0, 0, 40]))
    accepted = eef.execute([.10000001, 0, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(commands[0], [100, 0, 0, 0, 0, 40])
    np.testing.assert_allclose(accepted, [100, 0, 0, 0, 0, 40])


def test_out_of_range_ik_holds_and_next_step_uses_measured_pose(caplog):
    eef, commands = controller()
    eef.observe(np.array([0, 0, 0, 0, 0, 40]))
    accepted = eef.execute([.101, 0, 0, 0, 0, 0, 5])
    np.testing.assert_allclose(accepted, [0, 0, 0, 0, 0, 40])
    np.testing.assert_allclose(commands[0], accepted)
    assert 'shoulder_pan=101.000000 outside' in caplog.text
    assert eef.expected_pose[0, 3] == 0
    assert eef.expected_gripper == 40
    eef.execute([.001, 0, 0, 0, 0, 0, 5])
    np.testing.assert_allclose(commands[1], [1, 0, 0, 0, 0, 45])


def test_out_of_range_ik_does_not_hold_from_unsafe_measured_state():
    eef, commands = controller()
    eef.observe(np.array([101, 0, 0, 0, 0, 40]))
    with pytest.raises(ValueError, match='measured position exceeds calibrated limits'):
        eef.execute([.001, 0, 0, 0, 0, 0, 0])
    assert commands == []
