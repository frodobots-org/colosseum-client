from __future__ import annotations

import pytest

from colosseum_client import ColosseumClient


def test_client_registers_default_robot_spec():
    client = ColosseumClient("ws://router:8443", "clt_test", "robot-1")
    assert client.robot_spec.robot_type == "franka"
    assert client.robot_spec.joint_count == 7
    assert client.robot_spec.has_gripper
    assert client.robot_spec.control_hz == 15
    assert [(spec.name, spec.action_dim) for spec in client.robot_spec.action_spaces] == [
        ("joint_position", 8)
    ]


def test_client_accepts_custom_robot_spec():
    client = ColosseumClient(
        "ws://router:8443",
        "clt_test",
        "robot-2",
        robot_type="custom-arm",
        joint_count=6,
        has_gripper=False,
        control_hz=20,
        action_spaces={"joint_velocity": 6, "cartesian_position": 6},
    )
    assert client.robot_spec.joint_count == 6
    assert not client.robot_spec.has_gripper
    assert {spec.name: spec.action_dim for spec in client.robot_spec.action_spaces} == {
        "joint_velocity": 6,
        "cartesian_position": 6,
    }


def test_client_rejects_invalid_robot_spec():
    with pytest.raises(ValueError, match="joint_count"):
        ColosseumClient("ws://router:8443", "clt_test", "robot", joint_count=0)
    with pytest.raises(ValueError, match="action_spaces"):
        ColosseumClient("ws://router:8443", "clt_test", "robot", action_spaces={"joint_position": 0})


@pytest.mark.parametrize("robot_type", ["DROID", "droid"])
def test_driver_name_is_not_a_robot_identity(robot_type):
    with pytest.raises(ValueError, match="Use robot_type: franka"):
        ColosseumClient("ws://router:8443", "clt_test", "robot-1", robot_type=robot_type)
