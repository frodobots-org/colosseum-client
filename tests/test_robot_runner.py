from __future__ import annotations

import numpy as np
import pytest

from colosseum_client import ProtocolError, RobotObservation
from colosseum_client import colosseum_pb2 as pb
from colosseum_client.robot_runner import action_chunk, protobuf_observation
from colosseum_client.tensors import tensor_from_numpy, tensor_to_numpy


def test_robot_observation_is_serialized_for_policy():
    image = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    source = RobotObservation(
        images={"head_image": image},
        joints=np.arange(7, dtype=np.float32),
        gripper=np.array([0.5], dtype=np.float32),
        cartesian_position=np.arange(6, dtype=np.float32),
    )

    message = protobuf_observation(source, instruction="pick cube", control_step=12)

    assert message.control_step == 12
    assert message.instruction == "pick cube"
    np.testing.assert_array_equal(tensor_to_numpy(message.state["joint_position"]), source.joints)
    np.testing.assert_array_equal(tensor_to_numpy(message.state["gripper_position"]), [0.5])
    assert message.sensors[0].sensor_id == "head_image"
    assert message.sensors[0].encoding == pb.RAW_RGB
    assert message.sensors[0].data == image.tobytes()


def test_action_chunk_validates_timing_and_dimension():
    plan = pb.ActionPlan(
        start_step=4,
        valid_until_step=5,
        actions=tensor_from_numpy(np.zeros((2, 8), dtype=np.float32)),
    )
    assert action_chunk(plan, control_step=4).shape == (2, 8)

    plan.start_step = 3
    with pytest.raises(ProtocolError, match="starts at step"):
        action_chunk(plan, control_step=4)

    plan.start_step = 4
    plan.actions.CopyFrom(tensor_from_numpy(np.zeros((2, 7), dtype=np.float32)))
    with pytest.raises(ProtocolError, match="shape"):
        action_chunk(plan, control_step=4)


def test_extra_state_and_done_plans():
    source = RobotObservation(
        images={"front_image": np.zeros((2, 3, 3), dtype=np.uint8), "goal_image": np.zeros((2, 3, 3), dtype=np.uint8)},
        joints=np.zeros(0, dtype=np.float32), gripper=np.zeros(0, dtype=np.float32),
        cartesian_position=np.zeros(0, dtype=np.float32),
        extra_state={"navigation_state": np.array([1.23456789012, 2.0], dtype=np.float64)},
    )
    message = protobuf_observation(source, instruction="go", control_step=0)
    assert tensor_to_numpy(message.state["navigation_state"]).dtype == np.float64
    np.testing.assert_array_equal(tensor_to_numpy(message.state["navigation_state"]), [1.23456789012, 2.0])
    assert tensor_to_numpy(message.state["joint_position"]).size == 0
    assert {sensor.sensor_id for sensor in message.sensors} == {"front_image", "goal_image"}
    with pytest.raises(ValueError, match="duplicate observation state joint_position"):
        protobuf_observation(RobotObservation({}, np.zeros(0), np.zeros(0), np.zeros(0), {"joint_position": np.ones(1)}),
                             instruction="go", control_step=0)

    plan = pb.ActionPlan(start_step=4, valid_until_step=4, actions=tensor_from_numpy(np.zeros((1, 2), dtype=np.float32)), done=True)
    assert action_chunk(plan, control_step=4, expected_dim=2).shape == (1, 2)
    plan.actions.CopyFrom(tensor_from_numpy(np.array([[0.1, 0.0]], dtype=np.float32)))
    with pytest.raises(ProtocolError, match="done"):
        action_chunk(plan, control_step=4, expected_dim=2)
