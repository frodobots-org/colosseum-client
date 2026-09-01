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
