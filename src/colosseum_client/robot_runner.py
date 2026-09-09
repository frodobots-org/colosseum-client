from __future__ import annotations

import asyncio
import time

import numpy as np

from . import colosseum_pb2 as pb
from .client import ColosseumClient, ProtocolError
from .droid_robot import DroidRobot, RobotObservation
from .robot_config import RobotClientConfig
from .tensors import tensor_from_numpy, tensor_to_numpy


def protobuf_observation(
    observation: RobotObservation,
    *,
    instruction: str,
    control_step: int,
) -> pb.Observation:
    message = pb.Observation(
        robot_time_ns=time.time_ns(),
        control_step=control_step,
        instruction=instruction,
    )
    message.state["joint_position"].CopyFrom(tensor_from_numpy(observation.joints))
    message.state["gripper_position"].CopyFrom(tensor_from_numpy(observation.gripper))
    message.state["cartesian_position"].CopyFrom(
        tensor_from_numpy(observation.cartesian_position)
    )
    capture_time_ns = time.time_ns()
    for image_id, image in observation.images.items():
        rgb = np.ascontiguousarray(image, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"{image_id} must have RGB shape (height, width, 3)")
        frame = message.sensors.add(
            sensor_id=image_id,
            encoding=pb.RAW_RGB,
            capture_time_ns=capture_time_ns,
            width=rgb.shape[1],
            height=rgb.shape[0],
        )
        frame.data = rgb.tobytes()
    return message


def action_chunk(plan: pb.ActionPlan, *, control_step: int, expected_dim: int = 8) -> np.ndarray:
    actions = tensor_to_numpy(plan.actions)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2 or actions.shape[0] == 0 or actions.shape[1] != expected_dim:
        raise ProtocolError(f"policy action chunk must have shape (horizon, {expected_dim}), got {actions.shape}")
    if not np.all(np.isfinite(actions)):
        raise ProtocolError("policy action chunk contains non-finite values")
    if plan.start_step != control_step:
        raise ProtocolError(
            f"policy action starts at step {plan.start_step}, expected {control_step}"
        )
    if plan.valid_until_step < control_step + len(actions) - 1:
        raise ProtocolError("policy action chunk expires before its final action")
    return np.asarray(actions, dtype=np.float32)


async def run_robot(
    config: RobotClientConfig,
    instruction: str,
    *,
    robot: DroidRobot | None = None,
    max_control_steps: int | None = None,
) -> None:
    robot = robot or DroidRobot(
        config.cameras,
        action_space="joint_position",
        image_size=(config.image_width, config.image_height),
    )
    client = ColosseumClient(
        config.url,
        config.token,
        client_id="",
        robot_type="DROID",
        joint_count=7,
        has_gripper=True,
        control_hz=config.control_hz,
        action_spaces={"joint_position": 8},
    )
    control_step = 0
    period = 1.0 / config.control_hz
    try:
        metadata = await client.connect()
        if "joint_position" not in metadata.action_spaces:
            raise ProtocolError("matched policy does not support joint_position actions")
        print(
            f"connected client={client.client_id} policy={client.policy_id} "
            f"session={client.session_id}"
        )
        print('Action execution disabled: receiving actions and reading cameras only.', flush=True)
        while max_control_steps is None or control_step < max_control_steps:
            current = robot.get_observation()
            request = protobuf_observation(
                current,
                instruction=instruction,
                control_step=control_step,
            )
            plan = await client.infer(request, deadline_ms=config.deadline_ms)
            actions = action_chunk(plan, control_step=control_step)
            print(f'Received action chunk at step {control_step} (execution skipped):\n{actions.tolist()}', flush=True)

            for action in actions:
                if max_control_steps is not None and control_step >= max_control_steps:
                    break
                started = time.monotonic()
                # Temporary camera diagnostic: do not execute received actions.
                control_step += 1
                remaining = period - (time.monotonic() - started)
                if remaining > 0:
                    await asyncio.sleep(remaining)
    finally:
        try:
            await client.close()
        finally:
            robot.close()
