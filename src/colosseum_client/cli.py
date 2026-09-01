from __future__ import annotations

import argparse
import asyncio
import os
import time

import numpy as np

from . import colosseum_pb2 as pb
from .client import ColosseumClient
from .tensors import tensor_from_numpy, tensor_to_numpy


async def _run(args: argparse.Namespace) -> None:
    token = os.environ.get("COLOSSEUM_CLIENT_TOKEN")
    if not token:
        raise SystemExit("COLOSSEUM_CLIENT_TOKEN is required")
    client = ColosseumClient(
        args.router_url,
        token,
        args.client_id,
        robot_type=args.robot_type,
        joint_count=args.joint_count,
        has_gripper=not args.no_gripper,
        control_hz=args.control_hz,
    )
    try:
        metadata = await client.connect(requested_policy_id=args.policy_id)
        observation = pb.Observation(
            robot_time_ns=time.time_ns(),
            control_step=0,
            instruction=args.instruction,
        )
        observation.state["joint_position"].CopyFrom(
            tensor_from_numpy(np.zeros(args.joint_count, dtype=np.float32))
        )
        if not args.no_gripper:
            observation.state["gripper_position"].CopyFrom(
                tensor_from_numpy(np.zeros(1, dtype=np.float32))
            )
        if args.synthetic_image:
            width, height = 320, 240
            x = np.linspace(0, 255, width, dtype=np.uint8)[None, :]
            y = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
            rgb = np.empty((height, width, 3), dtype=np.uint8)
            rgb[:, :, 0] = x
            rgb[:, :, 1] = y
            rgb[:, :, 2] = 255 - x
            sensor = observation.sensors.add(
                sensor_id="head_image",
                encoding=pb.RAW_RGB,
                capture_time_ns=time.time_ns(),
                width=width,
                height=height,
            )
            sensor.data = rgb.tobytes()
        plan = await client.infer(observation, deadline_ms=args.deadline_ms)
        actions = tensor_to_numpy(plan.actions)
        print(
            f"policy={metadata.policy_id}@{metadata.policy_revision} "
            f"session={client.session_id} actions_shape={actions.shape}"
        )
        print(actions)
    finally:
        await client.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Send one synthetic Colosseum observation")
    parser.add_argument("--router-url", default="ws://127.0.0.1:8443")
    parser.add_argument("--client-id", default="robot-demo")
    parser.add_argument("--policy-id", default="demo-zero")
    parser.add_argument("--instruction", default="demo instruction")
    parser.add_argument("--deadline-ms", type=int, default=1000)
    parser.add_argument("--robot-type", default="DROID")
    parser.add_argument("--joint-count", type=int, default=7)
    parser.add_argument("--control-hz", type=int, default=15)
    parser.add_argument("--no-gripper", action="store_true")
    parser.add_argument("--synthetic-image", action="store_true")
    asyncio.run(_run(parser.parse_args()))


if __name__ == "__main__":
    main()
