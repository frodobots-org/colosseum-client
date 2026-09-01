from __future__ import annotations

import argparse
import asyncio

from .robot_config import RobotClientConfig
from .robot_runner import run_robot


def main() -> None:
    parser = argparse.ArgumentParser(description="Connect a DROID robot to Colosseum Router")
    parser.add_argument("config", nargs="?", default="configs/robot.yaml")
    args = parser.parse_args()
    config = RobotClientConfig.from_yaml(args.config)
    instruction = config.instruction or input("Instruction: ").strip()
    if not instruction:
        raise SystemExit("instruction cannot be empty")
    try:
        asyncio.run(run_robot(config, instruction))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
