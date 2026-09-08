from __future__ import annotations

import argparse
import asyncio

from .robot_config import RobotClientConfig
from .robot_runner import run_robot


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Colosseum robot evaluation")
    parser.add_argument("config", nargs="?", default="configs/robot.yaml")
    parser.add_argument("--track", choices=["open", "fine-tuning"])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", help="Resume an assigned evaluation or pending upload/result")
    group.add_argument("--abort", help="Report an interrupted evaluation")
    parser.add_argument("--inference-only", action="store_true", help="Use the original single-policy inference loop")
    args = parser.parse_args()
    config = RobotClientConfig.from_yaml(args.config)
    if not args.inference_only:
        from .evaluation import run_evaluation
        try:
            run_evaluation(config, track=args.track, resume=args.resume, abort=args.abort)
        except KeyboardInterrupt:
            pass
        return
    instruction = config.instruction or input("Instruction: ").strip()
    if not instruction:
        raise SystemExit("instruction cannot be empty")
    try:
        asyncio.run(run_robot(config, instruction))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
