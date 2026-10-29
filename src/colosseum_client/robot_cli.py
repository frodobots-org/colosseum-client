from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace

from .navigation import ROVER_TYPE
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
    parser.add_argument("--no-execute-action", action="store_true",
                        help="Receive and log actions without executing them on the robot")
    for name in ('instruction', 'scene', 'evaluator'):
        parser.add_argument('--' + name)
    parser.add_argument('--goal-gps', metavar='LAT,LON',
                        help='Earth Rover Open Track: destination coordinates in degrees')
    parser.add_argument('--goal-image', metavar='PATH',
                        help='Earth Rover Open Track: reference photo of the destination')
    parser.add_argument('--bot', metavar='SLUG',
                        help='Earth Rover: the FrodoBots bot to connect (otherwise adapter_config.bot_slug or a prompt)')
    parser.add_argument('--mission', metavar='SLUG',
                        help="Earth Rover: the bot's FrodoBots mission to ride on (otherwise chosen from a list)")
    parser.add_argument('--checkpoint', type=int, metavar='N',
                        help='Earth Rover: which mission checkpoint is the destination (default: the last)')
    args = parser.parse_args()
    config = RobotClientConfig.from_yaml(args.config)
    overrides = {name: getattr(args, name) for name in ('instruction', 'scene', 'evaluator')
                 if getattr(args, name) is not None}
    if overrides:
        config = replace(config, **overrides)
    if args.goal_gps or args.goal_image or args.bot or args.mission or args.checkpoint:
        if config.robot_type != ROVER_TYPE:
            raise SystemExit('--bot, --goal-gps, --goal-image, --mission and --checkpoint apply only to robot_type: earth_rover_mini_plus')
        settings = dict(config.adapter_config)
        if args.bot:
            settings['bot_slug'] = args.bot
        if args.mission:
            settings['mission_slug'] = args.mission
        if args.checkpoint:
            settings['checkpoint'] = args.checkpoint
        if args.goal_gps:
            try:
                latitude, longitude = (float(value) for value in args.goal_gps.split(','))
            except ValueError:
                raise SystemExit('--goal-gps expects LAT,LON in degrees') from None
            settings['goal_gps'] = [latitude, longitude]
        if args.goal_image:
            settings['goal_image'] = args.goal_image
        config = replace(config, adapter_config=settings)
    if args.inference_only and config.policy_server_url:
        raise SystemExit("policy_server_url requires evaluation mode so Router can assign the model")
    if not args.inference_only:
        from .evaluation import run_evaluation
        try:
            run_evaluation(config, track=args.track, resume=args.resume, abort=args.abort,
                           execute_action=not args.no_execute_action)
        except KeyboardInterrupt:
            pass
        return
    instruction = config.instruction or input("Instruction: ").strip()
    if not instruction:
        raise SystemExit("instruction cannot be empty")
    try:
        asyncio.run(run_robot(config, instruction, execute_action=not args.no_execute_action))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
