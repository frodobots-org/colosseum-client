#!/usr/bin/env python3
"""Calibrate one LINEAR_4310 gripper using I2RT; save a Client YAML fragment.

Run from the Client checkout, with I2RT installed in its environment:
    .venv/bin/python scripts/calibrate_yam_gripper.py --side left --channel can_left

This enables real hardware and moves the gripper. Support the arm throughout:
calibration happens during SDK construction, and SDK close releases motor torque.
No Router, model, cameras, or existing Client config are needed or modified.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--side', required=True, choices=('left', 'right'))
    parser.add_argument('--channel', required=True, help='CAN interface wired to this arm')
    parser.add_argument('--output', type=Path, help='New YAML fragment (default: yam-SIDE-gripper.yaml)')
    args = parser.parse_args(argv)
    if not args.channel.strip():
        parser.error('--channel must not be empty')
    if args.output is None:
        args.output = Path(f'yam-{args.side}-gripper.yaml')
    return args


def main(argv=None):
    args = parse_args(argv)
    # Reject a bad destination before importing the motor SDK or enabling hardware.
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}; choose another --output')
    if not args.output.parent.is_dir():
        raise FileNotFoundError(f'Output directory does not exist: {args.output.parent}')
    print(f'Arm: {args.side}; CAN: {args.channel}; gripper: LINEAR_4310')
    print('Stop other CAN clients. Remove objects and hands from the gripper travel.')
    print('Support the arm throughout initialization and shutdown. SDK close releases torque.')
    print('Calibration moves the gripper to both ends; SDK initialization may block.')
    if input('Type CALIBRATE to enable hardware: ').strip() != 'CALIBRATE':
        print('Cancelled; hardware was not opened.')
        return 0

    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import GripperType

    logging.basicConfig(level=logging.INFO)
    robot = None
    try:
        # Omitting an override enables the SDK's automatic endpoint detection.
        # Position hold starts after calibration, not during SDK construction.
        robot = get_yam_robot(channel=args.channel, gripper_type=GripperType.LINEAR_4310,
                              zero_gravity_mode=False)
        limits = np.asarray(robot.get_robot_info()['gripper_limits'], dtype=float)
        if limits.shape != (2,) or not np.isfinite(limits).all() or limits[0] == limits[1]:
            raise ValueError(f'SDK returned invalid gripper limits: {limits}')
        calibration_offset = float(robot.motor_chain.motor_offset[6])
        if not np.isfinite(calibration_offset):
            raise ValueError(f'SDK returned invalid gripper motor offset: {calibration_offset}')
        print(f'Detected SDK endpoints, in original order: {limits.tolist()}')
        print(f'Calibration motor offset (rad): {calibration_offset}')
        print('A timeout can also produce endpoints. Confirm the gripper physically reached both ends.')
        if input('Type SAVE only if full open/close travel was observed: ').strip() != 'SAVE':
            print('Calibration not accepted; no file saved.')
            return 1
        fragment = {'adapter_config': {
            f'{args.side}_gripper_limits': limits.tolist(),
            f'{args.side}_gripper_calibration_offset': calibration_offset,
        }}
        # Exclusive creation prevents overwriting a file created during calibration.
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write('# LINEAR_4310 SDK endpoints [closed, open]; preserve this order.\n')
            stream.write(f'# Arm: {args.side}. Recheck calibration after power cycling.\n')
            yaml.safe_dump(fragment, stream, sort_keys=False)
        print(f'Saved: {args.output.resolve()}')
        print('Merge both calibration entries into your existing adapter_config; do not replace the whole config.')
        input('Ensure the arm is supported; press Enter to release motor torque and exit: ')
        return 0
    finally:
        if robot is not None:
            robot.close()
        # If construction itself fails, the SDK owns any partially opened hardware.
        # This process cannot close a Robot object it never received.


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print('\nInterrupted. Check the physical arm and motor state before restarting.')
        raise SystemExit(130)
