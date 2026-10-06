"""Bimanual I2RT YAM / LINEAR_4310 and three RealSense cameras.

Actions: left six radians, left normalized gripper, right six radians, right
normalized gripper. Observations keep twelve joints and two grippers separate.
SDK close releases motor torque; it is not a powered position hold.
"""
import asyncio
import logging
import select
import sys
import time
import threading

import numpy as np

from ..robot_interface import Robot, RobotObservation

log = logging.getLogger(__name__)
CAMERAS = ('head_image', 'left_image', 'right_image')
_CONTROL_JOIN_TIMEOUT = 5.0


def _close_arm(arm):
    """Let I2RT stop its robot loop, then join CAN workers before socket close.

    The pinned I2RT DMChain discards its worker handle. Discover only workers
    bound to this chain; do not change the SDK globally or other arms' threads.
    """
    chain = getattr(arm, 'motor_chain', None)
    interface = getattr(chain, 'motor_interface', None)
    if interface is None:
        arm.close()
        return
    workers = [t for t in threading.enumerate()
               if getattr(getattr(t, '_target', None), '__self__', None) is chain]
    original_close = interface.close

    def close_after_workers():
        chain.running = False
        pending = []
        for worker in workers:
            if worker is not threading.current_thread():
                worker.join(timeout=_CONTROL_JOIN_TIMEOUT)
            if worker.is_alive():
                pending.append(worker.name)
        # Always attempt SDK motor/socket shutdown, even after a join timeout.
        # A timeout remains a reported failure, never a successful teardown.
        original_close()
        if pending:
            raise RuntimeError(f'I2RT CAN workers did not stop before shutdown: {pending}')

    interface.close = close_after_workers
    try:
        arm.close()
    finally:
        interface.close = original_close


def _vector(value, size, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must contain {size} finite values')
    return result.copy()


def _open_arm(channel, gripper_limits):
    try:
        from i2rt.robots.get_robot import get_yam_robot
        from i2rt.robots.utils import GripperType
    except ImportError as exc:
        raise RuntimeError('Install I2RT in the Client environment; see docs/yam.md') from exc
    return get_yam_robot(channel=channel, gripper_type=GripperType.LINEAR_4310,
                         gripper_limits_override=gripper_limits,
                         zero_gravity_mode=False)


class _RealSenseCamera:
    def __init__(self, serial, timeout_ms):
        try:
            import pyrealsense2 as rs
        except ImportError as exc:
            raise RuntimeError('Install colosseum-client[yam] for RealSense cameras') from exc
        self.pipeline, self.timeout_ms = rs.pipeline(), timeout_ms
        settings = rs.config()
        settings.enable_device(str(serial))
        settings.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        try:
            self.pipeline.start(settings)
        except BaseException:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            raise

    def read(self):
        frame = self.pipeline.wait_for_frames(self.timeout_ms).get_color_frame()
        if not frame:
            raise RuntimeError('YAM camera returned no color frame')
        return np.asanyarray(frame.get_data()).copy()

    def close(self):
        self.pipeline.stop()


class YAMRobot(Robot):
    joint_count = 12
    has_gripper = True
    action_dim = 14
    action_space_name = 'joint_position'

    def __init__(self, config):
        settings = dict(config.adapter_config)
        allowed = {'left_channel', 'right_channel', 'left_gripper_limits',
                   'right_gripper_limits', 'joint_low', 'joint_high',
                   'joint_max_step', 'joint_step_mode', 'camera_timeout_ms'}
        if settings.keys() - allowed:
            raise ValueError(f'Unknown YAM settings: {sorted(settings.keys() - allowed)}')
        self.low = _vector(settings.get('joint_low'), 12, 'joint_low')
        self.high = _vector(settings.get('joint_high'), 12, 'joint_high')
        self.max_step = _vector(settings.get('joint_max_step'), 12, 'joint_max_step')
        self.joint_step_mode = settings.get('joint_step_mode', 'reject')
        if self.joint_step_mode not in ('reject', 'clip', 'interpolate'):
            raise ValueError('joint_step_mode must be reject, clip or interpolate')
        if np.any(self.low >= self.high) or np.any(self.max_step <= 0):
            raise ValueError('YAM requires joint_low < joint_high and positive joint_max_step')
        if np.any(self.low > 0) or np.any(self.high < 0):
            raise ValueError('YAM normal finish requires zero within all joint limits')
        channels = [settings.get(f'{side}_channel') for side in ('left', 'right')]
        if any(not isinstance(c, str) or not c.strip() for c in channels) or channels[0] == channels[1]:
            raise ValueError('YAM requires distinct left_channel and right_channel')
        limits = [_vector(settings.get(f'{side}_gripper_limits'), 2, f'{side}_gripper_limits')
                  for side in ('left', 'right')]
        if any(v[0] == v[1] for v in limits):
            raise ValueError('Gripper limits must be distinct calibrated [closed, open] positions')
        if set(config.cameras) != set(CAMERAS) or len(set(config.cameras.values())) != 3:
            raise ValueError('YAM requires three distinct cameras: head_image, left_image, right_image')
        if config.control_hz != 30:
            raise ValueError('MolmoAct2 YAM requires control_hz: 30')
        timeout = settings.get('camera_timeout_ms', 1000)
        if type(timeout) is not int or not 1 <= timeout <= 10000:
            raise ValueError('camera_timeout_ms must be an integer in [1, 10000]')
        self.arms, self.cameras, self.closed = [], {}, False
        try:
            # Check cameras before enabling either arm. No automatic homing.
            for name in CAMERAS:
                self.cameras[name] = _RealSenseCamera(config.cameras[name], timeout)
                self.cameras[name].read()
            for channel, calibration in zip(channels, limits):
                self.arms.append(_open_arm(channel, calibration))
            self._read_positions()
        except BaseException:
            try:
                self.close()
            except Exception:
                log.exception('YAM initialization cleanup failed')
            raise

    def _read_positions(self):
        if self.closed:
            raise RuntimeError('YAM is closed')
        return [_vector(arm.get_joint_pos(), 7, 'I2RT joint position') for arm in self.arms]

    def get_observation(self):
        if self.closed:
            raise RuntimeError('YAM is closed')
        images = {name: camera.read() for name, camera in self.cameras.items()}
        left, right = self._read_positions()
        return RobotObservation(images, np.r_[left[:6], right[:6]].astype(np.float32),
                                np.array([left[6], right[6]], dtype=np.float32),
                                np.empty(0, dtype=np.float32))  # FK is unavailable, not zero.

    def _validated_action(self, action):
        value = _vector(action, 14, 'YAM action')
        joints = np.r_[value[:6], value[7:13]]
        if np.any(joints < self.low) or np.any(joints > self.high):
            raise ValueError('YAM action exceeds joint limits')
        if np.any(value[[6, 13]] < 0) or np.any(value[[6, 13]] > 1):
            raise ValueError('YAM grippers must be in [0, 1] (0 closed, 1 open)')
        return value

    def _interpolation(self, action):
        """MolmoAct-style linear targets, with a strict command spacing bound.

        Read actual feedback once at the start, then interpolate commands. This
        bounds command increments, not tracking error or physical joint speed.
        """
        target = self._validated_action(action)
        left, right = self._read_positions()
        start = self._validated_action(np.r_[left, right])
        limits = np.r_[self.max_step[:6], .01, self.max_step[6:], .01]
        intervals = max(1, int(np.ceil(np.max(np.abs(target - start) / limits))))
        if intervals > 100:
            raise ValueError(f'YAM interpolation requires {intervals} intervals (maximum 100); '
                             'check starting pose and target')
        # ceil + intervals+1 avoids upstream floor/linspace endpoint overshoot.
        return np.linspace(start, target, intervals + 1)[1:]

    def execute(self, action):
        if self.joint_step_mode != 'interpolate':
            return self._execute_single(action)
        for target in self._interpolation(action):
            started = time.monotonic()
            self._send_action(target)
            time.sleep(max(0, 1 / 30 - (time.monotonic() - started)))

    async def execute_async(self, action, robot_call):
        """Keep each SDK call on its owning thread; allow cancellation per tick."""
        if self.joint_step_mode != 'interpolate':
            return await robot_call(self._execute_single, action)
        targets = await robot_call(self._interpolation, action)
        for target in targets:
            started = time.monotonic()
            await robot_call(self._send_action, target)
            await asyncio.sleep(max(0, 1 / 30 - (time.monotonic() - started)))

    def _execute_single(self, action):
        value = self._validated_action(action)
        joints = np.r_[value[:6], value[7:13]]
        left, right = self._read_positions()
        measured = np.r_[left[:6], right[:6]]
        if self.joint_step_mode == 'clip' and (np.any(measured < self.low) or np.any(measured > self.high)):
            raise ValueError('YAM measured position exceeds joint limits; cannot clip action')
        delta = joints - measured
        exceeded = np.flatnonzero(np.abs(delta) > self.max_step)
        if exceeded.size:
            details = '; '.join(
                f'{"left" if i < 6 else "right"}_joint{i % 6 + 1}: '
                f'measured={measured[i]:.6f}, target={joints[i]:.6f}, '
                f'delta={delta[i]:.6f}, limit={self.max_step[i]:.6f}'
                for i in exceeded)
            if self.joint_step_mode != 'clip':
                raise ValueError('YAM action exceeds joint_max_step from measured position: ' + details)
            # Bound each target against fresh feedback, never the previous command.
            # Keep absolute limits strict; do not attempt recovery from outside them.
            limited = measured + np.clip(delta, -self.max_step, self.max_step)
            value[:6], value[7:13] = limited[:6], limited[6:]
            log.warning('YAM joint targets clipped (radians): %s', details)
        self._send_action(value)

    def _send_action(self, action):
        if self.closed:
            raise RuntimeError('YAM is closed')
        value = self._validated_action(action)
        try:
            # Both commands validated before either arm moves. CAN writes are sequential.
            self.arms[0].command_joint_pos(value[:7].copy())
            self.arms[1].command_joint_pos(value[7:].copy())
        except BaseException:
            try:
                self.close()
            except Exception:
                log.exception('YAM command failure cleanup failed')
            raise

    async def finish_trial(self, robot_call):
        """Return to zero; caller must gate execution and cancellation.

        Zero refers to calibrated joint angles, not a collision-checked park pose.
        Keep grippers unchanged. Await each short SDK operation on the robot's
        existing executor so cancellation cannot leave a homing thread running.
        """
        print('YAM return: moving arm joints to zero; grippers unchanged.', flush=True)
        left, right = await robot_call(self._read_positions)
        self._validated_action(np.r_[left, right])
        target = np.r_[left[:6], right[:6]]
        measured = target.copy()
        grippers = np.array([left[6], right[6]])
        # Fixed command ramp, independent of feedback lag (as in interpolation).
        # This limits commanded motion, not physical velocity or tracking error.
        increment = np.minimum(self.max_step, .15 / 30)
        deadline = time.monotonic() + 60
        settled = 0
        zero_sent = False
        while time.monotonic() < deadline:
            target = target - np.clip(target, -increment, increment)
            action = np.r_[target[:6], grippers[:1], target[6:], grippers[1:]]
            await robot_call(self._send_action, action)
            if not zero_sent and np.all(target == 0):
                zero_sent = True
                print('YAM zero target sent; waiting for measured position confirmation.', flush=True)
            await asyncio.sleep(1 / 30)
            left, right = await robot_call(self._read_positions)
            measured = np.r_[left[:6], right[:6]]
            settled = settled + 1 if np.all(target == 0) and np.all(np.abs(measured) <= .02) else 0
            if settled >= 3:
                print('YAM zero position reached; proceeding to torque shutdown.', flush=True)
                return
        details = '; '.join(
            f'{"left" if i < 6 else "right"}_joint{i % 6 + 1}: '
            f'measured={measured[i]:.6f}, target={target[i]:.6f}'
            for i in range(12))
        raise RuntimeError('YAM move to zero timed out; position was not confirmed; '
                           f'zero_target_sent={zero_sent}, tolerance=0.020000 rad; {details}')

    @staticmethod
    def _retry_zero_requested():
        if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.readline() != ''
        return False

    async def hold_for_recovery(self, robot_call):
        """Keep SDK position control alive after a failed return, until retried.

        No automatic torque release on a healthy connection. Cancellation is
        still an explicit shutdown and hardware errors cannot guarantee hold.
        """
        while True:
            left, right = await robot_call(self._read_positions)
            await robot_call(self._send_action, np.r_[left, right])
            print('YAM return failed: position hold commanded; automatic torque shutdown paused. '
                  'Keep this process running. Press Enter to retry return to zero. '
                  'Ctrl+C shuts down and releases torque.', flush=True)
            while not self._retry_zero_requested():
                await asyncio.sleep(.1)
            try:
                await self.finish_trial(robot_call)
                return
            except Exception as exc:
                if self.closed:
                    raise
                print(f'YAM return retry failed: {exc}', flush=True)

    def close(self):
        if self.closed:
            return
        self.closed = True
        errors = []
        for resource in [*self.arms, *self.cameras.values()]:
            try:
                if any(resource is arm for arm in self.arms):
                    _close_arm(resource)
                else:
                    resource.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError('YAM resource shutdown failed') from errors[0]
