"""Bimanual I2RT YAM / LINEAR_4310 and three RealSense cameras.

Actions: left six radians, left normalized gripper, right six radians, right
normalized gripper. Observations keep twelve joints and two grippers separate.
SDK close releases motor torque; it is not a powered position hold.
"""
import logging

import numpy as np

from ..robot_interface import Robot, RobotObservation

log = logging.getLogger(__name__)
CAMERAS = ('head_image', 'left_image', 'right_image')


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
                   'joint_max_step', 'camera_timeout_ms'}
        if settings.keys() - allowed:
            raise ValueError(f'Unknown YAM settings: {sorted(settings.keys() - allowed)}')
        self.low = _vector(settings.get('joint_low'), 12, 'joint_low')
        self.high = _vector(settings.get('joint_high'), 12, 'joint_high')
        self.max_step = _vector(settings.get('joint_max_step'), 12, 'joint_max_step')
        if np.any(self.low >= self.high) or np.any(self.max_step <= 0):
            raise ValueError('YAM requires joint_low < joint_high and positive joint_max_step')
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

    def execute(self, action):
        value = _vector(action, 14, 'YAM action')
        joints = np.r_[value[:6], value[7:13]]
        if np.any(joints < self.low) or np.any(joints > self.high):
            raise ValueError('YAM action exceeds joint limits')
        if np.any(value[[6, 13]] < 0) or np.any(value[[6, 13]] > 1):
            raise ValueError('YAM grippers must be in [0, 1] (0 closed, 1 open)')
        left, right = self._read_positions()
        if np.any(np.abs(joints - np.r_[left[:6], right[:6]]) > self.max_step):
            raise ValueError('YAM action exceeds joint_max_step from measured position')
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

    def close(self):
        if self.closed:
            return
        self.closed = True
        errors = []
        for resource in [*self.arms, *self.cameras.values()]:
            try:
                resource.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError('YAM resource shutdown failed') from errors[0]
