"""SO-ARM101 follower via LeRobot's Feetech driver, plus LeRobot cameras.

Actions and state use LeRobot's so101_follower convention: shoulder_pan,
shoulder_lift, elbow_flex, wrist_flex, wrist_roll in degrees, then the gripper
opening in [0, 100]. Observations keep the five joints and the gripper separate.
Close returns to the pose measured at start-up, then releases motor torque.
"""
import logging
import time

import numpy as np
from PIL import Image

from ..robot_interface import Robot, RobotObservation

log = logging.getLogger(__name__)
JOINTS = ('shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll')
MOTORS = (*JOINTS, 'gripper')
GRIPPER_RANGE = (0.0, 100.0)
_CAMERA_TYPES = ('realsense', 'opencv')
_RETURN_HZ = 30


def _vector(value, size, name):
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f'{name} must contain {size} finite values')
    return result.copy()


def _camera_type(camera_id, configured):
    if configured is not None:
        if configured not in _CAMERA_TYPES:
            raise ValueError(f'camera_types values must be one of {_CAMERA_TYPES}')
        return configured
    # RealSense devices are addressed by their numeric serial; V4L2 by index or path.
    return 'realsense' if camera_id.isdecimal() and len(camera_id) > 4 else 'opencv'


def _open_follower(port, robot_id, calibration_dir, cameras, max_relative_target):
    """cameras: role -> (type, id, width, height, fps). Returns a connected follower."""
    try:
        from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
        from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
        from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig
    except ImportError as exc:
        raise RuntimeError('Install colosseum-client[so101] for LeRobot; see docs/so101.md') from exc
    camera_configs = {}
    for role, (kind, camera_id, width, height, fps) in cameras.items():
        if kind == 'realsense':
            camera_configs[role] = RealSenseCameraConfig(
                serial_number_or_name=camera_id, width=width, height=height, fps=fps)
        else:
            camera_configs[role] = OpenCVCameraConfig(
                index_or_path=int(camera_id) if camera_id.isdecimal() else camera_id,
                width=width, height=height, fps=fps)
    follower = SO101Follower(SO101FollowerConfig(
        port=port, id=robot_id, calibration_dir=calibration_dir, cameras=camera_configs,
        use_degrees=True, max_relative_target=max_relative_target))
    try:
        # Never start LeRobot's interactive calibration from an evaluation.
        follower.connect(calibrate=False)
        if not follower.is_calibrated:
            raise RuntimeError(
                f'SO101 {robot_id!r} is not calibrated; run lerobot-calibrate before evaluating')
    except BaseException:
        try:
            follower.disconnect()
        except Exception:
            log.debug('SO101 cleanup after failed connection', exc_info=True)
        raise
    return follower


class SO101Robot(Robot):
    robot_name = 'SO101'
    joint_count = 5
    has_gripper = True
    action_dim = 6
    action_space_name = 'joint_position'

    def __init__(self, config):
        settings = dict(config.adapter_config)
        allowed = {'port', 'robot_id', 'calibration_dir', 'camera_types', 'camera_width',
                   'camera_height', 'camera_fps', 'joint_low', 'joint_high',
                   'max_relative_target', 'return_to_start', 'return_step_deg'}
        if settings.keys() - allowed:
            raise ValueError(f'Unknown SO101 settings: {sorted(settings.keys() - allowed)}')
        port, robot_id = settings.get('port'), settings.get('robot_id')
        if any(not isinstance(v, str) or not v.strip() for v in (port, robot_id)):
            raise ValueError('SO101 requires port and robot_id (the LeRobot calibration id)')
        calibration_dir = settings.get('calibration_dir')
        if calibration_dir is not None and (not isinstance(calibration_dir, str) or not calibration_dir.strip()):
            raise ValueError('calibration_dir must be a nonempty path')
        self.low = _vector(settings.get('joint_low'), 5, 'joint_low')
        self.high = _vector(settings.get('joint_high'), 5, 'joint_high')
        if np.any(self.low >= self.high):
            raise ValueError('SO101 requires joint_low < joint_high in degrees')
        max_relative_target = settings.get('max_relative_target')
        if max_relative_target is not None and (
                type(max_relative_target) not in (int, float) or not 0 < max_relative_target <= 180):
            raise ValueError('max_relative_target must be degrees in (0, 180]')
        self.return_to_start = settings.get('return_to_start', True)
        if type(self.return_to_start) is not bool:
            raise ValueError('return_to_start must be true or false')
        self.return_step = settings.get('return_step_deg', 1.5)
        if type(self.return_step) not in (int, float) or not 0 < self.return_step <= 10:
            raise ValueError('return_step_deg must be in (0, 10]')
        sizes = [settings.get(name, default) for name, default in
                 (('camera_width', 640), ('camera_height', 480), ('camera_fps', 30))]
        if any(type(v) is not int or v <= 0 for v in sizes):
            raise ValueError('camera_width, camera_height and camera_fps must be positive integers')
        camera_types = settings.get('camera_types', {})
        if not isinstance(camera_types, dict) or camera_types.keys() - set(config.cameras):
            raise ValueError('camera_types must map configured camera roles to realsense or opencv')
        if not config.cameras or len(set(config.cameras.values())) != len(config.cameras):
            raise ValueError('SO101 requires distinct camera IDs')
        cameras = {role: (_camera_type(camera_id, camera_types.get(role)), camera_id, *sizes)
                   for role, camera_id in config.cameras.items()}
        self.image_size = (config.image_width, config.image_height)
        self.follower, self.start, self.closed = None, None, False
        try:
            self.follower = _open_follower(port, robot_id, calibration_dir, cameras, max_relative_target)
            # The operator places the arm in its rest pose before each trial.
            self.start = self._positions(self.follower.get_observation())
        except BaseException:
            try:
                self.close()
            except Exception:
                log.exception('SO101 initialization cleanup failed')
            raise

    @staticmethod
    def _positions(raw):
        try:
            values = [raw[f'{motor}.pos'] for motor in MOTORS]
        except KeyError as exc:
            raise ValueError(f'LeRobot observation is missing {exc.args[0]}') from exc
        return _vector(values, 6, 'LeRobot motor position')

    def _image(self, raw, role):
        image = np.asarray(raw.get(role))
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f'SO101 camera {role} must return uint8 HWC RGB')
        if (image.shape[1], image.shape[0]) != self.image_size:
            image = np.asarray(Image.fromarray(image).resize(self.image_size, resample=Image.Resampling.LANCZOS))
        return np.ascontiguousarray(image)

    def get_observation(self):
        if self.closed:
            raise RuntimeError('SO101 is closed')
        raw = self.follower.get_observation()
        positions = self._positions(raw)
        images = {role: self._image(raw, role) for role in self.follower.cameras}
        return RobotObservation(images, positions[:5].astype(np.float32), positions[5:].astype(np.float32),
                                np.empty(0, dtype=np.float32))  # FK is unavailable, not zero.

    def _send(self, positions):
        self.follower.send_action({f'{motor}.pos': float(v) for motor, v in zip(MOTORS, positions)})

    def execute(self, action):
        if self.closed:
            raise RuntimeError('SO101 is closed')
        value = _vector(action, 6, 'SO101 action')
        # Learned policies routinely overshoot calibrated ends by a fraction of a
        # degree; clamp to the configured range instead of aborting the trial.
        value[:5] = np.clip(value[:5], self.low, self.high)
        value[5] = np.clip(value[5], *GRIPPER_RANGE)
        self._send(value)

    def _return_to_start(self):
        current = self._positions(self.follower.get_observation())
        steps = int(np.ceil(np.max(np.abs(self.start - current)) / self.return_step))
        for index in range(1, steps + 1):
            self._send(current + (self.start - current) * index / steps)
            time.sleep(1 / _RETURN_HZ)

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.follower is None:
            return
        errors = []
        if self.return_to_start and self.start is not None:
            try:
                self._return_to_start()
            except Exception as exc:
                errors.append(exc)
        try:
            self.follower.disconnect()
        except Exception as exc:
            errors.append(exc)
        if errors:
            raise RuntimeError('SO101 shutdown failed') from errors[0]
