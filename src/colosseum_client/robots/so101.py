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
_CONNECT_ATTEMPTS = 3
_TORQUE_OFF_RETRIES = 2
_SETTLE_SECONDS = 0.5
# A wrist USB camera can drop off the bus when its cable flexes; the kernel
# may need up to about a minute to re-enumerate it after a failed handshake.
_CAMERA_RECOVERY_SECONDS = 90
_CAMERA_RETRY_SECONDS = 1.0


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
    for attempt in range(1, _CONNECT_ATTEMPTS + 1):
        follower = SO101Follower(SO101FollowerConfig(
            port=port, id=robot_id, calibration_dir=calibration_dir, cameras=camera_configs,
            use_degrees=True, max_relative_target=max_relative_target))
        try:
            # Never start LeRobot's interactive calibration from an evaluation.
            follower.connect(calibrate=False)
            if not follower.is_calibrated:
                raise RuntimeError(
                    f'SO101 {robot_id!r} is not calibrated; run lerobot-calibrate before evaluating')
            return follower
        except BaseException as exc:
            try:
                _release(follower)
            except Exception as release_error:
                log.warning('SO101 cleanup after failed connection: %s', release_error)
            # Feetech buses occasionally corrupt one status packet ("Incorrect status
            # packet!"); LeRobot's set-up writes are not retried, so retry the connection.
            if not isinstance(exc, ConnectionError) or attempt == _CONNECT_ATTEMPTS:
                raise
            log.warning('SO101 connection attempt %d/%d failed (%s); retrying', attempt, _CONNECT_ATTEMPTS, exc)
            time.sleep(0.5)


def _release(follower):
    """Disable torque, then free the serial port and cameras, even after a motor alarm.

    LeRobot's disconnect raises at the first motor whose reply carries an alarm
    bit, before it closes the port and cameras. A gripper squeezing an object
    trips its overload protection and then flags every reply with an Overload
    alarm, although the servo still carries out the write. So disable torque on
    each motor, confirm it with a sync read (which ignores alarm bits), and
    release the rest. Raises only if torque may still be on.
    """
    try:
        follower.disconnect()
        return
    except Exception as exc:
        cause = exc
    bus, unconfirmed = follower.bus, []
    # LeRobot closes the port only after disabling torque, so a closed port needs nothing more.
    if bus.is_connected:
        unconfirmed = _disable_torque(bus)
        try:
            bus.disconnect(disable_torque=False)
        except Exception:
            log.warning('SO101 serial port did not close', exc_info=True)
    for camera in follower.cameras.values():
        try:
            camera.disconnect()
        except Exception:
            log.debug('SO101 camera release failed', exc_info=True)
    if unconfirmed:
        raise RuntimeError(f'SO101 torque may still be on for {", ".join(unconfirmed)}') from cause
    log.warning('SO101 disconnect reported %s; torque is confirmed off and the port and cameras are '
                'released', cause)


def _disable_torque(bus):
    """Write torque off to every motor; return the motors not confirmed off."""
    for motor in bus.motors:
        try:
            bus.write('Torque_Enable', motor, 0, num_retry=_TORQUE_OFF_RETRIES)
        except Exception as exc:
            log.debug('SO101 %s torque-off write: %s', motor, exc)
    try:
        torque = bus.sync_read('Torque_Enable', normalize=False, num_retry=_TORQUE_OFF_RETRIES)
    except Exception:
        log.warning('SO101 could not read back motor torque', exc_info=True)
        return list(bus.motors)
    return [motor for motor in bus.motors if torque.get(motor) != 0]


class SO101Robot(Robot):
    robot_name = 'SO101'
    joint_count = 5
    has_gripper = True
    action_dim = 6
    action_space_name = 'joint_position'

    def __init__(self, config, *, control_mode='joint_pos'):
        settings = dict(config.adapter_config)
        allowed = {'port', 'robot_id', 'calibration_dir', 'camera_types', 'camera_width',
                   'camera_height', 'camera_fps', 'joint_low', 'joint_high',
                   'max_relative_target', 'return_to_start', 'return_step_deg', 'start_poses',
                   'eef_urdf_path'}
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
        if control_mode not in {'joint_pos', 'eef_delta_pose'}:
            raise ValueError(f'unsupported SO101 control mode: {control_mode}')
        self.control_mode = control_mode
        self.eef = None
        if control_mode == 'eef_delta_pose':
            if not settings.get('eef_urdf_path'):
                raise ValueError('SO101 EEF control requires adapter_config.eef_urdf_path')
            self.action_dim = 7
        max_relative_target = settings.get('max_relative_target')
        if max_relative_target is not None and (
                type(max_relative_target) not in (int, float) or not 0 < max_relative_target <= 180):
            raise ValueError('max_relative_target must be degrees in (0, 180]')
        # LeRobot accepts only float or dict here; an integer raises TypeError
        # on the first send_action call. None disables its per-command clamp.
        if max_relative_target is not None:
            max_relative_target = float(max_relative_target)
        self.return_to_start = settings.get('return_to_start', True)
        if type(self.return_to_start) is not bool:
            raise ValueError('return_to_start must be true or false')
        self.return_step = settings.get('return_step_deg', 1.5)
        if type(self.return_step) not in (int, float) or not 0 < self.return_step <= 10:
            raise ValueError('return_step_deg must be in (0, 10]')
        start_poses = settings.get('start_poses', {})
        if not isinstance(start_poses, dict) or any(not isinstance(key, str) or not key for key in start_poses):
            raise ValueError('start_poses must map a model runtime_profile, subfolder or URL to six values')
        self.start_poses = {}
        for key, pose in start_poses.items():
            pose = _vector(pose, 6, f'start_poses[{key!r}]')
            if (np.any(pose[:5] < self.low) or np.any(pose[:5] > self.high)
                    or not GRIPPER_RANGE[0] <= pose[5] <= GRIPPER_RANGE[1]):
                raise ValueError(f'start_poses[{key!r}] is outside the joint limits')
            self.start_poses[key] = pose
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
        if control_mode == 'eef_delta_pose':
            from .so101_eef import SO101EEFController
            self.eef = SO101EEFController(self, settings['eef_urdf_path'])
        self.follower, self.start, self.closed = None, None, False
        self.eef_motion_failed = False
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
        raw = self._read_follower()
        positions = self._positions(raw)
        images = {role: self._image(raw, role) for role in self.follower.cameras}
        try:
            extra = self.eef.observe(positions) if self.eef is not None else {}
        except Exception:
            self.eef_motion_failed = True
            raise
        return RobotObservation(images, positions[:5].astype(np.float32), positions[5:].astype(np.float32),
                                np.empty(0, dtype=np.float32), extra)

    def _read_follower(self):
        """Read motors and cameras, reconnecting any camera whose USB link dropped."""
        try:
            return self.follower.get_observation()
        except Exception as exc:
            failed = self._failed_cameras()
            if not failed:
                raise
            log.warning('SO101 camera(s) %s stopped (%s); reconnecting', ', '.join(failed), exc)
            self._reconnect_cameras(failed, exc)
            return self.follower.get_observation()

    def _failed_cameras(self):
        failed = []
        for role, camera in self.follower.cameras.items():
            try:
                camera.read_latest()
            except Exception:
                failed.append(role)
        return failed

    def _reconnect_cameras(self, roles, cause):
        # The arm holds its last commanded pose under torque while we wait.
        deadline = time.monotonic() + _CAMERA_RECOVERY_SECONDS
        for role in roles:
            camera = self.follower.cameras[role]
            while True:
                try:
                    camera.disconnect()
                except Exception:
                    log.debug('SO101 camera %s cleanup before reconnect failed', role, exc_info=True)
                try:
                    camera.connect()
                    break
                except Exception:
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f'SO101 camera {role} did not reconnect within '
                                           f'{_CAMERA_RECOVERY_SECONDS}s; check its USB cable') from cause
                    log.debug('SO101 camera %s reconnect attempt failed', role, exc_info=True)
                    time.sleep(_CAMERA_RETRY_SECONDS)
            log.warning('SO101 camera %s reconnected', role)

    def _send(self, positions):
        return self.follower.send_action({f'{motor}.pos': float(v) for motor, v in zip(MOTORS, positions)})

    def begin_chunk(self):
        if self.eef is not None:
            self.eef.begin_chunk()

    def execute(self, action):
        if self.closed:
            raise RuntimeError('SO101 is closed')
        if self.eef is not None:
            try:
                return self.eef.execute(action)
            except Exception:
                self.eef_motion_failed = True
                raise
        value = _vector(action, 6, 'SO101 action')
        # Learned policies routinely overshoot calibrated ends by a fraction of a
        # degree; clamp to the configured range instead of aborting the trial.
        value[:5] = np.clip(value[:5], self.low, self.high)
        value[5] = np.clip(value[5], *GRIPPER_RANGE)
        self._send(value)

    def _joint_positions(self):
        """Motor positions only, so homing still works after a camera is lost."""
        raw = self.follower.bus.sync_read('Present_Position', num_retry=self.follower.config.num_read_retries)
        return self._positions({f'{motor}.pos': value for motor, value in raw.items()})

    def _move_to(self, target):
        current = self._joint_positions()
        steps = int(np.ceil(np.max(np.abs(target - current)) / self.return_step))
        for index in range(1, steps + 1):
            self._send(current + (target - current) * index / steps)
            time.sleep(1 / _RETURN_HZ)

    def _return_to_start(self):
        self._move_to(self.start)

    def prepare_for_model(self, model):
        """Move to the start pose configured for the assigned model, if any.

        Some checkpoints are deployed upstream from a fixed home pose and degrade
        from a folded rest pose. Shutdown still returns to the pose measured at
        start-up. Returns the matched start_poses key, or None.
        """
        if self.closed:
            raise RuntimeError('SO101 is closed')
        for key in (model.get('runtime_profile'), model.get('subfolder'), model.get('url')):
            if key and key in self.start_poses:
                log.info('SO101 moving to the start pose configured for %s', key)
                self._move_to(self.start_poses[key])
                time.sleep(_SETTLE_SECONDS)
                return key
        return None

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.follower is None:
            return
        errors = []
        if self.return_to_start and self.start is not None and not self.eef_motion_failed:
            try:
                self._return_to_start()
            except Exception as exc:
                errors.append(exc)
        try:
            _release(self.follower)
        except Exception as exc:
            errors.append(exc)
        if errors:
            raise RuntimeError('SO101 shutdown failed') from errors[0]
