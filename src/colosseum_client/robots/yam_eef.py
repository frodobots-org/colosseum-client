"""YAM Cartesian control through Inspect Robots, using the Client's calibrated driver."""

from __future__ import annotations

import numpy as np
from dataclasses import replace

from ..robot_interface import Robot, RobotObservation
from .yam import YAMRobot


class _CalibratedYAMDriver:
    """Give Inspect's EE controller the Client's already calibrated motor interface."""

    def __init__(self, robot: YAMRobot):
        self.robot = robot

    def get_joint_pos(self):
        left, right = self.robot._read_positions()
        return np.r_[left, right]

    def command_joint_pos(self, target):
        # Reject, rather than silently clip, a target that Inspect reports as sent.
        self.robot._send_action(target)

    def get_motor_temps(self):
        values = []
        for arm in self.robot.arms:
            values.extend(max(state.temp_mos, state.temp_rotor)
                          for state in arm.motor_chain.read_states())
        return np.asarray(values, dtype=np.float64)

    def close(self):
        self.robot.close()


class YAMEEFRobot(Robot):
    """Expose the native ``yam_eef`` embodiment to a Colosseum local trial."""

    joint_count = 12
    has_gripper = True
    action_dim = 14
    action_space_name = 'eef_abs_pose'

    def __init__(self, config, *, hardware_factory=YAMRobot, embodiment_factory=None):
        if config.robot_type != 'yam' or config.control_hz != 30:
            raise ValueError('YAM EE control requires robot_type: yam and control_hz: 30')
        settings = dict(config.adapter_config)
        eef_options = settings.pop('inspect_eef', {})
        if not isinstance(eef_options, dict):
            raise ValueError('adapter_config.inspect_eef must be a mapping')
        if 'eef_low' not in eef_options or 'eef_high' not in eef_options:
            raise ValueError('YAM EE requires calibrated inspect_eef.eef_low and eef_high')
        # The shared Client config may use commanded clipping for joint VLAs.
        # The EE adapter must never clip after Inspect reports its sent command:
        # that would make both its IK state and the recording disagree with CAN.
        settings['joint_step_mode'] = 'reject'
        settings['joint_limit_mode'] = 'reject'
        self.joint_step_mode = 'reject'
        if embodiment_factory is None:
            try:
                from inspect_robots_yam_eef import YamEEFEmbodiment
            except ImportError as exc:
                raise RuntimeError('Install inspect-robots-yam-eef in the Client environment; see docs/local-evals.md') from exc
            embodiment_factory = YamEEFEmbodiment
        # The calibrated Client driver owns both CAN channels and all cameras.
        # Inspect receives read/send adapters and never opens a second device.
        low = np.asarray(settings['joint_low'], dtype=np.float64)
        high = np.asarray(settings['joint_high'], dtype=np.float64)
        step = np.asarray(settings['joint_max_step'], dtype=np.float64)
        if any(value.shape != (12,) or not np.isfinite(value).all() for value in (low, high, step)):
            raise ValueError('YAM EE requires finite 12-D joint limits and steps')
        eef_low = np.asarray(eef_options['eef_low'], dtype=np.float64)
        eef_high = np.asarray(eef_options['eef_high'], dtype=np.float64)
        if (eef_low.shape != (14,) or eef_high.shape != (14,)
                or not np.isfinite(eef_low).all() or not np.isfinite(eef_high).all()
                or np.any(eef_low > eef_high)):
            raise ValueError('YAM EE requires finite ordered 14-D EEF workspace bounds')
        options = {
            'left_channel': settings['left_channel'],
            'right_channel': settings['right_channel'],
            'control_hz': 30,
            'control_interface': 'eef_pos',
            'cam_width': 640,
            'cam_height': 480,
            'joint_low': tuple(np.r_[low[:6], 0, low[6:], 0]),
            'joint_high': tuple(np.r_[high[:6], 1, high[6:], 1]),
            'step_limits': tuple(np.r_[step[:6], settings.get('gripper_max_step', .15),
                                       step[6:], settings.get('gripper_max_step', .15)]),
            # Colosseum already performs the attended start gate and scoring.
            'unattended': True,
        }
        forbidden = {'control_interface', 'control_hz', 'left_channel', 'right_channel',
                     'joint_low', 'joint_high', 'step_limits', 'unattended',
                     'top_depth_serial', 'left_depth_serial', 'right_depth_serial',
                     'driver_factory', 'camera_reader', 'operator', 'poll_end',
                     'kinematics_factory', 'depth_reader', 'sleep_fn', 'clock', 'status_fn'}
        if forbidden & eef_options.keys():
            raise ValueError('inspect_eef cannot override Client-owned hardware or action settings')
        options.update(eef_options)
        # Validate native workspace, orientation and homing settings before CAN
        # or cameras open. The embodiment parses the same flat options on reset.
        from inspect_robots_yam import YamConfig
        YamConfig.from_kwargs(**options)
        self.hardware = hardware_factory(replace(config, adapter_config=settings))
        self.closed = False
        self._observation = None

        def read_cameras(_cfg):
            return {native: self.hardware.cameras[client].read()
                    for native, client in (('top_cam', 'head_image'),
                                           ('left_cam', 'left_image'),
                                           ('right_cam', 'right_image'))}

        try:
            self.embodiment = embodiment_factory(
                driver_factory=lambda _cfg: _CalibratedYAMDriver(self.hardware),
                camera_reader=read_cameras,
                **options,
            )
        except BaseException:
            self.hardware.close()
            self.closed = True
            raise

    def prepare_for_trial(self, instruction: str, run_id: str):
        from inspect_robots.scene import Scene

        if self._observation is not None:
            raise RuntimeError('YAM EE trial was already prepared')
        if not instruction or not run_id:
            raise ValueError('YAM EE trial requires an instruction and run ID')
        self._observation = self.embodiment.reset(Scene(id=run_id, instruction=instruction))

    def get_observation(self):
        if self._observation is None:
            raise RuntimeError('YAM EE robot has not been reset for this trial')
        native = self._observation
        joint = np.asarray(native.state['joint_pos'], dtype=np.float32)
        eef = np.asarray(native.state['eef_state'], dtype=np.float32)
        if joint.shape != (14,) or eef.shape != (14,) or not np.isfinite(joint).all() or not np.isfinite(eef).all():
            raise ValueError('Inspect YAM EE observation must contain finite 14-D joint_pos and eef_state')
        images = {client: native.images[source]
                  for client, source in (('head_image', 'top_cam'),
                                         ('left_image', 'left_cam'),
                                         ('right_image', 'right_cam'))}
        return RobotObservation(images, joint[[*range(6), *range(7, 13)]],
                                joint[[6, 13]], eef)

    def execute(self, action):
        from inspect_robots.types import Action

        if self._observation is None:
            raise RuntimeError('YAM EE robot has not been reset for this trial')
        target = np.asarray(action, dtype=np.float64)
        if target.shape != (14,) or not np.isfinite(target).all():
            raise ValueError('YAM EE action must contain 14 finite values')
        result = self.embodiment.step(Action(target))
        self._observation = result.observation
        command = np.asarray(result.info.get('yam_joint_command'), dtype=np.float32)
        if command.shape != (14,) or not np.isfinite(command).all():
            raise RuntimeError('Inspect YAM EE step did not report a sent joint command')
        if result.terminated or result.truncated:
            raise RuntimeError(f'Inspect YAM EE trial ended: {result.termination_reason or "unknown"}')
        return command.copy()

    async def finish_trial(self, robot_call):
        await robot_call(self.close)

    def close(self):
        if self.closed:
            return
        try:
            self.embodiment.close()
        finally:
            self.closed = True
            if not self.hardware.closed:
                self.hardware.close()
