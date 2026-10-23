"""Robot-specific implementations sit behind this evaluation interface."""
from dataclasses import replace
from importlib import import_module
from importlib.metadata import entry_points
import numpy as np
from .robots.franka import DroidRobot
from .navigation import ACTION_SPACE, GOAL_GPS_KEY, NAV_STATE_KEY, NAV_STATE_NAMES, ROVER_TYPE
from .robot_config import RobotClientConfig
from .robot_interface import Robot, RobotObservation


ADAPTER_ENTRY_POINT_GROUP = 'colosseum_client.adapters'
_TODO_ADAPTERS = {
    'so101': 'colosseum_client.robots.so101:SO101Robot',
    'yam': 'colosseum_client.robots.yam:YAMRobot',
    'g1': 'colosseum_client.robots.g1:G1Robot',
    ROVER_TYPE: 'colosseum_client.robots.earth_rover_mini_plus:EarthRoverMiniPlusRobot',
}


def load_adapter_factory(name: str):
    """Resolve an installed entry point or an explicit module:factory reference."""
    target = _TODO_ADAPTERS.get(name, name)
    if ':' in target:
        module, attribute = target.split(':', 1)
        if not module or not attribute or ':' in attribute:
            raise ValueError('Adapter reference must be module:factory')
        factory = getattr(import_module(module), attribute)
    else:
        matches = list(entry_points(group=ADAPTER_ENTRY_POINT_GROUP, name=name))
        if not matches:
            raise ValueError(
                f'Robot adapter {name!r} is not installed; install a package with a matching robot entry point'
            )
        if len(matches) != 1:
            raise ValueError(f'Multiple installed adapters use {name!r}; keep only one implementation for this robot type')
        factory = matches[0].load()
    if not callable(factory):
        raise TypeError(f'Robot adapter {name!r} must be a callable factory(config)')
    return factory


def make_robot(config: RobotClientConfig, *, action_space: str = "joint_position", control_mode: str = "joint_pos") -> Robot:
    if config.test or config.robot_type == 'test':
        return TestRobot(config, control_mode=control_mode)
    if config.robot_type == 'yam' and action_space == 'eef_abs_pose':
        from .robots.yam_eef import YAMEEFRobot
        return YAMEEFRobot(config)
    if config.robot_type == 'yam' and 'inspect_eef' in config.adapter_config:
        config = replace(config, adapter_config={key: value for key, value in config.adapter_config.items()
                                                 if key != 'inspect_eef'})
    if config.robot_type == 'franka':
        return DroidRobot(config.cameras, action_space=action_space,
                          image_size=(config.image_width, config.image_height))
    if config.robot_type == 'so101' and control_mode == 'eef_delta_pose':
        return load_adapter_factory(config.robot_type)(config, control_mode=control_mode)
    return load_adapter_factory(config.robot_type)(config)



class TestRobot(Robot):
    """Same adapter contract; generates state/RGB and applies actions in memory only."""
    joint_count = 7
    has_gripper = True
    action_dim = 8
    action_space_name = 'joint_position'

    def __init__(self, config, *, control_mode='joint_pos'):
        import numpy as np
        self.robot_type = config.robot_type
        self.cameras = list(config.cameras)
        self.width, self.height = config.image_width, config.image_height
        self.joints = np.zeros(7, dtype=np.float32)
        self.gripper = np.zeros(1, dtype=np.float32)
        self.eef_state = np.zeros(14, dtype=np.float32)
        self.step = 0
        self.control_mode = control_mode
        self.goal_gps = None
        if self.robot_type == ROVER_TYPE:
            # Synthetic mobile base: no joints, a 2-D velocity command, dead-reckoned navigation state.
            self.joint_count, self.has_gripper = 0, False
            self.action_dim, self.action_space_name = 2, ACTION_SPACE
            self.joints = np.zeros(0, dtype=np.float32)
            self.gripper = np.zeros(0, dtype=np.float32)
            self.navigation = np.array([37.0, -122.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def prepare_for_task(self, task):
        import numpy as np
        goal = task.get('goal') or {}
        if self.robot_type == ROVER_TYPE and goal.get('type') == 'gps':
            self.goal_gps = np.array([goal['latitude'], goal['longitude']], dtype=np.float64)

    def configure_model(self, model):
        import numpy as np
        self.action_dim = model['action_dim']
        self.action_space_name = model['action_space']
        if self.robot_type == ROVER_TYPE:
            if self.action_dim != 2 or self.action_space_name != ACTION_SPACE:
                raise ValueError('Synthetic Earth Rover requires 2-D base_velocity')
            self.joint_count = 0
            return
        self.joint_count = 7
        if self.robot_type == 'yam' and self.action_space_name in {'joint_position', 'eef_abs_pose'}:
            if self.action_dim != 14:
                raise ValueError('Synthetic YAM requires 14-D actions')
            self.joint_count = 12
            self.gripper = np.zeros(2, dtype=np.float32)
        if self.robot_type == 'so101' and self.action_space_name == 'joint_position':
            if self.action_dim != 6:
                raise ValueError('Synthetic SO101 requires 6-D joint_position')
            self.joint_count = 5
            if self.control_mode == 'eef_delta_pose':
                self.action_dim = 7
        self.joints = np.zeros(self.joint_count, dtype=np.float32)

    def get_observation(self):
        import numpy as np
        images = {}
        for camera in self.cameras:
            image = np.zeros((self.height,self.width,3), dtype=np.uint8)
            image[:] = [32,128,224]
            image[:,self.step % self.width,:] = 255
            images[camera] = image
        if self.robot_type == ROVER_TYPE:
            extra = {NAV_STATE_KEY: self.navigation.copy()}
            if self.goal_gps is not None:
                extra[GOAL_GPS_KEY] = self.goal_gps.copy()
            return RobotObservation(images, self.joints.copy(), self.gripper.copy(), np.zeros(0, dtype=np.float32), extra)
        cartesian = self.eef_state.copy() if self.action_space_name == 'eef_abs_pose' else np.zeros(6,dtype=np.float32)
        extra = ({'eef_state': np.r_[np.zeros(6, dtype=np.float32), self.gripper].astype(np.float32),
                  'eef_quat_wxyz': np.array([1, 0, 0, 0], dtype=np.float32)}
                 if self.control_mode == 'eef_delta_pose' else {})
        return RobotObservation(images, self.joints.copy(), self.gripper.copy(), cartesian, extra)

    def execute(self, action):
        import numpy as np
        value = np.asarray(action,dtype=np.float32)
        if value.shape != (self.action_dim,) or not np.isfinite(value).all():
            raise ValueError('Invalid dummy robot action')
        if self.control_mode == 'eef_delta_pose':
            self.gripper = np.clip(self.gripper + value[-1:], 0, 100)
            self.step += 1
            return np.r_[self.joints, self.gripper].copy()
        if self.robot_type == 'yam' and self.action_dim == 14 and self.action_space_name == 'joint_position':
            self.joints = np.r_[value[:6], value[7:13]]
            self.gripper = value[[6, 13]].copy()
        elif self.robot_type == 'yam' and self.action_space_name == 'eef_abs_pose':
            self.eef_state = value.copy()
            self.gripper = value[[6, 13]].copy()
            self.step += 1
            return np.r_[self.joints[:6], self.gripper[:1], self.joints[6:], self.gripper[1:]]
        elif self.robot_type == ROVER_TYPE:
            linear, angular = np.clip(value, -1.0, 1.0)
            period = 0.1
            heading = (self.navigation[2] + angular * period * 90.0) % 360.0
            radians = np.deg2rad(heading)
            self.navigation[0] += linear * period * 1e-5 * np.cos(radians)
            self.navigation[1] += linear * period * 1e-5 * np.sin(radians)
            self.navigation[2], self.navigation[3] = heading, linear
            self.navigation[5:9] = linear * 100.0
        else:
            if self.action_space_name.startswith('joint'):
                self.joints = value[:-1].copy()
            self.gripper = value[-1:].copy()
        self.step += 1

    def close(self):
        pass
