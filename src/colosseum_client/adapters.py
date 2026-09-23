"""Robot-specific implementations sit behind this evaluation interface."""
from importlib import import_module
from importlib.metadata import entry_points
import numpy as np
from .robots.franka import DroidRobot
from .robot_config import RobotClientConfig
from .robot_interface import Robot, RobotObservation


ADAPTER_ENTRY_POINT_GROUP = 'colosseum_client.adapters'
_TODO_ADAPTERS = {
    'so101': 'colosseum_client.robots.so101:SO101Robot',
    'yam': 'colosseum_client.robots.yam:YAMRobot',
    'g1': 'colosseum_client.robots.g1:G1Robot',
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


def make_robot(config: RobotClientConfig, *, action_space: str = "joint_position") -> Robot:
    if config.test or config.robot_type == 'test':
        return TestRobot(config)
    if config.robot_type == 'franka':
        return DroidRobot(config.cameras, action_space=action_space,
                          image_size=(config.image_width, config.image_height))
    return load_adapter_factory(config.robot_type)(config)



class TestRobot(Robot):
    """Same adapter contract; generates state/RGB and applies actions in memory only."""
    joint_count = 7
    has_gripper = True
    action_dim = 8
    action_space_name = 'joint_position'

    def __init__(self, config):
        import numpy as np
        self.cameras = list(config.cameras)
        self.width, self.height = config.image_width, config.image_height
        self.joints = np.zeros(7, dtype=np.float32)
        self.gripper = np.zeros(1, dtype=np.float32)
        self.step = 0

    def configure_model(self, model):
        import numpy as np
        self.action_dim = model['action_dim']
        self.action_space_name = model['action_space']
        self.joint_count = 7
        self.joints = np.zeros(7, dtype=np.float32)

    def get_observation(self):
        import numpy as np
        images = {}
        for camera in self.cameras:
            image = np.zeros((self.height,self.width,3), dtype=np.uint8)
            image[:] = [32,128,224]
            image[:,self.step % self.width,:] = 255
            images[camera] = image
        return RobotObservation(images, self.joints.copy(), self.gripper.copy(), np.zeros(6,dtype=np.float32))

    def execute(self, action):
        import numpy as np
        value = np.asarray(action,dtype=np.float32)
        if value.shape != (self.action_dim,) or not np.isfinite(value).all():
            raise ValueError('Invalid dummy robot action')
        if self.action_space_name.startswith('joint'):
            self.joints = value[:-1].copy()
        self.gripper = value[-1:].copy()
        self.step += 1

    def close(self):
        pass
