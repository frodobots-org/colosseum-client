"""Robot-specific implementations sit behind this evaluation interface."""
from typing import Protocol
from .droid_robot import DroidRobot, RobotObservation


class RobotAdapter(Protocol):
    joint_count: int
    has_gripper: bool
    action_dim: int
    action_space_name: str
    def get_observation(self) -> RobotObservation: ...
    def execute(self, action): ...
    def close(self): ...


def make_robot(config) -> RobotAdapter:
    if config.test or config.robot_type == 'test':
        return TestRobot(config)
    if config.adapter == 'test':
        raise ValueError('Dummy adapter requires test: true')
    if config.adapter == 'droid':
        if config.robot_type not in {'franka', 'DROID'}:
            raise ValueError('The DROID adapter requires robot_type franka or DROID')
        return DroidRobot(config.cameras, image_size=(config.image_width, config.image_height))
    raise ValueError(f'Robot adapter {config.adapter!r} is not installed')


class TestRobot:
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
        self.joint_count = max(1, self.action_dim - 1)
        self.joints = np.zeros(self.joint_count, dtype=np.float32)

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
        self.joints, self.gripper = value[:-1].copy(),value[-1:].copy()
        self.step += 1

    def close(self):
        pass
