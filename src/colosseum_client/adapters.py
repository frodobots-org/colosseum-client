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
    if config.adapter == 'droid':
        if config.robot_type not in {'franka', 'DROID'}:
            raise ValueError('The DROID adapter requires robot_type franka or DROID')
        return DroidRobot(config.cameras, image_size=(config.image_width, config.image_height))
    raise ValueError(f'Robot adapter {config.adapter!r} is not installed')
