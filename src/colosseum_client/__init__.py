"""Colosseum robot client."""

from .client import ColosseumClient, ProtocolError
from .droid_robot import DroidRobot, RobotObservation
from .robot_config import RobotClientConfig

__all__ = [
    "ColosseumClient",
    "DroidRobot",
    "ProtocolError",
    "RobotClientConfig",
    "RobotObservation",
]
