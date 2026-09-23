"""Colosseum robot client."""

from .client import ColosseumClient, ProtocolError
from .robots.franka import DroidRobot
from .robot_interface import Robot, RobotObservation
from .robot_config import RobotClientConfig
from .adapters import make_robot

__all__ = [
    "ColosseumClient",
    "DroidRobot",
    "ProtocolError",
    "RobotClientConfig",
    "RobotObservation",
    "Robot",
    "make_robot",
]
