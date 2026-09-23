"""Compatibility imports; the Franka driver lives in robots.franka."""
from .robot_interface import RobotObservation
from .robots.franka import DroidRobot, RobotEnvironment

__all__ = ["DroidRobot", "RobotEnvironment", "RobotObservation"]
