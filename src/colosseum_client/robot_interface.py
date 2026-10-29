"""Shared robot contract, independent of any hardware SDK or model runtime."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Mapping
from dataclasses import field

import numpy as np


@dataclass(frozen=True)
class RobotObservation:
    """RGB images and state vectors in the adapter/model's agreed ordering.

    ``extra_state`` carries additional named vectors (a mobile base's
    ``navigation_state``, for example). They reach the policy as
    ``Observation.state[name]`` and are recorded with the trial. Images that are
    not configured cameras (such as a ``goal_image`` reference photo) are sent
    to the policy but never recorded or exported.
    """

    images: Mapping[str, np.ndarray]
    joints: np.ndarray
    gripper: np.ndarray
    cartesian_position: np.ndarray
    extra_state: Mapping[str, np.ndarray] = field(default_factory=dict)


class Robot(ABC):
    """Base class for hardware drivers and the synthetic test robot.

    Implement all three lifecycle methods before instantiation. Concrete drivers
    also supply the hardware metadata below, either as class or instance fields.

    Optional hooks the evaluation loop calls when present: ``prepare_for_task(task)``
    once per trial with the Router's task (goal, limits) before the first
    observation, and ``async finish_trial(robot_call)`` after the last step.
    """
    joint_count: int
    has_gripper: bool
    action_dim: int
    action_space_name: str

    @abstractmethod
    def get_observation(self) -> RobotObservation:
        """Read RGB images and state in the agreed model-facing format."""
        raise NotImplementedError

    @abstractmethod
    def execute(self, action: np.ndarray) -> np.ndarray | None:
        """Validate and execute one action; EEF drivers return accepted joint commands."""
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Safely stop/hold and release resources; allow repeated cleanup."""
        raise NotImplementedError
