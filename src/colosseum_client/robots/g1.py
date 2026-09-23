"""G1 hardware adapter TODO; this is not an operational driver."""
import numpy as np

from ..robot_interface import Robot, RobotObservation
from ..robot_config import RobotClientConfig


class G1Robot(Robot):
    robot_name = 'G1'
    # TODO: Select the SDK/network interface, controlled joint subset, hand
    # variant and command mode; define ownership with the balance/WBC controller.

    # TODO: Set these from the actual controlled joints/end effectors and policy
    # contract. Do not infer them from the robot's total physical joint count.
    joint_count: int
    has_gripper: bool
    action_dim: int
    action_space_name: str

    def __init__(self, config: RobotClientConfig) -> None:
        self.config = config
        self.settings = config.adapter_config
        # TODO: Validate settings and connect to the hardware on this thread.
        # Release any partially initialized resources if connection fails.
        raise NotImplementedError(f'{self.robot_name} hardware control is not implemented')

    def get_observation(self) -> RobotObservation:
        # TODO: Return camera-role -> RGB uint8 HWC images, joint/gripper vectors,
        # and Cartesian state using the current RobotObservation schema.
        raise NotImplementedError(f'{self.robot_name}: TODO read real observations')

    def execute(self, action: np.ndarray) -> None:
        # TODO: Validate shape, finite values, units, order and hardware limits;
        # map the policy command to the SDK. Implement bounded I/O and watchdogs.
        raise NotImplementedError(f'{self.robot_name}: TODO execute hardware action')

    def close(self) -> None:
        # TODO: Safely stop/hold the robot and release SDK/camera resources.
        # Make cleanup safe to repeat, including after partial initialization.
        raise NotImplementedError(f'{self.robot_name}: TODO stop and release hardware')
