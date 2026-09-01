from __future__ import annotations

import numpy as np
import pytest

from colosseum_client import DroidRobot


class FakeRobotEnv:
    def __init__(self) -> None:
        self.actions: list[np.ndarray] = []
        self.closed = False

    def get_observation(self):
        image = np.zeros((2, 3, 3), dtype=np.uint8)
        image[:, :, 0] = 10  # B
        image[:, :, 1] = 20  # G
        image[:, :, 2] = 30  # R
        return {
            "image": {"camera-123-left": image},
            "robot_state": {
                "joint_positions": np.arange(7, dtype=np.float32),
                "gripper_position": 0.25,
                "cartesian_position": np.arange(6, dtype=np.float32),
            },
        }

    def step(self, action):
        self.actions.append(np.asarray(action))

    def close(self):
        self.closed = True


def test_droid_robot_reads_images_and_separate_state():
    environment = FakeRobotEnv()
    robot = DroidRobot(
        {"head_image": "123"},
        image_size=(3, 2),
        environment=environment,
    )

    observation = robot.get_observation()

    assert observation.images["head_image"].shape == (2, 3, 3)
    np.testing.assert_array_equal(observation.images["head_image"][0, 0], [30, 20, 10])
    np.testing.assert_array_equal(observation.joints, np.arange(7, dtype=np.float32))
    np.testing.assert_array_equal(observation.gripper, [0.25])


def test_droid_robot_binarizes_gripper_and_clips_velocity():
    environment = FakeRobotEnv()
    robot = DroidRobot(
        {"head_image": "123"},
        action_space="joint_velocity",
        image_size=(3, 2),
        environment=environment,
    )

    robot.execute(np.array([2, -2, 0, 0, 0, 0, 0, 0.75], dtype=np.float32))

    np.testing.assert_array_equal(environment.actions[0], [1, -1, 0, 0, 0, 0, 0, 1])


def test_droid_robot_rejects_bad_action():
    robot = DroidRobot(
        {"head_image": "123"},
        image_size=(3, 2),
        environment=FakeRobotEnv(),
    )
    with pytest.raises(ValueError, match="8 values"):
        robot.execute(np.zeros(7))
