"""Robot I/O helpers shared by evaluation and inference-only runners."""


def read_observation(robot, step, phase):
    return robot.get_observation()


def execute_robot_action(robot, action, step, enabled=True):
    if enabled:
        robot.execute(action)
