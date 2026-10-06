"""Robot I/O helpers shared by evaluation and inference-only runners."""

import logging
import sys


async def close_resources(resources):
    """Attempt every cleanup; preserve a pending error or raise the first close error."""
    original = sys.exc_info()[1]
    first_error = None
    for label, close in resources:
        try:
            await close()
        except Exception as exc:
            # Exception text can contain credentials or signed URLs.
            logging.getLogger(__name__).warning('%s cleanup failed (%s)', label, type(exc).__name__)
            if first_error is None:
                first_error = exc
    if original is None and first_error is not None:
        raise first_error


def read_observation(robot, step, phase):
    return robot.get_observation()


def execute_robot_action(robot, action, step, enabled=True):
    if enabled:
        robot.execute(action)


async def execute_robot_action_async(robot, action, step, robot_call, enabled=True):
    if not enabled:
        return
    execute = getattr(robot, 'execute_async', None)
    if execute is not None:
        await execute(action, robot_call)
    else:
        await robot_call(execute_robot_action, robot, action, step, True)
