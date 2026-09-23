import pytest

from colosseum_client import Robot
from colosseum_client.adapters import TestRobot as SyntheticRobot
from colosseum_client.robots.franka import DroidRobot
from colosseum_client.robots.g1 import G1Robot
from colosseum_client.robots.so101 import SO101Robot
from colosseum_client.robots.yam import YAMRobot


def test_base_cannot_be_instantiated():
    with pytest.raises(TypeError, match='abstract'):
        Robot()


@pytest.mark.parametrize('missing', ['get_observation', 'execute', 'close'])
def test_each_lifecycle_method_is_required(missing):
    methods = {
        'get_observation': lambda self: None,
        'execute': lambda self, action: None,
        'close': lambda self: None,
    }
    methods.pop(missing)
    incomplete = type('IncompleteRobot', (Robot,), methods)
    with pytest.raises(TypeError, match=missing):
        incomplete()


@pytest.mark.parametrize('robot', [DroidRobot, SO101Robot, YAMRobot, G1Robot, SyntheticRobot])
def test_builtin_robots_inherit_and_override_lifecycle_methods(robot):
    assert issubclass(robot, Robot)
    assert not robot.__abstractmethods__
