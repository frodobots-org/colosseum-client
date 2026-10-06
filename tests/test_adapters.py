from importlib.metadata import EntryPoint
from pathlib import Path

import pytest
import yaml

from colosseum_client import RobotClientConfig, make_robot
from colosseum_client import adapters


def config(**kwargs):
    return RobotClientConfig(url='ws://localhost', token='test', cameras={'wrist': 'camera0'}, **kwargs)


@pytest.mark.parametrize('name', ['g1'])
def test_todo_adapters_fail_explicitly(name):
    with pytest.raises(NotImplementedError, match='hardware control is not implemented'):
        make_robot(config(robot_type=name))


@pytest.mark.parametrize('name', ['so101', 'yam', 'g1', 'missing_robot'])
def test_dummy_mode_does_not_load_hardware(name):
    robot = make_robot(config(robot_type=name, test=True))
    assert isinstance(robot, adapters.TestRobot)
    assert robot.get_observation().images['wrist'].shape == (288, 512, 3)


@pytest.fixture
def external_adapter(tmp_path, monkeypatch):
    (tmp_path / 'partner_robot.py').write_text('''
from colosseum_client import Robot, RobotObservation

class PartnerRobot(Robot):
    joint_count = 2
    has_gripper = False
    action_dim = 2
    action_space_name = 'joint_position'

    def __init__(self, config):
        self.settings = config.adapter_config
        self.closed = False

    def get_observation(self):
        import numpy as np
        return RobotObservation({'wrist': np.zeros((2, 2, 3), dtype=np.uint8)},
                                np.zeros(2), np.zeros(0), np.zeros(6))

    def execute(self, action):
        self.last_action = action

    def close(self):
        self.closed = True

not_a_factory = 1
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    # Avoid a cached temporary module leaking between tests.
    monkeypatch.delitem(__import__('sys').modules, 'partner_robot', raising=False)


def test_external_package_loads_by_robot_type(external_adapter, monkeypatch):
    def discover(**kwargs):
        assert kwargs == {'group': 'colosseum_client.adapters', 'name': 'partner'}
        return [EntryPoint(name='partner', value='partner_robot:PartnerRobot', group=kwargs['group'])]
    monkeypatch.setattr(adapters, 'entry_points', discover)
    robot = make_robot(config(robot_type='partner', adapter_config={'port': 'test-port'}))
    assert robot.settings == {'port': 'test-port'}
    assert robot.get_observation().joints.shape == (2,)
    robot.execute([0.1, 0.2])
    assert robot.last_action == [0.1, 0.2]
    robot.close()
    assert robot.closed


def test_missing_or_ambiguous_entry_point(monkeypatch):
    monkeypatch.setattr(adapters, 'entry_points', lambda **kwargs: [])
    with pytest.raises(ValueError, match='not installed'):
        make_robot(config(robot_type='unknown'))
    monkeypatch.setattr(adapters, 'entry_points', lambda **kwargs: [object(), object()])
    with pytest.raises(ValueError, match='Multiple installed adapters'):
        make_robot(config(robot_type='ambiguous'))


def test_non_callable_factory(external_adapter, monkeypatch):
    monkeypatch.setattr(adapters, 'entry_points', lambda **kwargs: [
        EntryPoint(name='partner', value='partner_robot:not_a_factory', group=adapters.ADAPTER_ENTRY_POINT_GROUP)
    ])
    with pytest.raises(TypeError, match='callable factory'):
        make_robot(config(robot_type='partner'))


def test_droid_still_uses_existing_factory(monkeypatch):
    calls = []
    monkeypatch.setattr(adapters, 'DroidRobot', lambda *args, **kwargs: calls.append((args, kwargs)))
    make_robot(config())
    assert calls == [(({'wrist': 'camera0'},), {
        'action_space': 'joint_position', 'image_size': (512, 288)
    })]


@pytest.mark.parametrize('settings', [{'robot_type': 'g1'}, {'test': True}])
async def test_legacy_runner_does_not_fall_back_to_droid(settings):
    from colosseum_client.robot_runner import run_robot
    with pytest.raises(ValueError, match='use evaluation mode'):
        await run_robot(config(**settings), 'test')


@pytest.mark.parametrize('name', ['g1'])
def test_shared_example_selects_todo_robot_without_opening_hardware(tmp_path, name):
    example = Path(__file__).parents[1] / 'configs' / 'robot.yaml.example'
    values = yaml.safe_load(example.read_text())
    values.update(robot_type=name, test=False, cameras={'head_image': 'camera0'})
    path = tmp_path / 'robot.yaml'
    path.write_text(yaml.safe_dump(values))
    settings = RobotClientConfig.from_yaml(path)
    assert settings.robot_type == name
    assert not settings.test
    with pytest.raises(NotImplementedError, match='hardware control is not implemented'):
        make_robot(settings)
