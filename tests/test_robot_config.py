from __future__ import annotations

import pytest

from colosseum_client import RobotClientConfig


def test_robot_config_loads_wss_and_camera_ids(tmp_path):
    path = tmp_path / "robot.yaml"
    path.write_text(
        "url: wss://router.example.com:8443\n"
        "token: clt_test\n"
        "cameras:\n"
        "  left_image: 123\n"
        "  head_image: '456'\n",
        encoding="utf-8",
    )

    config = RobotClientConfig.from_yaml(path)

    assert config.url == "wss://router.example.com:8443"
    assert config.cameras == {"left_image": "123", "head_image": "456"}
    assert config.control_hz == 15


def test_robot_config_rejects_unknown_camera_name(tmp_path):
    path = tmp_path / "robot.yaml"
    path.write_text(
        "url: ws://router:8443\n"
        "token: clt_test\n"
        "cameras:\n"
        "  wrist: 123\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="camera names"):
        RobotClientConfig.from_yaml(path)


@pytest.mark.parametrize('track,expected', [(1, 'open'), (2, 'fine-tuning')])
def test_evaluation_config(tmp_path, track, expected):
    path = tmp_path / 'evaluation.yaml'
    path.write_text(f'''router_url: wss://router.example.com
token: clt_test
scene: kitchen
evaluator: operator_1
track: {track}
policy_server_url: wss://local.example.com:8000
cameras:
  head_image: '123'
''')
    c = RobotClientConfig.from_yaml(path)
    assert c.track == expected and c.scene == 'kitchen' and c.evaluator == 'operator_1'
    assert c.policy_server_url == 'wss://local.example.com:8000'


@pytest.mark.parametrize('setting', ['track: true', 'track: 3', 'track: 1.0',
                                    'policy_server_url: https://localhost', 'policy_server_url: 123'])
def test_invalid_evaluation_config(tmp_path, setting):
    path = tmp_path / 'evaluation.yaml'
    path.write_text('url: ws://localhost\ntoken: test\ncameras: {head_image: "123"}\n' + setting)
    with pytest.raises(ValueError):
        RobotClientConfig.from_yaml(path)


@pytest.mark.parametrize('robot_type,adapter', [('franka','droid'), ('yam','yam')])
def test_robot_type_selects_identity_and_adapter_default(tmp_path, robot_type, adapter):
    path = tmp_path/'robot.yaml'
    path.write_text(f'url: ws://localhost\ntoken: test\ncameras: {{head_image: "test"}}\nrobot_type: {robot_type}\n')
    c = RobotClientConfig.from_yaml(path)
    assert c.robot_type == robot_type and c.adapter == adapter


def test_communication_config_needs_no_cameras(tmp_path):
    path = tmp_path/'test.yaml'
    path.write_text('robot_type: test\nrouter_url: wss://localhost:18443\npolicy_server_url: wss://localhost:18444\ntoken: test\ntrack: 2\n')
    c = RobotClientConfig.from_yaml(path)
    assert c.robot_type == 'test' and c.adapter == 'test' and c.cameras == {'head_image':'dummy'}


def test_test_robot_cli_uses_shared_evaluation(tmp_path, monkeypatch):
    from colosseum_client import robot_cli, evaluation
    path = tmp_path/'test.yaml'
    path.write_text('robot_type: test\nurl: wss://localhost\npolicy_server_url: wss://localhost:8000\ntoken: test\ntrack: 2\n')
    seen = []
    monkeypatch.setattr('sys.argv', ['colosseum-robot', str(path)])
    monkeypatch.setattr(evaluation, 'run_evaluation', lambda config, **kwargs: seen.append(config.robot_type))
    monkeypatch.setattr(robot_cli, 'run_robot', lambda *a, **k: pytest.fail('Must not access robot'))
    robot_cli.main()
    assert seen == ['test']

@pytest.mark.parametrize('robot', ['franka','yam'])
def test_dummy_flag_keeps_robot_identity(tmp_path, robot):
    from colosseum_client.adapters import make_robot, TestRobot
    path=tmp_path/'robot.yaml'
    path.write_text(f'robot_type: {robot}\ntest: true\nurl: ws://localhost\ntoken: t\n')
    config=RobotClientConfig.from_yaml(path)
    assert config.robot_type==robot and config.test is True
    adapter=make_robot(config)
    assert isinstance(adapter,TestRobot) and adapter.get_observation().images['head_image'].shape==(288,512,3)

@pytest.mark.parametrize('value', ['1','"true"','null'])
def test_test_flag_requires_boolean(tmp_path,value):
    path=tmp_path/'robot.yaml'
    path.write_text(f'robot_type: franka\ntest: {value}\nurl: ws://localhost\ntoken: t\n')
    with pytest.raises(ValueError,match='test must'):
        RobotClientConfig.from_yaml(path)
