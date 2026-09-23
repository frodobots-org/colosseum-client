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


def test_robot_config_accepts_custom_camera_name(tmp_path):
    path = tmp_path / "robot.yaml"
    path.write_text(
        "url: ws://router:8443\n"
        "token: clt_test\n"
        "cameras:\n"
        "  wrist: 123\n",
        encoding="utf-8",
    )

    assert RobotClientConfig.from_yaml(path).cameras == {'wrist': '123'}


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


@pytest.mark.parametrize('robot_type', ['franka', 'yam', 'so101', 'g1', 'my_robot'])
def test_robot_type_is_the_only_selector(tmp_path, robot_type):
    path = tmp_path/'robot.yaml'
    path.write_text(f'url: ws://localhost\ntoken: test\ncameras: {{head_image: "test"}}\nrobot_type: {robot_type}\n')
    c = RobotClientConfig.from_yaml(path)
    assert c.robot_type == robot_type
    assert c.adapter is None


def test_communication_config_needs_no_cameras(tmp_path):
    path = tmp_path/'test.yaml'
    path.write_text('robot_type: test\nrouter_url: wss://localhost:18443\npolicy_server_url: wss://localhost:18444\ntoken: test\ntrack: 2\n')
    c = RobotClientConfig.from_yaml(path)
    assert c.robot_type == 'test' and c.cameras == {'head_image':'dummy'}


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


def test_recording_defaults_enabled_and_requires_boolean(tmp_path):
    path = tmp_path / "robot.yaml"
    source = "router_url: ws://localhost:8000\ntoken: example\ntest: true\n"
    path.write_text(source)
    assert RobotClientConfig.from_yaml(path).recording is True
    path.write_text(source + "recording: false\n")
    assert RobotClientConfig.from_yaml(path).recording is False
    for value in ['1', 'null', '"true"']:
        path.write_text(source + f"recording: {value}\n")
        with pytest.raises(ValueError, match="recording must"):
            RobotClientConfig.from_yaml(path)
    path.write_text(source + "recording_format: lerobot\n")
    with pytest.raises(ValueError, match="unsupported keys"):
        RobotClientConfig.from_yaml(path)


@pytest.mark.parametrize('setting,message', [
    ('adapter_config: []', 'adapter_config must'),
    ('adapter_config: {1: value}', 'adapter_config must'),
    ('robot_type: DROID', 'Use robot_type: franka'),
    ('robot_type: droid', 'Use robot_type: franka'),
    ('adapter: so101', 'legacy adapter'),
    ('robot_type: []', 'robot_type must'),
    ('robot_type: ""', 'robot_type must'),
    ('cameras: {../outside: camera}', 'camera names must'),
    ('cameras: {1: camera}', 'camera names must'),
])
def test_invalid_extension_config(tmp_path, setting, message):
    path = tmp_path / 'robot.yaml'
    path.write_text('url: ws://localhost\ntoken: test\ntest: true\n' + setting)
    with pytest.raises(ValueError, match=message):
        RobotClientConfig.from_yaml(path)


def test_adapter_config_is_separate_from_client_settings(tmp_path):
    path = tmp_path / 'robot.yaml'
    path.write_text('''url: ws://localhost
token: test
robot_type: custom
cameras: {wrist_left: camera0}
adapter_config:
  address: localhost
  arms: [left, right]
''')
    config = RobotClientConfig.from_yaml(path)
    assert config.robot_type == 'custom'
    assert config.adapter_config == {'address': 'localhost', 'arms': ['left', 'right']}
    assert config.policy_server_url == ''


def test_legacy_droid_adapter_is_accepted_only_for_franka(tmp_path):
    path = tmp_path / 'robot.yaml'
    path.write_text(
        'url: ws://localhost\ntoken: test\ntest: true\nrobot_type: franka\nadapter: droid\n'
    )
    config = RobotClientConfig.from_yaml(path)
    assert config.robot_type == 'franka' and config.adapter == 'droid'
    path.write_text(
        'url: ws://localhost\ntoken: test\ntest: true\nrobot_type: yam\nadapter: droid\n'
    )
    with pytest.raises(ValueError, match='only'):
        RobotClientConfig.from_yaml(path)


def test_local_action_contract_is_explicit_and_requires_local_policy(tmp_path):
    path = tmp_path / 'robot.yaml'
    path.write_text('url: ws://localhost\ntoken: test\ntest: true\nuse_local_action_contract: true\n')
    with pytest.raises(ValueError, match='requires policy_server_url'):
        RobotClientConfig.from_yaml(path)
    path.write_text(
        'url: ws://localhost\ntoken: test\ntest: true\n'
        'policy_server_url: ws://localhost:8000\nuse_local_action_contract: true\n'
    )
    assert RobotClientConfig.from_yaml(path).use_local_action_contract is True


@pytest.mark.parametrize('value', ['1', 'null', '"true"'])
def test_skip_upload_requires_boolean(tmp_path, value):
    path = tmp_path / 'robot.yaml'
    source = 'router_url: ws://localhost:8000\ntoken: example\ntest: true\n'
    path.write_text(source)
    assert RobotClientConfig.from_yaml(path).skip_upload is False
    path.write_text(source + 'skip_upload: true\n')
    assert RobotClientConfig.from_yaml(path).skip_upload is True
    path.write_text(source + f'skip_upload: {value}\n')
    with pytest.raises(ValueError, match='skip_upload must'):
        RobotClientConfig.from_yaml(path)
