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
