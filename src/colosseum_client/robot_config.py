from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import yaml


@dataclass(frozen=True)
class RobotClientConfig:
    url: str
    token: str
    cameras: Mapping[str, str]
    robot_type: str = "franka"
    adapter: str = "droid"
    api_url: str = ""
    evaluation_dir: str = "eval_runs"
    max_trial_steps: int = 2700
    instruction: str = ""
    control_hz: int = 15
    deadline_ms: int = 2000
    image_width: int = 512
    image_height: int = 288

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RobotClientConfig":
        with Path(path).open("r", encoding="utf-8") as config_file:
            value = yaml.safe_load(config_file)
        if not isinstance(value, dict):
            raise ValueError("robot config must be a YAML mapping")
        allowed = {
            "robot_type", "adapter", "api_url", "evaluation_dir", "max_trial_steps",
            "url",
            "token",
            "cameras",
            "instruction",
            "control_hz",
            "deadline_ms",
            "image_width",
            "image_height",
        }
        unknown = value.keys() - allowed
        missing = {"url", "token", "cameras"} - value.keys()
        if unknown:
            raise ValueError(f"robot config contains unsupported keys: {', '.join(sorted(unknown))}")
        if missing:
            raise ValueError(f"robot config is missing: {', '.join(sorted(missing))}")
        if not isinstance(value["url"], str) or not value["url"].startswith(("ws://", "wss://")):
            raise ValueError("url must use ws:// or wss://")
        if not isinstance(value["token"], str) or not value["token"]:
            raise ValueError("token must be a non-empty string")
        cameras = value["cameras"]
        valid_camera_names = {"left_image", "right_image", "head_image"}
        if not isinstance(cameras, dict) or not cameras:
            raise ValueError("cameras must map image names to camera IDs")
        if cameras.keys() - valid_camera_names:
            raise ValueError("camera names must be left_image, right_image, or head_image")
        if any(not isinstance(camera_id, (str, int)) or not str(camera_id) for camera_id in cameras.values()):
            raise ValueError("camera IDs must be non-empty strings or integers")

        config = cls(
            url=value["url"],
            token=value["token"],
            cameras={name: str(camera_id) for name, camera_id in cameras.items()},
            robot_type=value.get("robot_type", "franka"),
            adapter=value.get("adapter", "droid"),
            api_url=value.get("api_url", ""),
            evaluation_dir=value.get("evaluation_dir", "eval_runs"),
            max_trial_steps=value.get("max_trial_steps", 2700),
            instruction=value.get("instruction", ""),
            control_hz=value.get("control_hz", 15),
            deadline_ms=value.get("deadline_ms", 2000),
            image_width=value.get("image_width", 512),
            image_height=value.get("image_height", 288),
        )
        numbers = (
            config.max_trial_steps,
            config.control_hz,
            config.deadline_ms,
            config.image_width,
            config.image_height,
        )
        if any(not isinstance(number, int) or isinstance(number, bool) or number <= 0 for number in numbers):
            raise ValueError("control_hz, deadline_ms, image_width, and image_height must be positive")
        if not isinstance(config.instruction, str):
            raise ValueError("instruction must be a string")
        if not all(isinstance(v, str) and v for v in (config.robot_type, config.adapter, config.evaluation_dir)):
            raise ValueError("robot_type, adapter and evaluation_dir must be nonempty strings")
        if not isinstance(config.api_url, str) or (config.api_url and not config.api_url.startswith(("https://", "http://"))):
            raise ValueError("api_url must use http:// or https://")
        return config
