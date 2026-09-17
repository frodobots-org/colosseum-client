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
    institution: str = ""
    scene: str = ""
    evaluator: str = ""
    track: str = ""
    policy_server_url: str = ""
    prepare_timeout: int = 1800
    test: bool = False
    robot_type: str = "franka"
    adapter: str = "droid"
    api_url: str = ""
    evaluation_dir: str = "eval_runs"
    recording: bool = True
    max_trial_steps: int = 2700
    instruction: str = ""
    control_hz: int = 15
    deadline_ms: int = 30000
    image_width: int = 512
    image_height: int = 288

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RobotClientConfig":
        with Path(path).open("r", encoding="utf-8") as config_file:
            value = yaml.safe_load(config_file)
        if not isinstance(value, dict):
            raise ValueError("robot config must be a YAML mapping")
        if "router_url" in value:
            if "url" in value and value["url"] != value["router_url"]:
                raise ValueError("url and router_url disagree")
            value["url"] = value.pop("router_url")
        raw_track = value.get("track", "")
        if type(raw_track) not in (str, int) or raw_track not in ("", 1, 2, "open", "fine-tuning"):
            raise ValueError("track must be 1 (Open) or 2 (Fine-tuning)")
        value["track"] = {1: "open", 2: "fine-tuning"}.get(raw_track, raw_track)
        if value.get("policy_server_url") is None:
            value["policy_server_url"] = ""
        if type(value.get("test", False)) is not bool:
            raise ValueError("test must be true or false")
        if value.get("robot_type") == "test":
            value["test"] = True  # Legacy configuration remains marked synthetic.
        if value.get("test", False):
            value.setdefault("cameras", {})
        allowed = {
            "institution", "test", "scene", "evaluator", "track", "policy_server_url", "prepare_timeout",
            "robot_type", "adapter", "api_url", "evaluation_dir", "recording", "max_trial_steps",
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
        if not isinstance(cameras, dict) or (not cameras and not value.get("test", False)):
            raise ValueError("cameras must map image names to camera IDs")
        if cameras.keys() - valid_camera_names:
            raise ValueError("camera names must be left_image, right_image, or head_image")
        if any(not isinstance(camera_id, (str, int)) or not str(camera_id) for camera_id in cameras.values()):
            raise ValueError("camera IDs must be non-empty strings or integers")

        robot_type = value.get("robot_type", "franka")
        if robot_type not in ("franka", "yam", "test", "DROID"):
            raise ValueError("robot_type must be franka, yam or test (legacy DROID is also accepted)")

        config = cls(
            url=value["url"],
            token=value["token"],
            cameras={name: str(camera_id) for name, camera_id in cameras.items()} or ({'head_image':'dummy'} if value.get('test', False) else {}),
            scene=value.get("scene", ""),
            institution=value.get("institution", ""),
            evaluator=value.get("evaluator", ""),
            track=value["track"],
            policy_server_url=value["policy_server_url"],
            prepare_timeout=value.get("prepare_timeout", 1800),
            robot_type=robot_type,
            test=value.get("test", False),
            adapter=value.get("adapter", robot_type if robot_type in {"yam", "test"} else "droid"),
            api_url=value.get("api_url", ""),
            evaluation_dir=value.get("evaluation_dir", "eval_runs"),
            recording=value.get("recording", True),
            max_trial_steps=value.get("max_trial_steps", 2700),
            instruction=value.get("instruction", ""),
            control_hz=value.get("control_hz", 15),
            deadline_ms=value.get("deadline_ms", cls.deadline_ms),
            image_width=value.get("image_width", 512),
            image_height=value.get("image_height", 288),
        )
        if type(config.recording) is not bool:
            raise ValueError("recording must be true or false")
        numbers = (
            config.prepare_timeout,
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
        from urllib.parse import urlsplit
        for name in ("url", "policy_server_url"):
            url = getattr(config, name)
            if name == "policy_server_url" and url == "":
                continue
            if not isinstance(url, str):
                raise ValueError(f"{name} must be a URL string")
            parts = urlsplit(url)
            if parts.scheme not in {"ws", "wss"} or not parts.hostname or parts.username or parts.password or parts.fragment:
                raise ValueError(f"{name} must be a ws:// or wss:// URL without credentials or fragment")
        if not isinstance(config.scene, str) or len(config.scene) > 4000:
            raise ValueError("scene must be a string of at most 4000 characters")
        if not isinstance(config.evaluator, str) or len(config.evaluator) > 200:
            raise ValueError("evaluator must be a string of at most 200 characters")
        if config.prepare_timeout > 1800:
            raise ValueError("prepare_timeout must not exceed 1800 seconds")
        if not isinstance(config.institution, str) or len(config.institution) > 200 or any(ord(c) < 32 for c in config.institution):
            raise ValueError('institution must be a string of at most 200 printable characters')
        return config
