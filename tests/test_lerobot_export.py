import json
import subprocess

import numpy as np
import pyarrow.parquet as pq
import pytest

from colosseum_client.droid_robot import RobotObservation
from colosseum_client.lerobot_export import export_trial
from colosseum_client.recording import TrialRecorder


def record(path):
    recorder = TrialRecorder(path, ["head_image", "left_image"])
    for i in range(3):
        obs = RobotObservation(
            {c: np.full((25, 31, 3), i * 50, dtype=np.uint8) for c in recorder.cameras},
            np.arange(7, dtype=np.float32) + i, np.array([i / 2]), np.arange(6))
        recorder.add(obs, np.arange(8) + i, captured_at=recorder.started + [0.1, 0.4, 1.1][i])
    recorder.add(obs)
    recorder.close()


def export(path, **kwargs):
    return export_trial(path, ["head_image", "left_image"], fps=15, robot_type="franka",
                        task="Open the lid", metadata={"test": True, **kwargs})


def test_v3_preserves_frame_action_alignment_and_real_capture_times(tmp_path):
    record(tmp_path)
    output = export(tmp_path)
    info = json.loads((output / "meta/info.json").read_text())
    assert info["codebase_version"] == "v3.0"
    assert info["total_episodes"] == 1 and info["total_frames"] == 3
    table = pq.read_table(output / "data/chunk-000/file-000.parquet").to_pydict()
    assert table["observation.state"][1] == list(np.arange(7) + 1) + [0.5]
    assert table["action"][2] == list(np.arange(8) + 2)
    np.testing.assert_allclose(table["observation.capture_timestamp"], [0.1, 0.4, 1.1])
    np.testing.assert_allclose(table["timestamp"], np.arange(3) / 15)
    episode = pq.read_table(output / "meta/episodes/chunk-000/file-000.parquet").to_pylist()[0]
    assert episode["dataset_from_index"] == 0 and episode["dataset_to_index"] == 3
    assert episode["tasks"] == ["Open the lid"]
    stats = json.loads((output / "meta/stats.json").read_text())
    assert stats["action"]["count"] == [3]
    for camera in ["head_image", "left_image"]:
        video = output / f"videos/observation.images.{camera}/chunk-000/file-000.mp4"
        stream = json.loads(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(video)]))["streams"][0]
        assert int(stream["nb_frames"]) == 3
        assert (stream["height"], stream["width"]) == (26, 32)
    assert len((tmp_path / "frames.jsonl").read_text().splitlines()) == 4
    assert export(tmp_path) == output
    with pytest.raises(ValueError, match="differs"):
        export(tmp_path, institution="changed")


def test_export_failure_is_atomic_and_retryable(tmp_path, monkeypatch):
    record(tmp_path)
    original = subprocess.run
    def fail(*args, **kwargs):
        raise RuntimeError("encoder failed")
    monkeypatch.setattr(subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="encoder failed"):
        export(tmp_path)
    assert not (tmp_path / "lerobot").exists()
    assert not list(tmp_path.glob(".lerobot-*"))
    monkeypatch.setattr(subprocess, "run", original)
    output = export(tmp_path)
    (output / "meta/stats.json").write_text("{}")
    with pytest.raises(ValueError, match="modified"):
        export(tmp_path)


def test_legacy_recording_does_not_invent_gripper_state(tmp_path):
    record(tmp_path)
    path = tmp_path / "frames.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    for row in records:
        del row["gripper"]
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    with pytest.raises(ValueError, match="gripper"):
        export(tmp_path)


def test_incomplete_recording_is_rejected(tmp_path):
    record(tmp_path)
    (tmp_path / "recording-status.json").write_text('{"complete": false, "frames": 4}')
    with pytest.raises(ValueError, match="incomplete"):
        export(tmp_path)
