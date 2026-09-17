"""Backfill datasets for owned, finished evaluations without executing a robot."""
import argparse
import json
from pathlib import Path
import re

from .evaluation import EvalAPI, save
from .robot_config import RobotClientConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("--evaluation-id", help="Upload one evaluation; defaults to all local datasets")
    args = parser.parse_args()
    if args.evaluation_id and not re.fullmatch(r"[A-Za-z0-9_-]+", args.evaluation_id):
        parser.error("Invalid evaluation ID")
    config = RobotClientConfig.from_yaml(args.config)
    root = Path(config.evaluation_dir).resolve()
    manifests = ([root / args.evaluation_id / "manifest.json"] if args.evaluation_id
                 else sorted(root.glob("*/manifest.json")))
    api = EvalAPI(config)
    count = 0
    try:
        for manifest in manifests:
            local = json.loads(manifest.read_text())
            paths = list(manifest.parent.glob("*/lerobot/meta/info.json"))
            if not paths:
                continue
            assignment = api.request("GET", f"/assignments/{manifest.parent.name}")
            for run in assignment["runs"]:
                dataset = manifest.parent / run["id"] / "lerobot"
                if not dataset.is_dir():
                    continue
                api.upload_dataset(run["id"], dataset)
                local.setdefault("datasets_uploaded", {})[run["id"]] = True
                save(manifest, local)
                count += 1
        print(f"Uploaded and verified {count} LeRobot datasets.")
    finally:
        api.close()


if __name__ == "__main__":
    main()
