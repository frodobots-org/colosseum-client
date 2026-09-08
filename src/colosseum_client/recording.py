"""Per-trial multi-camera recording with capture timing and restartable encoding."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np
from PIL import Image


class TrialRecorder:
    def __init__(self, path: Path, cameras: list[str]):
        if shutil.which('ffmpeg') is None:
            raise RuntimeError('Install ffmpeg before starting a trial')
        self.path = path
        self.cameras = cameras
        path.mkdir(parents=True, exist_ok=True)
        self.frames = 0
        self.started = time.monotonic()

    def add(self, observation, action=None):
        stamp = time.monotonic() - self.started
        for camera in self.cameras:
            if camera not in observation.images:
                raise ValueError(f'Missing required camera: {camera}')
            folder = self.path / camera
            folder.mkdir(exist_ok=True)
            Image.fromarray(np.asarray(observation.images[camera], dtype=np.uint8)).save(folder / f'{self.frames:06d}.png')
        with (self.path / 'frames.jsonl').open('a') as output:
            output.write(json.dumps({'frame': self.frames, 'seconds': stamp,
                'joints': np.asarray(observation.joints).tolist(),
                'action': None if action is None else np.asarray(action).tolist()}) + '\n')
        self.frames += 1

    @staticmethod
    def encode(path: Path, cameras: list[str]) -> dict[str, Path]:
        records = [json.loads(line) for line in (path / 'frames.jsonl').read_text().splitlines()]
        if not records:
            raise ValueError('No recorded frames')
        videos = {}
        for camera in cameras:
            folder = path / camera
            lines = []
            for index, row in enumerate(records):
                duration = records[index+1]['seconds'] - row['seconds'] if index+1 < len(records) else 1/15
                lines.extend([f"file '{row['frame']:06d}.png'", f'duration {max(duration, .001):.6f}'])
            lines.append(f"file '{records[-1]['frame']:06d}.png'")
            (folder / 'frames.txt').write_text('\n'.join(lines) + '\n')
            temp = folder / 'video.pending.mp4'
            subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat',
                '-safe', '1', '-i', 'frames.txt', '-vsync', 'vfr', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', temp.name],
                cwd=folder, check=True)
            final = folder / 'video.mp4'
            temp.replace(final)
            videos[camera] = final
        return videos
