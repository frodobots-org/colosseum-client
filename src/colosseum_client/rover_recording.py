import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import threading


class RoverRecording:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / 'observations').mkdir(exist_ok=True)
        self.lock = threading.Lock()
        self.frames = {'front': [], 'rear': []}

    def event(self, stream, value):
        with self.lock:
            with (self.folder / stream).open('a') as output:
                output.write(json.dumps(value, allow_nan=False) + '\n')

    def observation(self, observation, step):
        images = []
        for sensor in observation.sensors:
            name = f'{step:06d}-{sensor.sensor_id}.jpg'
            (self.folder / 'observations' / name).write_bytes(sensor.data)
            images.append(dict(name=name, sha256=hashlib.sha256(sensor.data).hexdigest(),
                               captured_at_ns=sensor.capture_time_ns))
            if sensor.sensor_id in {'front_image', 'rear_image'}:
                self.frames[sensor.sensor_id.split('_')[0]].append((name, observation.robot_time_ns / 1e9))
        from .tensors import tensor_to_numpy
        self.event('policy.jsonl', dict(kind='observation', step=step, robot_time_ns=observation.robot_time_ns,
            instruction=observation.instruction, images=images,
            state={name: tensor_to_numpy(value).tolist() for name, value in observation.state.items()}))

    def finish(self, summary):
        (self.folder / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
        for name in ('policy.jsonl', 'telemetry.jsonl'):
            if not (self.folder / name).exists():
                (self.folder / name).write_text('{}\n')
        for camera, frames in self.frames.items():
            if not frames:
                continue
            listing = []
            for index, (name, stamp) in enumerate(frames):
                duration = frames[index + 1][1] - stamp if index + 1 < len(frames) else 0.1
                listing += [f"file 'observations/{name}'", f'duration {max(.01, duration):.6f}']
            listing.append(f"file 'observations/{frames[-1][0]}'")
            (self.folder / f'{camera}.concat').write_text('\n'.join(listing))
            subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0',
                            '-i', f'{camera}.concat', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2',
                            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', f'{camera}.mp4'],
                           cwd=self.folder, check=True, timeout=120, capture_output=True)
        with tarfile.open(self.folder / 'observations.tar.gz', 'w:gz') as archive:
            archive.add(self.folder / 'observations', arcname='observations')

    @staticmethod
    def digest(path):
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                digest.update(chunk)
        return digest.hexdigest()
