import argparse
import asyncio
import base64
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import time
from urllib.parse import urlsplit

import httpx
import yaml

from .client import ColosseumClient
from .rover import ActionBuffer, RoverSDK, SimulatedRover, observation
from .rover_recording import RoverRecording


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    required = {'api_url', 'router_url', 'token', 'institution', 'runner_id', 'bot_slug', 'sdk_url',
                'max_linear', 'max_angular', 'qualified'}
    if not isinstance(config, dict) or not required <= config.keys():
        raise ValueError('Missing required rover configuration fields')
    if config['token'].startswith('env:'):
        config['token'] = os.environ[config['token'][4:]]
    local = {'localhost', '127.0.0.1', '::1'}
    for key, secure in [('api_url', 'https'), ('router_url', 'wss'), ('sdk_url', 'http')]:
        endpoint = urlsplit(config[key])
        if endpoint.username or endpoint.password or endpoint.query or endpoint.fragment or endpoint.path not in {'', '/'}:
            raise ValueError(f'{key} must be an origin without credentials, paths or query parameters')
        if not endpoint.hostname or (key == 'sdk_url' and endpoint.hostname not in local):
            raise ValueError('The SDK sidecar must be bound to localhost')
        allowed = {secure, 'http' if secure == 'https' else 'ws' if secure == 'wss' else 'http'} if endpoint.hostname in local else {secure}
        if endpoint.scheme not in allowed:
            raise ValueError(f'{key} requires a secure connection outside localhost')
    ActionBuffer(float(config['max_linear']), float(config['max_angular']))
    return config


class Runner:
    def __init__(self, config, *, armed=False, simulated=False):
        self.config = config
        self.armed = armed and (config['qualified'] is True or simulated)
        self.sdk = SimulatedRover(config['bot_slug']) if simulated else RoverSDK(config['sdk_url'], config['bot_slug'])
        self.api = httpx.AsyncClient(base_url=config['api_url'].rstrip('/'), timeout=2,
            headers={'Authorization': 'Bearer ' + config['token'], 'X-Colosseum-Institution': config['institution']})
        self.session = secrets.token_hex(16)
        self.folder = Path(config.get('evaluation_dir', 'eval_runs/earth-rover')).resolve()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.latest = None
        self.phase = 'connecting'
        self.message = ''
        self.stop_event = asyncio.Event()
        self.active = None
        self.handled = set()
        self.connected = False
        self.heartbeat_at = 0.0
        self.shutting_down = False
        self.ride_ended = False

    async def request(self, method, path, **kwargs):
        response = await self.api.request(method, path, **kwargs)
        response.raise_for_status()
        return response.json()

    async def sample_loop(self):
        while not self.shutting_down:
            try:
                self.latest = await self.sdk.sample()
                self.message = ''
            except Exception as exc:
                self.message = f'Camera/telemetry unavailable: {type(exc).__name__}'
            await asyncio.sleep(.05)

    def ready(self):
        return self.connected and self.latest is not None and self.latest.fresh() and not self.ride_ended

    async def heartbeat_loop(self):
        while not self.shutting_down:
            sample = self.latest
            payload = dict(session=self.session, bot_slug=self.config['bot_slug'], ready=self.ready(),
                armed=self.armed, simulated=self.sdk.simulated, rear_camera=self.sdk.rear_camera,
                identity=self.sdk.identity, checks={'camera_fresh': bool(sample and sample.fresh()),
                    'ride_id': self.sdk.identity.get('ride_id'), 'armed': self.armed,
                    'recording_started': self.sdk.identity.get('recording_started', False)},
                phase=self.phase, message=self.message,
                frame_base64=base64.b64encode(sample.images['front_image']).decode() if sample and len(sample.images.get('front_image', b'')) <= 400000 else '')
            try:
                reply = await self.request('POST', f"/api/eval/rover-runners/{self.config['runner_id']}/heartbeat", json=payload)
                self.heartbeat_at = time.monotonic()
                if reply.get('restart_required'):
                    self.armed = False
                    self.stop_event.set()
                    self.message = 'Connection was interrupted; inspect the rover and restart the runner'
                if reply['stop']:
                    self.stop_event.set()
                job = reply.get('job')
                if job and job['run_id'] not in self.handled and not reply['stop'] and (self.active is None or self.active.done()):
                    self.handled.add(job['run_id'])
                    self.stop_event = asyncio.Event()
                    self.active = asyncio.create_task(self.evaluate(job))
                if self.connected and reply.get('command') == 'end_session' and (self.active is None or self.active.done()):
                    await self.sdk.end()
                    self.ride_ended = True
                    self.armed = False
                    self.phase = 'session ended'
                    await self.request('POST', f"/api/eval/rover-runners/{self.config['runner_id']}/ack", params={'session': self.session})
            except Exception as exc:
                self.message = f'Admin connection unavailable: {type(exc).__name__}'
                if time.monotonic() - self.heartbeat_at > 1:
                    self.stop_event.set()
                    if self.heartbeat_at:
                        self.armed = False
                        self.message += '; restart the runner to re-arm'
            await asyncio.sleep(.25)

    async def stationary_hold(self, seconds):
        started = None
        deadline = time.monotonic() + seconds + 3
        while time.monotonic() < deadline:
            sample = self.latest
            values = sample.telemetry if sample and sample.fresh() else {}
            speed = values.get('speed')
            rpms = [values.get(f'rpm_{index}') for index in range(1, 5)]
            stopped = (isinstance(speed, (float, int)) and math.isfinite(speed) and abs(speed) <= .03
                       and all(isinstance(rpm, (float, int)) and math.isfinite(rpm) and abs(rpm) <= 2 for rpm in rpms))
            if stopped:
                started = started or time.monotonic()
                if time.monotonic() - started >= seconds:
                    return seconds
            else:
                started = None
            await asyncio.sleep(.1)
        return 0.0

    async def evaluate(self, job):
        run_id = job['run_id']
        recording = RoverRecording(self.folder / run_id)
        buffer = ActionBuffer(float(self.config['max_linear']), float(self.config['max_angular']))
        client = ColosseumClient(self.config['router_url'], self.config['token'], self.config['runner_id'],
            robot_type='earth_rover_mini_plus', joint_count=0, has_gripper=False, control_hz=10,
            action_spaces={'base_velocity': 2})
        start_utc = utc_now()
        start = time.monotonic()
        reason = 'preflight'
        declared_done = False
        health = dict(stop_confirmed=False, stop_held_s=0, within_timeout=False)
        policy = {}
        control_steps = 0
        latencies = []
        frame_stamps = {'front_image': set(), 'rear_image': set()}
        maximum_frame_age = 0.0
        maximum_telemetry_age = 0.0
        gps_fixes = 0
        inference = None
        control = None
        inference_error = ''
        self.phase = 'connecting policy'

        async def infer_loop():
            nonlocal policy, inference_error
            step = 0
            while not self.stop_event.is_set():
                sample = self.latest
                if sample is None or not sample.fresh():
                    await asyncio.sleep(.05)
                    continue
                obs = observation(sample, job, step)
                await asyncio.to_thread(recording.observation, obs, step)
                requested = time.monotonic()
                try:
                    plan = await client.infer(obs, deadline_ms=1000)
                    latencies.append(time.monotonic() - requested)
                    if len(plan.decision_metadata_json.encode()) > 32768:
                        raise ValueError('Oversize decision metadata')
                    metadata = json.loads(plan.decision_metadata_json or '{}')
                    if not isinstance(metadata, dict):
                        raise ValueError('Invalid decision metadata')
                    policy = metadata.get('policy') or policy
                    recording.event('policy.jsonl', dict(kind='decision', step=step, done=plan.done, metadata=metadata))
                    if metadata.get('error'):
                        inference_error = 'policy_error'
                        return
                    buffer.accept(plan, time.monotonic())
                    if plan.done:
                        return
                except ValueError:
                    inference_error = 'policy_error'
                    return
                except Exception:
                    inference_error = 'policy_disconnected'
                    return
                step += 1
                await asyncio.sleep(max(0, .1 - (time.monotonic() - requested)))

        async def control_loop():
            nonlocal control_steps, maximum_frame_age, maximum_telemetry_age, gps_fixes
            while not self.stop_event.is_set():
                tick = time.monotonic()
                sample = self.latest
                fresh = sample is not None and sample.fresh()
                action = buffer.current(tick) if fresh and not inference_error and tick - self.heartbeat_at < 1 else (0.0, 0.0)
                if sample and abs(float(sample.telemetry.get('speed') or 0)) > job['task']['navigation']['max_speed_mps']:
                    raise ValueError('Qualified speed cap exceeded')
                await self.sdk.control(*action)
                recording.event('policy.jsonl', dict(kind='command', utc=utc_now(), action=action, monotonic=tick))
                if sample:
                    maximum_frame_age = max(maximum_frame_age, time.time() - sample.captured['front_image'])
                    maximum_telemetry_age = max(maximum_telemetry_age, sample.telemetry_age_s + tick - sample.received)
                    for camera, stamp in sample.captured.items():
                        frame_stamps[camera].add(stamp)
                    latitude, longitude = sample.telemetry.get('latitude'), sample.telemetry.get('longitude')
                    gps_fixes += int(isinstance(latitude, (float, int)) and isinstance(longitude, (float, int))
                                     and -90 <= latitude <= 90 and -180 <= longitude <= 180)
                    recording.event('telemetry.jsonl', dict(utc=utc_now(), data=sample.telemetry,
                                                           frame_age_s=time.time() - sample.captured['front_image']))
                control_steps += 1
                await asyncio.sleep(max(0, .1 - (time.monotonic() - tick)))

        try:
            if not self.ready() or not self.armed or job['bot_slug'] != self.sdk.identity['bot_slug']:
                raise ValueError('Runner or bot is not ready')
            ready = await asyncio.wait_for(client.connect(evaluation_run=run_id, institution=self.config['institution']), 15)
            if len(ready.policy_metadata_json.encode()) > 32768:
                raise ValueError('Oversize policy identity')
            policy = json.loads(ready.policy_metadata_json or '{}')
            if not isinstance(policy, dict) or not all(policy.get(key) for key in ('name', 'model', 'policy_revision', 'prompt_version', 'configuration_version', 'controller')):
                raise ValueError('Navigation policy must declare model, commit, prompt, configuration and controller settings')
            start, start_utc = time.monotonic(), utc_now()
            identity = {key: self.sdk.identity[key] for key in ('bot_slug', 'bot_id', 'device_ref_id', 'ride_id')}
            await self.request('POST', f'/api/eval/rover-runs/{run_id}/started', json={
                **identity, 'start_utc': start_utc, 'sdk_version': self.sdk.version})
            self.phase = 'running'
            reason = 'timeout'
            inference = asyncio.create_task(infer_loop())
            control = asyncio.create_task(control_loop())
            while time.monotonic() - start < job['task']['navigation']['timeout_s']:
                if self.stop_event.is_set():
                    reason = 'operator_stop' if time.monotonic() - self.heartbeat_at < 1 else 'runner_disconnected'
                    break
                if not self.latest or not self.latest.fresh() or (control.done() and control.exception()):
                    reason = 'sdk_connection'
                    self.armed = False
                    break
                if inference.done() and inference.exception():
                    reason = 'recording_failure'
                    break
                if inference_error:
                    reason = inference_error
                    break
                if buffer.done:
                    reason, declared_done = 'done', True
                    break
                await asyncio.sleep(.05)
        except Exception as exc:
            self.message = f'Evaluation stopped: {type(exc).__name__}'
        finally:
            self.stop_event.set()
            for task in (inference, control):
                if task:
                    task.cancel()
            await asyncio.gather(*(task for task in (inference, control) if task), return_exceptions=True)
            self.phase = 'stopping'
            control_end = time.monotonic()
            try:
                health['stop_confirmed'] = await self.sdk.stop()
                health['stop_receipt_s'] = time.monotonic() - control_end
                if health['stop_confirmed']:
                    health['stop_held_s'] = await self.stationary_hold(job['task']['navigation']['stopped_s'])
                if health['stop_held_s'] < job['task']['navigation']['stopped_s']:
                    self.armed = False
                    reason = 'sdk_connection'
                    self.message = 'Stationary stop could not be verified; inspect the rover and restart the runner'
            except Exception:
                health['stop_confirmed'] = False
                self.armed = False
                self.message = 'Stop receipt unconfirmed; operator intervention required'
            health['within_timeout'] = time.monotonic() - start <= job['task']['navigation']['timeout_s']
            ordered = sorted(latencies)
            health.update(control_steps=control_steps, control_hz=control_steps / max(.1, control_end - start),
                          max_front_frame_age_s=maximum_frame_age if math.isfinite(maximum_frame_age) else None,
                          max_telemetry_age_s=maximum_telemetry_age if math.isfinite(maximum_telemetry_age) else None,
                          observed_camera_hz={camera: len(stamps) / max(.1, control_end-start) for camera, stamps in frame_stamps.items()},
                          gps_fix_ratio=gps_fixes / max(1, control_steps),
                          action_expiries=buffer.expiries, inference_p50_s=ordered[len(ordered)//2] if ordered else None,
                          inference_p95_s=ordered[min(len(ordered)-1, int(len(ordered)*.95))] if ordered else None)
            report = dict(reason=reason, end_utc=utc_now(), declared_done=declared_done, health=health, policy=policy)
            summary = dict(run_id=run_id, assignment_id=job['assignment_id'], start_utc=start_utc,
                end_utc=report['end_utc'], identity=self.sdk.identity, goal_id=job['goal']['id'],
                cameras=list(self.latest.images) if self.latest else [],
                goal_sha256=job['goal'].get('image_sha256'), simulated=self.sdk.simulated, report=report)
            try:
                await asyncio.to_thread(recording.finish, summary)
            except Exception as exc:
                report['health']['recording_error'] = type(exc).__name__
                report['reason'] = 'recording_failure'
                (recording.folder / 'summary.json').write_text(json.dumps(summary, indent=2))
            (recording.folder / 'finish-pending.json').write_text(json.dumps(report))
            try:
                await self.request('POST', f'/api/eval/rover-runs/{run_id}/finish', json=report)
                (recording.folder / 'finish-pending.json').unlink()
            except Exception:
                self.message = 'Result saved locally; retrying server upload'
            try:
                await client.close()
            except Exception:
                pass
            self.phase = 'idle' if health['stop_confirmed'] else 'stop unconfirmed'

    async def archive_loop(self):
        while not self.shutting_down:
            for folder in self.folder.glob('run_*'):
                try:
                    pending = folder / 'finish-pending.json'
                    if pending.exists():
                        await self.request('POST', f'/api/eval/rover-runs/{folder.name}/finish', json=json.loads(pending.read_text()))
                        pending.unlink()
                    summary_path = folder / 'summary.json'
                    if not summary_path.exists():
                        continue
                    summary = json.loads(summary_path.read_text())
                    # Upload the review evidence immediately. Native recording is
                    # finalized only after End session and retried independently.
                    for name in ('policy.jsonl', 'telemetry.jsonl', 'summary.json', 'front.mp4', 'rear.mp4',
                                 'observations.tar.gz', 'native.tar.gz', 'native-manifest.json'):
                        path = folder / name
                        if path.exists() and not (folder / (name + '.uploaded')).exists():
                            await self.upload(folder.name, path)
                            (folder / (name + '.uploaded')).write_text('verified\n')
                    if self.ride_ended and self.config.get('native_exporter') and not summary['simulated'] and not (folder / 'native.tar.gz').exists():
                        await asyncio.to_thread(subprocess.run, [self.config.get('native_python', 'python3'), self.config['native_exporter'],
                            '--run', str(summary_path), '--output', str(folder), '--source-bucket', self.config['source_bucket']], check=True, timeout=180)
                except Exception as exc:
                    self.message = f'Evidence upload pending; retrying ({type(exc).__name__})'
            await asyncio.sleep(3)

    async def upload(self, run_id, path):
        digest = await asyncio.to_thread(RoverRecording.digest, path)
        registered = await self.request('POST', f'/api/eval/rover-runs/{run_id}/artifacts',
            json=dict(name=path.name, size=path.stat().st_size, sha256=digest))
        if not registered['verified']:
            async def chunks():
                with path.open('rb') as source:
                    while True:
                        chunk = await asyncio.to_thread(source.read, 1024 * 1024)
                        if not chunk:
                            break
                        yield chunk
            url = registered['url']
            headers = {**registered['headers'], 'Content-Length': str(path.stat().st_size)}
            if url.startswith('/'):
                response = await self.api.put(url, content=chunks(), headers=headers, timeout=120)
            else:
                async with httpx.AsyncClient() as uploader:
                    response = await uploader.put(url, content=chunks(), headers=headers, timeout=120)
            if response.status_code != 412:
                response.raise_for_status()
            await self.request('POST', f'/api/eval/rover-runs/{run_id}/artifacts/{path.name}/complete')

    async def run(self):
        tasks = []
        try:
            # Claim the runner before touching the sidecar. A duplicate process
            # must not send its preflight stop to somebody else's active run.
            await self.request('POST', f"/api/eval/rover-runners/{self.config['runner_id']}/heartbeat", json=dict(
                session=self.session, bot_slug=self.config['bot_slug'], ready=False, armed=self.armed,
                simulated=self.sdk.simulated, identity={}, phase='connecting'))
            self.heartbeat_at = time.monotonic()
            tasks = [asyncio.create_task(self.heartbeat_loop())]
            await self.sdk.start()
            self.connected = True
            self.phase = 'idle'
            tasks += [asyncio.create_task(self.sample_loop()), asyncio.create_task(self.archive_loop())]
            await asyncio.gather(*tasks)
        finally:
            self.stop_event.set()
            if self.active and not self.active.done():
                await self.active
            if not self.ride_ended and (self.connected or getattr(self.sdk, 'acquired', False)):
                try:
                    await self.sdk.end()
                except Exception:
                    self.message = 'Session end failed; confirm physical stop and retry SDK /end-mission'
                    print(self.message)
            self.shutting_down = True
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.api.aclose()
            if hasattr(self.sdk, 'http'):
                await self.sdk.http.aclose()


def main():
    parser = argparse.ArgumentParser(description='Connect an Earth Rover runner to RoboColosseum admin')
    parser.add_argument('--config', required=True)
    parser.add_argument('--arm', action='store_true', help='Allow this runner to execute admin-started physical trials')
    parser.add_argument('--simulate', action='store_true', help='Use a labelled simulated rover without SDK/network hardware access')
    parser.add_argument('--archive-only', action='store_true', help='Retry evidence export/upload for ended rides; never connect to the SDK')
    args = parser.parse_args()
    runner = Runner(load_config(args.config), armed=args.arm, simulated=args.simulate)
    try:
        if args.archive_only:
            runner.ride_ended = True
            asyncio.run(runner.archive_loop())
        else:
            asyncio.run(runner.run())
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
