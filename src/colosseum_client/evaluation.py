"""Interactive Open/Fine-tuning evaluation with durable upload/result recovery."""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
from urllib.parse import quote
from pathlib import Path
import select
import sys
import time
from urllib.parse import urlsplit, urlunsplit

import httpx

from .adapters import make_robot
from .client import ColosseumClient, ProtocolError
from .recording import TrialRecorder
from .robot_runner import action_chunk, protobuf_observation


def api_url(config):
    parts = urlsplit(config.url)
    return config.api_url or urlunsplit(('https' if parts.scheme == 'wss' else 'http', parts.netloc, '', '', ''))


class NoAssignment(RuntimeError):
    pass


class EvalAPI:
    def __init__(self, config):
        self.client = httpx.Client(base_url=api_url(config).rstrip('/'),
            headers={'Authorization': f'Bearer {config.token}'}, timeout=30, trust_env=False)

    def request(self, method, path, body=None):
        response = self.client.request(method, '/api/eval' + path, json=body)
        if response.status_code == 409:
            detail = response.json().get('detail', '')
            if detail.startswith(('No trial available', 'No pair available')):
                raise NoAssignment(detail)
        if response.is_error:
            raise RuntimeError(f'Evaluation API {response.status_code}: {response.text}')
        return response.json()

    def upload(self, run_id, camera, path):
        with path.open('rb') as source:
            checksum = base64.b64encode(hashlib.file_digest(source, 'sha256').digest()).decode()
        route = f'/runs/{run_id}/videos/{camera}'
        target = self.request('POST', route + '/upload-url',
                              {'size': path.stat().st_size, 'checksum_sha256': checksum})
        if target.get('complete'):
            return
        if target['storage'] == 'local':
            with path.open('rb') as source:
                response = self.client.put('/api/eval' + route, content=source,
                    headers={'Content-Type': 'video/mp4'}, timeout=300)
            response.raise_for_status()
        elif target['storage'] == 's3':
            # Separate HTTP client: never forward the Colosseum bearer token to S3.
            with httpx.Client(timeout=300, trust_env=False) as upload_client, path.open('rb') as source:
                response = upload_client.put(target['url'], content=source, headers=target['headers'])
            if response.status_code not in {200, 201, 204, 412}:
                # Do not log the presigned URL (it grants temporary object access).
                raise RuntimeError(f'S3 upload failed ({response.status_code}); resume to retry')
            # 412 may mean a prior upload succeeded but its response was lost.
            self.request('POST', route + '/complete')
        else:
            raise RuntimeError('Unknown video storage mode')

    def close(self):
        self.client.close()


def save(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('w') as output:
        json.dump(document, output, indent=2)
        output.flush()
        os.fsync(output.fileno())
    temp.replace(path)


def progress(label):
    while True:
        raw = input(f'{label} partial success (0–100): ').strip()
        try:
            value = float(raw)
            if 0 <= value <= 100:
                return value / 100
        except ValueError:
            pass
        print('Enter a number from 0 to 100.')


def yes_no(label):
    while True:
        choice = input(label + ' (y/n): ').strip().lower()
        if choice in {'y', 'yes', 'n', 'no'}:
            return choice in {'y', 'yes'}


async def run_trial(config, assignment, run, path, api, *, robot_factory=make_robot):
    task = assignment['task']
    recorder = TrialRecorder(path, task['cameras'])
    robot = robot_factory(config)
    client = ColosseumClient(config.url, config.token, client_id='', robot_type=config.robot_type,
        joint_count=robot.joint_count, has_gripper=robot.has_gripper, control_hz=config.control_hz, action_spaces={robot.action_space_name: robot.action_dim})
    step = 0
    try:
        metadata = await client.connect(evaluation_run=run['id'])
        if robot.action_space_name not in metadata.action_spaces:
            raise ProtocolError('Assigned policy does not support this robot action space')
        print('Trial started. Press Enter to finish, or Ctrl-C to interrupt.')
        stopped = False
        while step < task['max_steps'] and not stopped:
            current = await asyncio.to_thread(robot.get_observation)
            observation = protobuf_observation(current, instruction=task['instruction'], control_step=step)
            plan = await client.infer(observation, deadline_ms=config.deadline_ms)
            actions = action_chunk(plan, control_step=step, expected_dim=robot.action_dim)
            for action in actions:
                if step >= task['max_steps']:
                    break
                if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
                    sys.stdin.readline()
                    if step > 0:
                        stopped = True
                        break
                started = time.monotonic()
                # One observation per action, including during action chunks.
                await asyncio.to_thread(recorder.add, current, action)
                await asyncio.to_thread(robot.execute, action)
                step += 1
                await asyncio.sleep(max(0, 1/config.control_hz - (time.monotonic() - started)))
                current = await asyncio.to_thread(robot.get_observation)
        await asyncio.to_thread(recorder.add, current)
        await asyncio.to_thread(api.request, 'POST', f"/runs/{run['id']}/finish")
    finally:
        try:
            await client.close()
        finally:
            robot.close()


def select_task(api, robot):
    tasks = api.request('GET', '/tasks?robot_id=' + quote(robot, safe=''))['tasks']
    if not tasks:
        raise NoAssignment('No Fine-tuning tasks available for this robot')
    print('\nAvailable Fine-tuning tasks:')
    for index, task in enumerate(tasks, 1):
        print(f"{index}. {task['instruction']}")
    while True:
        choice = input('Select task number: ').strip()
        if choice.isdecimal() and 1 <= int(choice) <= len(tasks):
            return tasks[int(choice) - 1]['id']
        print(f'Enter a number from 1 to {len(tasks)}.')


def run_evaluation(config, *, track=None, resume=None, abort=None):
    api = EvalAPI(config)
    root = Path(config.evaluation_dir).resolve()
    task_id = None
    try:
        if abort:
            reason = input('Reason for interrupted evaluation: ').strip()
            if not reason:
                raise ValueError('A reason is required')
            print(api.request('POST', f'/assignments/{abort}/abort', {'reason': reason}))
            return
        if resume:
            assignment = api.request('GET', f'/assignments/{resume}')
            track = assignment['track']
            if track == 'fine-tuning':
                task_id = assignment['task']['id']
        else:
            if track is None:
                while track not in {'open', 'fine-tuning'}:
                    value = input('Evaluation track [1: Open / 2: Fine-tuning]: ').strip().lower()
                    track = {'1': 'open', '2': 'fine-tuning', 'open': 'open', 'fine-tuning': 'fine-tuning'}.get(value)
            assignment = None
        while True:
            if assignment is None:
                if track == 'fine-tuning' and task_id is None:
                    try:
                        task_id = select_task(api, config.robot_type)
                    except NoAssignment as exc:
                        print(str(exc)); return
                instruction = (config.instruction or input('Instruction: ').strip()) if track == 'open' else ''
                scene = input('Scene / setup: ').strip() if track == 'open' else ''
                try:
                    assignment = api.request('POST', '/next', {'robot_id': config.robot_type, 'track': track,
                        'instruction': instruction, 'scene': scene, 'max_steps': config.max_trial_steps,
                        'cameras': list(config.cameras), **({'task_id': task_id} if track == 'fine-tuning' else {})})
                except NoAssignment as exc:
                    print(str(exc)); return
            folder = root / assignment['id']
            manifest = folder / 'manifest.json'
            local = json.loads(manifest.read_text()) if manifest.exists() else {'assignment': assignment, 'outcomes': {}, 'uploaded': []}
            save(manifest, local)
            task = assignment['task']
            print(f"\nEvaluation {assignment['id']} ({track})\nTask: {task['instruction']}\nSetup: {task['setup']}\nSuccess: {task['success_criteria']}\nPartial success: {task['partial_success_criteria']}")
            if assignment['state'] == 'completed':
                print('Already submitted.'); return
            if assignment['state'] != 'pending':
                raise RuntimeError('This assignment is closed')
            if set(task['cameras']) - set(config.cameras):
                raise ValueError('Required task cameras are missing from robot config')
            for run in assignment['runs']:
                run_path = folder / run['id']
                if run['state'] == 'assigned':
                    if (run_path / 'frames.jsonl').exists():
                        raise RuntimeError('Existing recording found; inspect this interrupted run instead of overwriting it')
                    input(f"\n{run['side']}: restore the scene to its initial setup, then press Enter to start.")
                    asyncio.run(run_trial(config, assignment, run, run_path, api))
                elif run['state'] != 'finished':
                    raise RuntimeError(f"Run {run['side']} was interrupted; use --abort {assignment['id']} to report it")
                if run['side'] not in local['outcomes']:
                    local['outcomes'][run['side']] = {'success': yes_no(f"{run['side']} succeeded?"),
                        'partial_success': progress(run['side'])}
                    save(manifest, local)
                if not all(f"{run['id']}/{c}" in local['uploaded'] for c in task['cameras']):
                    videos = TrialRecorder.encode(run_path, task['cameras'])
                    for camera, path in videos.items():
                        key = f"{run['id']}/{camera}"
                        if key not in local['uploaded']:
                            api.upload(run['id'], camera, path)
                            local['uploaded'].append(key); save(manifest, local)
            if 'result' not in local:
                preference = None
                if track == 'open':
                    while preference not in {'A', 'B', 'tie'}:
                        value = input('Preference (A/B/tie): ').strip()
                        preference = 'tie' if value.lower() == 'tie' else value.upper()
                local['result'] = {'outcomes': local['outcomes'], 'preference': preference,
                    'feedback': input('Feedback (optional): ').strip() if track == 'open' else ''}
                save(manifest, local)
            api.request('POST', f"/assignments/{assignment['id']}/result", local['result'])
            local['submitted'] = True; save(manifest, local)
            print('Videos and result submitted.')
            if track == 'open' and not yes_no('Continue with the next evaluation?'):
                return
            assignment = None
    except (Exception, KeyboardInterrupt):
        if 'assignment' in locals() and assignment:
            print(f"Saved locally. Resume uploads/results: --resume {assignment['id']}; report interruption: --abort {assignment['id']}")
        raise
    finally:
        api.close()
