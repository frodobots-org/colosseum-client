"""Interactive Open/Fine-tuning evaluation with durable upload/result recovery."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from functools import partial

import asyncio
import base64
import hashlib
import json
import os
from urllib.parse import quote
from pathlib import Path
import select
import sys
import ssl
import time
from urllib.parse import urlsplit, urlunsplit

import httpx

from .adapters import make_robot
from .diagnostics import read_observation, execute_robot_action
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
            headers={'Authorization': f'Bearer {config.token}', 'X-Colosseum-Institution': quote(config.institution.strip(),safe='')}, timeout=30, trust_env=False, verify=ssl.create_default_context())

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

    def upload_dataset(self, run_id, root):
        root = Path(root)
        files = []
        for path in sorted(root.rglob('*')):
            if path.is_file():
                with path.open('rb') as source:
                    checksum = base64.b64encode(hashlib.file_digest(source, 'sha256').digest()).decode()
                files.append({'path': path.relative_to(root).as_posix(),
                              'size': path.stat().st_size, 'checksum_sha256': checksum})
        route = f'/runs/{run_id}/dataset'
        self.request('POST', route, {'files': files})
        for index, file in enumerate(files, 1):
            query = '?path=' + quote(file['path'], safe='')
            target = self.request('POST', route + '/upload-url' + query)
            if target.get('complete'):
                continue
            path = root / file['path']
            print(f"Uploading dataset {index}/{len(files)}: {file['path']}", flush=True)
            if target['storage'] == 's3':
                with httpx.Client(timeout=600, trust_env=False) as upload_client, path.open('rb') as source:
                    try:
                        response = upload_client.put(target['url'], content=source, headers=target['headers'])
                    except httpx.HTTPError:
                        raise RuntimeError('Dataset S3 connection failed; resume to retry') from None
                if response.status_code not in {200, 201, 204, 412}:
                    raise RuntimeError(f'Dataset S3 upload failed ({response.status_code}); resume to retry')
            elif target['storage'] == 'local':
                with path.open('rb') as source:
                    response = self.client.put('/api/eval' + route + '/file' + query, content=source,
                                               headers={'Content-Type': 'application/octet-stream'}, timeout=600)
                response.raise_for_status()
            else:
                raise RuntimeError('Unknown dataset storage mode')
        self.request('POST', route + '/complete')

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


async def run_trial(config, assignment, run, path, api, *, robot_factory=make_robot, execute_action=True):
    # ZeroRPC/gevent connections must be created and used on the same OS thread.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='colosseum-robot') as executor:
        async def robot_call(function, *args):
            return await asyncio.get_running_loop().run_in_executor(executor, partial(function, *args))

        task = assignment['task']
        recorder = None
        robot = await robot_call(robot_factory, config)
        local_mode = assignment.get('inference_mode', 'remote') == 'local'
        client_options = dict(robot_type=config.robot_type, joint_count=robot.joint_count,
            has_gripper=robot.has_gripper, control_hz=config.control_hz,
            action_spaces={robot.action_space_name: robot.action_dim})
        if local_mode:
            from .local_policy import LocalPolicyClient
            client = LocalPolicyClient(config, api_url(config), **client_options)
        else:
            client = ColosseumClient(config.url, config.token, client_id='', **client_options)
        step = 0
        try:
            metadata = await client.connect(evaluation_run=run['id'], institution=config.institution)
            if config.test and local_mode:
                await robot_call(robot.configure_model, client.model)
            if robot.action_space_name not in metadata.action_spaces:
                raise ProtocolError('Assigned policy does not support this robot action space')
            if local_mode:
                save(path / 'policy.json', {'run_id': run['id'], 'model': client.model})
            if config.recording:
                save(path / 'recording-context.json', {
                    'robot_type': config.robot_type, 'fps': config.control_hz,
                    'action_space': robot.action_space_name,
                    'execution_enabled': execute_action, 'test': config.test})
                recorder = TrialRecorder(path, task['cameras'])
            print('Trial started. Press Enter to finish, or Ctrl-C to interrupt.')
            print('Recording enabled: frames are written by a background worker.' if recorder is not None
                  else 'Recording disabled.', flush=True)
            print(f'Action execution: {"enabled" if execute_action else "disabled"}.', flush=True)
            actions = None
            chunk_index = 0
            while step < task['max_steps']:
                if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
                    sys.stdin.readline()
                    if step > 0:
                        break
                started = time.monotonic()
                current = await robot_call(read_observation, robot, step, 'step_start')
                captured_at = time.monotonic()
                if actions is None or chunk_index >= len(actions):
                    observation = protobuf_observation(current, instruction=task['instruction'], control_step=step)
                    plan = await client.infer(observation, deadline_ms=config.deadline_ms)
                    actions = action_chunk(plan, control_step=step, expected_dim=robot.action_dim)
                    chunk_index = 0
                    print(f'Received action chunk at step {step} (execution {"enabled" if execute_action else "skipped"}):\n{actions.tolist()}', flush=True)
                action = actions[chunk_index]
                if recorder is not None:
                    recorder.add(current, action, captured_at=captured_at)
                if local_mode:
                    await client.before_action(step)
                await robot_call(execute_robot_action, robot, action, step, execute_action)
                chunk_index += 1
                step += 1
                # Include observation, inference and execution in the control period.
                await asyncio.sleep(max(0, 1/config.control_hz - (time.monotonic() - started)))
            if recorder is not None:
                final_observation = await robot_call(robot.get_observation)
                recorder.add(final_observation)
                print('Finishing recording...', flush=True)
                await asyncio.to_thread(recorder.close)
            if local_mode:
                await client.finish()
            else:
                await asyncio.to_thread(api.request, 'POST', f"/runs/{run['id']}/finish")
        finally:
            try:
                await client.close()
            finally:
                try:
                    await robot_call(robot.close)
                finally:
                    if recorder is not None:
                        await asyncio.to_thread(recorder.close)


def select_task(api, robot, test=False):
    tasks = api.request('GET', '/tasks?robot_id=' + quote(robot, safe='') + ('&test=true' if test else ''))['tasks']
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


def run_evaluation(config, *, track=None, resume=None, abort=None, execute_action=True):
    track = track or config.track or None
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
            previous = api.request('POST', '/reset')
            if previous.get('discarded'):
                print('Previous incomplete evaluation closed. Starting a new evaluation.')
            if track is None:
                while track not in {'open', 'fine-tuning'}:
                    value = input('Evaluation track [1: Open / 2: Fine-tuning]: ').strip().lower()
                    track = {'1': 'open', '2': 'fine-tuning', 'open': 'open', 'fine-tuning': 'fine-tuning'}.get(value)
            assignment = None
        while True:
            if assignment is None:
                if track == 'fine-tuning' and task_id is None:
                    try:
                        task_id = select_task(api, config.robot_type, config.test)
                    except NoAssignment as exc:
                        print(str(exc)); return
                instruction = (config.instruction or input('Instruction: ').strip()) if track == 'open' else ''
                scene = config.scene or (input('Scene / setup: ').strip() if track == 'open' else '')
                mode = 'local' if config.policy_server_url else 'remote'
                try:
                    assignment = api.request('POST', '/next', {'robot_id': config.robot_type, 'track': track,
                        'instruction': instruction, 'scene': scene, 'evaluator': config.evaluator, 'institution': config.institution.strip(),
                        'inference_mode': mode, 'test': config.test, 'recording': config.recording, 'max_steps': config.max_trial_steps,
                        'cameras': list(config.cameras), **({'task_id': task_id} if track == 'fine-tuning' else {})})
                except NoAssignment as exc:
                    print(str(exc)); return
            if bool(assignment.get('recording', True)) != config.recording:
                raise ValueError('Assignment recording flag differs from config; restore the original setting to resume')
            if bool(assignment.get('test', False)) != config.test:
                raise ValueError('Assignment test flag differs from config; finish or abort before switching')
            if assignment.get('inference_mode', 'remote') == 'local' and not config.policy_server_url:
                raise ValueError('This assignment requires policy_server_url in the config')
            folder = root / assignment['id']
            manifest = folder / 'manifest.json'
            local = json.loads(manifest.read_text()) if manifest.exists() else {'assignment': assignment, 'outcomes': {}, 'uploaded': []}
            save(manifest, local)
            task = assignment['task']
            print(f"\nTask: {task['instruction']}")
            if assignment['state'] == 'completed':
                print('Already submitted.'); return
            if assignment['state'] != 'pending':
                raise RuntimeError('This assignment is closed')
            if set(task['cameras']) - set(config.cameras):
                raise ValueError('Required task cameras are missing from robot config')
            for run in assignment['runs']:
                run_path = folder / run['id']
                if (run_path / 'recording-disabled.json').exists():
                    raise RuntimeError('This older diagnostic run cannot be resumed; restart the client for a new evaluation')
                if run['state'] == 'assigned':
                    if (run_path / 'frames.jsonl').exists():
                        raise RuntimeError('Existing recording found; inspect this interrupted run instead of overwriting it')
                    input(f"\n{run['side']}: restore the scene to its initial setup, then press Enter to start.")
                    asyncio.run(run_trial(config, assignment, run, run_path, api,
                                          execute_action=execute_action))
                elif run['state'] != 'finished':
                    raise RuntimeError(f"Run {run['side']} was interrupted; restart the client for a new evaluation")
                if (run_path / 'recording-disabled.json').exists():
                    raise RuntimeError('This diagnostic run cannot be submitted; restart the client for a new evaluation')
                if run['side'] not in local['outcomes']:
                    if track == 'open':
                        partial = progress(run['side'])
                        local['outcomes'][run['side']] = {'success': partial == 1.0, 'partial_success': partial}
                    else:
                        local['outcomes'][run['side']] = {'success': yes_no(f"{run['side']} succeeded?"),
                            'partial_success': progress(run['side'])}
                    save(manifest, local)
                if config.recording:
                    from .lerobot_export import export_trial
                    context_path = run_path / 'recording-context.json'
                    if not context_path.exists():
                        raise ValueError('This older recording lacks the context required for LeRobot export')
                    context = json.loads(context_path.read_text())
                    dataset = export_trial(run_path, task['cameras'], fps=context['fps'],
                        robot_type=context['robot_type'], task=task['instruction'],
                        metadata={'assignment_id': assignment['id'], 'run_id': run['id'],
                            'side': run['side'], 'track': track, 'test': context['test'],
                            'action_space': context['action_space'],
                            'execution_enabled': context['execution_enabled'],
                            'institution': assignment.get('institution', config.institution),
                            'scene': assignment.get('scene', config.scene),
                            'evaluator': assignment.get('evaluator', config.evaluator),
                            'outcome': local['outcomes'][run['side']]})
                    print(f'LeRobot dataset saved: {dataset}', flush=True)
            if 'result' not in local:
                preference = None
                if track == 'open':
                    while preference not in {'A', 'B', 'tie'}:
                        value = input('Preference (A/B/tie): ').strip()
                        preference = 'tie' if value.lower() == 'tie' else value.upper()
                local['result'] = {'outcomes': local['outcomes'], 'preference': preference,
                    'feedback': input('Feedback (optional): ').strip() if track == 'open' else ''}
                save(manifest, local)
            if config.recording:
                print('Evaluation complete. Uploading saved recordings.', flush=True)
            for run in assignment['runs']:
                run_path = folder / run['id']
                if config.recording:
                    dataset = run_path / 'lerobot'
                    # The server checks each object on retry; local flags alone do not prove durability.
                    api.upload_dataset(run['id'], dataset)
                    local.setdefault('datasets_uploaded', {})[run['id']] = True
                    save(manifest, local)
                    print('LeRobot dataset uploaded and verified.', flush=True)
            api.request('POST', f"/assignments/{assignment['id']}/result", local['result'])
            local['submitted'] = True; save(manifest, local)
            print('Videos and result submitted.' if config.recording else 'Evaluation result submitted (recording disabled).')
            if track == 'open' and not yes_no('Continue with the next evaluation?'):
                return
            assignment = None
    except (Exception, KeyboardInterrupt):
        if 'assignment' in locals() and assignment:
            print(f"Saved locally. Restart the client to begin a new evaluation; the incomplete evaluation will not be published.")
        raise
    finally:
        api.close()
