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

import threading

import httpx
import numpy as np

from .adapters import make_robot
from .diagnostics import read_observation, execute_robot_action_async, close_resources
from .client import ColosseumClient, ProtocolError
from .recording import TrialRecorder
from .http_diagnostics import log_http_error
from .navigation import ROVER_TYPE
from .robot_runner import action_chunk, protobuf_observation


def api_url(config):
    parts = urlsplit(config.url)
    return config.api_url or urlunsplit(('https' if parts.scheme == 'wss' else 'http', parts.netloc, '', '', ''))


class NoAssignment(RuntimeError):
    pass


def sha256_stream(source):
    digest = hashlib.sha256()
    for block in iter(lambda: source.read(1024 * 1024), b''):
        digest.update(block)
    return digest.digest()


def trial_cameras(config, task):
    """Use Router-required cameras plus explicit local recording cameras."""
    names = []
    for name in list(task.get('cameras') or []) + list(config.cameras):
        if name not in names:
            names.append(name)
    return names


class EvalAPI:
    def __init__(self, config):
        self.config = config
        self.log_dir = Path(config.evaluation_dir)
        self.client = httpx.Client(base_url=api_url(config).rstrip('/'),
            headers={'Authorization': f'Bearer {config.token}', 'X-Colosseum-Institution': quote(config.institution.strip(),safe='')}, timeout=30, trust_env=False, verify=ssl.create_default_context())

    def request(self, method, path, body=None):
        if method.upper() == 'POST' and path == '/next' and getattr(self.config, 'llm_api_urls', []):
            body = {**body, 'llm_api_urls': self.config.llm_api_urls,
                    'llm_providers': self.config.llm_providers}
        response = self.client.request(method, '/api/eval' + path, json=body,
            timeout=1800 if path.endswith('/dataset/complete') else 30)
        if response.status_code == 409:
            detail = response.json().get('detail', '')
            if detail.startswith(('No trial available', 'No pair available')):
                raise NoAssignment(detail)
        if response.is_error:
            reason = log_http_error(response, operation='Evaluation API', log_dir=self.log_dir)
            raise RuntimeError(f'Evaluation API {response.status_code}: {reason}')
        return response.json()

    def upload(self, run_id, camera, path):
        with path.open('rb') as source:
            checksum = base64.b64encode(sha256_stream(source)).decode()
        route = f'/runs/{run_id}/videos/{camera}'
        target = self.request('POST', route + '/upload-url',
                              {'size': path.stat().st_size, 'checksum_sha256': checksum})
        if target.get('complete'):
            return
        if target['storage'] == 'local':
            with path.open('rb') as source:
                response = self.client.put('/api/eval' + route, content=source,
                    headers={'Content-Type': 'video/mp4'}, timeout=300)
            if response.is_error:
                log_http_error(response, operation='Local video upload', log_dir=getattr(self, 'log_dir', None))
                raise RuntimeError(f'Local video upload failed ({response.status_code}); resume to retry')
        elif target['storage'] == 's3':
            # Separate HTTP client: never forward the Colosseum bearer token to S3.
            with httpx.Client(timeout=300, trust_env=False) as upload_client, path.open('rb') as source:
                response = upload_client.put(target['url'], content=source, headers=target['headers'])
            if response.status_code not in {200, 201, 204, 412}:
                log_http_error(response, operation='S3 video upload', log_dir=getattr(self, 'log_dir', None))
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
                    checksum = base64.b64encode(sha256_stream(source)).decode()
                files.append({'path': path.relative_to(root).as_posix(),
                              'size': path.stat().st_size, 'checksum_sha256': checksum})
        route = f'/runs/{run_id}/dataset'
        if getattr(getattr(self, 'config', None), 'dataset_url', ''):
            from .hub_upload import upload_episode
            source = upload_episode(self.config, run_id, root, files, self.request)
            self.request('POST', route, {'files': files, 'source': source})
            self.request('POST', route + '/complete')
            return
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
                    log_http_error(response, operation='Dataset S3 upload', log_dir=getattr(self, 'log_dir', None))
                    raise RuntimeError(f'Dataset S3 upload failed ({response.status_code}); resume to retry')
            elif target['storage'] == 'local':
                with path.open('rb') as source:
                    response = self.client.put('/api/eval' + route + '/file' + query, content=source,
                                               headers={'Content-Type': 'application/octet-stream'}, timeout=600)
                if response.is_error:
                    log_http_error(response, operation='Dataset local upload', log_dir=getattr(self, 'log_dir', None))
                    raise RuntimeError(f'Dataset local upload failed ({response.status_code}); resume to retry')
            else:
                raise RuntimeError('Unknown dataset storage mode')
        self.request('POST', route + '/complete')

    def close(self):
        self.client.close()


def ask(prompt, options=None):
    """Operator prompt. The terminal reads stdin; the local operator page (rover_ui) replaces this
    to show the same question as a form. ``options`` lists the valid answers as values or (value, label)."""
    return input(prompt)


class SessionStopped(RuntimeError):
    """The operator ended the session while a question was pending."""


# Set by the operator page: INTERRUPT ends the running trial, STOP ends the session before the next one.
INTERRUPT = threading.Event()
STOP = threading.Event()


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
        raw = ask(f'{label} partial success (0–100): ').strip()
        try:
            value = float(raw)
            if 0 <= value <= 100:
                return value / 100
        except ValueError:
            pass
        print('Enter a number from 0 to 100.')


def yes_no(label):
    while True:
        choice = ask(label + ' (y/n): ', options=[('y', 'Yes'), ('n', 'No')]).strip().lower()
        if choice in {'y', 'yes', 'n', 'no'}:
            return choice in {'y', 'yes'}


def rover_score(label):
    """Earth Rover outcome on the V4 0-4 rubric, mapped onto success/partial_success."""
    print('Navigation rubric: 0 no progress, 1 left the start area toward the goal, 2 substantial progress, '
          '3 reached the destination without completing, 4 full success (declared done, stopped for the hold '
          'time, arrived; no collision, boundary violation or human help).')
    while True:
        raw = ask(f'{label} navigation score (0-4): ', options=[(str(score), str(score)) for score in range(5)]).strip()
        if raw.isdecimal() and 0 <= int(raw) <= 4:
            score = int(raw)
            return {'success': score == 4, 'partial_success': score / 4}
        print('Enter a whole number from 0 to 4.')


def rover_bot(config):
    """The rover to connect: adapter_config.bot_slug, --bot, or a prompt, like the SDK's BOT_SLUG."""
    settings = config.adapter_config
    if config.test or settings.get('link', 'frodobots') != 'frodobots' or settings.get('bot_slug'):
        return config
    while True:
        slug = ask('Bot slug: ').strip()
        if slug:
            from dataclasses import replace
            return replace(config, adapter_config={**settings, 'bot_slug': slug})
        print('Enter the slug of the Earth Rover to connect.')


def rover_session(config):
    """A FrodoBots backend session for mission lookup, or None when the trial does not ride a real rover."""
    settings = config.adapter_config
    if config.test or settings.get('link', 'frodobots') != 'frodobots':
        return None
    from .robots.earth_rover.session import DEFAULT_API_URL, FrodobotsSession
    from .robots.earth_rover_mini_plus import resolve_secret
    return FrodobotsSession(resolve_secret(settings.get('sdk_token')), settings.get('bot_slug'),
                            api_url=settings.get('frodobots_api_url', DEFAULT_API_URL))


def select_rover_mission(config, session):
    """The bot's mission, like the SDK's start-mission: a configured slug, or a prompt over the bot's SDK-enabled missions."""
    missions = session.missions()
    wanted = config.adapter_config.get('mission_slug')
    if wanted:
        mission = next((mission for mission in missions if mission.get('slug') == wanted), None)
        if mission is None:
            raise ValueError(f'{wanted} is not an SDK-enabled active mission of {session.bot_slug}')
        return mission
    if not missions:
        raise NoAssignment(f'{session.bot_slug} has no SDK-enabled active missions; create one in the FrodoBots admin')
    print(f'\nMissions on {session.bot_slug}:')
    labels = [f"{mission.get('slug')}  [{mission.get('type')}, {mission.get('checkpoints_count') or 0} checkpoints]  {mission.get('name', '')}"
              for mission in missions]
    for index, label in enumerate(labels, 1):
        print(f'{index}. {label}')
    while True:
        choice = ask('Select mission number: ', options=[(str(index), label) for index, label in enumerate(labels, 1)]).strip()
        if choice.isdecimal() and 1 <= int(choice) <= len(missions):
            return missions[int(choice) - 1]
        print(f'Enter a number from 1 to {len(missions)}.')


def mission_goal(session, mission, checkpoint=None):
    """The destination is one of the mission's checkpoints: the last one unless adapter_config.checkpoint names another."""
    checkpoints = session.checkpoints(mission['slug'])
    if not checkpoints:
        raise ValueError(f"Mission {mission['slug']} has no checkpoints; use --goal-gps or --goal-image instead")
    if checkpoint is None:
        target = checkpoints[-1]
    else:
        target = next((item for item in checkpoints if int(item.get('sequence', 0)) == int(checkpoint)), None)
        if target is None:
            raise ValueError(f"Mission {mission['slug']} has no checkpoint {checkpoint}")
    return {'type': 'gps', 'latitude': float(target['latitude']), 'longitude': float(target['longitude']),
            'mission_slug': mission['slug'], 'checkpoint': int(target.get('sequence', len(checkpoints)))}


def select_rover_task(api, config):
    """Fine-tuning: the registered task for the chosen mission (prompting only among tasks of that mission)."""
    session = rover_session(config)
    settings = config.adapter_config
    if session is None or settings.get('goal_gps') is not None or settings.get('goal_image') is not None:
        return select_task(api, config.robot_type, config.test)
    try:
        mission = select_rover_mission(config, session)
    except NoAssignment as exc:
        # A bot without SDK-enabled missions: fall back to the Router's registered tasks (goal by coordinates).
        print(f'{exc}; choosing among the registered tasks instead.', flush=True)
        return select_task(api, config.robot_type, config.test)
    finally:
        session.close()
    tasks = api.request('GET', '/tasks?robot_id=' + quote(config.robot_type, safe='') + ('&test=true' if config.test else ''))['tasks']
    matching = [task for task in tasks if (task.get('goal') or {}).get('mission_slug') == mission['slug']]
    if not matching:
        raise NoAssignment(f"No Fine-tuning task is registered for mission {mission['slug']} "
                           f"(register it from its checkpoints, e.g. register-mission-task.sh {mission['slug']})")
    if len(matching) == 1:
        print(f"Task for mission {mission['slug']}: {matching[0]['instruction']}")
        return matching[0]['id']
    print(f"\nTasks for mission {mission['slug']}:")
    for index, task in enumerate(matching, 1):
        print(f"{index}. {task['instruction']}")
    while True:
        choice = ask('Select task number: ', options=[(str(index), task['instruction']) for index, task in enumerate(matching, 1)]).strip()
        if choice.isdecimal() and 1 <= int(choice) <= len(matching):
            return matching[int(choice) - 1]['id']
        print(f'Enter a number from 1 to {len(matching)}.')


def rover_goal(config):
    """Open Track destination: the chosen mission's checkpoint, or --goal-gps / --goal-image when given."""
    settings = config.adapter_config
    gps, image = settings.get('goal_gps'), settings.get('goal_image')
    if gps is None and image is None and config.test:
        gps = (0.0, 0.0)  # Synthetic trials need no real destination.
    if gps is None and image is None:
        session = rover_session(config)
        if session is None:
            raise ValueError('Open Track rover trials need --goal-gps LAT,LON or --goal-image PATH without a FrodoBots session')
        try:
            return mission_goal(session, select_rover_mission(config, session), settings.get('checkpoint'))
        finally:
            session.close()
    if gps is not None and image is not None:
        raise ValueError('Open Track rover trials take one of --goal-gps LAT,LON or --goal-image PATH, not both')
    mission = {'mission_slug': settings['mission_slug']} if settings.get('mission_slug') else {}
    if gps is not None:
        try:
            latitude, longitude = (float(value) for value in gps)
        except (TypeError, ValueError):
            raise ValueError('goal_gps must be latitude,longitude in degrees') from None
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError('goal_gps must be latitude,longitude in degrees')
        return {'type': 'gps', 'latitude': latitude, 'longitude': longitude, **mission}
    path = Path(image)
    with path.open('rb') as source:
        digest = sha256_stream(source).hex()
    return {'type': 'image', 'sha256': digest, 'file': path.name, **mission}


def rover_navigation(config):
    settings = config.adapter_config
    return {'timeout_s': int(settings.get('timeout_s', 300)), 'stopped_s': float(settings.get('stopped_s', 2))}


async def run_trial(config, assignment, run, path, api, *, robot_factory=make_robot,
                    execute_action=True, confirm_live=None):
    # ZeroRPC/gevent connections must be created and used on the same OS thread.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='colosseum-robot') as executor:
        async def robot_call(function, *args, **kwargs):
            return await asyncio.get_running_loop().run_in_executor(
                executor, partial(function, *args, **kwargs)
            )

        task = assignment['task']
        recorder = None
        robot = None
        local_mode = assignment.get('inference_mode', 'remote') == 'local'
        if local_mode:
            if not config.test and config.robot_type not in {'franka', 'yam', 'so101'}:
                raise ValueError('Local Policy evaluation currently supports robot_type: franka, yam or so101')
            from .local_policy import LocalPolicyClient
            joint_count, action_spaces = {
                'yam': (12, {'joint_position': 14, 'eef_abs_pose': 14}),
                'so101': (5, {'joint_position': 6}),
            }.get(config.robot_type, (7, {'joint_position': 8, 'joint_velocity': 8, 'cartesian_position': 7}))
            client = LocalPolicyClient(
                config, api_url(config), robot_type=config.robot_type,
                joint_count=joint_count, has_gripper=True, control_hz=config.control_hz,
                action_spaces=action_spaces,
            )
        else:
            robot = await robot_call(robot_factory, config)
            client_options = dict(robot_type=config.robot_type, joint_count=robot.joint_count,
                has_gripper=robot.has_gripper, control_hz=config.control_hz,
                action_spaces={robot.action_space_name: robot.action_dim})
            client = ColosseumClient(config.url, config.token, client_id='', **client_options)
        step = 0
        return_attempted = False
        return_confirmed = False
        try:
            if local_mode and not config.test and confirm_live is None:
                raise ProtocolError('Physical local trial requires operator confirmation before RobotEnv construction')
            options = {'before_start': confirm_live} if local_mode and not config.test else {}
            metadata = await client.connect(evaluation_run=run['id'], institution=config.institution, **options)
            native_mode = getattr(client, 'native_control_mode', 'joint_pos')
            trial_hz = (client.model['control_hz'] if local_mode and client.model.get('model_type') == 'llm'
                        else config.control_hz)
            if type(trial_hz) is not int or trial_hz <= 0:
                raise ProtocolError('Assigned LLM control_hz must be a positive integer')
            if local_mode:
                mode = native_mode
                if mode == 'eef_delta_pose' and not execute_action:
                    raise ValueError('SO101 EEF recording requires action execution to capture accepted joint commands')
                factory_options = {'action_space': client.model['action_space']}
                if mode == 'eef_delta_pose':
                    factory_options['control_mode'] = mode
                robot = await robot_call(robot_factory, config, **factory_options)
                prepare_trial = getattr(robot, 'prepare_for_trial', None)
                if prepare_trial is not None:
                    if not execute_action:
                        raise ValueError('YAM EE evaluation requires action execution')
                    await robot_call(prepare_trial, task['instruction'], run['id'])
                # Optional driver hook: a rig may start some models from their own home pose.
                prepare = getattr(robot, 'prepare_for_model', None)
                if prepare is not None and execute_action and not config.test:
                    start_pose = await robot_call(prepare, client.model)
                    if start_pose:
                        print(f'Moved to the start pose configured for {start_pose}.', flush=True)
            if config.test and local_mode:
                await robot_call(robot.configure_model, client.model)
            # Navigation adapters read the goal and limits from the Router's task.
            prepare_task = getattr(robot, 'prepare_for_task', None)
            if prepare_task is not None:
                await robot_call(prepare_task, task)
            if robot.action_space_name not in metadata.action_spaces:
                raise ProtocolError('Assigned policy does not support this robot action space')
            if local_mode:
                save(path / 'policy.json', {'run_id': run['id'], 'model': client.model})
            if config.recording:
                eef_control = robot.action_space_name == 'eef_abs_pose'
                save(path / 'recording-context.json', {
                    'robot_type': config.robot_type, 'fps': trial_hz,
                    'action_space': 'joint_position' if eef_control else robot.action_space_name,
                    'execution_enabled': execute_action, 'test': config.test,
                    'joint_step_mode': getattr(robot, 'joint_step_mode', None),
                    'native_control_mode': native_mode,
                    'action_recording': ('accepted_joint_commands_after_eef_ik'
                                         if local_mode and native_mode == 'eef_delta_pose'
                                         else 'post_ik_sent_joint_commands' if eef_control
                                         else 'policy_targets'),
                    'joint_max_step': config.adapter_config.get('joint_max_step')})
                record_cameras = trial_cameras(config, task)
                print(f"Recording cameras: {record_cameras} (Router task cameras: {task['cameras']})", flush=True)
                recorder = TrialRecorder(path, record_cameras)
            print('Trial started. Press Enter to finish, or Ctrl-C to interrupt.')
            print('Recording enabled: frames are written by a background worker.' if recorder is not None
                  else 'Recording disabled.', flush=True)
            print(f'Action execution: {"enabled" if execute_action else "disabled"}.', flush=True)
            actions = None
            chunk_index = 0
            plan_done = False
            # Mobile bases run against a wall clock; arms keep their step limit unless the task sets one.
            navigation = task.get('navigation') or {}
            timeout_s = navigation.get('timeout_s', 300 if robot.action_space_name == 'base_velocity' else None)
            deadline = None if timeout_s is None else time.monotonic() + float(timeout_s)
            while step < task['max_steps']:
                if sys.stdin.isatty() and select.select([sys.stdin], [], [], 0)[0]:
                    sys.stdin.readline()
                    if step > 0:
                        break
                if INTERRUPT.is_set():
                    INTERRUPT.clear()
                    print('Trial interrupted by the operator.', flush=True)
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    print(f'Trial reached its {timeout_s}s navigation timeout.', flush=True)
                    break
                started = time.monotonic()
                current = await robot_call(read_observation, robot, step, 'step_start')
                captured_at = time.monotonic()
                if actions is None or chunk_index >= len(actions):
                    observation = protobuf_observation(current, instruction=task['instruction'], control_step=step)
                    plan = await client.infer(observation, deadline_ms=config.deadline_ms)
                    actions = action_chunk(plan, control_step=step, expected_dim=robot.action_dim)
                    plan_done = bool(getattr(plan, 'done', False))
                    chunk_index = 0
                    begin_chunk = getattr(robot, 'begin_chunk', None)
                    if begin_chunk is not None:
                        await robot_call(begin_chunk)
                    print(f'Received action chunk at step {step}: {len(actions)} actions (execution {"enabled" if execute_action else "skipped"}).', flush=True)
                action = actions[chunk_index]
                eef_mode = local_mode and native_mode == 'eef_delta_pose'
                if local_mode:
                    await client.before_action(step)
                sent = await execute_robot_action_async(robot, action, step, robot_call, execute_action)
                if eef_mode:
                    # SO101 EEF: record the joint command the driver accepted after IK.
                    accepted = np.asarray(sent, dtype=np.float32)
                    if accepted.shape != (client.model['action_dim'],) or not np.isfinite(accepted).all():
                        raise ValueError('SO101 EEF driver did not report a valid accepted joint command')
                    if recorder is not None:
                        recorder.add(current, accepted, captured_at=captured_at)
                    path.mkdir(parents=True, exist_ok=True)
                    with (path / 'eef-actions.jsonl').open('a', encoding='utf-8') as target_log:
                        target_log.write(json.dumps({'step': step, 'eef_delta': np.asarray(action).tolist(),
                                                     'accepted_joint_command': accepted.tolist()}) + '\n')
                elif recorder is not None:
                    if robot.action_space_name == 'eef_abs_pose':
                        if sent is None:
                            raise RuntimeError('YAM EE step returned no sent joint command')
                        recorder.add(current, sent, captured_at=captured_at, policy_action=action)
                    else:
                        recorder.add(current, action, captured_at=captured_at)
                chunk_index += 1
                step += 1
                if plan_done:
                    print('Policy declared the task done; its stop command was executed.', flush=True)
                    break
                # Include observation, inference and execution in the control period.
                await asyncio.sleep(max(0, 1/trial_hz - (time.monotonic() - started)))
            if recorder is not None:
                final_observation = await robot_call(robot.get_observation)
                recorder.add(final_observation)
                print('Finishing recording...', flush=True)
                await asyncio.to_thread(recorder.close)
            # Synthetic trials and observation-only runs never home.
            finish_robot = getattr(robot, 'finish_trial', None)
            if execute_action and not config.test and finish_robot is not None:
                return_attempted = True
                await finish_robot(robot_call)
                return_confirmed = True
            if local_mode:
                await client.finish()
            else:
                await asyncio.to_thread(api.request, 'POST', f"/runs/{run['id']}/finish")
        except (Exception, asyncio.CancelledError) as trial_error:
            # YAM stays powered while recovering an ordinary trial error. Do not
            # mask the original failure or submit the failed run as successful.
            if (config.robot_type == 'yam' and execute_action and not config.test and not return_confirmed
                    and robot is not None and not getattr(robot, 'closed', False)):
                hold = getattr(robot, 'hold_for_recovery', None)
                if hold is not None:
                    if recorder is not None:
                        try:
                            await asyncio.to_thread(recorder.close)
                        except Exception:
                            print('Recording close failed during YAM recovery.', flush=True)
                    try:
                        if not return_attempted or isinstance(trial_error, asyncio.CancelledError):
                            print('YAM trial stopped: returning to zero before torque shutdown. '
                                  'Do not press Ctrl+C again; forced interruption may release torque.', flush=True)
                            try:
                                await robot.finish_trial(robot_call)
                            except Exception as return_error:
                                print(f'YAM return failed: {return_error}', flush=True)
                                if getattr(robot, 'closed', False):
                                    raise
                                await hold(robot_call)
                        else:
                            print(f'YAM return failed: {trial_error}', flush=True)
                            await hold(robot_call)
                    except asyncio.CancelledError:
                        # First Ctrl+C can arrive during recovery of an earlier
                        # ordinary error. Restart return instead of dropping torque.
                        print('YAM recovery interrupted: retrying return to zero. '
                              'Do not press Ctrl+C again; forced interruption may release torque.', flush=True)
                        try:
                            await robot.finish_trial(robot_call)
                        except Exception:
                            if getattr(robot, 'closed', False):
                                raise
                            await hold(robot_call)
                    except Exception as recovery_error:
                        print(f'YAM recovery failed; powered hold cannot be guaranteed: '
                              f'{recovery_error}', flush=True)
            raise
        finally:
            resources = [('Client', client.close)]
            if robot is not None:
                resources.append(('Robot', partial(robot_call, robot.close)))
            if recorder is not None:
                resources.append(('Recorder', partial(asyncio.to_thread, recorder.close)))
            await close_resources(resources)


def select_task(api, robot, test=False):
    tasks = api.request('GET', '/tasks?robot_id=' + quote(robot, safe='') + ('&test=true' if test else ''))['tasks']
    if not tasks:
        raise NoAssignment('No Fine-tuning tasks available for this robot')
    print('\nAvailable Fine-tuning tasks:')
    for index, task in enumerate(tasks, 1):
        print(f"{index}. {task['instruction']}")
    while True:
        choice = ask('Select task number: ', options=[(str(index), task['instruction']) for index, task in enumerate(tasks, 1)]).strip()
        if choice.isdecimal() and 1 <= int(choice) <= len(tasks):
            return tasks[int(choice) - 1]['id']
        print(f'Enter a number from 1 to {len(tasks)}.')


def run_evaluation(config, *, track=None, resume=None, abort=None, execute_action=True):
    track = track or config.track or None
    if config.robot_type == ROVER_TYPE and not (resume or abort):
        config = rover_bot(config)
    api = EvalAPI(config)
    root = Path(config.evaluation_dir).resolve()
    task_id = None
    try:
        if abort:
            reason = ask('Reason for interrupted evaluation: ').strip()
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
                    value = ask('Evaluation track [1: Open / 2: Fine-tuning]: ', options=[('1', 'Open'), ('2', 'Fine-tuning')]).strip().lower()
                    track = {'1': 'open', '2': 'fine-tuning', 'open': 'open', 'fine-tuning': 'fine-tuning'}.get(value)
            assignment = None
        while True:
            if STOP.is_set() and assignment is None:
                STOP.clear()
                print('Session stopped by the operator.', flush=True)
                return
            if assignment is None:
                if track == 'fine-tuning' and task_id is None:
                    try:
                        task_id = (select_rover_task(api, config) if config.robot_type == ROVER_TYPE
                                   else select_task(api, config.robot_type, config.test))
                    except NoAssignment as exc:
                        print(str(exc)); return
                instruction = (config.instruction or ask('Instruction: ').strip()) if track == 'open' else ''
                if track == 'open' and not instruction.strip():
                    raise ValueError('instruction cannot be empty')
                scene = config.scene or (ask('Scene / setup: ').strip() if track == 'open' else '')
                mode = 'local' if config.policy_server_url else 'remote'
                request = {'robot_id': config.robot_type, 'track': track,
                    'instruction': instruction, 'scene': scene, 'evaluator': config.evaluator, 'institution': config.institution.strip(),
                    'inference_mode': mode, 'test': config.test, 'recording': config.recording, 'max_steps': config.max_trial_steps,
                    'cameras': list(config.cameras), **({'task_id': task_id} if track == 'fine-tuning' else {})}
                if config.robot_type == ROVER_TYPE and track == 'open':
                    # Open Track rover trials name their destination; Fine-tuning tasks registered one.
                    request['goal'] = rover_goal(config)
                    request['navigation'] = rover_navigation(config)
                    request['max_steps'] = request['navigation']['timeout_s'] * config.control_hz
                try:
                    assignment = api.request('POST', '/next', request)
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
                    ask(f"\n{run['side']}: restore the scene to its initial setup, then press Enter to start.",
                        options=[('', 'Start trial')])
                    options = {'execute_action': execute_action}
                    if assignment.get('inference_mode', 'remote') == 'local' and not config.test:
                        def confirm_live(model, prepared_task):
                            print(
                                f"Prepared model: {model['url']}@{model['revision']} "
                                f"({model['action_space']})."
                            )
                            print(
                                f"Maximum control steps: "
                                f"{prepared_task.get('max_steps', task.get('max_steps'))}."
                            )
                            if config.robot_type == 'yam':
                                print('YAM initialization enables position control; shutdown releases motor torque.')
                            elif config.robot_type == 'so101':
                                print('SO101 initialization enables motor torque; shutdown returns to the '
                                      'start pose, then releases torque.')
                            else:
                                print('RobotEnv initialization may reset the robot before any model action.')
                            return yes_no('Confirm this physical trial may initialize the robot and start')
                        options['confirm_live'] = confirm_live
                    asyncio.run(run_trial(config, assignment, run, run_path, api, **options))
                elif run['state'] != 'finished':
                    raise RuntimeError(f"Run {run['side']} was interrupted; restart the client for a new evaluation")
                if (run_path / 'recording-disabled.json').exists():
                    raise RuntimeError('This diagnostic run cannot be submitted; restart the client for a new evaluation')
                if run['side'] not in local['outcomes']:
                    if config.robot_type == ROVER_TYPE:
                        local['outcomes'][run['side']] = rover_score(run['side'])
                    elif track == 'open':
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
                    dataset = export_trial(run_path, trial_cameras(config, task), fps=context['fps'],
                        robot_type=context['robot_type'], task=task['instruction'],
                        metadata={'assignment_id': assignment['id'], 'run_id': run['id'],
                            'side': run['side'], 'track': track, 'test': context['test'],
                            'action_space': context['action_space'],
                            'native_control_mode': context.get('native_control_mode', 'joint_pos'),
                            'action_recording': context.get('action_recording', 'policy_targets'),
                            'execution_enabled': context['execution_enabled'],
                            'institution': assignment.get('institution', config.institution),
                            'scene': assignment.get('scene', config.scene),
                            'evaluator': assignment.get('evaluator', config.evaluator),
                            'outcome': local['outcomes'][run['side']],
                            **({'goal': task['goal']} if task.get('goal') else {})})
                    print(f'LeRobot dataset saved: {dataset}', flush=True)
            if 'result' not in local:
                preference = None
                if track == 'open':
                    while preference not in {'A', 'B', 'tie'}:
                        value = ask('Preference (A/B/tie): ', options=[('A', 'A'), ('B', 'B'), ('tie', 'Tie')]).strip()
                        preference = 'tie' if value.lower() == 'tie' else value.upper()
                local['result'] = {'outcomes': local['outcomes'], 'preference': preference,
                    'feedback': ask('Feedback (optional): ').strip() if track == 'open' else ''}
                save(manifest, local)
            api.request('POST', f"/assignments/{assignment['id']}/result", local['result'])
            local['submitted'] = True; save(manifest, local)
            print('Evaluation result submitted.', flush=True)
            if config.recording and config.skip_upload:
                print('Dataset upload skipped. Recordings remain local. '
                      f'Upload later with colosseum-upload-dataset <config> --evaluation-id {assignment["id"]}', flush=True)
            elif config.recording:
                print('Uploading saved recordings. The result is already published.', flush=True)
            for run in assignment['runs']:
                run_path = folder / run['id']
                if config.recording and not config.skip_upload:
                    dataset = run_path / 'lerobot'
                    # The server checks each object on retry; local flags alone do not prove durability.
                    try:
                        api.upload_dataset(run['id'], dataset)
                    except (Exception, KeyboardInterrupt) as exc:
                        print(f'Dataset upload deferred ({type(exc).__name__}). The result is published and recordings remain local. '
                              f'Retry with colosseum-upload-dataset <config> --evaluation-id {assignment["id"]}', flush=True)
                        return
                    local.setdefault('datasets_uploaded', {})[run['id']] = True
                    save(manifest, local)
                    print('LeRobot dataset uploaded and verified.', flush=True)
            if config.recording and not config.skip_upload:
                print('Videos and result submitted.')
            elif not config.recording:
                print('Evaluation result submitted (recording disabled).')
            if track == 'open' and not yes_no('Continue with the next evaluation?'):
                return
            assignment = None
    except (Exception, KeyboardInterrupt):
        if 'assignment' in locals() and assignment:
            if 'local' in locals() and local.get('submitted'):
                print('Result already submitted. Recordings remain local for dataset backfill.')
            else:
                print('Saved locally. Use --resume to submit this evaluation; a normal restart closes it and starts a new one.')
        raise
    finally:
        api.close()
