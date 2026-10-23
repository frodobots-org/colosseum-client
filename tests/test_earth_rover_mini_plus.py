"""Earth Rover Mini+ adapter: contract, watchdog, goals, stop confirmation and the standard trial loop."""
import asyncio
import hashlib
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest
from PIL import Image

from colosseum_client import adapters, evaluation as e
from colosseum_client import colosseum_pb2 as pb
from colosseum_client.lerobot_export import export_trial
from colosseum_client.navigation import NAV_STATE_NAMES, ROVER_TYPE
from colosseum_client.robot_config import RobotClientConfig
from colosseum_client.robots import earth_rover_mini_plus as rover
from colosseum_client.robots.earth_rover import SimulatedRoverLink
from colosseum_client.robots.earth_rover.commands import command_payload
from colosseum_client.robots.earth_rover.telemetry import parse_report
from colosseum_client.tensors import tensor_from_numpy


def rover_config(tmp_path, cameras=None, **settings):
    return RobotClientConfig(url='ws://localhost', token='t', cameras=cameras or {'front_image': 'front'},
                             robot_type=ROVER_TYPE, control_hz=10, image_width=64, image_height=32,
                             evaluation_dir=str(tmp_path),
                             adapter_config={'link': 'simulated', 'command_ttl_s': 0.2, **settings})


@pytest.fixture
def robot(tmp_path):
    adapter = rover.EarthRoverMiniPlusRobot(rover_config(tmp_path))
    yield adapter
    if not adapter.closed:
        adapter.close()


def wait_until(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def test_configuration_is_validated(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match='front_image'):
        rover.EarthRoverMiniPlusRobot(rover_config(tmp_path, cameras={'head_image': '0'}))
    with pytest.raises(ValueError, match='control_hz'):
        rover.EarthRoverMiniPlusRobot(RobotClientConfig(url='ws://localhost', token='t', cameras={'front_image': 'f'},
                                                        robot_type=ROVER_TYPE, control_hz=15, adapter_config={'link': 'simulated'}))
    with pytest.raises(ValueError, match='Unsupported'):
        rover.EarthRoverMiniPlusRobot(rover_config(tmp_path, port='x'))
    with pytest.raises(ValueError, match='command_ttl_s'):
        rover.EarthRoverMiniPlusRobot(rover_config(tmp_path, command_ttl_s=9))
    with pytest.raises(ValueError, match='FRODOBOTS_SDK_TOKEN'):
        monkeypatch.delenv('FRODOBOTS_SDK_TOKEN', raising=False)
        rover.EarthRoverMiniPlusRobot(rover_config(tmp_path, sdk_token='env:FRODOBOTS_SDK_TOKEN'))
    monkeypatch.setenv('FRODOBOTS_SDK_TOKEN', 'secret')
    assert rover.resolve_secret('env:FRODOBOTS_SDK_TOKEN') == 'secret' and rover.resolve_secret('plain') == 'plain'


def test_observation_contract_and_goals(robot, tmp_path):
    assert (robot.joint_count, robot.has_gripper, robot.action_dim, robot.action_space_name) == (0, False, 2, 'base_velocity')
    observation = robot.get_observation()
    assert observation.images['front_image'].shape == (32, 64, 3)
    assert observation.joints.size == observation.gripper.size == observation.cartesian_position.size == 0
    state = observation.extra_state['navigation_state']
    assert state.shape == (len(NAV_STATE_NAMES),) and state.dtype == np.float64 and 'goal_gps' not in observation.extra_state
    assert 'goal_image' not in observation.images

    robot.prepare_for_task({'goal': {'type': 'gps', 'latitude': 1.5, 'longitude': 103.8}, 'navigation': {'stopped_s': 3}})
    observation = robot.get_observation()
    np.testing.assert_array_equal(observation.extra_state['goal_gps'], [1.5, 103.8])
    assert robot.stopped_s == 3
    with pytest.raises(ValueError, match='out of range'):
        robot.prepare_for_task({'goal': {'type': 'gps', 'latitude': 95, 'longitude': 0}})

    goals = tmp_path / 'goals'
    goals.mkdir()
    Image.new('RGB', (300, 200), (10, 20, 30)).save(goals / 'fountain.jpg')
    digest = hashlib.sha256((goals / 'fountain.jpg').read_bytes()).hexdigest()
    robot.goal_dir = goals
    with pytest.raises(ValueError, match='checksum'):
        robot.prepare_for_task({'goal': {'type': 'image', 'file': 'fountain.jpg', 'sha256': '0' * 64}})
    with pytest.raises(ValueError, match='plain file name'):
        robot.prepare_for_task({'goal': {'type': 'image', 'file': '../fountain.jpg', 'sha256': digest}})
    with pytest.raises(FileNotFoundError):
        robot.prepare_for_task({'goal': {'type': 'image', 'file': 'missing.jpg', 'sha256': digest}})
    robot.prepare_for_task({'goal': {'type': 'image', 'file': 'fountain.jpg', 'sha256': digest}})
    observation = robot.get_observation()
    assert observation.images['goal_image'].shape == (32, 64, 3) and 'goal_gps' not in observation.extra_state


def test_watchdog_sends_scaled_commands_then_zeros_on_its_own_thread(tmp_path):
    robot = rover.EarthRoverMiniPlusRobot(rover_config(tmp_path, max_linear=0.5, max_angular=0.25, command_ttl_s=0.15))
    try:
        link = robot.link
        robot.execute([0.8, -1.0])
        assert wait_until(lambda: any(command[:2] == (0.4, -0.25) for command in link.sent), 0.5)
        count = len(link.sent)
        time.sleep(0.25)
        assert len(link.sent) > count  # keeps re-sending while the command is fresh
        assert wait_until(lambda: link.sent[-1][:2] == (0.0, 0.0), 0.5)  # TTL expired without a new action
        assert robot.sender.expiries >= 1
        assert {command[2] for command in link.sent} == {robot.sender.ident}
        assert robot.sender.ident != threading.get_ident()
        with pytest.raises(ValueError, match='two finite values'):
            robot.execute([0.1, 0.2, 0.3])
        robot.execute([3.0, 0.0])  # clipped before scaling
        assert wait_until(lambda: link.sent[-1][:2] == (0.5, 0.0), 0.5)
    finally:
        robot.close()
    assert link.closed and link.stops >= 1 and link.sent[-1][:2] == (0.0, 0.0)
    robot.close()  # idempotent


def test_stale_feeds_and_sender_faults_end_the_trial(tmp_path, monkeypatch):
    monkeypatch.setattr(rover, 'make_link', lambda settings, cameras, size: SimulatedRoverLink(cameras, size, stale=True))
    robot = rover.EarthRoverMiniPlusRobot(rover_config(tmp_path))
    with pytest.raises(RuntimeError, match='stale'):
        robot.get_observation()
    robot.close()

    monkeypatch.setattr(rover, 'make_link', lambda settings, cameras, size: SimulatedRoverLink(cameras, size, telemetry=False))
    robot = rover.EarthRoverMiniPlusRobot(rover_config(tmp_path))
    with pytest.raises(RuntimeError, match='telemetry'):
        robot.get_observation()
    robot.close()

    class FailingLink(SimulatedRoverLink):
        def send(self, linear, angular):
            raise OSError('network down')
    monkeypatch.setattr(rover, 'make_link', lambda settings, cameras, size: FailingLink(cameras, size))
    robot = rover.EarthRoverMiniPlusRobot(rover_config(tmp_path))
    robot.get_observation()  # starts the link and the sender
    assert wait_until(lambda: robot._fault is not None)
    with pytest.raises(RuntimeError, match='OSError'):
        robot.get_observation()
    robot.close()

    class FlakyLink(SimulatedRoverLink):
        """One failed send in five is a transient hiccup, not a fault."""
        def send(self, linear, angular):
            if len(self.sent) % 5 == 4:
                self.sent.append((None, None, threading.get_ident(), time.monotonic()))
                raise OSError('timeout')
            super().send(linear, angular)
    monkeypatch.setattr(rover, 'make_link', lambda settings, cameras, size: FlakyLink(cameras, size))
    robot = rover.EarthRoverMiniPlusRobot(rover_config(tmp_path))
    robot.get_observation()
    assert wait_until(lambda: len(robot.link.sent) >= 12, 3.0)
    assert robot._fault is None and robot.sender.failures <= 1
    robot.get_observation()
    robot.close()


def test_unconfirmed_stop_is_an_error_after_cleanup(tmp_path, monkeypatch):
    monkeypatch.setattr(rover, 'make_link', lambda settings, cameras, size: SimulatedRoverLink(cameras, size, stop_ok=False))
    robot = rover.EarthRoverMiniPlusRobot(rover_config(tmp_path))
    robot.get_observation()
    with pytest.raises(RuntimeError, match='not confirmed'):
        robot.close()
    assert robot.link.closed and robot.link.stops == rover.STOP_ATTEMPTS
    assert not robot.sender.is_alive()


def test_finish_trial_requests_a_confirmed_stop_through_the_sender(robot):
    async def run():
        calls = []
        async def robot_call(function, *args):
            calls.append(function.__name__)
            return function(*args)
        robot.stopped_s = 0.01
        robot.get_observation()
        await robot.finish_trial(robot_call)
        return calls
    assert asyncio.run(run()) == ['confirmed_stop', '_speed']
    assert robot.link.stops == 1 and robot.link.sent[-1][2] == robot.sender.ident


def test_command_payload_and_report_parsing():
    assert json.loads(command_payload(2.0, -0.33333)) == {'linear': 1.0, 'angular': -0.333, 'lamp': 0}
    report = {'battery': 87, 'orientation': 271.5, 'speed': 0.4, 'latitude': 1.29, 'longitude': 103.77,
              'rpms': [[10, 11, 12, 13, 1.0], [20, 21, 22, 23, 2.0]], 'accels': [[0, 0, 1, 2.0]]}
    parsed = parse_report(json.dumps(report).encode())
    assert parsed == {'latitude': 1.29, 'longitude': 103.77, 'heading_deg': 271.5, 'speed': 0.4, 'battery': 87.0,
                      'rpm_1': 20.0, 'rpm_2': 21.0, 'rpm_3': 22.0, 'rpm_4': 23.0}
    with pytest.raises(ValueError, match='latitude'):
        parse_report(json.dumps({**report, 'latitude': None}))


class FakeClient:
    """Stands in for ColosseumClient: answers each observation with a forward chunk, then a done chunk."""

    def __init__(self, done_at):
        self.done_at = done_at
        self.observations = []

    async def connect(self, **kwargs):
        return SimpleNamespace(action_spaces=['base_velocity'])

    async def infer(self, observation, **kwargs):
        self.observations.append(observation)
        step = observation.control_step
        if step >= self.done_at:
            actions = np.zeros((1, 2), dtype=np.float32)
            return pb.ActionPlan(start_step=step, valid_until_step=step, actions=tensor_from_numpy(actions), done=True)
        actions = np.tile(np.array([[0.5, -0.2]], dtype=np.float32), (5, 1))
        return pb.ActionPlan(start_step=step, valid_until_step=step + 4, actions=tensor_from_numpy(actions))

    async def close(self):
        pass


def run_rover_trial(tmp_path, monkeypatch, *, task, done_at=5, goal_dir=None):
    client = FakeClient(done_at)
    monkeypatch.setattr(e, 'ColosseumClient', lambda *args, **kwargs: client)
    finished = []
    api = SimpleNamespace(request=lambda *args: finished.append(args))
    created = []
    def factory(config):
        robot = rover.EarthRoverMiniPlusRobot(config)
        if goal_dir is not None:
            robot.goal_dir = goal_dir
        created.append(robot)
        return robot
    config = rover_config(tmp_path, timeout_s=300)
    run_path = tmp_path / 'run-1'
    asyncio.run(e.run_trial(config, {'task': task}, {'id': 'run-1'}, run_path, api, robot_factory=factory))
    return client, created[0], finished, run_path


def test_trial_ends_on_done_with_a_confirmed_stop_and_exports(tmp_path, monkeypatch, capsys):
    task = {'instruction': 'Drive to the pin', 'cameras': ['front_image'], 'max_steps': 500,
            'goal': {'type': 'gps', 'latitude': 1.5, 'longitude': 103.8}, 'navigation': {'timeout_s': 300, 'stopped_s': 0.01}}
    client, robot, finished, run_path = run_rover_trial(tmp_path, monkeypatch, task=task, done_at=5)
    assert finished and finished[0][1] == '/runs/run-1/finish'
    assert 'declared the task done' in capsys.readouterr().out
    assert robot.closed and robot.link.stops >= 1 and robot.link.sent[-1][:2] == (0.0, 0.0)
    first = client.observations[0]
    assert {sensor.sensor_id for sensor in first.sensors} == {'front_image'}
    assert 'navigation_state' in first.state and 'goal_gps' in first.state
    rows = [json.loads(line) for line in (run_path / 'frames.jsonl').read_text().splitlines()]
    assert len(rows) == 7 and rows[0]['joints'] == [] and len(rows[0]['action']) == 2
    assert len(rows[0]['extra_state']['navigation_state']) == len(NAV_STATE_NAMES)
    output = export_trial(run_path, ['front_image'], fps=10, robot_type=ROVER_TYPE, task=task['instruction'],
                          metadata={'test': True, 'goal': task['goal']})
    info = json.loads((output / 'meta/info.json').read_text())
    assert info['features']['observation.state'] == {'dtype': 'float64', 'shape': [9], 'names': list(NAV_STATE_NAMES)}
    assert info['features']['action']['names'] == ['linear', 'angular']
    assert info['features']['observation.goal_gps']['names'] == ['goal_latitude', 'goal_longitude']
    assert 'observation.cartesian_position' not in info['features'] and 'observation.images.goal_image' not in info['features']
    table = pq.read_table(output / 'data/chunk-000/file-000.parquet').to_pydict()
    assert table['observation.goal_gps'][0] == [1.5, 103.8]
    np.testing.assert_allclose(table['action'][0], [0.5, -0.2], rtol=1e-6)
    assert len(table['action']) == 6 and table['action'][-1] == [0.0, 0.0]
    context = json.loads((output / 'meta/colosseum.json').read_text())
    assert context['goal'] == task['goal'] and 'turns left' in context['state_action_units']


def test_trial_ends_on_the_navigation_timeout(tmp_path, monkeypatch, capsys):
    task = {'instruction': 'Drive', 'cameras': ['front_image'], 'max_steps': 10000,
            'goal': {'type': 'gps', 'latitude': 0.0, 'longitude': 0.0}, 'navigation': {'timeout_s': 0.3, 'stopped_s': 0.01}}
    client, robot, finished, _ = run_rover_trial(tmp_path, monkeypatch, task=task, done_at=10 ** 6)
    assert 'navigation timeout' in capsys.readouterr().out and finished
    assert 1 <= len(client.observations) < 100 and robot.closed


def test_photo_goal_reaches_the_policy_but_not_the_recording(tmp_path, monkeypatch):
    goals = tmp_path / 'goals'
    goals.mkdir()
    Image.new('RGB', (120, 80), (200, 100, 50)).save(goals / 'plaza.png')
    digest = hashlib.sha256((goals / 'plaza.png').read_bytes()).hexdigest()
    task = {'instruction': 'Go to the plaza', 'cameras': ['front_image'], 'max_steps': 50,
            'goal': {'type': 'image', 'file': 'plaza.png', 'sha256': digest}, 'navigation': {'timeout_s': 300, 'stopped_s': 0.01}}
    client, robot, finished, run_path = run_rover_trial(tmp_path, monkeypatch, task=task, done_at=2, goal_dir=goals)
    first = client.observations[0]
    assert {sensor.sensor_id for sensor in first.sensors} == {'front_image', 'goal_image'}
    assert 'goal_gps' not in first.state
    assert not (run_path / 'goal_image').exists()


def test_stale_frames_abort_the_trial_and_still_stop(tmp_path, monkeypatch):
    monkeypatch.setattr(rover, 'make_link', lambda settings, cameras, size: SimulatedRoverLink(cameras, size, stale=True))
    task = {'instruction': 'Drive', 'cameras': ['front_image'], 'max_steps': 50,
            'goal': {'type': 'gps', 'latitude': 0.0, 'longitude': 0.0}}
    with pytest.raises(RuntimeError, match='stale'):
        run_rover_trial(tmp_path, monkeypatch, task=task)


def test_test_mode_rover_runs_the_synthetic_flow(tmp_path):
    config = RobotClientConfig(url='ws://localhost', token='t', cameras={'front_image': 'dummy'}, robot_type=ROVER_TYPE,
                               test=True, control_hz=10)
    robot = adapters.make_robot(config)
    assert isinstance(robot, adapters.TestRobot) and robot.joint_count == 0 and robot.action_dim == 2
    robot.prepare_for_task({'goal': {'type': 'gps', 'latitude': 2.0, 'longitude': 3.0}})
    observation = robot.get_observation()
    assert observation.extra_state['navigation_state'].shape == (9,) and list(observation.extra_state['goal_gps']) == [2.0, 3.0]
    robot.execute(np.array([0.5, 0.0], dtype=np.float32))
    assert robot.get_observation().extra_state['navigation_state'][3] == 0.5
    with pytest.raises(ValueError):
        robot.execute(np.zeros(8, dtype=np.float32))
    robot.close()



def test_frodobots_session_mimics_the_sdk_token_and_ride_calls():
    import httpx
    from colosseum_client.robots.earth_rover.session import FrodobotsSession
    calls = []
    credentials = {'APP_ID': 'app', 'CHANNEL_NAME': 'ride_bot_frodobot_x', 'USERID': 42, 'RTC_TOKEN': 'rtc', 'RTM_TOKEN': 'rtm',
                   'BOT_UID': 'frodobot_x', 'BOT_TYPE': 'mini', 'SPECTATOR_USERID': 421, 'SPECTATOR_RTC_TOKEN': 'rtc-spectator'}
    def handler(request):
        calls.append((request.method, request.url.path, json.loads(request.content or b'{}')))
        assert request.headers['authorization'] == 'Bearer secret'
        return httpx.Response(200, json=credentials)
    transport = httpx.MockTransport(handler)
    session = FrodobotsSession('secret', 'frodobot_x', api_url='https://api.example/api/v1', transport=transport)
    ride = session.start_ride()
    assert ride.user_id == 42 and ride.bot_uid == 'frodobot_x' and ride.channel == 'ride_bot_frodobot_x'
    # The second RTC identity the SDK dashboard watches with; the headless page keeps USERID.
    assert ride.spectator() == {'appid': 'app', 'channel': 'ride_bot_frodobot_x', 'uid': 421, 'rtcToken': 'rtc-spectator'}
    session.end_ride()
    assert calls == [('POST', '/api/v1/sdk/token', {'bot_slug': 'frodobot_x'})]
    with_mission = FrodobotsSession('secret', 'frodobot_x', 'open-mission', api_url='https://api.example/api/v1', transport=transport)
    with_mission.start_ride()
    with_mission.end_ride()
    assert calls[1:] == [('POST', '/api/v1/sdk/start_ride', {'bot_slug': 'frodobot_x', 'mission_slug': 'open-mission'}),
                         ('POST', '/api/v1/sdk/end_ride', {'bot_slug': 'frodobot_x', 'mission_slug': 'open-mission'})]
    with pytest.raises(ValueError, match='bot_slug'):
        FrodobotsSession('secret', '', transport=transport)
    credentials.pop('SPECTATOR_USERID')  # a backend without the spectator identity: the page falls back to snapshots
    assert FrodobotsSession('secret', 'frodobot_x', api_url='https://api.example/api/v1', transport=transport).start_ride().spectator() is None

    def lookup(request):
        if request.url.path.endswith('/sdk/missions'):
            assert request.url.params['bot_slug'] == 'frodobot_x'
            return httpx.Response(200, json={'missions': [{'slug': 'loop', 'type': 'checkpoints', 'checkpoints_count': 2}]})
        assert request.method == 'POST' and request.url.path.endswith('/sdk/checkpoints_list')
        assert json.loads(request.content) == {'bot_slug': 'frodobot_x', 'mission_slug': 'loop'}
        return httpx.Response(200, json={'checkpoints_list': [{'id': 2, 'sequence': 2, 'latitude': '1.2', 'longitude': '2.2'},
                                                               {'id': 1, 'sequence': 1, 'latitude': '1.1', 'longitude': '2.1'}],
                                         'latest_scanned_checkpoint': 0})
    lookups = FrodobotsSession('secret', 'frodobot_x', api_url='https://api.example/api/v1', transport=httpx.MockTransport(lookup))
    assert lookups.missions()[0]['slug'] == 'loop'
    assert [item['sequence'] for item in lookups.checkpoints('loop')] == [1, 2]
    ride = FrodobotsSession('secret', 'frodobot_x', api_url='https://api.example/api/v1', transport=transport)
    ride.start_ride('loop')
    assert calls[-1] == ('POST', '/api/v1/sdk/start_ride', {'bot_slug': 'frodobot_x', 'mission_slug': 'loop'})



def test_link_starts_from_the_task_mission_like_the_sdk(tmp_path):
    robot = rover.EarthRoverMiniPlusRobot(rover_config(tmp_path))
    assert not robot.started and robot.sender is None and not robot.link.started
    robot.close()  # never connected: nothing to stop, no error
    assert robot.link.stops == 0

    robot = rover.EarthRoverMiniPlusRobot(rover_config(tmp_path))
    robot.prepare_for_task({'goal': {'type': 'gps', 'latitude': 1.0, 'longitude': 2.0, 'mission_slug': 'courtyard-loop', 'checkpoint': 3}})
    assert robot.started and robot.link.mission_slug == 'courtyard-loop'
    robot.close()

    robot = rover.EarthRoverMiniPlusRobot(rover_config(tmp_path, mission_slug='configured-mission'))
    robot.get_observation()  # no task goal: the configured mission, like MISSION_SLUG in the SDK
    assert robot.link.mission_slug == 'configured-mission'
    robot.close()

    robot = rover.EarthRoverMiniPlusRobot(rover_config(tmp_path))
    robot.get_observation()  # neither: token-only session
    assert robot.link.mission_slug is None
    robot.close()


class FakeSession:
    bot_slug = 'frodobot_x'

    def __init__(self):
        self.closed = False
        self.mission_list = [
            {'slug': 'courtyard-loop', 'name': 'Courtyard loop', 'type': 'checkpoints', 'checkpoints_count': 3},
            {'slug': 'plaza', 'name': 'Plaza', 'type': 'checkpoints', 'checkpoints_count': 1},
        ]

    def missions(self):
        return list(self.mission_list)

    def checkpoints(self, slug):
        return {'courtyard-loop': [{'id': 1, 'sequence': 1, 'latitude': '1.1', 'longitude': '2.1'},
                                   {'id': 2, 'sequence': 2, 'latitude': '1.2', 'longitude': '2.2'},
                                   {'id': 3, 'sequence': 3, 'latitude': '1.3', 'longitude': '2.3'}],
                'plaza': [{'id': 9, 'sequence': 1, 'latitude': '5.0', 'longitude': '6.0'}]}[slug]

    def close(self):
        self.closed = True


def test_open_track_goal_comes_from_the_chosen_mission(tmp_path, monkeypatch):
    session = FakeSession()
    monkeypatch.setattr(e, 'rover_session', lambda config: session)
    answers = iter(['x', '9', '1'])
    monkeypatch.setattr('builtins.input', lambda prompt: next(answers))
    config = rover_config(tmp_path, link='frodobots')
    goal = e.rover_goal(config)
    assert goal == {'type': 'gps', 'latitude': 1.3, 'longitude': 2.3, 'mission_slug': 'courtyard-loop', 'checkpoint': 3}
    assert session.closed
    assert e.rover_goal(rover_config(tmp_path, link='frodobots', mission_slug='courtyard-loop', checkpoint=2)) == {
        'type': 'gps', 'latitude': 1.2, 'longitude': 2.2, 'mission_slug': 'courtyard-loop', 'checkpoint': 2}
    with pytest.raises(ValueError, match='not an SDK-enabled'):
        e.rover_goal(rover_config(tmp_path, link='frodobots', mission_slug='unknown'))
    # Manual goals keep working and carry the configured mission for the ride.
    manual = e.rover_goal(rover_config(tmp_path, link='frodobots', goal_gps=[3.0, 4.0], mission_slug='plaza'))
    assert manual == {'type': 'gps', 'latitude': 3.0, 'longitude': 4.0, 'mission_slug': 'plaza'}


def test_fine_tuning_task_is_matched_to_the_chosen_mission(tmp_path, monkeypatch):
    session = FakeSession()
    monkeypatch.setattr(e, 'rover_session', lambda config: session)
    class API:
        def request(self, method, path):
            assert method == 'GET' and path.startswith('/tasks?robot_id=earth_rover_mini_plus')
            return {'tasks': [{'id': 'mission-plaza', 'instruction': 'Go to the plaza', 'goal': {'type': 'gps', 'mission_slug': 'plaza'}},
                              {'id': 'other', 'instruction': 'Other', 'goal': {'type': 'gps', 'mission_slug': 'elsewhere'}}]}
    monkeypatch.setattr('builtins.input', lambda prompt: pytest.fail('no prompt expected with a configured mission'))
    assert e.select_rover_task(API(), rover_config(tmp_path, link='frodobots', mission_slug='plaza')) == 'mission-plaza'
    with pytest.raises(e.NoAssignment, match='courtyard-loop'):
        e.select_rover_task(API(), rover_config(tmp_path, link='frodobots', mission_slug='courtyard-loop'))



def test_bot_slug_is_asked_for_when_not_configured(tmp_path, monkeypatch):
    answers = iter(['', 'frodobot_x'])
    monkeypatch.setattr('builtins.input', lambda prompt: next(answers))
    config = e.rover_bot(rover_config(tmp_path, link='frodobots', bot_slug=''))
    assert config.adapter_config['bot_slug'] == 'frodobot_x'
    configured = rover_config(tmp_path, link='frodobots', bot_slug='frodobot_y')
    assert e.rover_bot(configured) is configured
    assert e.rover_bot(rover_config(tmp_path)) is not None  # simulated link: no prompt



def test_frodobots_link_switches_between_preview_and_ride(monkeypatch):
    import httpx
    from colosseum_client.robots.earth_rover import frodobots_link as module
    from colosseum_client.robots.earth_rover.session import FrodobotsSession
    calls = []
    def handler(request):
        body = json.loads(request.content or b'{}')
        calls.append((request.url.path.rsplit('/', 1)[-1], body.get('mission_slug')))
        channel = 'ride_42' if body.get('mission_slug') else 'ride_bot_frodobot_x'
        return httpx.Response(200, json={'APP_ID': 'app', 'CHANNEL_NAME': channel, 'USERID': 7, 'RTC_TOKEN': 'rtc', 'RTM_TOKEN': 'rtm',
                                         'BOT_UID': 'frodobot_x', 'BOT_TYPE': 'mini', 'message': 'ok'})
    class Video:
        instances = []
        def __init__(self, credentials, cameras): self.channel = credentials.channel; self.closed = False; Video.instances.append(self)
        def start(self): pass
        def wait_for(self, camera, timeout): return True
        def latest(self): return {'front_image': (np.zeros((2, 2, 3), np.uint8), time.monotonic())}
        def close(self): self.closed = True
    class Telemetry:
        def __init__(self, credentials): self.closed = False
        def start(self): pass
        def latest(self): return ({name: 1.0 for name in NAV_STATE_NAMES}, time.monotonic())
        def close(self): self.closed = True
    class Sender:
        def __init__(self, credentials): self.destination = credentials.bot_uid; self.stops = 0; self.closed = False
        def send(self, linear, angular): pass
        def stop(self, confirm=True): self.stops += 1; return True
        def close(self): self.closed = True
    import colosseum_client.robots.earth_rover.video as video_module
    import colosseum_client.robots.earth_rover.telemetry as telemetry_module
    import colosseum_client.robots.earth_rover.commands as commands_module
    monkeypatch.setattr(video_module, 'AgoraVideo', Video)
    monkeypatch.setattr(telemetry_module, 'RtmTelemetry', Telemetry)
    monkeypatch.setattr(commands_module, 'RtmRestSender', Sender)
    link = module.FrodobotsRoverLink({'sdk_token': 'secret', 'bot_slug': 'frodobot_x', 'first_frame_timeout_s': 1, 'transport': 'native'},
                                     ['front_image'], (2, 2))
    link.session = FrodobotsSession('secret', 'frodobot_x', api_url='https://api.example/api/v1', transport=httpx.MockTransport(handler))
    link.start()  # preview
    assert link.started and link.mission_slug is None and Video.instances[-1].channel == 'ride_bot_frodobot_x'
    link.start()  # idempotent
    assert len(Video.instances) == 1
    link.start('plaza')  # switch to the ride: new channel, old media closed
    assert link.mission_slug == 'plaza' and Video.instances[-1].channel == 'ride_42' and Video.instances[0].closed
    link.start('plaza')  # already riding it
    assert len(Video.instances) == 2
    link.end_trial()  # stop, release the ride, back to the preview
    assert link.started and link.mission_slug is None and Video.instances[-1].channel == 'ride_bot_frodobot_x'
    assert [call for call in calls if call[0] in ('start_ride', 'end_ride')] == [('start_ride', 'plaza'), ('end_ride', 'plaza')]
    link.close()
    assert not link.started and Video.instances[-1].closed



def test_bot_without_missions_falls_back_to_registered_tasks(tmp_path, monkeypatch):
    session = FakeSession()
    session.mission_list = []
    monkeypatch.setattr(e, 'rover_session', lambda config: session)
    class API:
        def request(self, method, path):
            return {'tasks': [{'id': 'gps-15m', 'instruction': 'Drive 15 m north', 'goal': {'type': 'gps'}}]}
    monkeypatch.setattr('builtins.input', lambda prompt: '1')
    assert e.select_rover_task(API(), rover_config(tmp_path, link='frodobots')) == 'gps-15m'
    assert session.closed


def test_browser_open_waits_for_a_frame_from_the_new_channel(monkeypatch):
    """Switching preview -> ride reuses the transport; the preview's last frame must not pass as the ride's first."""
    from colosseum_client.robots.earth_rover.browser import BrowserTransport
    from colosseum_client.robots.earth_rover.telemetry import TelemetryStore
    from colosseum_client.robots.earth_rover.video import FrameStore
    transport = BrowserTransport(cameras=['front_image'])
    monkeypatch.setattr(transport, 'start', lambda: None)
    monkeypatch.setattr(transport, '_call', lambda coroutine, timeout: coroutine.close())  # no Chrome in tests
    transport.frames.put('front_image', np.zeros((2, 2, 3), dtype=np.uint8))  # the preview page's last frame
    credentials = SimpleNamespace(app_id='a', channel='ride_1', user_id=7, rtc_token='r', rtm_token='m', bot_uid='bot')
    with pytest.raises(RuntimeError, match='No front camera frame'):
        transport.open(credentials, first_frame_timeout=0.3)
    threading.Timer(0.1, lambda: transport.frames.put('front_image', np.zeros((2, 2, 3), dtype=np.uint8))).start()
    transport.open(credentials, first_frame_timeout=2.0)  # a frame decoded after the join counts
    store = FrameStore()
    store.put('front_image', np.zeros((1, 1, 3), dtype=np.uint8))
    store.clear()
    assert store.latest() == {} and store.counts == {}
    reports = TelemetryStore()
    reports.update({'battery': 1})
    reports.clear()
    assert reports.latest() is None and reports.received == 0


def test_end_trial_waits_out_the_ride_release_and_retries_the_preview(monkeypatch):
    """end_ride queues a shadow write that clears the rover's channel; the preview is requested after it lands."""
    from colosseum_client.robots.earth_rover import frodobots_link as module
    monkeypatch.setattr(module, 'PREVIEW_RESTORE_DELAY_S', 0.0)
    calls = []
    link = module.FrodobotsRoverLink({'sdk_token': 'secret', 'bot_slug': 'frodobot_x'}, ['front_image'], (2, 2))
    link.session = SimpleNamespace(mission_slug='plaza', end_ride=lambda: calls.append('end_ride'))
    link.started = True
    monkeypatch.setattr(link, 'stop', lambda confirm=True: True)

    def flaky_open(mission_slug):
        calls.append(('open', mission_slug))
        if sum(1 for call in calls if call[0] == 'open') == 1:
            raise RuntimeError('No front camera frame within 15s')
        link.started = True
    monkeypatch.setattr(link, '_open', flaky_open)
    link.end_trial()
    assert calls == ['end_ride', ('open', ''), ('open', '')] and link.started

    def dead_open(mission_slug):
        calls.append(('open', mission_slug))
        raise RuntimeError('still nothing')
    calls.clear()
    link.started = True
    monkeypatch.setattr(link, '_open', dead_open)
    link.end_trial()  # both attempts failing leaves the link idle without raising: the page says to reconnect
    assert not link.started and calls == ['end_ride', ('open', ''), ('open', '')]
    assert link.diagnostics()['started'] is False and link.diagnostics()['frames'] == {'front_image': {'count': 0, 'age_s': None}}


def test_staleness_limits_are_configurable(tmp_path, monkeypatch):
    monkeypatch.setattr(rover, 'make_link', lambda settings, cameras, size: SimulatedRoverLink(cameras, size, stale=True))
    tolerant = rover.EarthRoverMiniPlusRobot(rover_config(tmp_path, front_max_age_s=20, telemetry_max_age_s=20))
    try:
        assert tolerant.get_observation().images['front_image'].shape == (32, 64, 3)  # 10 s old is fine under 20 s
    finally:
        tolerant.close()
    strict = rover.EarthRoverMiniPlusRobot(rover_config(tmp_path, front_max_age_s=1))
    try:
        with pytest.raises(RuntimeError, match=r'stale \(10\.\ds old; publishers \[1000, 1001\], front frames'):
            strict.get_observation()  # the error carries what the link saw, for the session log
    finally:
        strict.close()
    with pytest.raises(ValueError, match='front_max_age_s'):
        rover.EarthRoverMiniPlusRobot(rover_config(tmp_path, front_max_age_s=0))
