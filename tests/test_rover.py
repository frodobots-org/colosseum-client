import base64
import json
import math
import time

import httpx
import numpy as np
import pytest

from colosseum_client import colosseum_pb2 as pb
from colosseum_client.rover import ActionBuffer, RoverSDK, SimulatedRover, observation
from colosseum_client.rover_recording import RoverRecording
from colosseum_client.tensors import tensor_from_numpy, tensor_to_numpy


def plan(actions, **kwargs):
    return pb.ActionPlan(actions=tensor_from_numpy(np.asarray(actions, dtype=np.float32)), control_hz=10, **kwargs)


def test_action_expiry_limits_done_and_malformed():
    buffer = ActionBuffer(.15, .25)
    buffer.accept(plan([[9, -9]] * 3), 10)
    assert buffer.current(10.1) == pytest.approx((.15, -.25))
    assert buffer.current(10.31) == (0, 0)
    assert buffer.expiries == 1
    assert buffer.current(99) == (0, 0) and buffer.expiries == 1
    for actions, done in [([[1, 1]], True), ([[float('nan'), 0]], False), ([[0]], False)]:
        with pytest.raises(ValueError):
            buffer.accept(plan(actions, done=done), 11)
    buffer.accept(plan([[0, 0]], done=True), 11)
    assert buffer.done and buffer.current(11) == (0, 0)


@pytest.mark.asyncio
async def test_photo_contract_is_allowlist_and_gps_is_separate():
    rover = SimulatedRover()
    sample = await rover.sample()
    sample.telemetry.update(latitude=52, longitude=13, hidden_judge=123, api_key='secret')
    job = {'goal': {'instruction': 'Go to photo', 'navigation': {'goal_type': 'image'},
                    'image_base64': base64.b64encode(rover.frame).decode()}}
    obs = observation(sample, job, 3)
    assert {'front_image', 'goal_image'} == {image.sensor_id for image in obs.sensors}
    assert not {'latitude', 'longitude', 'hidden_judge', 'api_key'} & obs.state.keys()
    job['goal'].update(latitude=1, longitude=2, navigation={'goal_type': 'gps'})
    gps = observation(sample, job, 4)
    np.testing.assert_equal(tensor_to_numpy(gps.state['goal_gps']), [1, 2])
    assert 'goal_image' not in {image.sensor_id for image in gps.sensors}


@pytest.mark.asyncio
async def test_sdk_identity_and_missing_telemetry_fail_closed():
    rover = SimulatedRover('the-intended-bot')
    calls = []
    def handle(request):
        calls.append(request.url.path)
        data = {'/status': {'configured_bot_slug': 'the-intended-bot', 'configured_mission_slug': 'eval',
                            'control_watchdog_s': .5, 'sdk_version': 'fixture', 'telemetry_age_s': None},
                '/start-mission': {'identity': rover.identity}, '/stop': {'stop_confirmed': True},
                '/data': {'speed': 0}, '/v2/front': {'front_frame': base64.b64encode(rover.frame).decode(), 'timestamp': time.time()}}
        return httpx.Response(200 if request.url.path in data else 404, json=data.get(request.url.path, {}))
    sdk = RoverSDK('http://sdk', 'the-intended-bot')
    await sdk.http.aclose()
    sdk.http = httpx.AsyncClient(base_url='http://sdk', transport=httpx.MockTransport(handle))
    await sdk.start()
    assert sdk.identity['ride_id'] == 1 and '/stop' in calls
    sample = await sdk.sample()
    assert not sample.fresh() and math.isinf(sample.telemetry_age_s)
    sdk.bot_slug = 'wrong-bot'
    with pytest.raises(ValueError, match='does not match'):
        await sdk.start()
    assert calls.count('/start-mission') == 1
    await sdk.http.aclose()


@pytest.mark.asyncio
async def test_recording_replay_and_exact_observation_evidence(tmp_path):
    rover = SimulatedRover()
    sample = await rover.sample()
    sample.images['rear_image'] = rover.frame
    sample.captured['rear_image'] = time.time()
    job = {'goal': {'instruction': 'Photo mission', 'navigation': {'goal_type': 'image'},
                    'image_base64': base64.b64encode(rover.frame).decode()}}
    recording = RoverRecording(tmp_path)
    recording.observation(observation(sample, job, 1), 1)
    recording.finish({'simulated': True})
    assert (tmp_path/'front.mp4').stat().st_size > 0
    assert (tmp_path/'rear.mp4').stat().st_size > 0
    assert (tmp_path/'observations.tar.gz').stat().st_size > 0
    event = json.loads((tmp_path/'policy.jsonl').read_text())
    assert 'latitude' not in event['state']
    assert len(event['images']) == 3


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['slow_policy', 'model_failure', 'sdk_loss'])
async def test_runner_independent_control_and_failure_outcomes(tmp_path, monkeypatch, failure):
    import asyncio
    from colosseum_client import rover_runner
    identity = dict(name='Fixture', model='fixture', policy_revision='a'*40, prompt_version='fixture-v1',
                    configuration_version=1, controller={'linear': .15})
    class Policy:
        def __init__(self, *args, **kwargs): self.calls = 0
        async def connect(self, **kwargs):
            return pb.SessionReady(policy_metadata_json=json.dumps(identity))
        async def infer(self, obs, **kwargs):
            self.calls += 1
            if self.calls == 1:
                return plan([[.15, 0]] * 3)
            await asyncio.sleep(.65)
            if failure == 'model_failure':
                return plan([[0, 0]], decision_metadata_json='{"error":"model unavailable"}')
            return plan([[0, 0]], done=True)
        async def close(self): pass
    monkeypatch.setattr(rover_runner, 'ColosseumClient', Policy)
    runner = rover_runner.Runner(dict(api_url='http://unused.invalid', router_url='ws://unused.invalid', token='fixture',
        institution='Fixture', runner_id='fixture', bot_slug='simulated-mini', sdk_url='http://unused.invalid',
        max_linear=.15, max_angular=.25, qualified=False, evaluation_dir=str(tmp_path)), armed=True, simulated=True)
    runner.connected = True
    runner.latest = await runner.sdk.sample()
    runner.heartbeat_at = time.monotonic()
    commands, reports = [], []
    control = runner.sdk.control
    async def command(linear, angular):
        commands.append((time.monotonic(), linear, angular))
        if failure == 'sdk_loss' and len(commands) > 4:
            raise OSError('connection lost')
        await control(linear, angular)
    runner.sdk.control = command
    async def request(method, path, **kwargs):
        reports.append((path, kwargs['json']))
        return {}
    runner.request = request
    async def samples():
        while True:
            runner.latest = await runner.sdk.sample()
            runner.heartbeat_at = time.monotonic()
            await asyncio.sleep(.03)
    sampling = asyncio.create_task(samples())
    job = dict(run_id='run_fixture', assignment_id='fixture', bot_slug='simulated-mini',
        goal=dict(id='photo', instruction='Navigate', image_base64=base64.b64encode(runner.sdk.frame).decode(), navigation={'goal_type':'image'}),
        task={'navigation': {'timeout_s': 10, 'stopped_s': 2, 'max_speed_mps': .4}})
    try:
        await runner.evaluate(job)
    finally:
        sampling.cancel()
        await asyncio.gather(sampling, return_exceptions=True)
        await runner.api.aclose()
    final = reports[-1][1]
    assert final['reason'] == {'slow_policy':'done', 'model_failure':'policy_error', 'sdk_loss':'sdk_connection'}[failure]
    assert final['health']['stop_confirmed'] and final['health']['stop_held_s'] == 2
    assert final['policy']['model'] == 'fixture'
    assert runner.sdk.linear == 0
    if failure != 'sdk_loss':
        assert len(commands) >= 7
        assert max(b[0]-a[0] for a,b in zip(commands,commands[1:])) < .25
        assert any(command[1] == 0 for command in commands[4:])
        assert final['health']['action_expiries'] >= 1
