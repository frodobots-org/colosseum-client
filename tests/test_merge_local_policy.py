"""Regression coverage for local-policy branch integration."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from colosseum_client import evaluation
from colosseum_client.diagnostics import close_resources
from colosseum_client.robot_config import RobotClientConfig


@pytest.mark.parametrize('field', ['policy_ping_interval', 'policy_ping_timeout'])
@pytest.mark.parametrize('value', [0, -1, True, '60', 1.5])
def test_policy_keepalive_rejects_invalid_values(field, value):
    with pytest.raises(ValueError, match=field):
        RobotClientConfig(url='ws://router', token='test', cameras={}, **{field: value})


def test_local_example_and_yaml_keepalive(tmp_path):
    example = Path(__file__).resolve().parents[1] / 'configs/robot.open-local-eval.yaml.example'
    config = RobotClientConfig.from_yaml(example)
    assert config.test and config.robot_type == 'franka' and config.adapter is None
    assert (config.policy_ping_interval, config.policy_ping_timeout) == (20, 60)
    custom = tmp_path / 'robot.yaml'
    custom.write_text(example.read_text().replace('policy_ping_interval: 20', 'policy_ping_interval: 7')
                      .replace('policy_ping_timeout: 60', 'policy_ping_timeout: 15'))
    config = RobotClientConfig.from_yaml(custom)
    assert (config.policy_ping_interval, config.policy_ping_timeout) == (7, 15)


@pytest.mark.parametrize('original', [None, ValueError('original inference failure'), asyncio.CancelledError()])
async def test_cleanup_attempts_all_resources_and_preserves_original_error(original, caplog):
    events = []
    first_close_error = RuntimeError('signed-url-secret')
    async def first():
        events.append('first')
        raise first_close_error
    async def second():
        events.append('second')
        raise OSError('second-secret')
    with pytest.raises(type(original) if original is not None else RuntimeError) as caught:
        try:
            if original is not None:
                raise original
        finally:
            await close_resources([('First resource', first), ('Second resource', second)])
    assert caught.value is (original if original is not None else first_close_error)
    assert events == ['first', 'second']
    assert 'First resource cleanup failed (RuntimeError)' in caplog.text
    assert 'Second resource cleanup failed (OSError)' in caplog.text
    assert 'secret' not in caplog.text


async def test_trial_cleanup_does_not_mask_robot_failure(tmp_path, monkeypatch):
    events = []
    failure = ValueError('observation failed')
    class Client:
        async def connect(self, **kwargs):
            return SimpleNamespace(action_spaces=['joint_position'])
        async def close(self):
            events.append('client closed')
            raise RuntimeError('client close failed')
    class Robot:
        joint_count = 7
        has_gripper = True
        action_dim = 8
        action_space_name = 'joint_position'
        def get_observation(self):
            raise failure
        def close(self):
            events.append('robot closed')
            raise RuntimeError('robot close failed')
    monkeypatch.setattr(evaluation, 'ColosseumClient', lambda *args, **kwargs: Client())
    config = RobotClientConfig(url='ws://router', token='test', cameras={}, recording=False)
    with pytest.raises(ValueError) as caught:
        await evaluation.run_trial(config, {'task': {'cameras': [], 'max_steps': 1}},
                                   {'id': 'run'}, tmp_path, None, robot_factory=lambda _: Robot())
    assert caught.value is failure
    assert events == ['client closed', 'robot closed']


def test_blank_instruction_never_requests_assignment(tmp_path, monkeypatch):
    requests = []
    class API:
        def __init__(self, config): pass
        def request(self, method, path, body=None):
            requests.append(path)
            assert path == '/reset'
            return {'discarded': None}
        def close(self): pass
    monkeypatch.setattr(evaluation, 'EvalAPI', API)
    config = RobotClientConfig(url='ws://router', token='test', cameras={},
                               instruction='   ', scene='desk', evaluation_dir=str(tmp_path))
    with pytest.raises(ValueError, match='instruction cannot be empty'):
        evaluation.run_evaluation(config, track='open')
    assert requests == ['/reset']


def test_http_diagnostics_do_not_persist_credentials(tmp_path, capsys):
    config = RobotClientConfig(url='ws://router', token='client-secret', cameras={},
                               evaluation_dir=str(tmp_path))
    api = evaluation.EvalAPI(config)
    api.client.close()
    def fail(request):
        return httpx.Response(403, headers={'x-request-id': 'request-123',
            'location': 'https://example.com/file?signature=url-secret',
            'set-cookie': 'auth=cookie-secret', 'authorization': 'Bearer header-secret'},
            json={'detail': 'body-secret'}, request=request)
    api.client = httpx.Client(base_url='http://router', transport=httpx.MockTransport(fail))
    with pytest.raises(RuntimeError, match='403: Forbidden') as caught:
        api.request('POST', '/next')
    api.close()
    files = list(tmp_path.glob('http-error-*.json'))
    assert len(files) == 1
    assert json.loads(files[0].read_text()) == {
        'operation': 'Evaluation API', 'status': 403, 'error': 'Forbidden', 'x-request-id': 'request-123'}
    assert files[0].stat().st_mode & 0o777 == 0o600
    output = capsys.readouterr().out + files[0].read_text() + str(caught.value)
    assert 'secret' not in output
