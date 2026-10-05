"""Cloud keys only travel from Client to Policy Server."""
import json
import httpx
import pytest
from colosseum_client.llm_credentials import local_preparation
from colosseum_client.robot_config import RobotClientConfig
from colosseum_client.local_protocol import encode_control, decode_control


def config(**kw):
    return RobotClientConfig(url='wss://router.example', token='client-token', cameras={},
                             policy_server_url='ws://127.0.0.1:8000', **kw)


def test_env_resolution_repr_and_missing_key(monkeypatch):
    monkeypatch.setenv('TEST_XAI_KEY', 'secret-key')
    monkeypatch.delenv('TEST_MISSING_KEY', raising=False)
    cfg = config(llm_api_keys={'xai': 'env:TEST_XAI_KEY', 'openai': 'env:TEST_MISSING_KEY'})
    assert cfg.llm_api_keys == {'xai': 'secret-key'}
    assert cfg.llm_api_urls == ['https://api.x.ai/v1']
    assert 'secret-key' not in repr(cfg)
    assert config().llm_api_urls == []


def test_rejects_cleartext_remote_key_transport():
    with pytest.raises(ValueError, match='wss'):
        RobotClientConfig(url='wss://router.example', token='token', cameras={},
            policy_server_url='ws://192.168.1.2:8000', llm_api_keys={'xai': 'secret'})


def test_only_selected_key_in_local_protobuf():
    original = dict(type='prepare', run_id='run', model=dict(model_type='llm',
        name='grok-4.7', url='https://api.x.ai/v1', revision='rig-v1',
        action_space='joint_position', action_dim=8, control_hz=15, max_horizon=150))
    prepared = local_preparation(original, {'xai': 'xai-secret', 'openai': 'openai-secret'}, 'wss://policy.example')
    decoded = decode_control(encode_control(prepared))
    assert decoded['api_key'] == 'xai-secret'
    assert 'api_key' not in original and 'openai-secret' not in json.dumps(decoded)
    assert decoded['model']['name'] == 'grok-4.7'
    with pytest.raises(ValueError, match='no matching'):
        local_preparation(original, {'openai': 'other'}, 'wss://policy.example')
    vla = {'model': {'url': 'https://huggingface.co/org/model'}}
    assert local_preparation(vla, {'xai': 'secret'}, 'ws://remote.example') == vla


def test_router_request_contains_only_capability_urls():
    from colosseum_client.evaluation import EvalAPI
    cfg = config(llm_api_keys={'xai': 'unique-secret-key'})
    seen = []
    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={})
    api = EvalAPI(cfg)
    api.client.close()
    api.client = httpx.Client(base_url='https://router.example', transport=httpx.MockTransport(handler))
    try:
        api.request('POST', '/next', {'inference_mode': 'local'})
    finally:
        api.client.close()
    assert seen == [{'inference_mode': 'local', 'llm_api_urls': ['https://api.x.ai/v1']}]
    assert 'secret' not in json.dumps(seen)


def test_yaml_loading(tmp_path, monkeypatch):
    monkeypatch.setenv('TEST_XAI_KEY', 'secret')
    path = tmp_path/'robot.yaml'
    path.write_text('url: wss://router.example\ntoken: test\ntest: true\npolicy_server_url: ws://localhost:8000\nllm_api_keys:\n  xai: env:TEST_XAI_KEY\n')
    assert RobotClientConfig.from_yaml(path).llm_api_keys == {'xai': 'secret'}
