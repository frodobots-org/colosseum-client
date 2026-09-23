import json
import base64
import hashlib
import sys
from types import SimpleNamespace

import pytest

from colosseum_client.robot_config import RobotClientConfig
from colosseum_client.evaluation import EvalAPI
from colosseum_client import hub_upload as ms


def test_modelscope_yaml_and_secret_repr(tmp_path):
    path = tmp_path / 'robot.yaml'
    path.write_text('url: ws://localhost\ntoken: client\ncameras: {head_image: x}\n'
                    'dataset_url: https://modelscope.cn/datasets/owner/data\n'
                    'dataset_token: private-test-secret\n')
    config = RobotClientConfig.from_yaml(path)
    assert config.dataset_token == 'private-test-secret'
    assert 'private-test-secret' not in repr(config)
    path.write_text(path.read_text().replace('dataset_token: private-test-secret',
                                           'dataset_token_file: secrets/ms-token'))
    assert RobotClientConfig.from_yaml(path).dataset_token_file == str(tmp_path / 'secrets/ms-token')


@pytest.mark.parametrize('url', ['http://modelscope.cn/datasets/a/b',
    'https://modelscope.cn.evil/datasets/a/b', 'https://secret@modelscope.cn/datasets/a/b',
    'https://modelscope.cn/datasets/a/b?token=secret', 'https://modelscope.cn/datasets/a/b/files',
    '../a/b', None])
def test_reject_invalid_destinations(url):
    with pytest.raises(ValueError):
        RobotClientConfig('ws://localhost', 'client', {}, dataset_url=url)


@pytest.mark.parametrize('host,provider', [('modelscope.cn', 'modelscope'), ('huggingface.co', 'huggingface')])
def test_url_selects_provider_without_changing_router(host, provider):
    c = RobotClientConfig('wss://router.example', 'router-token', {},
        dataset_url=f'https://{host}/datasets/owner/data.v1', dataset_token='dataset-secret')
    assert c.url == 'wss://router.example'
    assert ms.dataset_target(c.dataset_url) == (provider, 'owner/data.v1')


@pytest.mark.parametrize('remote_state', ['missing', 'same', 'different', 'private', 'gated'])
def test_huggingface_upload_lfs_git_hashes_and_reporting(tmp_path, monkeypatch, remote_state):
    root = tmp_path / 'lerobot'; root.mkdir()
    (root / 'small.json').write_bytes(b'{}')
    (root / 'video.mp4').write_bytes(b'video-bytes')
    files = [{'path': p.name, 'size': p.stat().st_size,
              'checksum_sha256': base64.b64encode(hashlib.sha256(p.read_bytes()).digest()).decode()}
             for p in sorted(root.iterdir())]
    prefix = 'episodes/e/r/lerobot'
    uploads, calls = [], []
    class HF:
        def __init__(self, **kw):
            assert kw == {'token': 'hf-secret', 'endpoint': 'https://huggingface.co'}
        def dataset_info(self, **kw):
            assert kw == dict(repo_id='owner/data', revision='main', files_metadata=True, token=False)
            siblings = []
            if remote_state in {'same', 'different'}:
                siblings = [SimpleNamespace(rfilename=prefix + '/small.json', size=2,
                    blob_id=hashlib.sha1(b'blob 2\0{}').hexdigest(), lfs=None),
                    SimpleNamespace(rfilename=prefix + '/video.mp4', size=11, blob_id='unused',
                        lfs=SimpleNamespace(sha256=hashlib.sha256(b'video-bytes').hexdigest()))]
                if remote_state == 'different':
                    siblings[1].lfs.sha256 = 'bad'
            return SimpleNamespace(sha='a' * 40, private=remote_state == 'private',
                gated=remote_state == 'gated', siblings=siblings)
        def upload_folder(self, **kw):
            assert kw['repo_type'] == 'dataset' and kw['revision'] == 'main'
            assert kw['parent_commit'] == 'a' * 40 and kw['path_in_repo'] == prefix
            assert kw['allow_patterns'] == ['small.json', 'video.mp4']
            uploads.append(kw)
            return SimpleNamespace(oid='b' * 40)
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(HfApi=HF))
    # Both provider env vars exist: only the matching provider's token is used.
    monkeypatch.setenv('HF_TOKEN', 'hf-secret')
    monkeypatch.setenv('MODELSCOPE_API_TOKEN', 'ms-secret')
    api = EvalAPI.__new__(EvalAPI)
    api.config = RobotClientConfig('ws://router', 'router-token', {},
                                  dataset_url='https://huggingface.co/datasets/owner/data')
    def request(method, route, body=None):
        calls.append((method, route, body))
        assert 'hf-secret' not in json.dumps(body)
        if method == 'GET':
            return {'registered': False, 'path': prefix}
        return {'complete': True}
    api.request = request
    if remote_state in {'different', 'private', 'gated'}:
        with pytest.raises(RuntimeError, match='huggingface upload failed'):
            api.upload_dataset('r', root)
        assert len(calls) == 1 and not uploads
    else:
        api.upload_dataset('r', root)
        expected = {'provider': 'huggingface', 'repo': 'owner/data',
                    'revision': ('b' if remote_state == 'missing' else 'a') * 40}
        assert calls[-2][2] == {'files': files, 'source': expected}
        assert len(uploads) == (remote_state == 'missing')
        # A receipt for HF must never be reused for the same repo name on ModelScope.
        api.config = RobotClientConfig('ws://router', 'router-token', {},
                                       dataset_url='https://modelscope.cn/datasets/owner/data')
        with pytest.raises(RuntimeError, match='receipt'):
            api.upload_dataset('r', root)


@pytest.mark.parametrize('credential', ['inline', 'file', 'env'])
def test_upload_reports_location_without_token_and_resume_skips_upload(tmp_path, monkeypatch, credential):
    root = tmp_path / 'lerobot'
    root.mkdir()
    (root / 'example.json').write_text('{}')
    kwargs = {}
    if credential == 'inline':
        kwargs['dataset_token'] = 'ms-secret'
    elif credential == 'file':
        token = tmp_path / 'token'
        token.write_text('ms-secret\n')
        kwargs['dataset_token_file'] = str(token)
    else:
        monkeypatch.setenv('MODELSCOPE_API_TOKEN', 'ms-secret')
    api = EvalAPI.__new__(EvalAPI)
    api.config = RobotClientConfig('ws://localhost', 'router-secret', {},
                                  dataset_url='https://modelscope.cn/datasets/owner/data', **kwargs)
    calls, uploads = [], []
    prefix = 'episodes/ev_abc/run_abc/lerobot'
    def request(method, route, body=None):
        calls.append((method, route, body))
        assert 'ms-secret' not in json.dumps(body)
        assert '/upload-url' not in route
        if method == 'GET':
            return {'registered': False, 'path': prefix}
        return {'complete': True}
    api.request = request
    class Hub:
        def __init__(self, **kw):
            assert kw == {'token': 'ms-secret', 'endpoint': 'https://modelscope.cn'}
        def list_repo_files(self, *a, **kw):
            return []
        def upload_folder(self, *args, **kw):
            assert kw['path_in_repo'] == prefix
            assert kw['use_cache'] is False
            uploads.append(args)
    monkeypatch.setitem(sys.modules, 'modelscope_hub', SimpleNamespace(HubApi=Hub))
    monkeypatch.setattr(ms, 'public_revision', lambda repo: 'a' * 40)
    api.upload_dataset('run_abc', root)
    registration = calls[-2][2]
    assert registration['source'] == {'provider': 'modelscope', 'repo': 'owner/data', 'revision': 'a' * 40}
    assert calls[-1][1].endswith('/complete')
    api.upload_dataset('run_abc', root)
    assert len(uploads) == 1  # Server failure/retry uses local receipt, not a new commit.
    assert 'ms-secret' not in (tmp_path / 'dataset-upload.json').read_text()
    (root / 'example.json').write_text('{"changed":true}')
    with pytest.raises(RuntimeError, match='receipt'):
        api.upload_dataset('run_abc', root)


def test_registered_source_resume_and_storage_conflict(tmp_path):
    cfg = SimpleNamespace(dataset_url='https://modelscope.cn/datasets/owner/data')
    source = {'provider': 'modelscope', 'repo': 'owner/data', 'revision': 'a' * 40, 'path': 'episodes/a/r/lerobot'}
    status = {'registered': True, 'path': source['path'], 'source': source, 'files': []}
    assert ms.upload_episode(cfg, 'r', tmp_path, [], lambda *a: status) == {
        'provider': 'modelscope', 'repo': 'owner/data', 'revision': 'a' * 40}
    status['source'] = None
    with pytest.raises(RuntimeError, match='different storage'):
        ms.upload_episode(cfg, 'r', tmp_path, [], lambda *a: status)


def test_existing_identical_files_need_no_new_commit(tmp_path, monkeypatch):
    cfg = SimpleNamespace(dataset_url='https://modelscope.cn/datasets/owner/data', dataset_token='ms-secret',
                          dataset_token_file='')
    content = b'{}'
    file = {'path': 'meta/info.json', 'size': len(content),
            'checksum_sha256': base64.b64encode(hashlib.sha256(content).digest()).decode()}
    prefix = 'episodes/a/r/lerobot'
    class Hub:
        def __init__(self, **kw): pass
        def list_repo_files(self, *a, **kw):
            assert kw['revision'] == 'b' * 40
            return [SimpleNamespace(path=prefix + '/' + file['path'], size=len(content),
                                    sha256=hashlib.sha256(content).hexdigest())]
        def upload_folder(self, *a, **kw):
            pytest.fail('Existing identical files must not create a new upload/commit')
    monkeypatch.setitem(sys.modules, 'modelscope_hub', SimpleNamespace(HubApi=Hub))
    monkeypatch.setattr(ms, 'public_revision', lambda repo: 'b' * 40)
    source = ms.upload_episode(cfg, 'r', tmp_path / 'lerobot', [file],
        lambda *a: {'registered': False, 'path': prefix})
    assert source == {'provider': 'modelscope', 'repo': 'owner/data', 'revision': 'b' * 40}


def test_sdk_failure_redacts_credentials_and_does_not_register(tmp_path, monkeypatch, capsys):
    cfg = SimpleNamespace(dataset_url='https://modelscope.cn/datasets/owner/data', dataset_token='secret-token',
                          dataset_token_file='')
    class Hub:
        def __init__(self, **kw):
            raise RuntimeError('Authorization: secret-token')
    monkeypatch.setitem(sys.modules, 'modelscope_hub', SimpleNamespace(HubApi=Hub))
    with pytest.raises(RuntimeError) as error:
        ms.upload_episode(cfg, 'run', tmp_path, [], lambda *a: {'registered': False, 'path': 'episodes/a/run/lerobot'})
    assert 'secret-token' not in str(error.value)
    assert 'secret-token' not in str(capsys.readouterr())
    assert not (tmp_path.parent / 'dataset-upload.json').exists()
