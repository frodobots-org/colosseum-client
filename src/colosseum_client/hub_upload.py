"""Direct uploads to public dataset hubs; credentials stay on Client."""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit


HUBS = {'modelscope.cn': 'modelscope', 'www.modelscope.cn': 'modelscope',
        'huggingface.co': 'huggingface'}


def dataset_target(value):
    if not isinstance(value, str):
        raise ValueError('dataset_url must be a ModelScope or Hugging Face dataset URL')
    parts = urlsplit(value)
    if (parts.scheme != 'https' or parts.netloc not in HUBS or parts.query
            or parts.fragment or not parts.path.startswith('/datasets/')):
        raise ValueError('Use https://modelscope.cn/datasets/owner/name or https://huggingface.co/datasets/owner/name')
    repo = parts.path[len('/datasets/'):].rstrip('/')
    if not re.fullmatch(r'[A-Za-z0-9_-]{1,100}/[A-Za-z0-9_][A-Za-z0-9_.-]{0,99}', repo):
        raise ValueError('dataset_url must identify a repository, without a file path')
    return HUBS[parts.netloc], repo


def public_revision(repo):
    """Resolve ModelScope's public master branch to an immutable commit."""
    try:
        result = subprocess.run(['git', '-c', 'credential.helper=', 'ls-remote',
            f'https://modelscope.cn/datasets/{repo}.git', 'refs/heads/master'],
            capture_output=True, text=True, timeout=60,
            env={**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GIT_CONFIG_GLOBAL': os.devnull,
                 'GIT_CONFIG_NOSYSTEM': '1'})
        revision = result.stdout.split()[0] if result.returncode == 0 and result.stdout.split() else ''
    except (OSError, subprocess.SubprocessError):
        revision = ''
    if not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise RuntimeError('Cannot resolve public ModelScope master commit; check dataset visibility and Git installation')
    return revision


def _modelscope(repo, token, root, prefix, files, run_id):
    from modelscope_hub import HubApi
    api = HubApi(token=token, endpoint='https://modelscope.cn')
    revision = public_revision(repo)
    remote = {f.path: f for f in api.list_repo_files(repo, 'dataset', revision=revision)}
    missing = []
    for file in files:
        existing = remote.get(prefix + '/' + file['path'])
        expected = base64.b64decode(file['checksum_sha256']).hex()
        if existing and (existing.size != file['size'] or existing.sha256 != expected):
            raise RuntimeError('Destination contains different content')
        if not existing:
            missing.append(file['path'])
    if missing:
        api.upload_folder(repo, 'dataset', root, path_in_repo=prefix,
            allow_patterns=missing, use_cache=False, disable_tqdm=True,
            commit_message=f'Publish evaluation {run_id}')
        revision = public_revision(repo)
    return revision


def _huggingface(repo, token, root, prefix, files, run_id):
    from huggingface_hub import HfApi
    api = HfApi(token=token, endpoint='https://huggingface.co')
    info = api.dataset_info(repo_id=repo, revision='main', files_metadata=True, token=False)
    if info.private or info.gated:
        raise RuntimeError('Dataset must be public and ungated')
    revision = info.sha
    remote = {f.rfilename: f for f in info.siblings}
    missing = []
    for file in files:
        existing = remote.get(prefix + '/' + file['path'])
        if existing:
            if existing.lfs:
                expected = base64.b64decode(file['checksum_sha256']).hex()
                same = existing.lfs.sha256 == expected
            else:
                # Small files have a Git blob SHA-1, not an LFS SHA-256.
                digest = hashlib.sha1(f"blob {file['size']}\0".encode())
                with (root / file['path']).open('rb') as stream:
                    while chunk := stream.read(65536):
                        digest.update(chunk)
                same = existing.blob_id == digest.hexdigest()
            if existing.size != file['size'] or not same:
                raise RuntimeError('Destination contains different content')
        else:
            missing.append(file['path'])
    if missing:
        commit = api.upload_folder(repo_id=repo, repo_type='dataset', folder_path=str(root),
            path_in_repo=prefix, allow_patterns=missing, revision='main', parent_commit=revision,
            commit_message=f'Publish evaluation {run_id}')
        revision = commit.oid
    if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-f]{40}', revision):
        raise RuntimeError('No immutable Hugging Face commit returned')
    return revision


def upload_episode(config, run_id, root, files, request):
    provider, repo = dataset_target(config.dataset_url)
    # Checks ownership, recording flag, run state and feature support before upload.
    status = request('GET', f'/runs/{run_id}/dataset')
    prefix = status['path']
    if status['registered']:
        source = status.get('source')
        if (not source or source['provider'] != provider or source['repo'] != repo
                or status['files'] != files):
            raise RuntimeError('This run is already registered with different storage or content')
        return {k: source[k] for k in ('provider', 'repo', 'revision')}
    receipt_path = root.parent / 'dataset-upload.json'
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if (receipt['files'] != files or receipt['source']['repo'] != repo
                or receipt['source']['provider'] != provider or receipt['path'] != prefix):
            raise RuntimeError('Dataset receipt is bound to different content or repository')
        return receipt['source']
    token_env = 'MODELSCOPE_API_TOKEN' if provider == 'modelscope' else 'HF_TOKEN'
    try:
        token = config.dataset_token or (
            Path(config.dataset_token_file).expanduser().read_text().strip()
            if config.dataset_token_file else os.environ.get('DATASET_TOKEN', '').strip()
            or os.environ.get(token_env, '').strip())
    except OSError:
        raise RuntimeError('Cannot read dataset_token_file') from None
    if not token:
        raise RuntimeError(f'Set dataset_token, dataset_token_file, DATASET_TOKEN or {token_env} on Client')
    print(f'Uploading evaluation dataset to {provider}: {repo}/{prefix}', flush=True)
    previous_logging = logging.root.manager.disable
    try:
        # SDK errors may contain auth headers or upload URLs; do not emit them.
        logging.disable(logging.CRITICAL)
        upload = _modelscope if provider == 'modelscope' else _huggingface
        revision = upload(repo, token, root, prefix, files, run_id)
        source = {'provider': provider, 'repo': repo, 'revision': revision}
    except ImportError:
        raise RuntimeError(f'Install hub support: pip install "colosseum-client[{provider}]"') from None
    except Exception:
        raise RuntimeError(f'{provider} upload failed; check public dataset access, destination contents and token, then retry') from None
    finally:
        logging.disable(previous_logging)
    from .evaluation import save
    save(receipt_path, {'files': files, 'path': prefix, 'source': source})
    return source
