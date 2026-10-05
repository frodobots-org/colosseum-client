"""BYOK credentials stay on the Client -> Policy Server connection."""
import ipaddress
import os
from collections.abc import Mapping
from urllib.parse import urlsplit

API_URLS = {
    'openai': 'https://api.openai.com/v1',
    'xai': 'https://api.x.ai/v1',
    'anthropic': 'https://api.anthropic.com/v1',
}


def resolve_keys(value):
    if not isinstance(value, Mapping) or set(value) - API_URLS.keys():
        raise ValueError('llm_api_keys must map openai, xai, or anthropic to a key or env:VARIABLE')
    result = {}
    for provider, key in value.items():
        if not isinstance(key, str):
            raise ValueError('llm_api_keys values must be strings')
        if key.startswith('env:'):
            key = os.environ.get(key[4:], '')
        if key.strip():
            result[provider] = key.strip()
    return result


def require_secure_policy_url(url):
    parsed = urlsplit(url)
    try:
        loopback = ipaddress.ip_address(parsed.hostname or '').is_loopback
    except ValueError:
        loopback = parsed.hostname == 'localhost'
    if parsed.scheme != 'wss' and not (parsed.scheme == 'ws' and loopback):
        raise ValueError('LLM keys require wss:// or a loopback ws:// SSH tunnel to Policy Server')


def local_preparation(preparation, keys, policy_url):
    """Return a copy carrying only this model's key; never alter Router data."""
    model = preparation['model']
    if model.get('model_type', 'vla') != 'llm':
        return preparation
    require_secure_policy_url(policy_url)
    provider = next((name for name, url in API_URLS.items() if url == model['url']), None)
    key = keys.get(provider)
    if not key:
        raise ValueError('Assigned LLM has no matching Client API key')
    return {**preparation, 'api_key': key}
