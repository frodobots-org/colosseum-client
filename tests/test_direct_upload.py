import base64
import hashlib
import httpx
import pytest
from colosseum_client.evaluation import EvalAPI


@pytest.mark.parametrize('status', [200, 412, 403])
def test_direct_upload_does_not_forward_client_credentials(tmp_path, monkeypatch, status):
    path = tmp_path/'video.mp4'
    path.write_bytes(b'example video bytes')
    calls = []; sent = []
    api = EvalAPI.__new__(EvalAPI)
    def request(method, route, body=None):
        calls.append(route)
        if route.endswith('upload-url'):
            assert body['checksum_sha256'] == base64.b64encode(hashlib.sha256(path.read_bytes()).digest()).decode()
            return {'storage':'s3', 'url':'https://bucket.example/video?secret=abc', 'headers':{'Content-Type':'video/mp4','If-None-Match':'*'}}
        return {'complete':True}
    api.request = request
    def send(req):
        sent.append(req)
        assert 'authorization' not in req.headers
        assert req.read() == path.read_bytes()
        return httpx.Response(status)
    constructor = httpx.Client
    monkeypatch.setattr(httpx, 'Client', lambda **kwargs: constructor(transport=httpx.MockTransport(send), **kwargs))
    if status == 403:
        with pytest.raises(RuntimeError, match='S3 upload failed') as error:
            api.upload('run', 'head_image', path)
        assert 'secret' not in str(error.value)
        assert len(calls)==1
    else:
        api.upload('run', 'head_image', path)
        assert calls[-1].endswith('/complete')
    assert len(sent)==1
