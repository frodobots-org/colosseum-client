import base64
import hashlib
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from colosseum_client.evaluation import EvalAPI


@pytest.mark.parametrize("failure", [False, True])
def test_dataset_upload_preserves_files_and_retries_without_leaking_credentials(tmp_path, monkeypatch, failure):
    path = tmp_path / "data/chunk-000/file-000.parquet"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"example")
    api = EvalAPI.__new__(EvalAPI)
    calls = []
    def request(method, route, body=None):
        calls.append(route)
        if route.endswith("/dataset"):
            assert body["files"] == [{"path": "data/chunk-000/file-000.parquet", "size": 7,
                "checksum_sha256": base64.b64encode(hashlib.sha256(b"example").digest()).decode()}]
            return {"complete": False}
        if "/upload-url" in route:
            assert parse_qs(urlsplit(route).query)["path"] == ["data/chunk-000/file-000.parquet"]
            return {"storage": "s3", "url": "https://s3.example/file?secret=private",
                    "headers": {"Content-Type": "application/vnd.apache.parquet", "If-None-Match": "*"}}
        return {"complete": True}
    api.request = request
    def send(request):
        assert "authorization" not in request.headers
        assert request.read() == b"example"
        return httpx.Response(403 if failure else 412)
    constructor = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: constructor(transport=httpx.MockTransport(send), **kwargs))
    if failure:
        with pytest.raises(RuntimeError) as error:
            api.upload_dataset("run", tmp_path)
        assert "private" not in str(error.value)
        assert not calls[-1].endswith("/complete")
    else:
        api.upload_dataset("run", tmp_path)
        assert calls[-1].endswith("/complete")
