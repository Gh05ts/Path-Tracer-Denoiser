import base64
import hashlib
from io import BytesIO
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import unquote

import pytest

from wavefront_denoiser.b2_archive import B2SceneArchiver, file_hashes
from wavefront_denoiser.b2_native import B2NativeClient, B2RequestError


@pytest.fixture
def b2_api(monkeypatch):
    """Simulate the HTTPS API, including consuming streaming request bodies."""
    from wavefront_denoiser import b2_native

    class Response(BytesIO):
        def __init__(self, value):
            super().__init__(json.dumps(value).encode())

    class API:
        objects = {}
        calls = []
        allowed_buckets = [{"id": "bucket-id", "name": "bucket"}]
        capabilities = ["listFiles", "writeFiles", "listBuckets"]
        name_prefix = None
        auth_count = 0
        upload_url_count = 0
        uploads = 0
        page_size = 10000
        expire_once = False
        upload_errors = []
        corrupt = False
        missing_hash = False

        def fail(self, url, status, code):
            # Response messages must never be reflected in errors or logs.
            body = {"code": code, "message": "SECRET-key-and-token-must-not-be-printed"}
            raise HTTPError(url, status, "B2 failure", {}, BytesIO(json.dumps(body).encode()))

        def metadata(self, name):
            data = self.objects[name]
            return {"action": "upload", "bucketId": "bucket-id", "fileName": name, "fileId": f"id-{name}",
                    "contentSha1": hashlib.sha1(data).hexdigest(), "contentLength": len(data)}

        def open(self, request, timeout):
            assert timeout == 300 and request.full_url.startswith("https://")
            operation = request.full_url.rsplit("/", 1)[-1]
            headers = {key.lower(): value for key, value in request.header_items()}
            self.calls.append(operation)
            if operation == "b2_authorize_account":
                assert headers["authorization"] == "Basic " + base64.b64encode(b"test-id:test-key").decode()
                self.auth_count += 1
                return Response({"accountId": "account-id", "authorizationToken": f"token-{self.auth_count}",
                    "apiInfo": {"storageApi": {"apiUrl": "https://api001.backblazeb2.com", "allowed": {
                        "buckets": self.allowed_buckets, "capabilities": self.capabilities, "namePrefix": self.name_prefix}}}})
            if operation == "upload":
                assert hasattr(request.data, "read"), "Uploads must stream rather than buffer the EXR"
                assert request.data.tell() == 0, "Retries must reopen the input at byte zero"
                assert headers["authorization"] == f"upload-token-{self.upload_url_count}"
                if self.upload_errors:
                    request.data.read(1)
                    status = self.upload_errors.pop(0)
                    self.fail(request.full_url, status, "expired_auth_token" if status == 401 else "service_unavailable")
                blocks = []
                while block := request.data.read(1024):
                    blocks.append(block)
                data = b"".join(blocks)
                assert int(headers["content-length"]) == len(data)
                assert headers["x-bz-content-sha1"] == hashlib.sha1(data).hexdigest()
                name = unquote(headers["x-bz-file-name"])
                self.objects[name] = data
                self.uploads += 1
                response = self.metadata(name)
                if self.corrupt:
                    self.objects[name] = b"corrupted after upload response"
                return Response(response)
            assert headers["authorization"] == f"token-{self.auth_count}"
            payload = json.loads(request.data)
            if self.expire_once:
                self.expire_once = False
                self.fail(request.full_url, 401, "expired_auth_token")
            if operation == "b2_list_buckets":
                assert payload == {"accountId": "account-id", "bucketName": "bucket"}
                return Response({"buckets": [{"bucketName": "bucket", "bucketId": "bucket-id"}]})
            assert payload["bucketId"] == "bucket-id"
            if operation == "b2_get_upload_url":
                self.upload_url_count += 1
                return Response({"uploadUrl": "https://upload.backblazeb2.com/upload",
                                 "authorizationToken": f"upload-token-{self.upload_url_count}"})
            if operation == "b2_list_file_names":
                names = sorted(name for name in self.objects if name.startswith(payload["prefix"])
                               and name >= payload.get("startFileName", ""))
                limit = min(payload["maxFileCount"], self.page_size)
                files = [self.metadata(name) for name in names[:limit]]
                if self.missing_hash:
                    for file in files:
                        file["contentSha1"] = "none"
                return Response({"files": files, "nextFileName": names[limit] if len(names) > limit else None})
            raise AssertionError(operation)

    api = API()
    monkeypatch.setenv("B2_APPLICATION_KEY_ID", "test-id")
    monkeypatch.setenv("B2_APPLICATION_KEY", "test-key")
    monkeypatch.setattr(b2_native, "urlopen", api.open)
    monkeypatch.setattr(b2_native.time, "sleep", lambda _: None)
    return api


@pytest.fixture
def native_scene(tmp_path, make_frame):
    root = tmp_path / "dataset"
    group = "bmw_clip"
    for i in range(2):
        make_frame(str(root / group / f"frame_{i:06d}"), index=i, sequence=group, previous=i > 0)
    (root / "capture_lock.json").write_text('{"renderer_sha256": "locked"}')
    (root / "split.json").write_text('{"data_root": "."}')
    return root, {group: {"scene": "bmw", "frames": [{}, {}]}}


def test_python_only_scene_archive_streams_verifies_and_removes_images(native_scene, b2_api, monkeypatch):
    root, configs = native_scene
    # Native archiving must never launch an external executable.
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("External command in native B2 archiving"))
    archiver = B2SceneArchiver(root, "b2://bucket/denoiser-v4")
    archiver.preflight()
    assert archiver.archive_scene("bmw", configs)
    assert not list(root.glob("*/frame_*/frame.exr"))
    receipt = json.loads((root / "b2-archives/bmw.json").read_text())
    assert receipt["destination"] == "b2://bucket/denoiser-v4"
    for name, record in receipt["files"].items():
        assert hashlib.sha256(b2_api.objects[f"denoiser-v4/{name}"]).hexdigest() == record["sha256"]
    assert json.loads(b2_api.objects["denoiser-v4/b2-archives/bmw.json"]) == receipt
    calls = len(b2_api.calls)
    assert archiver.archive_scene("bmw", configs)
    assert len(b2_api.calls) == calls


@pytest.mark.parametrize("missing_hash", [False, True])
def test_native_failed_verification_preserves_images_and_resumes(native_scene, b2_api, missing_hash):
    root, configs = native_scene
    archiver = B2SceneArchiver(root, "b2://bucket/denoiser-v4")
    archiver.preflight()
    b2_api.corrupt = not missing_hash
    b2_api.missing_hash = missing_hash
    with pytest.raises(ValueError, match="checksum mismatch"):
        archiver.archive_scene("bmw", configs)
    assert len(list(root.glob("*/frame_*/frame.exr"))) == 2
    b2_api.corrupt = b2_api.missing_hash = False
    # Corruption includes the simulated capture lock. Remove that invalid
    # remote object explicitly so the collision guard does not overwrite it.
    if not missing_hash:
        del b2_api.objects["denoiser-v4/capture_lock.json"]
    assert archiver.archive_scene("bmw", configs)
    assert not list(root.glob("*/frame_*/frame.exr"))


def test_native_upload_retries_reopen_stream_and_refresh_endpoint(tmp_path, b2_api):
    client = B2NativeClient("bucket", "dataset")
    client.preflight()
    path = tmp_path / "data with spaces.exr"
    path.write_bytes(b"image content" * 500)
    b2_api.upload_errors = [503, 401]
    client.upload_file(path, path.name, file_hashes(path))
    assert b2_api.upload_url_count == 3
    assert b2_api.objects[f"dataset/{path.name}"] == path.read_bytes()


def test_expired_account_token_is_renewed_for_multi_day_capture(b2_api):
    client = B2NativeClient("bucket", "dataset")
    client.preflight()
    b2_api.expire_once = True
    assert client.list_files() == {}
    assert b2_api.auth_count == 2


def test_native_listings_paginate_and_matching_uploads_are_reused(tmp_path, b2_api):
    client = B2NativeClient("bucket", "dataset")
    client.preflight()
    b2_api.page_size = 2
    inventory = {}
    for i in range(5):
        name = f"frame_{i:06d}.exr"
        path = tmp_path / name
        path.write_bytes(bytes([i]) * 128)
        b2_api.objects[f"dataset/{name}"] = path.read_bytes()
        inventory[name] = file_hashes(path)
    client.upload_files(tmp_path, list(inventory), inventory)
    client.verify_files(inventory)
    assert b2_api.uploads == 0
    assert len(client.list_files()) == 5


def test_native_prefix_collision_does_not_overwrite_other_dataset(native_scene, b2_api):
    root, configs = native_scene
    archiver = B2SceneArchiver(root, "b2://bucket/denoiser-v4")
    archiver.preflight()
    b2_api.objects["denoiser-v4/capture_lock.json"] = b"different dataset"
    with pytest.raises(ValueError, match="different capture lock"):
        archiver.archive_scene("bmw", configs)
    assert b2_api.uploads == 0
    assert len(list(root.glob("*/frame_*/frame.exr"))) == 2


@pytest.mark.parametrize("restricted", [False, True])
def test_native_bucket_resolution_with_restricted_and_unrestricted_keys(b2_api, restricted):
    if restricted:
        b2_api.capabilities = ["listFiles", "writeFiles"]
    else:
        b2_api.allowed_buckets = None
    client = B2NativeClient("bucket", "dataset")
    client.preflight()
    assert client.bucket_id == "bucket-id"
    assert ("b2_list_buckets" in b2_api.calls) == (not restricted)


@pytest.mark.parametrize("restriction", ["bucket", "prefix", "capabilities"])
def test_native_key_restrictions_fail_before_upload(b2_api, restriction):
    if restriction == "bucket":
        b2_api.allowed_buckets = [{"id": "other-id", "name": "other-bucket"}]
    elif restriction == "prefix":
        b2_api.name_prefix = "another-prefix/"
    else:
        b2_api.capabilities = ["listFiles"]
    with pytest.raises(ValueError):
        B2NativeClient("bucket", "dataset").preflight()
    assert b2_api.uploads == 0


def test_native_missing_environment_credentials_are_actionable(b2_api, monkeypatch):
    monkeypatch.delenv("B2_APPLICATION_KEY", raising=False)
    with pytest.raises(ValueError, match="B2_APPLICATION_KEY_ID and B2_APPLICATION_KEY"):
        B2NativeClient("bucket", "dataset").preflight()
    assert not b2_api.calls


def test_native_http_errors_do_not_disclose_response_messages_or_secrets(b2_api, monkeypatch):
    from wavefront_denoiser import b2_native
    def denied(request, timeout):
        b2_api.fail(request.full_url, 403, "unauthorized")
    monkeypatch.setattr(b2_native, "urlopen", denied)
    with pytest.raises(B2RequestError) as error:
        B2NativeClient("bucket", "dataset").preflight()
    assert "403" in str(error.value)
    assert "test-key" not in str(error.value) and "SECRET" not in str(error.value)


def test_native_access_check_tool_does_not_upload_or_modify_capture(tmp_path, b2_api, capsys):
    import importlib.util
    path = Path(__file__).parents[1] / "tools/check_b2_archive.py"
    spec = importlib.util.spec_from_file_location("check_b2_archive", path)
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    assert tool.main(["b2://bucket/dataset"]) == 0
    assert b2_api.uploads == 0 and not b2_api.objects
    output = capsys.readouterr().out
    assert "access verified" in output
    assert "test-key" not in output and "token-" not in output


@pytest.mark.parametrize("destination", ["b2://", "b2://bucket/../elsewhere", "b2://user:password@bucket/prefix",
                                         "b2://bucket/prefix?key=secret", "b2://bucket/prefix#fragment"])
def test_native_invalid_destinations_are_rejected(tmp_path, destination):
    with pytest.raises(ValueError, match="b2://BUCKET/PREFIX"):
        B2SceneArchiver(tmp_path, destination)
