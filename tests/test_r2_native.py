import base64
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qs, unquote, urlsplit
from xml.etree import ElementTree as ET

import pytest

from wavefront_denoiser.b2_archive import file_hashes
from wavefront_denoiser.r2_archive import R2SceneArchiver
from wavefront_denoiser.r2_native import EMPTY_SHA256, R2NativeClient, R2RequestError, signed_headers
from wavefront_denoiser.capture_history import compatible_receipt_lock, revision_parent, sha256


def revised_capture(root):
    scene = root / "bmw-scene.json"
    scene.write_text('{"name":"bmw"}')
    config = {"scene":"bmw", "scene_file":str(scene)}
    old = {"renderer_sha256":"old", "configs":{"bmw_clip":config},
           "assets":{str(scene):sha256(scene)}}
    history = root / "capture-history"
    history.mkdir()
    original = json.dumps(old).encode()
    digest = hashlib.sha256(original).hexdigest()
    parent = f"capture-history/{digest}.json"
    (root / parent).write_bytes(original)
    validation = history / "bmw-check.json"
    validation.write_text('{"checked_views":2}')
    current = dict(old, renderer_sha256="new", revision={"parent_lock":parent,
        "parent_sha256":digest, "reused_scenes":["bmw"], "reset_scenes":["bistro"],
        "validation":{"bmw":{"path":"capture-history/bmw-check.json", "sha256":sha256(validation)}}})
    (root / "capture_lock.json").write_text(json.dumps(current))
    return current, parent, original


def test_revised_lock_preserves_cloud_ancestry_before_replacing_live_lock(tmp_path, r2_api):
    lock, parent, original = revised_capture(tmp_path)
    r2_api.objects["run/capture_lock.json"] = original
    client = R2NativeClient("bucket", "run")
    client.preflight()
    names = ["capture_lock.json", parent, "capture-history/bmw-check.json"]
    inventory = {n:file_hashes(tmp_path/n) for n in names}
    client.upload_files(tmp_path, names, inventory)
    puts = [unquote(urlsplit(u).path).removeprefix('/bucket/run/') for method,u in r2_api.calls if method=="PUT"]
    assert puts.index(parent) < puts.index("capture_lock.json")
    assert puts.index("capture-history/bmw-check.json") < puts.index("capture_lock.json")
    assert r2_api.objects[f"run/{parent}"] == original
    assert r2_api.objects["run/capture_lock.json"] == (tmp_path/"capture_lock.json").read_bytes()
    assert compatible_receipt_lock(tmp_path,"bmw",lock["revision"]["parent_sha256"],sha256(tmp_path/"capture_lock.json"))
    assert not compatible_receipt_lock(tmp_path,"bistro",lock["revision"]["parent_sha256"],sha256(tmp_path/"capture_lock.json"))
    before = r2_api.uploads
    client.upload_files(tmp_path,names,inventory)
    assert r2_api.uploads == before


@pytest.mark.parametrize("corrupt", ["parent","validation","assets","configs","source"])
def test_capture_revision_rejects_changed_reuse_evidence(tmp_path, corrupt):
    lock,parent,_ = revised_capture(tmp_path)
    if corrupt == "parent":
        (tmp_path/parent).write_text('{}')
    elif corrupt == "validation":
        (tmp_path/"capture-history/bmw-check.json").write_text('{}')
    elif corrupt == "assets":
        lock["assets"][str(tmp_path/"bmw-scene.json")] = "changed"
    elif corrupt == "source":
        (tmp_path/"bmw-scene.json").write_text('{"name":"changed"}')
    else:
        lock["configs"] = {"bmw_clip":dict(lock["configs"]["bmw_clip"],samples=99)}
    with pytest.raises(ValueError):
        revision_parent(tmp_path,lock)


def test_cloud_revision_refuses_unrelated_remote_lock(tmp_path,r2_api):
    _,parent,_ = revised_capture(tmp_path)
    r2_api.objects["run/capture_lock.json"] = b'unrelated dataset'
    client = R2NativeClient("bucket","run")
    client.preflight()
    names = ["capture_lock.json",parent,"capture-history/bmw-check.json"]
    with pytest.raises(ValueError,match="different capture lock"):
        client.upload_files(tmp_path,names,{n:file_hashes(tmp_path/n) for n in names})
    assert r2_api.uploads == 0


def scope_revision(root):
    old, _, _ = revised_capture(root)
    parent_hash = sha256(root/"capture_lock.json")
    parent = f"capture-history/{parent_hash}.json"
    (root/parent).write_bytes((root/"capture_lock.json").read_bytes())
    new = dict(old, revision={"kind": "scope_reduction", "parent_lock": parent,
        "parent_sha256": parent_hash, "preserved_groups": ["bmw_clip"],
        "reused_scenes": ["bmw"], "reset_scenes": [],
        "validation": old["revision"]["validation"]})
    (root/"capture_lock.json").write_text(json.dumps(new))
    return new


def test_scope_revision_keeps_older_scene_receipts_and_cloud_lock(tmp_path, r2_api):
    new = scope_revision(tmp_path)
    intermediate = json.loads((tmp_path/new["revision"]["parent_lock"]).read_text())
    original = tmp_path/intermediate["revision"]["parent_lock"]
    r2_api.objects["run/capture_lock.json"] = original.read_bytes()
    assert compatible_receipt_lock(tmp_path, "bmw", sha256(original), sha256(tmp_path/"capture_lock.json"))
    names = ["capture_lock.json"] + [p.relative_to(tmp_path).as_posix() for p in (tmp_path/"capture-history").glob("*")]
    client = R2NativeClient("bucket", "run")
    client.preflight()
    client.upload_files(tmp_path, names, {n: file_hashes(tmp_path/n) for n in names})
    assert r2_api.objects["run/capture_lock.json"] == (tmp_path/"capture_lock.json").read_bytes()


@pytest.mark.parametrize("change", ["renderer", "preserved_group", "ancestor"])
def test_scope_revision_rejects_changed_capture_provenance(tmp_path, change):
    from wavefront_denoiser.capture_history import revision_ancestors
    new = scope_revision(tmp_path)
    if change == "renderer":
        new["renderer_sha256"] = "other"
    elif change == "preserved_group":
        new["configs"] = {"bmw_clip": dict(new["configs"]["bmw_clip"], samples=8)}
    else:
        parent = json.loads((tmp_path/new["revision"]["parent_lock"]).read_text())
        (tmp_path/parent["revision"]["parent_lock"]).write_text("changed")
    with pytest.raises(ValueError):
        revision_ancestors(tmp_path, new)


def test_sigv4_matches_published_aws_s3_signature():
    # AWS's Get Bucket (List Objects) example is an independent signing vector:
    # https://docs.aws.amazon.com/AmazonS3/latest/developerguide/sig-v4-header-based-auth.html
    headers = signed_headers("GET", "https://examplebucket.s3.amazonaws.com/?max-keys=2&prefix=J",
        "AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", EMPTY_SHA256,
        now=datetime(2013, 5, 24, tzinfo=timezone.utc), region="us-east-1")
    assert headers["authorization"].endswith(
        "Signature=34b48302e7b5fa45bde8084f4b7868a86f0a534bc59db6670ed5711ef69dc6f7")


@pytest.fixture
def r2_api(monkeypatch):
    from wavefront_denoiser import r2_native

    class Response(BytesIO):
        def __init__(self, body=b"", headers=None):
            super().__init__(body)
            self.headers = headers or {}

    class API:
        objects = {}
        calls = []
        uploads = 0
        upload_errors = []
        corrupt = False
        multipart_etags = False
        bad_upload_etag = False
        fail_receipt = False
        page_size = 1000

        def md5(self, name):
            return hashlib.md5(self.objects[name], usedforsecurity=False).hexdigest()

        def fail(self, url, status, code):
            body = f"<Error><Code>{code}</Code><Message>SECRET-r2-token-not-for-logs</Message></Error>".encode()
            raise HTTPError(url, status, "R2 failure", {}, BytesIO(body))

        def open(self, request, timeout):
            assert timeout == 300
            parsed = urlsplit(request.full_url)
            assert parsed.scheme == "https" and parsed.netloc.endswith(".r2.cloudflarestorage.com")
            headers = {key.lower(): value for key, value in request.header_items()}
            assert headers["authorization"].startswith("AWS4-HMAC-SHA256 Credential=test-r2-key/")
            assert "/auto/s3/aws4_request" in headers["authorization"]
            assert headers["host"] == parsed.netloc
            method = request.get_method()
            self.calls.append((method, request.full_url))
            if method == "GET" and parsed.path != "/bucket":
                key = unquote(parsed.path.removeprefix("/bucket/"))
                if key not in self.objects:
                    self.fail(request.full_url, 404, "NoSuchKey")
                return Response(self.objects[key], {"Content-Length": str(len(self.objects[key]))})
            if method == "PUT":
                assert hasattr(request.data, "read") and request.data.tell() == 0
                key = unquote(parsed.path.removeprefix("/bucket/"))
                if self.upload_errors:
                    request.data.read(1)
                    self.fail(request.full_url, self.upload_errors.pop(0), "SlowDown")
                if self.fail_receipt and key.endswith("r2-archives/bmw.json"):
                    self.fail(request.full_url, 503, "ServiceUnavailable")
                blocks = []
                while block := request.data.read(1024):
                    blocks.append(block)
                data = b"".join(blocks)
                assert int(headers["content-length"]) == len(data)
                assert headers["x-amz-content-sha256"] == hashlib.sha256(data).hexdigest()
                assert headers["x-amz-meta-sha256"] == hashlib.sha256(data).hexdigest()
                assert headers["content-md5"] == base64.b64encode(hashlib.md5(data, usedforsecurity=False).digest()).decode()
                assert headers["x-amz-storage-class"] == "STANDARD"
                self.objects[key] = data
                self.uploads += 1
                etag = self.md5(key)
                if self.corrupt:
                    self.objects[key] = b"corrupted after PUT response"
                return Response(headers={"ETag": f'"{"bad-etag" if self.bad_upload_etag else etag}"'})
            assert method == "GET" and parsed.path == "/bucket"
            assert headers["x-amz-content-sha256"] == EMPTY_SHA256
            query = parse_qs(parsed.query, keep_blank_values=True)
            assert query["list-type"] == ["2"]
            prefix = query["prefix"][0]
            names = sorted(key for key in self.objects if key.startswith(prefix))
            offset = int(query.get("continuation-token", ["0"])[0].split(" ")[0])
            limit = min(int(query["max-keys"][0]), self.page_size)
            root = ET.Element("ListBucketResult", xmlns="http://s3.amazonaws.com/doc/2006-03-01/")
            for key in names[offset:offset+limit]:
                item = ET.SubElement(root, "Contents")
                ET.SubElement(item, "Key").text = key
                ET.SubElement(item, "Size").text = str(len(self.objects[key]))
                ET.SubElement(item, "ETag").text = f'"{self.md5(key)}{"-2" if self.multipart_etags else ""}"'
            more = offset+limit < len(names)
            ET.SubElement(root, "IsTruncated").text = "true" if more else "false"
            if more:
                ET.SubElement(root, "NextContinuationToken").text = f"{offset+limit} +/="
            return Response(ET.tostring(root))

    api = API()
    monkeypatch.setenv("R2_ACCOUNT_ID", "a" * 32)
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "test-r2-key")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "test-r2-secret")
    monkeypatch.delenv("R2_ENDPOINT_URL", raising=False)
    monkeypatch.setattr(r2_native, "urlopen", api.open)
    monkeypatch.setattr(r2_native.time, "sleep", lambda _: None)
    return api


@pytest.fixture
def r2_scene(tmp_path, make_frame):
    root = tmp_path / "dataset"
    group = "bmw_clip"
    for i in range(2):
        make_frame(str(root / group / f"frame_{i:06d}"), index=i, sequence=group, previous=i > 0)
    (root / "capture_lock.json").write_text('{"renderer_sha256": "locked"}')
    (root / "split.json").write_text('{"data_root": "."}')
    return root, {group: {"scene": "bmw", "frames": [{}, {}]}}


def test_r2_scene_archives_all_data_before_deleting_exrs(r2_scene, r2_api, monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("External executable in R2 upload"))
    root, configs = r2_scene
    archiver = R2SceneArchiver(root, "r2://bucket/denoiser-v4")
    archiver.preflight()
    assert archiver.archive_scene("bmw", configs)
    assert not list(root.glob("*/frame_*/frame.exr"))
    assert len(list(root.glob("*/frame_*/manifest.json"))) == 2
    receipt = json.loads((root / "r2-archives/bmw.json").read_text())
    assert receipt["format"] == "wavefront-r2-scene-archive"
    for name, record in receipt["files"].items():
        data = r2_api.objects[f"denoiser-v4/{name}"]
        assert hashlib.sha256(data).hexdigest() == record["sha256"]
        assert hashlib.md5(data, usedforsecurity=False).hexdigest() == record["md5"]
    assert json.loads(r2_api.objects["denoiser-v4/r2-archives/bmw.json"]) == receipt
    calls = len(r2_api.calls)
    assert archiver.archive_scene("bmw", configs)
    assert len(r2_api.calls) == calls


@pytest.mark.parametrize("failure", ["corrupt", "multipart_etags", "bad_upload_etag", "fail_receipt"])
def test_r2_verification_failure_keeps_images_and_resumes(r2_scene, r2_api, failure):
    root, configs = r2_scene
    archiver = R2SceneArchiver(root, "r2://bucket/denoiser-v4")
    archiver.preflight()
    setattr(r2_api, failure, True)
    with pytest.raises((ValueError, R2RequestError)):
        archiver.archive_scene("bmw", configs)
    assert len(list(root.glob("*/frame_*/frame.exr"))) == 2
    setattr(r2_api, failure, False)
    if failure == "corrupt":
        del r2_api.objects["denoiser-v4/capture_lock.json"]
    assert archiver.archive_scene("bmw", configs)
    assert not list(root.glob("*/frame_*/frame.exr"))


def test_r2_retries_reopen_streams_and_encode_object_paths(tmp_path, r2_api):
    client = R2NativeClient("bucket", "dataset")
    client.preflight()
    image = tmp_path / "image.exr"
    image.write_bytes(b"frame data" * 500)
    r2_api.upload_errors = [503, 429]
    name = "clip/space + unicode-\u03bb.exr"
    client.upload_file(image, name, file_hashes(image))
    assert r2_api.objects[f"dataset/{name}"] == image.read_bytes()
    puts = [url for method, url in r2_api.calls if method == "PUT"]
    assert len(puts) == 3 and "space%20%2B%20unicode-%CE%BB.exr" in puts[0]


def test_r2_listings_paginate_and_matching_uploads_are_reused(tmp_path, r2_api):
    client = R2NativeClient("bucket", "dataset")
    client.preflight()
    r2_api.page_size = 2
    inventory = {}
    for i in range(5):
        path = tmp_path / f"frame_{i:06d}.exr"
        path.write_bytes(bytes([i]) * 128)
        r2_api.objects[f"dataset/{path.name}"] = path.read_bytes()
        inventory[path.name] = file_hashes(path)
    client.upload_files(tmp_path, list(inventory), inventory)
    client.verify_files(inventory)
    assert r2_api.uploads == 0 and len(client.list_files()) == 5
    assert any("continuation-token=2%20%2B%2F%3D" in url for _, url in r2_api.calls)


def test_r2_prefix_collision_is_rejected_before_any_upload(r2_scene, r2_api):
    root, configs = r2_scene
    archiver = R2SceneArchiver(root, "r2://bucket/denoiser-v4")
    archiver.preflight()
    r2_api.objects["denoiser-v4/capture_lock.json"] = b"another dataset"
    with pytest.raises(ValueError, match="different capture lock"):
        archiver.archive_scene("bmw", configs)
    assert r2_api.uploads == 0
    assert len(list(root.glob("*/frame_*/frame.exr"))) == 2


def test_r2_interrupted_deletion_resumes_from_verified_receipt(r2_scene, r2_api, monkeypatch):
    root, configs = r2_scene
    archiver = R2SceneArchiver(root, "r2://bucket/denoiser-v4")
    archiver.preflight()
    unlink = Path.unlink
    removed = []
    def stop_during_deletion(path, *args, **kwargs):
        if path.name == "frame.exr":
            if removed:
                raise KeyboardInterrupt
            removed.append(path)
        return unlink(path, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", stop_during_deletion)
        with pytest.raises(KeyboardInterrupt):
            archiver.archive_scene("bmw", configs)
    assert len(list(root.glob("*/frame_*/frame.exr"))) == 1
    assert archiver.archive_scene("bmw", configs)
    assert not list(root.glob("*/frame_*/frame.exr"))


def test_r2_receipt_is_bound_to_account_even_with_same_bucket_name(r2_scene, r2_api, monkeypatch):
    root, configs = r2_scene
    archiver = R2SceneArchiver(root, "r2://bucket/denoiser-v4")
    archiver.preflight()
    archiver.archive_scene("bmw", configs)
    monkeypatch.setenv("R2_ACCOUNT_ID", "b" * 32)
    other_account = R2SceneArchiver(root, "r2://bucket/denoiser-v4")
    other_account.preflight()
    with pytest.raises(ValueError, match="receipt does not match"):
        other_account.archive_scene("bmw", configs)


def test_r2_missing_credentials_fail_before_network(r2_api, monkeypatch):
    monkeypatch.delenv("R2_SECRET_ACCESS_KEY")
    with pytest.raises(ValueError, match="R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY"):
        R2NativeClient("bucket", "dataset").preflight()
    assert not r2_api.calls


def test_r2_supports_jurisdiction_specific_s3_endpoint(r2_api, monkeypatch):
    monkeypatch.setenv("R2_ENDPOINT_URL", "https://" + "a" * 32 + ".eu.r2.cloudflarestorage.com")
    monkeypatch.delenv("R2_ACCOUNT_ID")
    R2NativeClient("bucket", "dataset").preflight()
    assert ".eu.r2.cloudflarestorage.com" in r2_api.calls[0][1]


def test_r2_error_messages_do_not_expose_server_messages_or_credentials(r2_api, monkeypatch):
    from wavefront_denoiser import r2_native
    def denied(request, timeout):
        r2_api.fail(request.full_url, 403, "AccessDenied")
    monkeypatch.setattr(r2_native, "urlopen", denied)
    with pytest.raises(R2RequestError) as error:
        R2NativeClient("bucket", "dataset").preflight()
    assert "403" in str(error.value)
    assert "test-r2-secret" not in str(error.value) and "SECRET" not in str(error.value)


def test_r2_check_tool_reads_optional_launcher_settings_without_uploads(r2_api, monkeypatch, capsys):
    import importlib.util
    import sys
    from types import SimpleNamespace
    tools = Path(__file__).parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    # Do not import a developer's credential-bearing local launcher settings.
    monkeypatch.setitem(sys.modules,"generate_r2_dataset",SimpleNamespace(BUCKET_NAME="bucket",PREFIX="dataset"))
    spec = importlib.util.spec_from_file_location("check_r2_archive", tools / "check_r2_archive.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    assert tool.main(["r2://bucket/dataset"]) == 0
    assert r2_api.uploads == 0 and not r2_api.objects
    assert "bucket listing verified" in capsys.readouterr().out


@pytest.mark.parametrize("destination", ["r2://", "r2://bucket/../elsewhere", "r2://user:secret@bucket/prefix",
                                         "r2://bucket/prefix?key=secret", "b2://bucket/prefix"])
def test_r2_rejects_invalid_destinations(tmp_path, destination):
    with pytest.raises(ValueError, match="r2://BUCKET/PREFIX"):
        R2SceneArchiver(tmp_path, destination)
