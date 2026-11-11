"""Streaming Backblaze B2 Native API client using only Python's standard library."""
from __future__ import annotations

import base64
from http.client import HTTPException
import json
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


class B2RequestError(RuntimeError):
    def __init__(self, status: int, code: str):
        self.status = status
        self.code = code if isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_]+", code) else "request_failed"
        super().__init__(f"B2 request failed (HTTP {status}, {self.code}); local EXRs were retained")

    @property
    def temporary(self):
        return self.status == 0 or self.status in (408, 429) or self.status >= 500


class B2NativeClient:
    def __init__(self, bucket: str, prefix: str):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.bucket_id = None
        self.account_id = None
        self.api_url = None
        self.token = None
        self.upload = None

    @staticmethod
    def _request(url, headers, data=None):
        """A file-like body plus Content-Length streams through urllib/http.client."""
        if not url.startswith("https://"):
            raise ValueError("B2 API endpoints must use HTTPS")
        request = Request(url, headers=headers, data=data, method="POST" if data is not None else "GET")
        try:
            with urlopen(request, timeout=300) as response:
                return json.loads(response.read())
        except HTTPError as error:
            try:
                code = json.loads(error.read()).get("code", "request_failed")
            except (ValueError, AttributeError):
                code = "request_failed"
            finally:
                error.close()
            # Never echo response messages, credentials, tokens, or request headers.
            raise B2RequestError(error.code, code) from None
        except (URLError, OSError, HTTPException):
            raise B2RequestError(0, "connection_failed") from None
        except ValueError:
            raise B2RequestError(0, "invalid_json_response") from None

    def _json(self, url, headers, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        if data is not None:
            headers = {**headers, "Content-Type": "application/json"}
        for attempt in range(3):
            try:
                return self._request(url, headers, data)
            except B2RequestError as error:
                if not error.temporary or attempt == 2:
                    raise
                time.sleep(2 ** attempt)

    def authorize(self):
        key_id = os.environ.get("B2_APPLICATION_KEY_ID")
        key = os.environ.get("B2_APPLICATION_KEY")
        if not key_id or not key:
            raise ValueError("Set B2_APPLICATION_KEY_ID and B2_APPLICATION_KEY in your environment")
        basic = base64.b64encode(f"{key_id}:{key}".encode()).decode()
        response = self._json("https://api.backblazeb2.com/b2api/v4/b2_authorize_account",
                              {"Authorization": f"Basic {basic}"})
        storage = response["apiInfo"]["storageApi"]
        allowed = storage["allowed"]
        capabilities = set(allowed["capabilities"])
        required = {"listFiles", "writeFiles"}
        if not required <= capabilities:
            raise ValueError("B2 application key requires listFiles and writeFiles capabilities")
        if not self.object_name("").startswith(allowed.get("namePrefix") or ""):
            raise ValueError("B2 destination does not match the application key's allowed name prefix")
        self.account_id = response["accountId"]
        self.token = response["authorizationToken"]
        self.api_url = storage["apiUrl"].rstrip("/")
        self.upload = None
        buckets = allowed.get("buckets")
        if buckets is not None:
            matching = [bucket["id"] for bucket in buckets if bucket.get("name") == self.bucket]
            if not matching and not any(bucket.get("name") is None for bucket in buckets):
                raise ValueError("B2 application key does not allow the requested bucket")
        else:
            matching = []
        if matching:
            self.bucket_id = matching[0]
        else:
            # Restricted keys with bucket names in their auth response need no
            # listBuckets capability. Unrestricted keys resolve by name here.
            if "listBuckets" not in capabilities:
                raise ValueError("B2 key needs listBuckets, or a named bucket restriction, to resolve this bucket")
            result = self._json(f"{self.api_url}/b2api/v4/b2_list_buckets", {"Authorization": self.token},
                                {"accountId": self.account_id, "bucketName": self.bucket})
            matches = [bucket["bucketId"] for bucket in result["buckets"] if bucket["bucketName"] == self.bucket]
            if not matches:
                raise ValueError("Requested B2 bucket was not found")
            self.bucket_id = matches[0]

    def _api(self, operation, **payload):
        for attempt in range(2):
            try:
                return self._json(f"{self.api_url}/b2api/v4/{operation}",
                                  {"Authorization": self.token}, payload)
            except B2RequestError as error:
                if attempt or error.status != 401 or error.code not in ("bad_auth_token", "expired_auth_token"):
                    raise
                self.authorize()

    def object_name(self, relative: str) -> str:
        return f"{self.prefix}/{relative}" if self.prefix else relative

    def preflight(self):
        self.authorize()
        self._api("b2_list_file_names", bucketId=self.bucket_id, prefix=self.object_name(""), maxFileCount=1)
        self.upload = self._api("b2_get_upload_url", bucketId=self.bucket_id)

    def list_files(self) -> dict:
        files, seen = {}, set()
        start = None
        prefix = self.object_name("")
        while True:
            payload = {"bucketId": self.bucket_id, "prefix": prefix, "maxFileCount": 10000}
            if start is not None:
                payload["startFileName"] = start
            result = self._api("b2_list_file_names", **payload)
            for file in result["files"]:
                name = file["fileName"]
                if not name.startswith(prefix) or name[len(prefix):] in files:
                    raise ValueError("Unexpected or duplicate filename in B2 listing; local EXRs were retained")
                if file.get("action") == "upload":
                    files[name[len(prefix):]] = file
            start = result.get("nextFileName")
            if start is None:
                return files
            if start in seen:
                raise ValueError("B2 file listing did not advance; local EXRs were retained")
            seen.add(start)

    @staticmethod
    def matches(remote, expected):
        return (remote is not None and remote.get("contentSha1") == expected["sha1"]
                and remote.get("contentLength") == expected["size"])

    def upload_file(self, path: Path, relative: str, expected: dict):
        if expected["size"] > 5_000_000_000:
            raise ValueError("Native B2 archiving supports individual files up to 5 GB")
        name = self.object_name(relative)
        for attempt in range(3):
            try:
                if self.upload is None:
                    self.upload = self._api("b2_get_upload_url", bucketId=self.bucket_id)
                headers = {"Authorization": self.upload["authorizationToken"],
                           "X-Bz-File-Name": quote(name, safe="/"),
                           "X-Bz-Content-Sha1": expected["sha1"],
                           "Content-Length": str(expected["size"]), "Content-Type": "b2/x-auto"}
                with path.open("rb") as source:
                    result = self._request(self.upload["uploadUrl"], headers, source)
                if (not self.matches(result, expected) or result.get("fileName") != name
                        or result.get("bucketId") != self.bucket_id):
                    raise ValueError("B2 upload response checksum/size/name mismatch; local EXRs were retained")
                return
            except B2RequestError as error:
                # Always acquire a fresh upload endpoint after a transient error
                # or rejected upload token, and reopen the file at byte zero.
                self.upload = None
                if attempt == 2 or not (error.temporary or error.status == 401):
                    raise
                time.sleep(2 ** attempt)

    def upload_files(self, root: Path, available: list[str], inventory: dict):
        remote = self.list_files()
        lock = "capture_lock.json"
        if lock in remote and lock in inventory and not self.matches(remote[lock], inventory[lock]):
            raise ValueError("B2 prefix already contains a different capture lock; choose a new prefix")
        # Publish provenance first, so interrupted initial uploads still bind
        # the destination prefix before any image is placed there.
        for i, relative in enumerate(sorted(available, key=lambda name: name != lock), 1):
            if not self.matches(remote.get(relative), inventory[relative]):
                self.upload_file(root / relative, relative, inventory[relative])
                if relative.endswith(".exr") or i == len(available):
                    print(f"B2 upload {i}/{len(available)}: {relative}", flush=True)

    def verify_files(self, inventory: dict):
        remote = self.list_files()
        if any(not self.matches(remote.get(name), record) for name, record in inventory.items()):
            raise ValueError("B2 archive checksum mismatch or missing remote files; local EXRs were retained")
