"""R2 conditional writes for durable claims and immutable clip payloads."""

import hashlib
import json
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
import time

from ..b2_archive import file_hashes
from ..r2_native import R2RequestError, signed_headers


def encoded(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


class R2Store:
    def __init__(self, client):
        self.client = client

    def get(self, key):
        try:
            body, headers = self.client._request("GET", key)
        except R2RequestError as error:
            if error.status == 404:
                return None
            raise
        etag = headers.get("etag", "").strip('"')
        if etag != hashlib.md5(body, usedforsecurity=False).hexdigest():
            raise ValueError("R2 coordinator JSON checksum mismatch")
        value = json.loads(body)
        if not isinstance(value, dict):
            raise ValueError("R2 coordinator record must be an object")
        return value, etag

    def put(self, key, value, *, etag=None):
        """Create if absent, or replace exactly the observed ETag.

        Reconcile a retried successful write whose initial response was lost.
        https://developers.cloudflare.com/r2/api/s3/conditional-operations/
        """
        body = encoded(value)
        expected_etag = hashlib.md5(body, usedforsecurity=False).hexdigest()
        with tempfile.TemporaryDirectory(prefix="clip-coordinator-") as name:
            path = Path(name) / "record.json"
            path.write_bytes(body)
            headers = {"content-length": str(len(body)), "content-type": "application/json",
                       "if-match" if etag else "if-none-match": f'"{etag}"' if etag else "*"}
            try:
                _, response = self.client._request("PUT", key, path=path,
                    payload_hash=hashlib.sha256(body).hexdigest(), headers=headers)
                if response.get("etag", "").strip('"') != expected_etag:
                    raise ValueError("R2 coordinator write checksum mismatch")
                return True
            except R2RequestError as error:
                if error.status != 412:
                    raise
                current = self.get(key)
                return current is not None and current[0] == value

    def head(self, key):
        try:
            _, headers = self.client._request("HEAD", key)
        except R2RequestError as error:
            if error.status == 404:
                return None
            raise
        return {"etag": headers.get("etag", "").strip('"'), "size": int(headers["content-length"])}

    def upload(self, path, key, expected):
        """Never overwrite a conflicting object, including after a worker crash."""
        remote = self.head(key)
        if remote is None:
            try:
                self.client.upload_file(Path(path), key, expected, create_only=True)
            except R2RequestError as error:
                if error.status != 412:
                    raise
            remote = self.head(key)
        if not self.client.matches(remote, expected):
            raise ValueError(f"R2 object conflicts with this clip; retained local files: {key}")

    def verify(self, inventory):
        for key, expected in inventory.items():
            if not self.client.matches(self.head(key), expected):
                raise ValueError(f"R2 clip verification failed; retained local files: {key}")

    def download(self, key, target, expected):
        """Stream a verified bundle; avoid loading the response into RAM."""
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = target.with_name(target.name + ".partial")
        for attempt in range(3):
            url = self.client._url(key, None)
            headers = signed_headers("GET", url, self.client.access_key, self.client.secret_key,
                                     hashlib.sha256(b"").hexdigest())
            try:
                with urlopen(Request(url, headers=headers), timeout=300) as response, staging.open("wb") as stream:
                    while block := response.read(1024 * 1024):
                        stream.write(block)
                if file_hashes(staging) != expected:
                    raise ValueError("Downloaded metadata bundle checksum mismatch")
                staging.replace(target)
                return
            except (URLError, HTTPError, OSError):
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
