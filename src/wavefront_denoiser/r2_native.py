"""Streaming R2 S3 uploads with AWS Signature V4 and no third-party dependencies."""
from __future__ import annotations

import base64
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import hmac
from http.client import HTTPException
import os
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen
from xml.etree import ElementTree as ET


EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()


class R2RequestError(RuntimeError):
    def __init__(self, status: int, code: str):
        self.status = status
        self.code = code if re.fullmatch(r"[A-Za-z0-9_]+", code) else "request_failed"
        super().__init__(f"R2 request failed (HTTP {status}, {self.code}); local EXRs were retained")

    @property
    def temporary(self):
        return self.status == 0 or self.status in (408, 429) or self.status >= 500


def signed_headers(method: str, url: str, access_key: str, secret_key: str, payload_hash: str,
                   headers: dict | None = None, *, now: datetime | None = None,
                   region: str = "auto") -> dict:
    """Sign a path-style S3 request. URI/query must already be percent encoded."""
    parsed = urlsplit(url)
    now = now or datetime.now(timezone.utc)
    timestamp, day = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    values = {key.lower(): " ".join(str(value).split()) for key, value in (headers or {}).items()}
    values.update(host=parsed.netloc, **{"x-amz-date": timestamp, "x-amz-content-sha256": payload_hash})
    names = ";".join(sorted(values))
    canonical_headers = "".join(f"{key}:{values[key]}\n" for key in sorted(values))
    canonical = "\n".join((method, parsed.path or "/", parsed.query, canonical_headers, names, payload_hash))
    scope = f"{day}/{region}/s3/aws4_request"
    to_sign = "\n".join(("AWS4-HMAC-SHA256", timestamp, scope, hashlib.sha256(canonical.encode()).hexdigest()))
    signing_key = ("AWS4" + secret_key).encode()
    for part in (day, region, "s3", "aws4_request"):
        signing_key = hmac.new(signing_key, part.encode(), hashlib.sha256).digest()
    signature = hmac.new(signing_key, to_sign.encode(), hashlib.sha256).hexdigest()
    values["authorization"] = (f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, "
                               f"SignedHeaders={names}, Signature={signature}")
    return values


class R2NativeClient:
    def __init__(self, bucket: str, prefix: str):
        self.bucket, self.prefix = bucket, prefix.strip("/")
        self.endpoint = None
        self.access_key = None
        self.secret_key = None

    def preflight(self):
        self.access_key = os.environ.get("R2_ACCESS_KEY_ID")
        self.secret_key = os.environ.get("R2_SECRET_ACCESS_KEY")
        if not self.access_key or not self.secret_key:
            raise ValueError("Set R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY in your environment")
        endpoint = os.environ.get("R2_ENDPOINT_URL")
        if not endpoint:
            account = os.environ.get("R2_ACCOUNT_ID", "")
            if not re.fullmatch(r"[0-9a-fA-F]{32}", account):
                raise ValueError("Set R2_ACCOUNT_ID to your 32-character Cloudflare account ID (or set R2_ENDPOINT_URL)")
            endpoint = f"https://{account}.r2.cloudflarestorage.com"
        parsed = urlsplit(endpoint)
        if (parsed.scheme != "https" or not parsed.hostname or not parsed.hostname.endswith(".r2.cloudflarestorage.com")
                or parsed.username or parsed.password or parsed.port or parsed.path.strip("/")
                or parsed.query or parsed.fragment):
            raise ValueError("R2_ENDPOINT_URL must be your HTTPS R2 S3 endpoint, without a bucket/path")
        self.endpoint = endpoint.rstrip("/")
        # Listing checks credentials and read/list permission without uploading
        # a probe object. The token must also grant Object Read & Write for PUT.
        body, _ = self._request("GET", query={"list-type": "2", "prefix": self.object_name(""), "max-keys": "1"})
        if ET.fromstring(body).tag.rsplit("}", 1)[-1] != "ListBucketResult":
            raise ValueError("Unexpected R2 bucket listing response")

    def object_name(self, relative: str):
        return f"{self.prefix}/{relative}" if self.prefix else relative

    def _url(self, relative: str | None, query: dict | None):
        key = f"/{self.bucket}" + (f"/{self.object_name(relative)}" if relative is not None else "")
        uri = quote(key, safe="/-_.~")
        pairs = sorted((quote(str(key), safe="-_.~"), quote(str(value), safe="-_.~"))
                       for key, value in (query or {}).items())
        return self.endpoint + uri + ("?" + "&".join(f"{key}={value}" for key, value in pairs) if pairs else "")

    def _request(self, method: str, relative: str | None = None, *, query: dict | None = None,
                 path: Path | None = None, payload_hash: str = EMPTY_SHA256, headers: dict | None = None):
        url = self._url(relative, query)
        for attempt in range(3):
            try:
                signed = signed_headers(method, url, self.access_key, self.secret_key, payload_hash, headers)
                # urllib streams file bodies when Content-Length is supplied;
                # retries reopen the file instead of resending its unread tail.
                with (path.open("rb") if path is not None else nullcontext(None)) as source:
                    request = Request(url, method=method, headers=signed, data=source)
                    with urlopen(request, timeout=300) as response:
                        return response.read(), {key.lower(): value for key, value in response.headers.items()}
            except HTTPError as error:
                try:
                    code = ET.fromstring(error.read()).findtext("Code") or "request_failed"
                except ET.ParseError:
                    code = "request_failed"
                finally:
                    error.close()
                failure = R2RequestError(error.code, code)
            except (URLError, OSError, HTTPException):
                failure = R2RequestError(0, "connection_failed")
            if not failure.temporary or attempt == 2:
                raise failure from None
            time.sleep(2 ** attempt)

    def list_files(self):
        files, seen = {}, set()
        token = None
        prefix = self.object_name("")
        while True:
            query = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
            if token is not None:
                query["continuation-token"] = token
            body, _ = self._request("GET", query=query)
            root = ET.fromstring(body)
            for item in root.findall("{*}Contents"):
                name = item.findtext("{*}Key")
                if name is None or not name.startswith(prefix) or name[len(prefix):] in files:
                    raise ValueError("Invalid or duplicate filename in R2 listing; local EXRs were retained")
                files[name[len(prefix):]] = {"etag": (item.findtext("{*}ETag") or "").strip('"'),
                                           "size": int(item.findtext("{*}Size"))}
            if root.findtext("{*}IsTruncated") == "false":
                return files
            token = root.findtext("{*}NextContinuationToken")
            if not token or token in seen:
                raise ValueError("R2 file listing did not advance; local EXRs were retained")
            seen.add(token)

    @staticmethod
    def matches(remote, expected):
        # Only single PUT ETags count as full-file MD5 hashes. A multipart ETag
        # or arbitrary user metadata must never pass content verification.
        return (remote is not None and re.fullmatch(r"[0-9a-fA-F]{32}", remote.get("etag", "")) is not None
                and remote["etag"].lower() == expected["md5"] and remote.get("size") == expected["size"])

    def upload_file(self, path: Path, relative: str, expected: dict):
        if expected["size"] > 5_000_000_000:
            raise ValueError("Native R2 archiving supports individual files up to 5 GB")
        headers = {"content-length": str(expected["size"]), "content-type": "application/octet-stream",
                   "content-md5": base64.b64encode(bytes.fromhex(expected["md5"])).decode(),
                   "x-amz-storage-class": "STANDARD", "x-amz-meta-sha256": expected["sha256"]}
        _, response = self._request("PUT", relative, path=path, payload_hash=expected["sha256"], headers=headers)
        if response.get("etag", "").strip('"').lower() != expected["md5"]:
            raise ValueError("R2 upload ETag checksum mismatch; local EXRs were retained")

    def upload_files(self, root: Path, available: list[str], inventory: dict):
        remote = self.list_files()
        lock = "capture_lock.json"
        if lock in remote and lock in inventory and not self.matches(remote[lock], inventory[lock]):
            raise ValueError("R2 prefix already contains a different capture lock; choose a new prefix")
        for i, relative in enumerate(sorted(available, key=lambda name: name != lock), 1):
            if not self.matches(remote.get(relative), inventory[relative]):
                self.upload_file(root / relative, relative, inventory[relative])
                if relative.endswith(".exr") or i == len(available):
                    print(f"R2 upload {i}/{len(available)}: {relative}", flush=True)

    def verify_files(self, inventory: dict):
        remote = self.list_files()
        if any(not self.matches(remote.get(name), record) for name, record in inventory.items()):
            raise ValueError("R2 archive checksum mismatch or missing remote files; local EXRs were retained")
