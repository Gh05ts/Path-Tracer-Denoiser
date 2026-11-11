"""R2 transport for the same verified, resumable scene archiving lifecycle as B2."""
from pathlib import Path, PurePosixPath
import re
from urllib.parse import urlsplit

from .b2_archive import B2SceneArchiver
from .r2_native import R2NativeClient


class R2SceneArchiver(B2SceneArchiver):
    provider = "r2"
    receipt_format = "wavefront-r2-scene-archive"

    def __init__(self, output: Path, destination: str):
        parsed = urlsplit(destination)
        prefix = parsed.path.strip("/")
        if (parsed.scheme != "r2" or not re.fullmatch(r"[a-z0-9][a-z0-9.-]*[a-z0-9]", parsed.netloc)
                or parsed.query or parsed.fragment or ".." in PurePosixPath(prefix).parts
                or "\n" in destination or "\r" in destination):
            raise ValueError("--r2-archive must be r2://BUCKET/PREFIX")
        self.output = output.resolve()
        self.destination = f"r2://{parsed.netloc}" + (f"/{prefix}" if prefix else "")
        self.directory = self.output / "r2-archives"
        self.native = R2NativeClient(parsed.netloc, prefix)

    def receipt_context(self):
        if not self.native.endpoint:
            raise ValueError("Run R2 preflight before archiving")
        # Bucket names are unique only within an R2 account. Bind receipts to
        # the account/jurisdiction too before trusting missing local EXRs.
        return {"r2_endpoint": self.native.endpoint}
