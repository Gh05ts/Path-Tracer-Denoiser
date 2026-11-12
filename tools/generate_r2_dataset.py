#!/usr/bin/env python3
"""Capture and archive completed scenes to R2; optionally configure values here."""
import os
import sys

from wavefront_denoiser.dataset_generation import main

# Optional local settings. Empty credentials use the existing environment.
ACCOUNT_ID = "6c378bae4254c5caad6be25154089c8a"
ACCESS_KEY_ID = "d04ec47d593c13f9b0c08d93844759ad"
SECRET_ACCESS_KEY = "42c5f8a1c1e1b1bb1fe942a164c121dcf6a9bc90d7fde6948e841c5273ea79d5"
BUCKET_NAME = "denoiser"
PREFIX = "denoiser-v4"
# Optional for jurisdiction-specific buckets; copy the bucket's S3 endpoint.
ENDPOINT_URL = ""

for name, value in (
    ("R2_ACCOUNT_ID", ACCOUNT_ID), ("R2_ACCESS_KEY_ID", ACCESS_KEY_ID),
    ("R2_SECRET_ACCESS_KEY", SECRET_ACCESS_KEY), ("R2_ENDPOINT_URL", ENDPOINT_URL),
):
    if value:
        os.environ[name] = value


if __name__ == "__main__":
    args = sys.argv[1:]
    explicit_archive = any(arg.split("=", 1)[0] in ("--r2-archive", "--b2-archive") for arg in args)
    if not explicit_archive and "--help" not in args and "-h" not in args:
        if not BUCKET_NAME:
            raise SystemExit("Set BUCKET_NAME in tools/generate_r2_dataset.py or pass --r2-archive r2://BUCKET/PREFIX")
        args = ["--r2-archive", f"r2://{BUCKET_NAME}/{PREFIX}", *args]
    raise SystemExit(main(args))
