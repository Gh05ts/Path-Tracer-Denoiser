#!/usr/bin/env python3
"""Capture and archive completed scenes to R2; optionally configure values here."""
import os
import sys

from wavefront_denoiser.dataset_generation import main

# Optional local settings. Empty credentials use the existing environment.
ACCOUNT_ID = "xxx"
ACCESS_KEY_ID = "xxx"
SECRET_ACCESS_KEY = "xxx"
BUCKET_NAME = "xxx"
PREFIX = "xxx"
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
