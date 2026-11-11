#!/usr/bin/env python3
"""Capture the dataset; rerun the same command to verify/skip completed frames."""
import os
import sys

from wavefront_denoiser.dataset_generation import main

BUCKET_NAME = "Denoiser"
KEY_ID = "00574fd95d23aff0000000001"
APPLICATION_KEY = "K005pf1WjcLeYZbEFpliTxaIbNkpfOQ"

os.environ["B2_APPLICATION_KEY_ID"] = KEY_ID
os.environ["B2_APPLICATION_KEY"] = APPLICATION_KEY

if __name__ == "__main__":
    raise SystemExit(main([
        "--b2-archive", f"b2://{BUCKET_NAME}/denoiser-v4",
        *sys.argv[1:],
    ]))