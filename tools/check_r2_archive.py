#!/usr/bin/env python3
"""Check R2 credentials and listing access without touching a running capture."""
import argparse
from pathlib import Path

from generate_r2_dataset import BUCKET_NAME, PREFIX
from wavefront_denoiser.r2_archive import R2SceneArchiver


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", nargs="?", metavar="r2://BUCKET/PREFIX",
                        default=f"r2://{BUCKET_NAME}/{PREFIX}" if BUCKET_NAME else None,
                        help="Defaults to the settings in tools/generate_r2_dataset.py")
    args = parser.parse_args(argv)
    if not args.destination:
        parser.error("Pass r2://BUCKET/PREFIX or set BUCKET_NAME in tools/generate_r2_dataset.py")
    try:
        R2SceneArchiver(Path.cwd(), args.destination).preflight()
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"R2 access check failed: {error}\n")
    print("R2 authentication and bucket listing verified. Use Object Read & Write credentials for uploads. "
          "No files were uploaded or removed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
