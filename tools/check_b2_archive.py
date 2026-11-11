#!/usr/bin/env python3
"""Check B2 credentials and bucket access without touching a running capture."""
import argparse
from pathlib import Path

from wavefront_denoiser.b2_archive import B2SceneArchiver


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", metavar="b2://BUCKET/PREFIX")
    args = parser.parse_args(argv)
    try:
        B2SceneArchiver(Path.cwd(), args.destination).preflight()
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"B2 access check failed: {error}\n")
    print("B2 authorization, bucket listing and upload access verified. No files were uploaded or removed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
