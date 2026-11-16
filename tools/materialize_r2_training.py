#!/usr/bin/env python3
"""Download selected decoded arrays from a prepared R2 snapshot for local training."""
import argparse
import json

from wavefront_denoiser.r2_streaming import materialize_decoded_snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", help="Metadata snapshot created by prepare_r2_training.py")
    args = parser.parse_args(argv)
    try:
        result = materialize_decoded_snapshot(args.snapshot)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"Local data download failed: {error}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
