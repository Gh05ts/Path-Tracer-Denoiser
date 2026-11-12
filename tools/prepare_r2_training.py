#!/usr/bin/env python3
"""Prepare a local metadata-only snapshot for joint R2 streaming training."""
import argparse
import json

from wavefront_denoiser.r2_streaming import prepare_r2_metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="r2://BUCKET/PREFIX")
    parser.add_argument("output", help="New local metadata directory")
    parser.add_argument("--layout", choices=("inline", "v5"), default="inline")
    parser.add_argument("--train-groups", nargs="+")
    parser.add_argument("--validation-groups", nargs="+")
    args = parser.parse_args(argv)
    try:
        report = prepare_r2_metadata(args.source, args.output, layout=args.layout,
            train_groups=args.train_groups, validation_groups=args.validation_groups)
    except (OSError, RuntimeError, ValueError) as error:
        parser.exit(1, f"R2 metadata preparation failed: {error}\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
