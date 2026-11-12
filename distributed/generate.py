#!/usr/bin/env python3
"""Entry point for distributed whole-clip v4 capture."""
from pathlib import Path
import sys

if sys.version_info < (3, 10):
    raise SystemExit(
        "Distributed generation requires Python 3.10+; this interpreter is {}.{}.{}. "
        "Activate a compatible environment before running python3 "
        "(on the primary machine: source /tmp/ntasad-venv/bin/activate).".format(*sys.version_info[:3])
    )

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wavefront_denoiser.distributed.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
