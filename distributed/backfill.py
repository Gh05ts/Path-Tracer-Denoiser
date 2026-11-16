#!/usr/bin/env python3
"""Entry point for deferred layer and decoded-cache backfills."""
from pathlib import Path
import sys

if sys.version_info < (3, 10):
    raise SystemExit("Backfills require Python 3.10+; activate your dataset environment first.")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wavefront_denoiser.distributed.backfill_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
