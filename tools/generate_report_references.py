#!/usr/bin/env python3
"""Separate 4096-SPP references with the same cameras and 4-SPP input samples."""
from wavefront_denoiser.dataset_generation import main

if __name__ == "__main__":
    raise SystemExit(main(reports=True))
