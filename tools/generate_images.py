#!/usr/bin/env python3
"""Export noisy radiance and reference RGB layers from an EXR to display PNGs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import Imath
import numpy as np
import OpenEXR
from PIL import Image

from wavefront_denoiser.preview import display_rgb


def generate_images(source: Path, output: Path | None = None, *, exposure: float = 1.0) -> list[Path]:
    source = Path(source)
    if output is None:
        directory, stem = source.parent, source.stem
    elif output.suffix.lower() == ".png" and not output.is_dir():
        directory, stem = output.parent, output.stem
    else:
        directory, stem = output, source.stem

    layers = ("radiance", "reference")
    exr = OpenEXR.InputFile(str(source))
    try:
        header = exr.header()
        window = header["dataWindow"]
        width = window.max.x - window.min.x + 1
        height = window.max.y - window.min.y + 1
        required = [f"{layer}.{color}" for layer in layers for color in "RGB"]
        missing = [name for name in required if name not in header["channels"]]
        if missing:
            raise ValueError("Missing EXR channels: " + ", ".join(missing))
        for name in required:
            channel = header["channels"][name]
            if channel.xSampling != 1 or channel.ySampling != 1:
                raise ValueError(f"Subsampled EXR channel is unsupported: {name}")
        images = []
        for layer in layers:
            planes = exr.channels(
                [f"{layer}.{color}" for color in "RGB"],
                Imath.PixelType(Imath.PixelType.FLOAT),
            )
            rgb = np.stack([
                np.frombuffer(plane, dtype=np.float32).reshape(height, width)
                for plane in planes
            ], axis=-1)
            images.append(Image.fromarray(display_rgb(rgb, exposure)))
    finally:
        exr.close()

    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"{stem}-{layer}.png" for layer in layers]
    for image, path in zip(images, paths):
        image.save(path)
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("exr", type=Path, help="EXR containing radiance.R/G/B and reference.R/G/B")
    parser.add_argument(
        "-o", "--output", type=Path,
        help="Output directory or .png base filename (default: beside the EXR). "
             "Adds -radiance.png and -reference.png; existing PNGs are replaced.",
    )
    parser.add_argument("--exposure", type=float, default=1.0,
                        help="Positive display multiplier (default: 1). Does not modify HDR data.")
    args = parser.parse_args(argv)
    try:
        for path in generate_images(args.exr, args.output, exposure=args.exposure):
            print(path.resolve())
    except (OSError, ValueError, RuntimeError) as error:
        print(f"generate_images: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
