"""Dataset validation, unified NTASAD training, evaluation and HDR inference."""

import argparse
import json
import subprocess
import sys

from .frames import FrameFormatError, load_frame
from .sequences import discover_sequences


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HDR datasets and the spatial/temporal NTASAD denoiser")
    subcommands = parser.add_subparsers(dest="command", required=True)
    lighting = subcommands.add_parser("lighting-study", help="Capture the historical matched-budget lighting study")
    lighting.add_argument("config")
    lighting.add_argument("output")
    lighting.add_argument("--capture-executable")
    lighting.add_argument("--renderer-root")
    validate = subcommands.add_parser("validate", help="Check metadata, image checksums, channels, and frame history")
    validate.add_argument("root")
    train = subcommands.add_parser("train", help="Train the unified NTASAD denoiser on sequence windows")
    train.add_argument("config")
    train.add_argument("output")
    train.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    train.add_argument("--resume")
    joint = subcommands.add_parser("train-adaptive", help="Jointly train the NTASAD sampler and denoiser on v5 layers")
    joint.add_argument("config")
    joint.add_argument("output")
    joint.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    joint.add_argument("--resume")
    denoise = subcommands.add_parser("denoise", help="Denoise a still using the unified model's spatial branch")
    denoise.add_argument("checkpoint")
    denoise.add_argument("input")
    denoise.add_argument("output")
    denoise.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    sequence = subcommands.add_parser("denoise-sequence", help="Stream clips with spatial/temporal filtering and history resets")
    sequence.add_argument("checkpoint")
    sequence.add_argument("root")
    sequence.add_argument("output")
    sequence.add_argument("--sequence-id", action="append")
    sequence.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    evaluate = subcommands.add_parser("evaluate", help="Compare a checkpoint with noisy inputs on selected groups")
    evaluate.add_argument("checkpoint")
    evaluate.add_argument("root", nargs="?")
    selection = evaluate.add_mutually_exclusive_group(required=True)
    selection.add_argument("--sequence-id", action="append")
    selection.add_argument("--split-manifest")
    evaluate.add_argument("--partition", choices=("validation", "test"), default="validation")
    evaluate.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    audit = subcommands.add_parser("split-audit", help="Check whole-scene train/validation/test separation without loading test pixels")
    audit.add_argument("manifest")
    audit.add_argument("--allow-missing-test", action="store_true")
    converge = subcommands.add_parser("reference-convergence", help="Capture independent reference estimates and measure their noise")
    converge.add_argument("config")
    converge.add_argument("output")
    converge.add_argument("--capture-executable", required=True)
    converge.add_argument("--renderer-root")
    converge.add_argument("--plot", action="store_true")
    capture_split = subcommands.add_parser("capture-split", help="Capture the preregistered split using a qualified reference budget")
    capture_split.add_argument("manifest")
    capture_split.add_argument("convergence_report")
    capture_split.add_argument("--capture-executable", required=True)
    capture_split.add_argument("--renderer-root")
    args = parser.parse_args(argv)
    try:
        if args.command == "lighting-study":
            from .lighting_experiment import prepare_experiment

            if not args.capture_executable:
                raise ValueError("Capture phase needs --capture-executable")
            report = prepare_experiment(args.config, args.output, args.capture_executable, args.renderer_root)
            print(json.dumps({key: value for key, value in report.items() if key != "inventory"}, indent=2, allow_nan=False))
            return 0
        if args.command == "train-adaptive":
            from .joint_training import train_joint

            report = train_joint(args.config, args.output, args.device, resume=args.resume)
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0
        if args.command == "train":
            from .training import train_denoiser

            report = train_denoiser(args.config, args.output, args.device, resume=args.resume)
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0
        if args.command == "denoise":
            from .runtime import denoise_image

            report = denoise_image(args.checkpoint, args.input, args.output, args.device)
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0
        if args.command == "denoise-sequence":
            from .runtime import denoise_sequence

            report = denoise_sequence(args.checkpoint, args.root, args.output, args.sequence_id, args.device)
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0
        if args.command == "evaluate":
            from .runtime import evaluate_partition, evaluate_sequences

            if args.split_manifest:
                if args.root is not None:
                    raise ValueError("Dataset root comes from --split-manifest")
                report = evaluate_partition(args.checkpoint, args.split_manifest, args.partition, args.device)
            else:
                if args.root is None or args.partition != "validation":
                    raise ValueError("Group evaluation needs a root; strict test evaluation needs --split-manifest")
                report = evaluate_sequences(args.checkpoint, args.root, args.sequence_id, args.device)
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0
        if args.command == "split-audit":
            from .dataset_splits import audit_dataset_split

            report = audit_dataset_split(args.manifest, allow_missing_test=args.allow_missing_test)
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0
        if args.command == "reference-convergence":
            from .reference_convergence import plot_reference_convergence, run_reference_convergence

            report = run_reference_convergence(args.config, args.output, args.capture_executable, args.renderer_root)
            if args.plot:
                from pathlib import Path
                plot_reference_convergence(report, Path(args.output) / "convergence.png")
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0
        if args.command == "capture-split":
            from .dataset_splits import capture_independent_dataset

            report = capture_independent_dataset(args.manifest, args.convergence_report, args.capture_executable, args.renderer_root)
            print(json.dumps(report, indent=2, allow_nan=False))
            return 0
        sequences = discover_sequences(args.root)
        summaries = []
        for name, paths in sequences.items():
            peak = 0.0
            for path in paths:
                frame = load_frame(path)
                peak = max(peak, float(frame.buffers["radiance"].max()))
            summaries.append({"sequence_id": name, "frames": len(paths),
                              "width": frame.metadata["width"], "height": frame.metadata["height"],
                              "peak_input_radiance": peak})
        print(json.dumps({"validated_frames": sum(len(paths) for paths in sequences.values()),
                          "sequences": summaries}, indent=2))
        return 0
    except ImportError as error:
        print(f"wavefront-denoiser: install training dependencies with pip install -e '.[train]': {error}", file=sys.stderr)
        return 1
    except (FrameFormatError, RuntimeError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"wavefront-denoiser: {error}", file=sys.stderr)
        return 1
