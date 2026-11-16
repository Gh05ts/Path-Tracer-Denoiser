#!/usr/bin/env python3
"""Write a portable Markdown pilot report from completed joint training outputs."""
import argparse
import json
import math
from pathlib import Path
import statistics


def make_report(run):
    run = Path(run)
    summary = json.loads((run/"summary.json").read_text())
    history = [json.loads(line) for line in (run/"history.jsonl").read_text().splitlines()]
    config, performance = summary["config"], summary["performance"]
    rows = performance["epochs"]
    gib = lambda n: "unavailable" if n is None else f"{n/2**30:.3f} GiB"
    lines = ["# H100 joint sampler/denoiser pilot", "",
        "Diagnostic pilot using completed v4 training/validation clips; backfills and test pixels excluded.", "",
        f"- Device: {performance['gpu_name'] or summary['device']}",
        f"- PyTorch: {performance['torch_version']}; CUDA: {performance['cuda_version']}; cuDNN: {performance['cudnn_version']}",
        f"- Networks: sampler {config['sampler_channels']} channels; denoiser {config['denoiser_channels']} channels",
        f"- Radiance gradient floor: {config.get('radiance_gradient_floor',0)} (backward-only stabilization; forward values unchanged)",
        f"- Training: batch {config['batch_size']}; crop {config['patch_size']}; {config['window_length']} frames/window; {config['patches_per_window']} crops/window",
        f"- Validation: batch {config.get('validation_batch_size',1)}; crop {config['validation_patch_size']}; {config['validation_crops_per_window']} crops/window",
        f"- Validation schedule: initial; first {config.get('validation_warmup_epochs',0)} epochs; every {config.get('validation_every_epochs',1)} epochs; final",
        f"- Resume checkpoint: every {config.get('checkpoint_every_epochs',1)} epochs and final; best model saved on validation improvement",
        f"- Completed epochs: {summary['completed_epochs']}; optimizer steps: {summary['global_step']}",
        f"- Total training invocation: {performance['total_seconds']/60:.2f} minutes",
        f"- Setup/inventory: {performance['setup_seconds']:.2f} seconds; initial validation: {performance['initial_validation_seconds']:.2f} seconds",
        f"- Peak process RAM (process lifetime RSS): {gib(performance['process_lifetime_peak_rss_bytes'])}",
        f"- Peak PyTorch allocated GPU memory: {gib(performance['cuda_peak_allocated_bytes'])}",
        f"- Peak PyTorch reserved GPU memory: {gib(performance['cuda_peak_reserved_bytes'])}",
        f"- Test pixels loaded: {summary['test_pixels_loaded']}", "",
        "GPU allocator peaks cover this run, including model setup; they exclude memory used outside PyTorch. RAM is process-lifetime peak RSS. These are crop-training measurements, not full-resolution inference requirements.", "",
        "## Selected clips", ""]
    for partition in ("train", "validation"):
        lines.append(f"- {partition}: {', '.join(config[partition+'_groups'] or ['entire registered partition'])}")
    root = Path(config["data_root"])
    arrays = list((root/"decoded").glob("*/frame_*/*.npy"))
    if arrays:
        lines += [f"- Local decoded payload: {sum(p.stat().st_size for p in arrays)/1e9:.3f} GB ({len(arrays)} arrays)"]
    lines += ["", "## Epoch measurements", "",
        "| Epoch | Steps | Training seconds | Data wait seconds | Validation seconds | Checkpoint seconds | Total seconds | Train loss | Validation loss | Sampler grad norm | Denoiser grad norm |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for record in history:
        t, train, val = record["timing"], record["train"], record["validation"]
        val_loss = 'not evaluated' if val is None else f"{val['loss']:.6g}"
        lines.append(f"| {record['epoch']} | {t['training_steps']} | {t['training_seconds']:.2f} | {t['training_data_seconds']:.2f} | {t['validation_seconds']:.2f} | {t['checkpoint_seconds']:.2f} | {t['epoch_seconds']:.2f} | {train['loss']:.6g} | {val_loss} | {train['sampler_gradient_norm']:.6g} | {train['denoiser_gradient_norm']:.6g} |")
    last = history[-1]
    lines += ["", "## Allocation checks", ""]
    for partition in ("train", "validation"):
        m = last[partition]
        lines.append(f"- {partition}: expected SPP {m['expected_spp']:.5g}; actual SPP {m['actual_spp']:.5g}; zero fraction {m['zero_fraction']:.5g}; saturated fraction {m['saturated_fraction']:.5g}; unavailable fraction {m['unavailable_fraction']:.5g}")
    gradients = all(r["train"][k] > 0 for r in history for k in ("sampler_gradient_norm", "denoiser_gradient_norm"))
    lines += [f"- Both networks have nonzero mean gradient norms in every recorded epoch: {gradients}",
        "- The trainer rejects nonfinite gradients/weights. Finishing this diagnostic does not establish convergence or an advantage over uniform sampling."]
    # Metadata declares the full inventory even when only a few clips have local arrays.
    capture = root/"capture_lock.json"
    if capture.exists():
        lock = json.loads(capture.read_text())
        membership = {s["id"]:s["partition"] for s in lock["plan"]["scenes"]}
        counts = {p:sum(max(0,len(c["frames"])-config["window_length"]+1) for c in lock["configs"].values()
                       if membership[c["scene"]] == p) for p in ("train", "validation")}
        train_steps = math.ceil(counts["train"]*config["patches_per_window"]/config["batch_size"])
        val_windows = counts["validation"]*config["validation_crops_per_window"]
        stable = rows[1:] or rows
        train_per_step = statistics.mean(r["training_seconds"]/r["training_steps"] for r in stable)
        evaluated = [r for r in stable if r['validation_windows']] or [r for r in rows if r['validation_windows']]
        val_per_window = sum(r['validation_seconds'] for r in evaluated)/sum(r['validation_windows'] for r in evaluated)
        training_cost = train_steps*train_per_step
        validation_cost = val_windows*val_per_window
        from wavefront_denoiser.joint_training import validation_due,checkpoint_due
        writes = sum(int(r.get('resume_checkpoint_saved',False))+int(r.get('best_checkpoint_saved',False)) for r in stable)
        checkpoint_per_write = sum(r['checkpoint_seconds'] for r in stable)/writes if writes else 0.
        best_rate = sum(r.get('best_checkpoint_saved',False) for r in stable)/len(evaluated)
        def projected_seconds(epochs):
            schedule = dict(config,epochs=epochs)
            validations = sum(validation_due(e,schedule) for e in range(1,epochs+1))
            checkpoints = sum(checkpoint_due(e,schedule) for e in range(1,epochs+1))
            checkpoint_cost = (checkpoints+best_rate*validations)*checkpoint_per_write if writes else epochs*statistics.mean(r['checkpoint_seconds'] for r in stable)
            return epochs*training_cost + validations*validation_cost + checkpoint_cost
        estimated = projected_seconds(config['epochs'])/config['epochs']
        lines += ["", "## Full-dataset timing extrapolation", "",
            f"Uses epochs 2 onward when available: {train_steps} training batches and {val_windows} validation windows per full epoch.", "",
            f"- Estimated full epoch (schedule average at {config['epochs']} epochs): **{estimated/60:.2f} minutes**",
            *[f"- {epochs} epochs: {projected_seconds(epochs)/3600:.2f} hours" for epochs in (100,300,1000)], "",
            "This is a throughput extrapolation, not a convergence forecast or a guaranteed runtime. The small local dataset can remain warm in the OS page cache; the full dataset may have higher I/O costs. Add transfer, startup, evaluation and reporting time separately."]
    lines += ["", "## Artifacts", "", "Keep config.json, inventory.json, history.jsonl, summary.json, model.pt and last.pt with this report. A changed dataset inventory requires a new experiment; the pilot checkpoint cannot be resumed unchanged after adding backfills.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args(argv)
    try:
        report = make_report(args.run)
    except (OSError, KeyError, ValueError) as error:
        parser.exit(1, f"Cannot generate pilot report (training must complete first): {error}\n")
    output = args.run/"report.md"
    output.write_text(report)
    print(report)
    print(f"Report saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
