#!/usr/bin/env python3
"""Bounded, local-data-only NTASAD component diagnostics (no production resume)."""
import argparse
from contextlib import contextmanager
import gc
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from wavefront_denoiser.adaptive_data import read_v5_dataset
from wavefront_denoiser.checkpoint import _cpu_tree, _rng_state
from wavefront_denoiser.dataset_generation import digest
from wavefront_denoiser.joint import JointNTASAD
from wavefront_denoiser.joint_checkpoint import load_joint_checkpoint
from wavefront_denoiser.joint_data import JointPatchDataset, joint_dataset_read_lock, joint_inventory
from wavefront_denoiser.joint_loading import joint_batches
from wavefront_denoiser.joint_training import _batch, _metrics, finite_group_flags, training_config, validation_due
from wavefront_denoiser.loss import sequence_loss
from wavefront_denoiser.ntasad_radiance import decode_radiance, radiance_gradient_policy
from wavefront_denoiser.sample_layers import replay_sample_layers

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent
ARMS = {
    "baseline": {"mode": "joint", "initialization": "fresh"},
    "uniform": {"mode": "uniform", "initialization": "fresh"},
    "frozen": {"mode": "frozen_denoiser", "initialization": "best-trained"},
    "warm-joint": {"mode": "joint", "initialization": "best-trained"},
    "sampler-lr-low": {"mode": "joint", "initialization": "fresh", "sampler_lr_multiplier": .1},
    "sampler-clip": {"mode": "joint", "initialization": "fresh", "sampler_clip_norm": 1.},
}
MODEL_KEYS = ("denoiser_channels", "sampler_channels", "average_spp", "max_spp",
              "allocation_policy", "zero_policy", "reconstruction_version")


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def memory_sample():
    fields = {line.split(":")[0]: int(line.split()[1])*1024
              for line in Path("/proc/meminfo").read_text().splitlines() if len(line.split()) >= 2}
    result = subprocess.run(["nvidia-smi", "--id=0", "--query-gpu=memory.total,memory.used",
                             "--format=csv,noheader,nounits"], text=True, capture_output=True, timeout=3)
    if result.returncode:
        raise RuntimeError("NVIDIA memory monitoring failed; refusing to continue without the memory guard")
    total, used = [int(x.strip())*2**20 for x in result.stdout.strip().split(",")]
    return {"host_total_bytes": fields["MemTotal"], "host_used_bytes": fields["MemTotal"]-fields["MemAvailable"],
            "gpu_total_bytes": total, "gpu_used_bytes": used}


def tensor_summary(value):
    """Deterministic strided quantiles: at most 8192 values, no RNG changes."""
    flat = value.detach().flatten().float()
    sample = flat[::max(1, math.ceil(flat.numel()/8192))]
    quantiles = torch.quantile(sample, sample.new_tensor([0., .01, .5, .99, 1.])).cpu().tolist()
    return {"min": float(flat.min()), "max": float(flat.max()), "mean": float(flat.mean()),
            "sampled_quantiles_0_1_50_99_100": quantiles, "quantile_sample_count": sample.numel()}


class Trace:
    def __init__(self):
        self.enabled = False
        self.reset()

    def reset(self):
        self.logits, self.renders, self.features = [], [], {}

    def feature_hook(self, name):
        def hook(module, args):
            if self.enabled:
                value = args[0].detach()
                lo = value.amin((0, 2, 3)).cpu().tolist()
                hi = value.amax((0, 2, 3)).cpu().tolist()
                self.features.setdefault(name, []).append({"channel_min": lo, "channel_max": hi})
        return hook

    def backward_stats(self, reference):
        result, gradients = [], []
        for index, (image, counts) in enumerate(self.renders):
            if index == 0:
                continue
            gradient = image.grad
            if gradient is None:
                result.append({"frame": index, "render_gradient_present": False})
                gradients.append(torch.zeros_like(counts, dtype=image.dtype))
                continue
            count_gradient = (gradient * (reference[:, index]-image.detach()) /
                              counts.clamp_min(1).to(image.dtype)).sum(1, keepdim=True)
            gradients.append(count_gradient.detach())
            near = (image.detach() > 0) & (image.detach() <= 1e-6)
            black = image.detach() == 0
            bright = image.detach() >= 1
            result.append({"frame": index, "render_gradient_present": True,
                "render_gradient_norm": float(gradient.norm()),
                "count_surrogate_gradient_norm": float(count_gradient.norm()),
                "count_gradient_on_zero_norm": float(count_gradient.masked_select(counts == 0).norm()),
                "near_black_gradient_norm": float(gradient.masked_select(near).norm()),
                "exact_black_gradient_norm": float(gradient.masked_select(black).norm()),
                "bright_ge_1_gradient_norm": float(gradient.masked_select(bright).norm()),
                "near_black_fraction": float(near.float().mean()),
                "bright_ge_1_fraction": float(bright.float().mean())})
        return result, gradients


class DiagnosticJoint(JointNTASAD):
    """Keep the production unroll intact; wrap only method boundaries for tracing.

    Uniform mode supplies constant logits to the existing allocator. Diagnostic
    checkpoints have a distinct format so a generic joint loader cannot silently
    interpret this override as learned allocation.
    """
    def __init__(self, spec, mode="joint"):
        super().__init__(**spec)
        self.diagnostic_mode = mode
        self.trace = Trace()
        original_logits, original_step = self.sampler.predict_logits, self.denoiser._step

        def logits(guides, history, **kwargs):
            value = (torch.zeros_like(guides["albedo"][:, :1]) if mode == "uniform"
                     else original_logits(guides, history, **kwargs))
            if self.trace.enabled:
                if value.requires_grad:
                    value.retain_grad()
                self.trace.logits.append(value)
            return value

        def step(buffers, history, **kwargs):
            if self.trace.enabled:
                value = buffers["radiance"]
                if value.requires_grad:
                    value.retain_grad()
                self.trace.renders.append((value, buffers["sample_count"]))
            return original_step(buffers, history, **kwargs)

        self.sampler.predict_logits, self.denoiser._step = logits, step
        self.sampler.encoder0.register_forward_pre_hook(self.trace.feature_hook("sampler"))
        self.denoiser.network.encoder0.register_forward_pre_hook(self.trace.feature_hook("denoiser"))
        if mode == "uniform":
            self.sampler.requires_grad_(False)
        if mode == "frozen_denoiser":
            self.denoiser.requires_grad_(False)


def optimizer_for(model, config, arm):
    groups = []
    for name in ("sampler", "denoiser"):
        parameters = [p for p in getattr(model, name).parameters() if p.requires_grad]
        if parameters:
            groups.append({"params": parameters, "name": name,
                           "lr": config["learning_rate"] * (arm.get("sampler_lr_multiplier", 1.)
                                                           if name == "sampler" else 1.)})
    return torch.optim.Adam(groups)


def checkpoint(path, model, optimizer, config, arm, epoch, step, generator, inventory_sha):
    payload = {"format": "wavefront-collapse-diagnostic", "version": 1,
        "model_spec": model.specification(), "arm": arm, "config": config,
        "completed_epochs": epoch, "global_step": step, "inventory_sha256": inventory_sha,
        "state_dict": _cpu_tree(model.state_dict()), "optimizer": _cpu_tree(optimizer.state_dict()),
        "rng": _cpu_tree(_rng_state(generator)), "diagnostic_only": True,
        "progress_note": "Mid-epoch snapshots are replay evidence, not exact-resume checkpoints."}
    temporary = path.with_suffix(".partial")
    torch.save(payload, temporary)
    temporary.replace(path)


def dataset(config, partition):
    validation = partition == "validation"
    return JointPatchDataset(config["data_root"], partition=partition,
        sequence_ids=config[partition+"_groups"], length=config["window_length"],
        patch_size=config["validation_patch_size"] if validation else config["patch_size"],
        patches_per_window=config["validation_crops_per_window"] if validation else config["patches_per_window"],
        augment=not validation, cache_maps=config.get("loader_policy") is not None,
        cache_frames=config.get("decoded_cache_frames", 16),
        fast_crops=config.get("loader_policy") == "prefetch_fused_v1")


def batches(data, config, generator, shuffle=False):
    return joint_batches(data, batch_size=config["batch_size"] if shuffle else config.get("validation_batch_size", 1),
        generator=generator, shuffle=shuffle, prefetch=config.get("loader_policy") in ("prefetch_v1", "prefetch_fused_v1"),
        prefetch_batches=config.get("prefetch_batches", 2))


def crop_evidence(output, batch, layers, prediction, counts, reference, model):
    from PIL import Image, ImageDraw
    from wavefront_denoiser.preview import display_rgb
    output.mkdir(parents=True, exist_ok=True)
    uniform_count = torch.full_like(counts[:1, -1], model.average_spp)
    images = [("Uniform noisy", replay_sample_layers(layers[:1, -1], uniform_count)[0]),
              ("Allocated noisy", replay_sample_layers(layers[:1, -1], counts[:1, -1])[0]),
              ("Denoised", decode_radiance(prediction.radiance[:1, -1])[0]),
              ("Reference", reference[0, -1])]
    height, width = reference.shape[-2:]
    strip = Image.new("RGB", (4*width, height+24), (30, 30, 30))
    draw = ImageDraw.Draw(strip)
    for index, (label, value) in enumerate(images):
        rgb = value.detach().cpu().numpy().transpose(1, 2, 0)
        strip.paste(Image.fromarray(display_rgb(rgb, 1.)), (index*width, 24))
        draw.text((index*width+3, 4), label, fill="white")
    strip.save(output/"comparison.png")
    np.save(output/"sample-count.npy", counts[0, -1, 0].cpu().numpy())
    frame_metrics = sequence_loss(prediction.radiance[:1], reference[:1], transformed=True)
    write(output/"preview.json", {"sequence_id": batch["sequence_id"][0], "crop_size": width,
        "mode": "fixed_validation_crop_not_full_frame", "actual_final_frame_paths": int(counts[:1, -1].sum()),
        "uniform_final_frame_paths": int(uniform_count.sum()),
        "metrics": {k: float(v) for k, v in frame_metrics.items()},
        "note": "Allocation and recurrence are crop-local, not native full-image evidence."})


@torch.inference_mode()
def evaluate(model, data, config, device, deadline, *, uniform=False, previews=None):
    """Bounded evaluation, explicitly label partial evaluation and actual budgets."""
    model.eval()
    model.trace.enabled = False
    totals, windows, paths, pixels = {}, 0, 0, 0
    preview_groups, scenes = set(), set()
    if previews is not None:
        for group in config["validation_groups"] or data.descriptor["partitions"]["validation"]:
            scene = data.descriptor["groups"][group]["scene"]
            if scene not in scenes:
                scenes.add(scene); preview_groups.add(group)
    with batches(data, config, torch.Generator().manual_seed(0)) as loader:
        for batch in loader:
            if time.monotonic() >= deadline:
                break
            buffers, layers, reference = _batch(batch, device)
            if uniform:
                counts = torch.full_like(buffers["sample_count"], model.average_spp)
                current = torch.stack([replay_sample_layers(layers[:, t], counts[:, t])
                                       for t in range(layers.shape[1])], 1)
                prediction = model.denoiser.forward_sequence(dict(buffers, radiance=current, sample_count=counts),
                                                             transformed=True)
                metrics = sequence_loss(prediction.radiance, reference, transformed=True)
                metrics.update(actual_spp=counts[:, 1:].float().mean(), zero_fraction=reference.new_tensor(0.),
                    saturated_fraction=(counts[:, 1:] == model.max_spp).float().mean(),
                    unavailable_fraction=(~prediction.valid[:, 1:]).float().mean())
            else:
                prediction = model.forward_sequence(buffers, layers, transformed=True)
                counts = prediction.counts
                metrics = _metrics(prediction, reference, model.max_spp)
            n = reference.shape[0]
            windows += n
            paths += int(counts[:, 1:].sum()); pixels += counts[:, 1:].numel()
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.) + float(value)*n
            group = batch["sequence_id"][0]
            if group in preview_groups:
                crop_evidence(previews/group, batch, layers, prediction, counts, reference, model)
                preview_groups.remove(group)
    return {"metrics": {k: v/windows for k, v in totals.items()} if windows else None,
            "windows": windows, "total_windows": len(data), "complete": windows == len(data),
            "learned_frame_paths": paths, "learned_frame_pixels": pixels}


def compare_historical(history, original):
    previous = {r["epoch"]: r for r in original}
    result = []
    for row in history:
        old = previous.get(row["epoch"])
        if old is not None and row["complete"]:
            result.append({"epoch": row["epoch"], "deltas": {
                key: row["train"][key]-old["train"][key]
                for key in ("loss", "zero_fraction", "sampler_gradient_norm", "denoiser_gradient_norm")}})
    return result


def train_arm(base, config, plan, name, inventory_sha, overall_deadline, device="cuda"):
    arm = dict(ARMS[name], name=name)
    output = base/name
    output.mkdir()
    started = time.monotonic()
    write(base/"progress.json", {"arm": name, "stage": "initializing"})
    deadline = min(started+plan["arm_seconds"], overall_deadline-plan["report_reserve_seconds"])
    random.seed(config["seed"]); np.random.seed(config["seed"]); torch.manual_seed(config["seed"])
    generator = torch.Generator().manual_seed(config["seed"])
    model = DiagnosticJoint({k: config[k] for k in MODEL_KEYS}, arm["mode"]).to(device)
    initial_path = Path(plan["source_run"])/"train/initial.pt"
    if arm["initialization"] == "fresh" and initial_path.exists():
        original = torch.load(initial_path, map_location="cpu", weights_only=True)["state_dict"]
        if any(not torch.equal(value.detach().cpu(), original[key]) for key, value in model.state_dict().items()):
            raise ValueError("Fresh initialization differs from the source pilot; inspect source/runtime before comparing")
        del original
    if arm["initialization"] == "best-trained":
        # Read without constructing a second randomly initialized model: preserve
        # the fresh-control CPU RNG/crop stream, but deliberately reset optimizer.
        payload = torch.load(Path(plan["source_run"])/"train/best-trained.pt", map_location="cpu", weights_only=True)
        if payload["model_spec"] != model.specification():
            raise ValueError("Warm-start model specification differs")
        model.load_state_dict(payload["state_dict"])
        arm.update(source_epoch=payload["epoch"], source_global_step=payload["global_step"],
                   source_checkpoint_sha256=digest(Path(plan["source_run"])/"train/best-trained.pt"))
        del payload
    optimizer = optimizer_for(model, config, arm)
    arm.update(optimizer="fresh Adam", optimizer_groups=[{k: v for k, v in g.items() if k != "params"}
                                                       for g in optimizer.param_groups])
    write(output/"arm.json", arm)
    train, validation = dataset(config, "train"), dataset(config, "validation")
    history, step, completed, anomaly = [], 0, 0, None
    stop = "epoch_target"
    try:
        checkpoint(output/"initial.pt", model, optimizer, config, arm, 0, 0, generator, inventory_sha)
        initial_start = time.monotonic()
        initial = evaluate(model, validation, config, device, deadline-5, previews=output/"previews/initial")
        evaluation_reserve = max(5., time.monotonic()-initial_start)
        best = initial if initial["complete"] else None
        best_epoch = 0 if best else None
        write(output/"initial-validation.json", initial)
        checkpoint(output/"best.pt", model, optimizer, config, arm, 0, 0, generator, inventory_sha)
        for epoch in range(1, plan["max_epochs"]+1):
            if time.monotonic()+2*evaluation_reserve+5 >= deadline:
                stop = "arm_time_budget"; break
            model.train()
            totals, windows, updates = {}, 0, 0
            complete = True
            epoch_started = time.monotonic()
            with batches(train, config, generator, shuffle=True) as loader:
                for batch in loader:
                    if time.monotonic()+2*evaluation_reserve+5 >= deadline:
                        complete = False; stop = "arm_time_budget"; break
                    buffers, layers, reference = _batch(batch, device)
                    model.trace.reset(); model.trace.enabled = True
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model.forward_sequence(buffers, layers, training_reference=reference, transformed=True)
                    metrics = _metrics(prediction, reference, model.max_spp)
                    metrics["loss"].backward()
                    networks = [(key, [p.grad for p in getattr(model, key).parameters() if p.grad is not None])
                                for key in ("sampler", "denoiser")]
                    if not all(finite_group_flags([grads for _, grads in networks], device)):
                        raise ValueError(f"Nonfinite gradients before update {step+1}")
                    for key, grads in networks:
                        metrics[key+"_gradient_norm"] = (torch.stack([g.detach().norm() for g in grads]).norm()
                                                        if grads else reference.new_tensor(0.))
                    numbers = {k: float(v.detach()) for k, v in metrics.items()}
                    reasons = [key for key, limit in plan["anomaly_thresholds"].items() if numbers[key] >= limit]
                    capture = anomaly is None and bool(reasons)
                    detailed = capture or (step+1) % plan["detail_every_steps"] == 0
                    frames, count_gradients = model.trace.backward_stats(reference) if detailed else (None, None)
                    row = {"step": step+1, "epoch": epoch, "elapsed_seconds": time.monotonic()-started, **numbers}
                    if detailed:
                        row.update(frame_gradients=frames, features=model.trace.features,
                            logits=[tensor_summary(v) for v in model.trace.logits],
                            logit_gradient_norms=[float(v.grad.norm()) if v.grad is not None else None
                                                  for v in model.trace.logits],
                            expected=tensor_summary(prediction.expected[:, 1:]),
                            counts=tensor_summary(prediction.counts[:, 1:]))
                    if capture:
                        anomaly = {"step": step+1, "epoch": epoch, "reasons": reasons, "metrics": numbers}
                        checkpoint(output/"anomaly-before.pt", model, optimizer, config, arm, completed,
                                   step, generator, inventory_sha)
                        torch.save(_cpu_tree({"batch": batch, "counts": prediction.counts,
                            "count_gradients": count_gradients}), output/"anomaly-batch.pt")
                        write(output/"anomaly.json", anomaly)
                    if "sampler_clip_norm" in arm:
                        torch.nn.utils.clip_grad_norm_(model.sampler.parameters(), arm["sampler_clip_norm"],
                                                       error_if_nonfinite=True)
                    optimizer.step()
                    if not all(finite_group_flags([list(model.parameters())], device)):
                        raise ValueError(f"Nonfinite weights after update {step+1}")
                    step += 1; updates += 1
                    if step % plan["detail_every_steps"] == 0:
                        write(base/"progress.json", {"arm": name, "stage": "training", "epoch": epoch, "step": step,
                            "loss": numbers["loss"], "zero_fraction": numbers["zero_fraction"]})
                    if capture:
                        checkpoint(output/"anomaly-after.pt", model, optimizer, config, arm, completed,
                                   step, generator, inventory_sha)
                    append(output/"steps.jsonl", row)
                    n = reference.shape[0]; windows += n
                    for key, value in numbers.items():
                        totals[key] = totals.get(key, 0.)+value*n
                    model.trace.enabled = False
                    model.trace.reset()
                    del prediction, metrics, buffers, layers, reference
            if windows == len(train):
                completed = epoch
            else:
                complete = False
            model.trace.enabled = False
            evaluation = None
            if complete and (validation_due(epoch, config) or epoch == plan["max_epochs"]) and time.monotonic()+5 < deadline:
                t0 = time.monotonic()
                evaluation = evaluate(model, validation, config, device, deadline-5,
                                      previews=output/f"previews/epoch-{epoch:02d}")
                evaluation_reserve = max(evaluation_reserve, time.monotonic()-t0)
                if evaluation["complete"] and (best is None or evaluation["metrics"]["loss"] < best["metrics"]["loss"]):
                    best, best_epoch = evaluation, epoch
                    checkpoint(output/"best.pt", model, optimizer, config, arm, completed, step, generator, inventory_sha)
            record = {"epoch": epoch, "global_step": step, "complete": complete, "updates": updates,
                      "windows": windows, "total_windows": len(train), "seconds": time.monotonic()-epoch_started,
                      "train": {k: v/windows for k, v in totals.items()} if windows else None,
                      "validation": evaluation}
            append(output/"history.jsonl", record); history.append(record)
            checkpoint(output/"last.pt", model, optimizer, config, arm, completed, step, generator, inventory_sha)
            print(json.dumps({"arm": name, **record}), flush=True)
            if not complete:
                break
        write(base/"progress.json", {"arm": name, "stage": "final_validation", "step": step})
        # Reuse a complete evaluation only if no optimizer updates followed it.
        final = (history[-1]["validation"] if history and history[-1]["complete"]
                 and history[-1]["validation"] is not None and history[-1]["validation"]["complete"]
                 else evaluate(model, validation, config, device, deadline-3, previews=output/"previews/final"))
        if final["complete"] and (best is None or final["metrics"]["loss"] < best["metrics"]["loss"]):
            best, best_epoch = final, completed
            checkpoint(output/"best.pt", model, optimizer, config, arm, completed, step, generator, inventory_sha)
        # Same frozen denoiser, independent uniform recurrent history; only full
        # evaluations are comparable. Integer budget differences are reported.
        uniform = evaluate(model, validation, config, device, deadline-2, uniform=True,
                           previews=output/"previews/uniform-final")
        checkpoint(output/"last.pt", model, optimizer, config, arm, completed, step, generator, inventory_sha)
        original = [json.loads(s) for s in (Path(plan["source_run"])/"train/history.jsonl").read_text().splitlines()]
        summary = {"arm": arm, "completed_epochs": completed, "global_step": step, "stop_reason": stop,
            "elapsed_seconds": time.monotonic()-started, "initial": initial, "final": final,
            "uniform_final": uniform, "best": best, "best_epoch": best_epoch, "anomaly": anomaly,
            "historical_epoch_deltas": compare_historical(history, original) if name == "baseline" else None,
            "exact_resume_supported": False, "test_pixels_loaded": False}
        write(output/"summary.json", summary)
        return summary
    finally:
        model.trace.enabled = False; model.trace.reset()
        train.close(); validation.close()


@torch.inference_mode()
def forced_prediction(model, buffers, layers, counts):
    state, outputs = None, []
    for t in range(layers.shape[1]):
        frame = {k: v[:, t] for k, v in buffers.items()}
        frame["sample_count"] = counts[:, t]
        frame["radiance"] = replay_sample_layers(layers[:, t], counts[:, t])
        state = model.denoiser._step(frame, state)
        outputs.append(state.radiance)
    return torch.stack(outputs, 1)


def probe_snapshot(output, device="cuda", deadline=float("inf")):
    """Fixed-bank, budget-preserving transfer probe; not an unbiased gradient test."""
    payload = torch.load(output/"anomaly-before.pt", map_location="cpu", weights_only=True)
    saved = torch.load(output/"anomaly-batch.pt", map_location="cpu", weights_only=True)
    model = DiagnosticJoint(payload["model_spec"], payload["arm"]["mode"]).to(device).eval()
    model.load_state_dict(payload["state_dict"])
    # First window only: predictions are independent across batch elements.
    batch = {"buffers": {k: v[:1] for k, v in saved["batch"]["buffers"].items()},
             "layers": saved["batch"]["layers"][:1]}
    buffers, layers, reference = _batch(batch, device)
    counts = saved["counts"][:1].to(device)
    gradient = saved["count_gradients"][-1][:1].flatten().to(device)
    baseline = float(sequence_loss(forced_prediction(model, buffers, layers, counts), reference, transformed=True)["loss"])
    flat = counts[:, -1].flatten()
    donors = torch.where(flat > 0)[0]
    receivers = torch.where(flat < model.max_spp)[0]
    donors = donors[gradient[donors].argsort(descending=True)[:4]]
    receivers = receivers[gradient[receivers].argsort()[:4]]
    results = []
    for donor, receiver in zip(donors.tolist(), receivers.tolist()):
        if donor == receiver or time.monotonic() >= deadline:
            continue
        changed = counts.clone()
        changed[:, -1].view(-1)[donor] -= 1
        changed[:, -1].view(-1)[receiver] += 1
        loss = float(sequence_loss(forced_prediction(model, buffers, layers, changed), reference, transformed=True)["loss"])
        results.append({"donor_flat_index": donor, "receiver_flat_index": receiver,
            "donor_count_before": int(flat[donor]), "receiver_count_before": int(flat[receiver]),
            "surrogate_delta_scaled_by_original_batch_mean": float(gradient[receiver]-gradient[donor]),
            "measured_loss_delta_first_window": loss-baseline,
            "paths_before": int(counts.sum()), "paths_after": int(changed.sum())})
    result = {"baseline_first_window_loss": baseline, "transfers": results,
        "note": "Fixed sample bank, final-frame one-sample transfers, frozen pre-update denoiser. "
                "Surrogate magnitudes use the original batch mean; compare signs only. "
                "Discrete differences are not an unbiased expected-gradient or smooth-derivative test."}
    write(output/"transfer-probe.json", result)
    return result


def checked_plan(path):
    plan = read(path)
    required = {"format", "version", "source_run", "config", "arms", "max_epochs", "total_seconds", "arm_seconds",
                "report_reserve_seconds", "torch_allocator_fraction", "monitor_stop_fraction", "memory_limit_fraction",
                "detail_every_steps", "anomaly_thresholds"}
    if set(plan) != required or plan["format"] != "wavefront-collapse-plan" or plan["version"] != 1:
        raise ValueError("Invalid diagnostic plan schema")
    for key in ("source_run", "config"):
        plan[key] = str((Path(path).resolve().parent/plan[key]).resolve())
    for key in ("max_epochs", "total_seconds", "arm_seconds", "report_reserve_seconds", "detail_every_steps"):
        if type(plan[key]) is not int or plan[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if plan["report_reserve_seconds"] >= plan["total_seconds"]:
        raise ValueError("Report reserve must be smaller than total budget")
    if (not isinstance(plan["arms"], list) or not plan["arms"] or len(set(plan["arms"])) != len(plan["arms"])
            or any(a not in ARMS for a in plan["arms"])):
        raise ValueError("Unknown or repeated arm")
    if not (plan["memory_limit_fraction"] == .8 and
            0 < plan["torch_allocator_fraction"] < plan["monitor_stop_fraction"] < .8):
        raise ValueError("Memory policy must leave headroom below 80%")
    if set(plan["anomaly_thresholds"]) != {"sampler_gradient_norm", "zero_fraction", "unavailable_fraction"}:
        raise ValueError("Invalid anomaly threshold metrics")
    if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in plan["anomaly_thresholds"].values()):
        raise ValueError("Anomaly thresholds must be positive finite numbers")
    config = training_config(plan["config"])
    if "r2_streaming" in config:
        raise ValueError("Diagnostic requires existing local pixels; remote preparation is disabled")
    if Path(config["data_root"]).resolve() != (Path(plan["source_run"])/"data").resolve():
        raise ValueError("Diagnostic must reuse the source run data view")
    if read(Path(plan["source_run"])/"train/config.json") != config:
        raise ValueError("Baseline config differs from original run config")
    read_v5_dataset(config["data_root"])
    return plan, config


def preflight(plan, config):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Run the final command on the GPU host; no CPU training fallback.")
    for name in ("initial.pt", "best-trained.pt"):
        _, payload = load_joint_checkpoint(Path(plan["source_run"])/"train"/name)
        if payload["config"] != config:
            raise ValueError("Source checkpoint config mismatch")
        if payload["dataset_split"] != read(Path(plan["source_run"])/"train/inventory.json"):
            raise ValueError("Source checkpoint inventory mismatch")
        if name == "initial.pt" and (payload["epoch"] != 0 or payload["global_step"] != 0):
            raise ValueError("Expected the original fresh initialization checkpoint")
        if name == "best-trained.pt" and payload["epoch"] != 5:
            raise ValueError("Expected the reviewed epoch-5 warm-start checkpoint")
    sample = memory_sample()
    if any(sample[k+"_used_bytes"] >= sample[k+"_total_bytes"]*plan["monitor_stop_fraction"] for k in ("host", "gpu")):
        raise RuntimeError("Memory already exceeds the diagnostic stop threshold")


@contextmanager
def execution_policy(config):
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(config["cpu_threads"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
        with radiance_gradient_policy(config.get("radiance_gradient_floor", 0.)):
            yield
    finally:
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn
        torch.set_num_threads(threads)


def worker(base, plan, config):
    deadline = time.monotonic()+plan["total_seconds"]-5
    torch.cuda.set_per_process_memory_fraction(plan["torch_allocator_fraction"], 0)
    with execution_policy(config), joint_dataset_read_lock(config["data_root"]):
        pinned = read(base/"source-state.json")["files_sha256"]
        if any(digest(REPO/path) != sha for path, sha in pinned.items()):
            raise ValueError("Source changed after launcher snapshot; use a new output")
        write(base/"progress.json", {"stage": "verifying_selected_data"})
        # Verify all selected bytes once; shared lock spans all arms. No test
        # image bytes or network sources are opened.
        inventory = joint_inventory(config["data_root"], {p: config[p+"_groups"] for p in ("train", "validation")})
        if inventory != read(Path(plan["source_run"])/"train/inventory.json"):
            raise ValueError("Verified inventory differs from the original pilot")
        write(base/"inventory.json", inventory)
        inventory_sha = digest(base/"inventory.json")
        for name in plan["arms"]:
            if time.monotonic()+plan["report_reserve_seconds"]+5 >= deadline:
                write(base/(name+"-skipped.json"), {"reason": "overall_time_budget"}); continue
            train_arm(base, config, plan, name, inventory_sha, deadline)
            gc.collect(); torch.cuda.empty_cache()
        baseline = base/"baseline"
        if (baseline/"anomaly-before.pt").is_file() and time.monotonic()+15 < deadline:
            probe_snapshot(baseline, deadline=deadline-2)


def report(base, status):
    plan = read(base/"plan.json")
    rows = ["# Quality-collapse diagnostic", "", f"Launcher status: **{status}**.", "",
        "Development validation only; fixed-SPP references are not convergence-qualified. "
        "Uniform controls share nominal 4 SPP; compare recorded actual path totals. "
        "Partial evaluations are explicitly excluded from quality conclusions.", "",
        "| Arm | Completed epochs | Updates | Seconds | Initial loss | Final loss | Uniform final loss | First anomaly |",
        "|---|---:|---:|---:|---:|---:|---:|---|"]
    results = {}
    def loss(evaluation):
        return f"{evaluation['metrics']['loss']:.6f}" if evaluation and evaluation["complete"] else "incomplete"
    for name in plan["arms"]:
        directory = base/name
        if (directory/"summary.json").exists():
            s = read(directory/"summary.json"); results[name] = s
            anomaly = s["anomaly"]
            rows.append(f"| {name} | {s['completed_epochs']} | {s['global_step']} | {s['elapsed_seconds']:.1f} | "
                        f"{loss(s['initial'])} | {loss(s['final'])} | {loss(s['uniform_final'])} | "
                        f"{anomaly['step'] if anomaly else 'none'} |")
        else:
            state = "interrupted; inspect history/steps" if directory.exists() else "not started / budget exhausted"
            rows.append(f"| {name} | — | — | — | — | — | — | {state} |")
    rows += ["", "- baseline vs uniform starts from identical fresh weights and uses the same CPU RNG/crop stream.",
        "- frozen starts from epoch 5 with fresh Adam; compare its initial/final change. "
        "Use optional warm-joint for a matched epoch-5 joint control.",
        "- Anomaly thresholds trigger evidence capture, not a diagnosis or automatic training fix.",
        "- Every update is in ARM/steps.jsonl; epoch completion and validation coverage are in ARM/history.jsonl.",
        "- ARM/anomaly-{before,after}.pt and anomaly-batch.pt reproduce the first captured abnormal update. "
        "Diagnostic checkpoints require this workflow; generic joint resume is intentionally unsupported.",
        "- Baseline historical epoch deltas are in baseline/summary.json. "
        "A failed reproduction makes causal conclusions premature.",
        "- If captured and time permits, baseline/transfer-probe.json tests small exact-budget transfers "
        "on one fixed sample bank; disagreement is a clue, not proof of a wrong expected gradient.",
        "- No conclusion about repaired training follows from this diagnostic alone."]
    for name in plan["arms"]:
        pictures = sorted((base/name/"previews").glob("*/*/comparison.png"))
        if pictures:
            rows += ["", f"## {name}: fixed validation crops", "",
                     "Crop-local allocation and recurrence; these do not replace native full-image evaluation.", ""]
            for picture in pictures:
                relative = picture.relative_to(base)
                rows += [f"{picture.parent.parent.name} · {picture.parent.name}", "", f"![Comparison]({relative})", ""]
    (base/"report.md").write_text("\n".join(rows)+"\n")
    write(base/"summary.json", {"status": status, "arms": results})


def launch(base, plan_path, plan, config):
    import shutil
    preflight(plan, config)
    if base.exists():
        raise FileExistsError(f"Output already exists; choose a new --output: {base}")
    runs = (WORKSPACE/"runs").resolve()
    if not base.is_relative_to(runs):
        raise ValueError("New outputs must be under quality_workspace/runs")
    base.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(base.parent).free < 2_000_000_000:
        raise ValueError("Need 2 GB free output headroom; source arrays are never copied")
    base.mkdir()
    write(base/"plan.json", plan); write(base/"config.json", config)
    files = sorted((REPO/"src/wavefront_denoiser").glob("*.py")) + [Path(__file__), REPO/"tools/run_quality_pilot.py"]
    (base/"source.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary", "HEAD"], cwd=REPO))
    source = base/"source"; source.mkdir()
    for path in files:
        destination = source/path.relative_to(REPO)
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, destination)
    write(base/"source-state.json", {"git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "files_sha256": {str(p.relative_to(REPO)): digest(p) for p in files},
        "source_checkpoints": {n: digest(Path(plan["source_run"])/"train"/n) for n in ("initial.pt", "best-trained.pt")},
        "python": sys.executable, "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda})
    started = time.monotonic(); status = "failed"; process = None
    print(f"Diagnostic: {base}\nLog: {base/'run.log'}\nOverall worker deadline: {plan['total_seconds']/60:.0f} minutes", flush=True)
    progress_seen, progress_at = None, 0.
    try:
        with (base/"run.log").open("w") as log, (base/"resources.jsonl").open("w") as audit:
            process = subprocess.Popen([sys.executable, "-u", str(Path(__file__)), "--worker", "--plan", str(base/"plan.json"),
                                        "--output", str(base)], stdout=log, stderr=subprocess.STDOUT, cwd=REPO,
                                       start_new_session=True)
            while process.poll() is None:
                sample = {"elapsed_seconds": time.monotonic()-started, **memory_sample()}
                audit.write(json.dumps(sample)+"\n"); audit.flush()
                if (base/"progress.json").exists() and time.monotonic()-progress_at >= 10:
                    progress = read(base/"progress.json")
                    if progress != progress_seen:
                        print(f"{sample['elapsed_seconds']:.0f}s: {json.dumps(progress)}", flush=True)
                        progress_seen = progress
                    progress_at = time.monotonic()
                if any(sample[k+"_used_bytes"] >= sample[k+"_total_bytes"]*plan["monitor_stop_fraction"]
                       for k in ("host", "gpu")):
                    status = "memory_guard_stopped"; break
                if time.monotonic()-started >= plan["total_seconds"]:
                    status = "overall_deadline_stopped"; break
                time.sleep(.5)
            else:
                status = "completed" if process.returncode == 0 else f"worker_failed_exit_{process.returncode}"
                if status == "completed" and any(base.glob("*-skipped.json")):
                    status = "completed_with_skipped_arms"
    except KeyboardInterrupt:
        status = "interrupted"
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
        write(base/"status.json", {"status": status, "wall_seconds": time.monotonic()-started})
        report(base, status)
    print(f"{status}: {base/'report.md'}", flush=True)
    return 0 if status.startswith("completed") else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=WORKSPACE/"configs/quality-collapse-diagnostic.json")
    parser.add_argument("--output", type=Path, default=WORKSPACE/"runs/quality-collapse-diagnostic")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--report-only", action="store_true", help="Rebuild a report from existing partial/completed outputs, without CUDA")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.report_only:
        base = args.output.resolve()
        report(base, read(base/"status.json")["status"] if (base/"status.json").exists() else "interrupted_unknown")
        print(base/"report.md")
        return 0
    plan, config = checked_plan(args.plan)
    if args.plan_only:
        print(json.dumps({"output": str(args.output.resolve()), "plan": plan,
            "train_clips": len(config["train_groups"]), "validation_clips": len(config["validation_groups"]),
            "needs_cuda_for_execution": True, "arms": {a: ARMS[a] for a in plan["arms"]}}, indent=2))
        return 0
    if args.worker:
        worker(args.output.resolve(), plan, config); return 0
    return launch(args.output.resolve(), args.plan, plan, config)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Diagnostic failed: {error}", file=sys.stderr)
        raise SystemExit(1)
