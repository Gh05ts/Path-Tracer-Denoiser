#!/usr/bin/env python3
"""Prepare a frozen larger-data quality pilot, train to a time budget, and preview."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
WORKSPACE = REPO.parent


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def gpu_memory():
    result = subprocess.run(["nvidia-smi", "--id=0", "--query-gpu=memory.total,memory.used",
                             "--format=csv,noheader,nounits"], text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError("No accessible NVIDIA GPU. Run this launcher in the GPU-enabled environment.")
    return [int(x.strip())*2**20 for x in result.stdout.strip().split(",")]


def memory_sample(use_gpu):
    fields = {line.split(":")[0]: int(line.split()[1])*1024
              for line in Path("/proc/meminfo").read_text().splitlines() if len(line.split()) >= 2}
    result = {"host_total_bytes":fields["MemTotal"],
              "host_used_bytes":fields["MemTotal"]-fields["MemAvailable"]}
    if use_gpu:
        result["gpu_total_bytes"],result["gpu_used_bytes"] = gpu_memory()
    return result


def monitored(args, base, phase, plan, *, use_gpu=True):
    log = base/(phase+".log")
    print(f"{phase}: {log}", flush=True)
    limit = plan["monitor_stop_fraction"]
    started = time.monotonic()
    def check(sample):
        for kind in ("host", "gpu") if use_gpu else ("host",):
            if sample[kind+"_used_bytes"] >= sample[kind+"_total_bytes"]*limit:
                raise RuntimeError(f"{kind} memory reached the {limit:.0%} stop threshold (user limit 80%). See {log}")
    check(memory_sample(use_gpu))
    with log.open("w") as stream, (base/(phase+"-resources.jsonl")).open("w") as audit:
        process = subprocess.Popen([sys.executable, "-u", str(Path(__file__)), "--worker", phase,
            "--output", str(base), *args], cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                sample = {"elapsed_seconds":time.monotonic()-started,**memory_sample(use_gpu)}
                audit.write(json.dumps(sample)+"\n"); audit.flush()
                check(sample)
                time.sleep(0.5)
            if process.returncode:
                raise RuntimeError(f"{phase} failed (exit {process.returncode}). See {log}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill(); process.wait()


def selections(config, descriptor):
    requested = config["train_groups"]+config["validation_groups"]
    result = {p:[] for p in ("train", "validation")}
    for group in requested:
        partition = next((p for p,names in descriptor["partitions"].items() if group in names),None)
        if partition not in result:
            raise ValueError(f"Requested group is missing or reserved: {group}")
        result[partition].append(group)
    return result


def reuse_arrays(snapshot, old_root):
    """Reuse only arrays with the same receipt-bound size and declared digest.

    Materialization subsequently verifies the actual bytes, including symlinks.
    No source sidecars or original arrays are written.
    """
    from wavefront_denoiser.r2_streaming import SOURCE_RECORD
    source = read(snapshot/SOURCE_RECORD)
    for names in source["selected"].values():
        for group in names:
            for record in sorted((snapshot/"decoded"/group).glob("frame_*/manifest.json")):
                relative = record.parent.relative_to(snapshot)
                old_manifest = old_root/relative/"manifest.json"
                if not old_manifest.is_file():
                    continue
                old = read(old_manifest)
                for filename,expected in read(record)["arrays"].items():
                    existing, destination = old_root/relative/filename, snapshot/relative/filename
                    if (old.get("arrays",{}).get(filename) == expected and existing.is_file()
                            and existing.stat().st_size == expected["size"] and not destination.exists()):
                        destination.symlink_to(existing.resolve())


def make_view(snapshot, output, config, plan):
    from wavefront_denoiser.adaptive_data import read_v5_dataset
    from wavefront_denoiser.dataset_generation import digest
    descriptor = read_v5_dataset(snapshot)
    partitions = {p:list(names) for p,names in descriptor["partitions"].items()}
    for scene in plan["train_scenes"]+plan["validation_scenes"]:
        target = "train" if scene in plan["train_scenes"] else "validation"
        groups = [g for g,v in descriptor["groups"].items() if v["scene"] == scene]
        if any(g in partitions["test"] for g in groups):
            raise ValueError("An experiment cannot move a reserved test scene")
        for p in ("train", "validation"):
            partitions[p] = [g for g in partitions[p] if g not in groups]
        partitions[target].extend(groups)
    split = {"format":"wavefront-experiment-split", "version":1,
        "source_manifest_sha256":digest(snapshot/"decoded_cache_lock.json"),
        "source_partitions":descriptor["partitions"],"partitions":partitions,
        "purpose":"User-selected fresh BMW/Living Room train; Dragon/Bathroom development validation",
        "previous_use":"Dragon previously appeared in throughput diagnostics; not an untouched test scene"}
    if output.exists():
        if read(output/"experiment-split.json") != split:
            raise ValueError("Existing experiment view differs from the requested split")
    else:
        output.mkdir()
        for path in snapshot.iterdir():
            if not path.name.startswith("."):
                (output/path.name).symlink_to(path.resolve(),target_is_directory=path.is_dir())
        write(output/"experiment-split.json",split)
    effective = read_v5_dataset(output)
    for p in ("train", "validation"):
        if set(config[p+"_groups"])-set(effective["partitions"][p]):
            raise ValueError("Requested selections disagree with experiment view")


def prepare(base, config, plan):
    from wavefront_denoiser.adaptive_data import read_v5_dataset
    from wavefront_denoiser.r2_streaming import prepare_r2_metadata, materialize_decoded_snapshot
    # Original local metadata chooses original source partitions; remote
    # preparation rechecks the current locks and every selected archive receipt.
    original = read_v5_dataset(Path(plan["reuse_arrays_from"]))
    selected = selections(config,original)
    snapshot = base/"source-data"
    if not snapshot.exists():
        prepare_r2_metadata(plan["source"],snapshot,layout="decoded",
            train_groups=selected["train"],validation_groups=selected["validation"])
    record = read(snapshot/"r2-source.json")
    if record["source"] != plan["source"] or record["selected"] != selected:
        raise ValueError("Existing snapshot has different selections/source; choose a new output")
    reuse_arrays(snapshot,Path(plan["reuse_arrays_from"]))
    required = 0; total = 0
    for p,names in selected.items():
        for group in names:
            for manifest in (snapshot/"decoded"/group).glob("frame_*/manifest.json"):
                for filename,array in read(manifest)["arrays"].items():
                    total += array["size"]
                    if not (manifest.parent/filename).exists():
                        required += array["size"]
    print(f"Selected payload: {total/1e9:.2f} GB; additional download: {required/1e9:.2f} GB",flush=True)
    if shutil.disk_usage(base).free < required+4_000_000_000:
        raise ValueError("Insufficient disk for the selected arrays plus 4 GB output headroom")
    report = materialize_decoded_snapshot(snapshot)
    write(base/"materialization.json",report)
    make_view(snapshot,base/"data",config,plan)


def evidence(base, plan):
    from wavefront_denoiser.dataset_generation import digest
    preview = __import__("preview_joint_pilot")
    run = base/"train"
    summary = read(run/"summary.json")
    root = base/"previews"; root.mkdir(exist_ok=True)
    results = []
    for checkpoint in plan["preview_checkpoints"]:
        for group in plan["preview_groups"]:
            output = root/(checkpoint.removesuffix(".pt")+"-"+group)
            if output.exists():
                result = read(output/"preview.json")
                if result["checkpoint_sha256"] != digest(run/checkpoint):
                    raise ValueError("Existing preview belongs to a different checkpoint")
            else:
                result = preview.export_preview(run,output,checkpoint=checkpoint,sequence_id=group,
                    device_name="cuda",full_frame=True)
            results.append({"path":str(output.relative_to(base)),**result})
            print(f"Full-frame preview ready: {output}",flush=True)
    write(base/"quality-evidence.json",{"training_summary":summary,"previews":results,"notes":plan["notes"]})
    rows = ["# Two-hour quality pilot", "", *plan["notes"], "",
            f"Completed epochs: {summary['completed_epochs']}; optimizer steps: {summary['global_step']}.",
            f"Best validation epoch (including initialization): {summary['best_epoch']}.",
            f"Best trained epoch: {summary.get('best_trained_epoch')}.", "",
            "Uniform controls use the same frozen denoiser. Nominal budget is 4 SPP; actual integer totals are recorded.",
            "These development scenes and fixed-budget references do not establish paper quality or sampler superiority.", "",
            "| Checkpoint | Clip | Joint spatial L1 | Uniform spatial L1 | Joint temporal L1 | Uniform temporal L1 | Actual SPP | Preview |",
            "|---|---|---:|---:|---:|---:|---:|---|"]
    for r in results:
        j,u = r["joint_metrics"],r["uniform_denoiser_metrics"]
        rows.append(f"| {r['checkpoint']} (epoch {r['checkpoint_epoch']}) | {r['sequence_id']} | "
            f"{j['spatial_l1']:.6f} | {u['spatial_l1']:.6f} | {j['temporal_l1']:.6f} | {u['temporal_l1']:.6f} | "
            f"{r['actual_spp']:.4f} | [Full image]({r['path']}/comparison.png) · [Animation]({r['path']}/comparison.gif) |")
    history = [json.loads(line) for line in (run/"history.jsonl").read_text().splitlines()]
    gradients = {name:{"min":min(r["train"][name+"_gradient_norm"] for r in history),
                       "max":max(r["train"][name+"_gradient_norm"] for r in history)}
                 for name in ("sampler","denoiser")}
    rows += ["", "Finite gradient norms recorded during training:", "", json.dumps(gradients), "",
             "Allocation zeros, saturation, unavailable outputs, actual path totals, and checkpoint identity are in each preview.json."]
    (base/"quality-report.md").write_text("\n".join(rows)+"\n")
    # Portable local image viewer; no server, upload, or external dependencies.
    import html
    cards = "\n".join(f'<h2>{html.escape(r["checkpoint"])} · {html.escape(r["sequence_id"])}</h2>'
        f'<a href="{r["path"]}/comparison.png"><img loading="lazy" src="{r["path"]}/comparison.png"></a>'
        f'<p><a href="{r["path"]}/comparison.gif">Temporal animation</a> · '
        f'<a href="{r["path"]}/sample-count.png">Sampling map</a> · '
        f'<a href="{r["path"]}/preview.json">Metrics and provenance</a></p>' for r in results)
    (base/"preview.html").write_text('<!doctype html><meta charset="utf-8"><title>Quality pilot</title>'
        '<style>body{background:#181818;color:#eee;font:16px system-ui;margin:24px}img{width:100%}a{color:#9cf}</style>'
        '<h1>BMW/Living Room → Dragon/Bathroom</h1><p>Native full-image exports. '
        'Columns: uniform noisy · adaptive noisy · uniform + frozen denoiser · joint denoised · reference.</p>'+cards)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=WORKSPACE/"configs/quality-bmw-living-2h.json")
    parser.add_argument("--plan",type=Path,default=WORKSPACE/"configs/quality-bmw-living-2h-plan.json")
    parser.add_argument("--output",type=Path,default=WORKSPACE/"runs/bmw-living-quality-2h")
    parser.add_argument("--plan-only",action="store_true")
    parser.add_argument("--prepare-only",action="store_true")
    parser.add_argument("--preview-only",action="store_true")
    parser.add_argument("--worker",choices=("prepare","train","preview"),help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    base = args.output.resolve()
    plan,config = read(args.plan),read(args.config)
    if not (plan["memory_limit_fraction"] == .8
            and 0 < plan["torch_allocator_fraction"] < plan["monitor_stop_fraction"] < .8):
        raise ValueError("Memory policy must leave headroom below the user's 80% limit")
    config["data_root"] = str(base/"data")
    if args.plan_only:
        from wavefront_denoiser.joint_training import training_config
        training_config(args.config)
        print(json.dumps({"output":str(base),"train_clips":len(config["train_groups"]),
            "validation_clips":len(config["validation_groups"]),"config":config,"plan":plan},indent=2))
        return 0
    if args.worker:
        if args.worker == "prepare":
            prepare(base,config,plan)
        else:
            import torch
            torch.set_num_threads(config["cpu_threads"])
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required; no CPU training fallback")
            torch.cuda.set_per_process_memory_fraction(plan["torch_allocator_fraction"],0)
            if args.worker == "train":
                from wavefront_denoiser.joint_training import train_joint
                train_joint(base/"train-config.json",base/"train","cuda")
            else:
                evidence(base,plan)
        return 0
    if base.exists() and (base/"train").exists() and not args.preview_only:
        raise FileExistsError("Training already started here. Use --preview-only or choose a new --output.")
    if not args.prepare_only:
        gpu_memory()
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot access CUDA in this environment")
    base.mkdir(parents=True,exist_ok=True)
    # Child processes use the same exact plan/config paths as the parent.
    extra = ["--config",str(args.config.resolve()),"--plan",str(args.plan.resolve())]
    if not args.preview_only:
        write(base/"train-config.json",config)
        write(base/"experiment-plan.json",plan)
        files = list((REPO/"src/wavefront_denoiser").glob("*.py"))+[Path(__file__),REPO/"tools/preview_joint_pilot.py"]
        write(base/"source-state.json",{"git_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),
            "files_sha256":{str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
            "python":sys.executable,"memory_policy":{k:plan[k] for k in
                ("memory_limit_fraction","monitor_stop_fraction","torch_allocator_fraction")}})
        monitored(extra,base,"prepare",plan,use_gpu=False)
        if args.prepare_only:
            print(f"Data ready. Start training with: bash {WORKSPACE/'run_quality_2h.sh'} --output {base}")
            return 0
        monitored(extra,base,"train",plan)
        subprocess.run([sys.executable,str(REPO/"tools/report_joint_pilot.py"),str(base/"train")],check=True,cwd=REPO)
    monitored(extra,base,"preview",plan)
    print(f"Done. Report: {base/'quality-report.md'}\nFull-image viewer: {base/'preview.html'}",flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError,ValueError,RuntimeError,subprocess.CalledProcessError) as error:
        print(f"Quality pilot failed: {error}",file=sys.stderr)
        raise SystemExit(1)
