#!/usr/bin/env python3
"""Compare local joint-training batches in isolated, reproducible processes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()


def summarize(summary, warmup_epochs):
    rows = summary["performance"]["epochs"][warmup_epochs:]
    if not rows:
        raise ValueError("Need measured epochs after warmup")
    train_seconds = sum(r["training_seconds"] for r in rows)
    val_seconds = sum(r["validation_seconds"] for r in rows)
    windows = sum(r["training_windows"] for r in rows)
    val_windows = sum(r["validation_windows"] for r in rows)
    config = summary["config"]
    return {
        "measured_epochs": len(rows),
        "training_windows": windows,
        "optimizer_updates": sum(r["training_steps"] for r in rows),
        "training_windows_per_second": windows / train_seconds,
        "training_pixels_per_second": windows * config["window_length"] * config["patch_size"]**2 / train_seconds,
        "validation_windows_per_second": val_windows / val_seconds,
        "validation_pixels_per_second": val_windows * config["window_length"] * config["validation_patch_size"]**2 / val_seconds,
        "training_seconds": train_seconds,
        "training_data_wait_seconds": sum(r["training_data_seconds"] for r in rows),
        "validation_seconds": val_seconds,
        "checkpoint_seconds": sum(r["checkpoint_seconds"] for r in rows),
        "cuda_peak_allocated_bytes": summary["performance"]["cuda_peak_allocated_bytes"],
        "cuda_peak_reserved_bytes": summary["performance"]["cuda_peak_reserved_bytes"],
        "process_peak_rss_bytes": summary["performance"]["process_lifetime_peak_rss_bytes"],
        "initial_validation": summary["initial_validation"],
    }


def host_memory():
    fields = {line.split(':')[0]: int(line.split()[1])*1024
              for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split()) >= 2}
    return fields['MemTotal'], fields['MemAvailable']


def run_case(command, log, env, fraction, device, audit_path):
    if fraction is None:
        return subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    samples, violation = [], None
    started = time.monotonic()
    with subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT) as process:
        try:
            while process.poll() is None:
                total, available = host_memory()
                sample = {'elapsed_seconds':time.monotonic()-started,
                          'host_total_bytes':total,'host_available_bytes':available}
                if device == 'cuda':
                    gpu = subprocess.check_output(['nvidia-smi','--id=0',
                        '--query-gpu=memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
                    gpu_total,gpu_used = [int(value.strip())*2**20 for value in gpu.strip().split(',')]
                    sample.update(gpu_total_bytes=gpu_total,gpu_used_bytes=gpu_used)
                if total-available >= total*fraction:
                    violation = 'host RAM limit'
                if sample.get('gpu_used_bytes',0) >= sample.get('gpu_total_bytes',1)*fraction:
                    violation = 'GPU memory limit'
                samples.append(sample)
                if violation:
                    process.terminate()
                    break
                time.sleep(.5)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            write_json(audit_path,{'limit_fraction':fraction,'poll_seconds':.5,
                'violation':violation,'samples':samples})
    return 124 if violation else process.returncode


def state_digest(value):
    """Stable byte identity for checkpoint tensors, RNG and optimizer trees."""
    import torch
    h = hashlib.sha256()
    def visit(item):
        h.update(type(item).__name__.encode()+b':')
        if isinstance(item,torch.Tensor):
            h.update(str((item.dtype,tuple(item.shape))).encode())
            h.update(item.contiguous().numpy().tobytes())
        elif isinstance(item,dict):
            for key in sorted(item,key=lambda key:(type(key).__name__,str(key))):
                visit(key); visit(item[key])
        elif isinstance(item,(list,tuple)):
            for child in item:
                visit(child)
        else:
            h.update(json.dumps(item,allow_nan=False).encode())
        h.update(b';')
    visit(value)
    return h.hexdigest()


def worker(config, output, device, memory_fraction=None):
    import torch
    import wavefront_denoiser
    from wavefront_denoiser.joint_training import train_joint

    # Capture the policies the trainer leaves unchanged. Deterministic/cuDNN
    # flags selected by the trainer are also stored in every checkpoint.
    software = {
        "python": sys.version, "python_executable": sys.executable,
        "import_path": wavefront_denoiser.__file__,
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "matmul_precision": torch.get_float32_matmul_precision(),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "training_deterministic_algorithms": True,
        "training_cudnn_deterministic": True,
        "training_cudnn_benchmark": False,
        "parameter_dtype": "float32", "autocast": False,
    }
    if device == 'cuda' and memory_fraction is not None:
        # Leave context/external allocations and another 512 MiB outside the
        # allocator budget. The parent also monitors whole-device usage.
        free,total = torch.cuda.mem_get_info()
        budget = int(total*memory_fraction)-(total-free)-512*2**20
        if budget <= 0:
            raise RuntimeError('Insufficient GPU headroom under the memory cap')
        torch.cuda.set_per_process_memory_fraction(budget/total)
        software['memory_limit'] = {'fraction':memory_fraction,'allocator_budget_bytes':budget,
            'initial_nonallocator_bytes':total-free,'margin_bytes':512*2**20}
    summary = train_joint(config, output, device)
    write_json(output / "software.json", software)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--sweep", choices=("train", "validation", "loader", "cache", "crops", "safety"), default="train")
    parser.add_argument("--batches", type=int, nargs="+")
    parser.add_argument("--cache-frames", type=int, nargs="+")
    parser.add_argument("--prefetch-depths", type=int, nargs="+")
    parser.add_argument("--max-memory-fraction", type=float)
    parser.add_argument("--discard-checkpoints", action="store_true",
                        help="Keep state hashes and reports; remove this run's generated .pt files after each successful case")
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--measured-epochs", type=int, default=20)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        worker(args.config, args.output, args.device,args.max_memory_fraction)
        return 0
    if args.warmup_epochs < 1 or args.measured_epochs < 1:
        parser.error("Need positive warmup and measured epoch counts")
    if args.max_memory_fraction is not None and not 0 < args.max_memory_fraction <= .8:
        parser.error("Memory limit must be greater than zero and at most 0.8")
    if args.sweep in ("loader","cache","crops","safety") and args.batches:
        parser.error("Loader sweep fixes batch size from the config; omit --batches")
    if args.sweep != "cache" and (args.cache_frames or args.prefetch_depths):
        parser.error("Cache sizes/depths require --sweep cache")
    cache_frames,depths = args.cache_frames or [8,16,32,64],args.prefetch_depths or [2]
    for values in (cache_frames,depths):
        if any(value < 1 for value in values) or len(set(values)) != len(values):
            parser.error("Cache sizes and depths must be unique positive integers")
    batches = [] if args.sweep in ("loader","cache","crops","safety") else (args.batches or ([4, 8, 16, 32] if args.sweep == "train" else [1, 4, 8]))
    if any(b < 1 for b in batches) or len(set(batches)) != len(batches):
        parser.error("Batches must be unique positive integers")
    config_path, output = args.config.resolve(), args.output.resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error("Benchmark output must be outside the source checkout")
    if output.exists():
        parser.error("Choose a fresh benchmark output directory")
    if git("status", "--porcelain", "--untracked-files=all"):
        parser.error("Commit source changes before benchmarking")
    source = {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"), "dirty": False}
    from wavefront_denoiser.joint_training import training_config
    config = training_config(config_path)
    if "r2_streaming" in config:
        parser.error("Use existing local decoded data for this benchmark")
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this process; run on the GPU host")
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), CUBLAS_WORKSPACE_CONFIG=":4096:8")
    config["epochs"] = args.warmup_epochs + args.measured_epochs
    write_json(output / "manifest.json", {
        **source, "source_root": str(ROOT), "config_path": str(config_path),
        "input_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "base_config": config, "sweep": args.sweep, "batches": batches,
        "loader_cases": ["synchronous", "cached_v1", "prefetch_v1"] if args.sweep == "loader" else None,
        "crop_cases": ["prefetch_v1","prefetch_fused_v1"] if args.sweep=='crops' else None,
        "warmup_epochs": args.warmup_epochs, "measured_epochs": args.measured_epochs,
        "cache_frames":cache_frames if args.sweep=='cache' else None,
        "prefetch_depths":depths if args.sweep=='cache' else None,
        "memory_limit_fraction":args.max_memory_fraction,"discard_checkpoints":args.discard_checkpoints,
        "cache_policy": "OS cache uncontrolled; later cases may benefit from earlier reads",
        "schedule": "Same seed, initial weights and loader RNG reset in each fresh process; fixed validation crops",
    })
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    (output / "pip-freeze.txt").write_text(freeze.stdout + freeze.stderr)
    try:
        hardware = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        (output / "nvidia-smi.txt").write_text(hardware.stdout + hardware.stderr)
    except FileNotFoundError:
        (output / "nvidia-smi.txt").write_text("nvidia-smi unavailable\n")
    results, inventory_sha = [], None
    cases = (["synchronous", "cached_v1", "prefetch_v1"] if args.sweep == "loader" else ["prefetch_v1","prefetch_fused_v1"] if args.sweep=="crops" else
             [(frames,depth) for frames in cache_frames for depth in depths] if args.sweep=='cache' else ["legacy", "grouped", "scheduled-grouped"] if args.sweep=="safety" else batches)
    for case in cases:
        if git("rev-parse", "HEAD") != source["commit"] or git("status", "--porcelain", "--untracked-files=all"):
            raise RuntimeError("Source changed during benchmark")
        case_config = dict(config)
        if args.sweep in ("loader","crops"):
            name = f"loader-{case}"
            batch = config["batch_size"]
            case_config.pop("loader_policy",None)
            if args.sweep == 'loader':
                case_config.pop("decoded_cache_frames",None)
                case_config.pop("prefetch_batches",None)
            if case != "synchronous":
                case_config["loader_policy"] = case
        elif args.sweep == 'safety':
            name = f'safety-{case}'
            batch = config['batch_size']
            for key in ('finite_check_policy','validation_every_epochs','validation_warmup_epochs'):
                case_config.pop(key,None)
            if case != 'legacy':
                case_config['finite_check_policy']='grouped_v1'
            if case == 'scheduled-grouped':
                case_config.update(validation_every_epochs=5,validation_warmup_epochs=5)
        elif args.sweep == 'cache':
            frames,depth = case
            name = f'cache-{frames}-prefetch-{depth}'
            batch = config['batch_size']
            case_config.update(loader_policy='prefetch_v1',decoded_cache_frames=frames,prefetch_batches=depth)
        else:
            batch = case
            name = f"{args.sweep}-batch-{batch}"
            case_config["batch_size" if args.sweep == "train" else "validation_batch_size"] = batch
        case_path = output / (name + ".json")
        write_json(case_path, case_config)
        run = output / name
        command = [sys.executable, str(Path(__file__).resolve()), str(case_path), str(run), "--worker", "--device", args.device]
        if args.max_memory_fraction is not None:
            command += ['--max-memory-fraction',str(args.max_memory_fraction)]
        with (output / (name + ".log")).open("w") as log:
            returncode = run_case(command,log,env,args.max_memory_fraction,args.device,output/(name+'-resources.json'))
        if returncode:
            write_json(output / "failure.json", {"case": name, "returncode": returncode, "command": command})
            raise RuntimeError(f"{name} failed; see {output / (name + '.log')}")
        if git("rev-parse", "HEAD") != source["commit"] or git("status", "--porcelain", "--untracked-files=all"):
            raise RuntimeError("Source changed during benchmark; results are invalid")
        digest = hashlib.sha256((run / "inventory.json").read_bytes()).hexdigest()
        if inventory_sha is not None and digest != inventory_sha:
            raise RuntimeError("Dataset inventory changed between cases")
        inventory_sha = digest
        summary = json.loads((run / "summary.json").read_text())
        metrics = summarize(summary, args.warmup_epochs)
        history = [json.loads(line) for line in (run / "history.jsonl").read_text().splitlines()]
        metrics["last_validation"] = history[-1]["validation"]
        payload = torch.load(run/'last.pt',map_location='cpu',weights_only=True)
        metrics['model_state_sha256'] = state_digest(payload['state_dict'])
        metrics['training_state_sha256'] = state_digest(payload['training'])
        metrics['mean_epoch_seconds'] = sum(r['epoch_seconds'] for r in summary['performance']['epochs'][args.warmup_epochs:])/args.measured_epochs
        results.append({"case": name, "batch": batch, "loader_policy":case_config.get("loader_policy"),
                        "decoded_cache_frames":case_config.get('decoded_cache_frames',16),
                        "prefetch_batches":case_config.get('prefetch_batches',2),
                        "inventory_sha256": digest, **metrics})
        write_json(output / "comparison.json", results)
        lines = ["# Joint batch benchmark", "", "Warmup epochs excluded from throughput; allocator/RSS peaks cover the whole case.",
                 "OS cache is uncontrolled. These local subset results do not measure cold/full-dataset I/O or convergence.", "",
                 "| Case | Train windows/s | Validation windows/s | Updates | Peak GPU GiB |", "| --- | ---: | ---: | ---: | ---: |"]
        for r in results:
            peak = r["cuda_peak_allocated_bytes"]
            memory = "n/a" if peak is None else f"{peak / 2**30:.3f}"
            lines.append(f"| {r['case']} | {r['training_windows_per_second']:.3f} | {r['validation_windows_per_second']:.3f} | {r['optimizer_updates']} | {memory} |")
        (output / "comparison.md").write_text("\n".join(lines) + "\n")
        if args.discard_checkpoints:
            for checkpoint in (run/'last.pt',run/'model.pt'):
                checkpoint.unlink()
        print(name, json.dumps(metrics, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
