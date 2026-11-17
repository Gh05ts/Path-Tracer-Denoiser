#!/usr/bin/env python3
"""Run a Python module/script with a wall deadline and whole-host/device guard."""
import argparse
import hashlib
import importlib.metadata
import platform
import json
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time


def sample(gpu_id=None):
    fields = {line.split(':')[0]: int(line.split()[1])*1024
              for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split()) >= 2}
    result = {'host_total': fields['MemTotal'], 'host_used': fields['MemTotal']-fields['MemAvailable']}
    if gpu_id is not None:
        output = subprocess.check_output(['nvidia-smi', '--id='+gpu_id,
            '--query-gpu=memory.total,memory.used', '--format=csv,noheader,nounits'], text=True, timeout=3)
        total, used = [int(v.strip())*2**20 for v in output.strip().split(',')]
        result.update(gpu_total=total, gpu_used=used)
    return result


def check_memory(values):
    if any(values[k+'_used'] >= .79*values[k+'_total'] for k in ('host', 'gpu') if k+'_total' in values):
        raise RuntimeError('Whole-host or device memory reached 79%; stopping below the 80% limit')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--hours', type=float, required=True)
    p.add_argument('--log', type=Path, required=True)
    p.add_argument('--gpu-id', default='0', help='Physical NVIDIA index or UUID; worker sees it as cuda:0')
    p.add_argument('--cpu', action='store_true')
    entry = p.add_mutually_exclusive_group(required=True)
    entry.add_argument('--module'); entry.add_argument('--script')
    p.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('args', nargs=argparse.REMAINDER)
    a = p.parse_args(argv)
    if not 0 < a.hours < 1e4: p.error('--hours must be positive and finite')
    command_args = a.args[1:] if a.args[:1] == ['--'] else a.args
    if a.worker:
        if not a.cpu:
            import torch
            values = sample(a.gpu_id); check_memory(values)
            fraction = min(.75, (.79*values['gpu_total']-values['gpu_used']-512*2**20)/values['gpu_total'])
            if fraction <= .05: raise RuntimeError('Insufficient GPU headroom')
            torch.cuda.set_per_process_memory_fraction(fraction, 0)
        sys.argv = [a.module or a.script, *command_args]
        if a.module: runpy.run_module(a.module, run_name='__main__', alter_sys=True)
        else: runpy.run_path(a.script, run_name='__main__')
        return 0
    if a.log.exists() or a.log.with_suffix(a.log.suffix+'.status.json').exists():
        raise FileExistsError('Choose a new log path')
    a.log.parent.mkdir(parents=True, exist_ok=True)
    check_memory(sample(None if a.cpu else a.gpu_id))
    repo = Path(__file__).resolve().parents[1]
    def git(*args):
        try:
            r = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True)
            return r.stdout.strip() if r.returncode == 0 else None
        except OSError: return None
    versions = {}
    for package in ('torch', 'numpy', 'Pillow', 'OpenEXR'):
        try: versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package] = None
    provenance = {'source_commit': git('rev-parse', 'HEAD'), 'source_tree': git('rev-parse', 'HEAD^{tree}'),
                  'source_dirty': bool(git('status', '--porcelain')), 'python': platform.python_version(),
                  'packages': versions, 'entrypoint': a.module or a.script,
                  'source_sha256': {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for folder in ('src', 'tools') for p in sorted((repo/folder).rglob('*.py'))}}
    a.log.with_suffix(a.log.suffix+'.provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='' if a.cpu else a.gpu_id,
               CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUNBUFFERED='1')
    args = list(sys.argv[1:] if argv is None else argv)
    # --worker must precede REMAINDER.
    args.insert(0, '--worker')
    started = time.monotonic(); peaks = {}; reason = 'completed'; child = None
    with a.log.open('x') as log:
        child = subprocess.Popen([sys.executable, __file__, *args], env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                values = sample(None if a.cpu else a.gpu_id)
                for key, value in values.items(): peaks[key] = max(peaks.get(key, 0), value)
                check_memory(values)
                if time.monotonic()-started >= a.hours*3600:
                    raise TimeoutError('Wall deadline reached')
                time.sleep(1)
        except BaseException as error:
            reason = type(error).__name__+': '+str(error)
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
        code = child.wait()
    status = {'reason': reason, 'exit_code': code, 'elapsed_seconds': time.monotonic()-started,
              'peak_observed_bytes': peaks, 'memory_stop_fraction': .79,
              'note': 'One-second polling plus query time; unobserved peaks are not bounded.'}
    a.log.with_suffix(a.log.suffix+'.status.json').write_text(json.dumps(status, indent=2)+'\n')
    print(json.dumps(status, indent=2))
    return code if reason == 'completed' else 1


if __name__ == '__main__': raise SystemExit(main())
