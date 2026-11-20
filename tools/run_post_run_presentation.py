#!/usr/bin/env python3
"""Render documented representative clips and back up their artifacts to R2."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

from wavefront_denoiser.b2_archive import file_hashes
from wavefront_denoiser.model_artifacts import download_bundle, verify_bundle
from wavefront_denoiser.r2_native import R2NativeClient

REPO = Path(__file__).resolve().parents[1]
MODEL_R2 = 'r2://denoiser/model-runs/final-01-selected'
TEST_GROUP = 'sponza_view_00_circle_slow'


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def read_json(path):
    return json.loads(path.read_text())


def checked_report(root, stage, checkpoint_hash):
    folder = root/stage
    report = read_json(folder/'presentation.json')
    status = read_json(root/(stage+'.log.status.json'))
    if (not report.get('complete') or report.get('checkpoint_sha256') != checkpoint_hash
            or status.get('reason') != 'completed' or status.get('exit_code') != 0):
        raise ValueError(f'{stage}: incomplete run or checkpoint mismatch')
    expected = (['bathroom_view_00_circle_slow', 'classroom_view_00_circle_slow']
                if stage == 'validation' else [TEST_GROUP])
    if sorted(report['groups']) != expected:
        raise ValueError(f'{stage}: unexpected representative clip selection')
    for clip in report['clips']:
        if clip['partition'] != ('validation' if stage == 'validation' else 'test'):
            raise ValueError('Validation/test partition mismatch')
        frames = clip['frames']
        if len(frames) != 8 or [r['frame'] for r in frames] != list(range(8)):
            raise ValueError('Expected eight ordered frames per documented clip')
        for frame in frames:
            if frame['adaptive_paths'] != frame['uniform_paths'] or frame['actual_spp'] != 4:
                raise ValueError('Presentation path budgets differ')
        for name in ('index.md', 'evaluation-protocol.json'):
            if not (folder/name).is_file():
                raise ValueError(f'Missing {name}')
        for name in ('comparison-native.png', 'comparison.gif', 'controls.png', 'metrics.json',
                     'adaptive-denoised.exr', 'uniform-denoised.exr', 'reference.exr'):
            if not (folder/clip['group']/name).is_file():
                raise ValueError(f'Missing clip artifact: {name}')
    return report


def render(root, stage, checkpoint_hash, hours, chunk_rows):
    output = root/stage
    log = root/(stage+'.log')
    if output.exists() or log.exists():
        # A completed stage can be reused after an upload failure. Preserve any
        # partial output and require a new run name for an interrupted render.
        try:
            return checked_report(root, stage, checkpoint_hash)
        except (OSError, ValueError, KeyError) as error:
            raise RuntimeError(f'{stage} is incomplete; choose a new --name and retain this attempt') from error
    command = [sys.executable, str(REPO/'tools/run_with_memory_guard.py'),
               '--hours', str(hours), '--log', str(log), '--script',
               str(REPO/'tools/generate_presentation.py'), '--', '--bundle', str(root/'model'),
               '--output', str(output), '--device', 'cuda', '--hdr',
               '--kernel-chunk-rows', str(chunk_rows)]
    if stage == 'sponza':
        command += ['--include-test', '--groups', TEST_GROUP]
    print('Rendering:', stage, flush=True)
    subprocess.run(command, cwd=REPO, check=True)
    return checked_report(root, stage, checkpoint_hash)


def package(root, upload_dir):
    archive = upload_dir/'presentation.tar.gz'
    manifest_path = upload_dir/'manifest.json'
    if archive.exists() or manifest_path.exists():
        manifest = read_json(manifest_path)
        if file_hashes(archive) != manifest['archive']:
            raise ValueError('Existing presentation archive changed')
        for name, record in manifest['files'].items():
            if file_hashes(root/name) != record:
                raise ValueError('Presentation outputs changed after packaging')
        return archive, manifest_path
    upload_dir.mkdir(parents=True, exist_ok=True)
    files = {p.relative_to(root).as_posix(): file_hashes(p)
             for p in sorted(root.rglob('*')) if p.is_file()}
    partial = upload_dir/'presentation.tar.gz.partial'
    print('Packaging', len(files), 'files', flush=True)
    with tarfile.open(partial, 'w:gz', compresslevel=1) as tar:
        for name in files:
            tar.add(root/name, arcname=root.name+'/'+name, recursive=False)
    # Check the archive payload itself, not just the source files.
    with tarfile.open(partial, 'r:gz') as tar:
        members = tar.getmembers()
        if len(members) != len(files):
            raise ValueError('Archive member count differs')
        for member in members:
            relative = Path(member.name).relative_to(root.name).as_posix()
            expected = files[relative]
            digest = hashlib.sha256()
            with tar.extractfile(member) as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    digest.update(block)
            if member.size != expected['size'] or digest.hexdigest() != expected['sha256']:
                raise ValueError('Archive payload verification failed')
    partial.rename(archive)
    write_json(manifest_path, {
        'format': 'wavefront-presentation-archive-v1', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'root_directory': root.name, 'model_r2': MODEL_R2,
        'checkpoint_sha256': read_json(root/'validation/presentation.json')['checkpoint_sha256'],
        'epoch': read_json(root/'validation/presentation.json')['epoch'],
        'coverage': 'Representative presentation: two validation clips and one fixed Sponza test clip',
        'metrics_policy': 'Keep validation and test separate; this is not full-split evaluation',
        'files': files, 'archive': file_hashes(archive),
    })
    return archive, manifest_path


def upload(archive, manifest_path, name):
    prefix = 'presentation-runs/'+name
    client = R2NativeClient('denoiser', prefix)
    client.preflight()
    inventory = {p.name: file_hashes(p) for p in (archive, manifest_path)}
    remote = client.list_files()
    for filename, record in inventory.items():
        if filename in remote and not client.matches(remote[filename], record):
            raise ValueError('Remote contents differ; choose a new --name')
    for path in (archive, manifest_path):
        if path.name not in remote:
            print('Uploading:', path.name, inventory[path.name]['size'], 'bytes', flush=True)
            client.upload_file(path, path.name, inventory[path.name], create_only=True)
    client.verify_files(inventory)
    # Read back both objects and verify SHA-256 before declaring success.
    with tempfile.TemporaryDirectory(prefix='presentation-recovery-', dir=archive.parent) as temp:
        for filename, record in inventory.items():
            print('Verifying recovery:', filename, flush=True)
            client.download_file(filename, Path(temp)/filename, sha256=record['sha256'], size=record['size'])
    receipt = {'destination': 'r2://denoiser/'+prefix, 'verified': True,
               'remote_readback_sha256_verified': True, 'files': inventory,
               'completed_utc': datetime.now(timezone.utc).isoformat()}
    write_json(archive.parent/'upload-receipt.json', receipt)
    print(json.dumps(receipt, indent=2), flush=True)


def snapshot_source(root):
    source_dir = root/'inference-source'
    sources = sorted((REPO/'src').rglob('*.py'))
    sources += [REPO/name for name in (
        'tools/generate_presentation.py', 'tools/run_with_memory_guard.py',
        'tools/run_post_run_presentation.py', 'tools/launch_post_run_presentation.sh',
        'docs/post-run-presentation.md', 'docs/checkpoint-selection-and-r2-use.md', 'pyproject.toml')]
    for source in sources:
        target = source_dir/source.relative_to(REPO)
        if target.exists():
            if source.read_bytes() != target.read_bytes():
                raise ValueError('Inference source changed; choose a new --name')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='final-01-presentation-01',
                        help='New artifact identity; a completed render can be reused to retry upload')
    parser.add_argument('--run-root', type=Path,
                        default=Path(os.environ.get('QUALITY_RUN_ROOT', str(REPO.parent/'runs'))))
    parser.add_argument('--final-root', type=Path, default=REPO.parent/'runs/final-01')
    parser.add_argument('--hours', type=float, default=2, help='Guard deadline per rendering stage')
    parser.add_argument('--kernel-chunk-rows', type=int, default=64)
    args = parser.parse_args()
    if not args.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in args.name):
        parser.error('--name must use letters, digits, hyphens or underscores')
    root = args.run_root.resolve()/args.name
    upload_dir = args.run_root.resolve()/(args.name+'-upload')
    # Check credentials before creating a run or using GPU time.
    client = R2NativeClient('denoiser', 'presentation-runs/'+args.name)
    client.preflight()
    root.mkdir(parents=True, exist_ok=True)
    # Hold this lock throughout rendering, packaging and upload.
    lock = (root/'.pipeline.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError('A presentation pipeline with this name is already running') from None
    snapshot_source(root)
    if not (root/'model').exists():
        print('Restoring verified model from', MODEL_R2, flush=True)
        download_bundle(MODEL_R2, root/'model')
    verify_bundle(root/'model')
    checkpoint_hash = hashlib.sha256((root/'model/weights.pt').read_bytes()).hexdigest()
    render(root, 'validation', checkpoint_hash, args.hours, args.kernel_chunk_rows)
    render(root, 'sponza', checkpoint_hash, args.hours, args.kernel_chunk_rows)
    # Preserve both invocations' actual guard records; model export only copied
    # the original train.log.* filenames.
    evidence = root/'training-evidence'
    evidence.mkdir(exist_ok=True)
    final_root = args.final_root.resolve()
    for name in ('plan.json', 'train.json', 'train.log', 'train.log.status.json', 'train.log.provenance.json',
                 'train-resume-01.log', 'train-resume-01.log.status.json', 'train-resume-01.log.provenance.json',
                 'train/config.json', 'train/summary.json', 'train/history.jsonl', 'train/inventory.json',
                 'train-resume-01/config.json', 'train-resume-01/summary.json',
                 'train-resume-01/history.jsonl', 'train-resume-01/inventory.json'):
        source = final_root/name
        if source.is_file() and not (evidence/name).exists():
            (evidence/name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, evidence/name)
    snapshot_source(root)
    archive, manifest_path = package(root, upload_dir)
    upload(archive, manifest_path, args.name)
    print('COMPLETE. Validation index:', root/'validation/index.md', flush=True)
    print('Test index:', root/'sponza/index.md', flush=True)
    print('Verified upload receipt:', upload_dir/'upload-receipt.json', flush=True)


if __name__ == '__main__':
    main()
