"""Portable inference bundles in the dataset R2 bucket, with verified payloads."""
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
from urllib.parse import urlsplit

from .b2_archive import file_hashes
from .joint_checkpoint import load_joint_checkpoint, save_joint_checkpoint
from .r2_archive import R2SceneArchiver

FORMAT = 'wavefront-model-bundle-v1'


def safe_relative(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or '..' in path.parts or '\\' in name
            or path.as_posix() != name or name in ('.', 'manifest.json')):
        raise ValueError('Unsafe bundle member path')
    return path


def build_bundle(checkpoint, metadata, output, *, include_resume=False, results=None):
    """Keep original model/policies/provenance; strip Adam/RNG for inference."""
    checkpoint, metadata, output = Path(checkpoint), Path(metadata), Path(output)
    if output.exists(): raise FileExistsError('Bundle output already exists')
    model, payload = load_joint_checkpoint(checkpoint)
    source = json.loads((metadata/'r2-source.json').read_text())
    from .joint_data import joint_inventory
    inventory = joint_inventory(metadata, source['selected'], verify_images=False)
    expected = dict(payload['dataset_split'])
    # Streamed training adds transport fields to the same frozen inventory.
    for key in ('r2_snapshot_sha256', 'image_verification'): expected.pop(key, None)
    if expected != inventory:
        raise ValueError('Model inventory differs from metadata; export the exact training snapshot')
    output.mkdir(parents=True)
    save_joint_checkpoint(output/'weights.pt', model, config=payload['config'],
                          dataset_split=payload['dataset_split'], epoch=payload['epoch'],
                          global_step=payload['global_step'])
    from .adaptive_data import read_v5_dataset
    descriptor = read_v5_dataset(metadata)
    top_names = {'capture_lock.json', 'sample_layers_lock.json', 'decoded_cache_lock.json',
                 'dataset.json', 'split.json', 'r2-source.json', 'experiment-split.json'}
    directories = set(descriptor['groups']) | {'decoded', 'sample_layers', 'base', 'layers', 'frames',
                                               'sample_layer_configs', 'r2-archives'}
    for path in sorted(metadata.rglob('*.json')):
        relative = path.relative_to(metadata)
        if relative.as_posix() not in top_names and relative.parts[0] not in directories: continue
        target = output/'metadata'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    if include_resume: shutil.copyfile(checkpoint, output/'resume.pt')
    if results:
        for name in ('config.json', 'summary.json', 'inventory.json', 'history.jsonl'):
            path = Path(results)/name
            if path.is_file():
                (output/'evidence').mkdir(exist_ok=True)
                shutil.copyfile(path, output/'evidence'/name)
        for name in ('plan.json', 'train.log.status.json', 'train.log.provenance.json'):
            path = Path(results).parent/name
            if path.is_file():
                (output/'evidence').mkdir(exist_ok=True)
                shutil.copyfile(path, output/'evidence'/name)
    repo = Path(__file__).resolve().parents[2]
    def git(*args):
        try:
            r = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True)
            return r.stdout.strip() if r.returncode == 0 else None
        except OSError: return None
    manifest = {'format': FORMAT, 'source_checkpoint': file_hashes(checkpoint),
                'epoch': payload['epoch'], 'global_step': payload['global_step'],
                'model_spec': payload['model_spec'], 'dataset_source': source['source'],
                'selected': source['selected'], 'export_source_commit': git('rev-parse', 'HEAD'),
                'export_source_dirty': bool(git('status', '--porcelain')),
                'inference_weights': 'weights.pt', 'resume_checkpoint': 'resume.pt' if include_resume else None,
                'files': {p.relative_to(output).as_posix(): file_hashes(p)
                          for p in sorted(output.rglob('*')) if p.is_file()}}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True)+'\n')
    return manifest



def verify_bundle(bundle):
    bundle = Path(bundle)
    manifest = json.loads((bundle/'manifest.json').read_text())
    if manifest.get('format') != FORMAT or not isinstance(manifest.get('files'), dict):
        raise ValueError('Unknown model bundle format')
    for name, record in manifest['files'].items():
        safe_relative(name)
        if file_hashes(bundle/name) != record: raise ValueError('Local bundle checksum changed')
    return manifest


def upload_bundle(bundle, destination):
    bundle = Path(bundle)
    manifest = verify_bundle(bundle)
    data = urlsplit(manifest['dataset_source']); target = urlsplit(destination)
    if target.netloc != data.netloc: raise ValueError('Model destination must use the dataset bucket')
    a, b = data.path.strip('/'), target.path.strip('/')
    if not b or a == b or a.startswith(b+'/') or b.startswith(a+'/'):
        raise ValueError('Use a separate model-runs prefix; dataset prefix must stay untouched')
    client = R2SceneArchiver(bundle, destination).native
    client.preflight(); remote = client.list_files()
    inventory = dict(manifest['files'], **{'manifest.json': file_hashes(bundle/'manifest.json')})
    # Validate everything and detect collisions before any upload.
    for name, record in inventory.items():
        if name != 'manifest.json': safe_relative(name)
        if file_hashes(bundle/name) != record: raise ValueError('Local bundle checksum changed')
        if name in remote and not client.matches(remote[name], record):
            raise ValueError('R2 prefix contains different artifacts; choose a new run identity')
    for name, record in inventory.items():
        if name not in remote: client.upload_file(bundle/name, name, record, create_only=True)
    client.verify_files(inventory)
    receipt = {'destination': destination.rstrip('/'), 'verified': True,
               'manifest_sha256': inventory['manifest.json']['sha256'], 'files': len(inventory)}
    return receipt


def download_bundle(source, output):
    output = Path(output)
    if output.exists(): raise FileExistsError('Download output already exists')
    client = R2SceneArchiver(output, source).native
    client.preflight()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.model-download-', dir=output.parent) as temp:
        root = Path(temp)/'bundle'; root.mkdir()
        client.download_file('manifest.json', root/'manifest.json', max_bytes=16*2**20)
        manifest = json.loads((root/'manifest.json').read_text())
        if manifest.get('format') != FORMAT or not isinstance(manifest.get('files'), dict):
            raise ValueError('Unknown model bundle format')
        if len(manifest['files']) > 20000 or sum(v['size'] for v in manifest['files'].values()) > 10_000_000_000:
            raise ValueError('Bundle exceeds download limits')
        for name, record in manifest['files'].items():
            safe_relative(name)
            client.download_file(name, root/name, sha256=record['sha256'], size=record['size'])
        if not (root/'weights.pt').is_file() or not (root/'metadata/r2-source.json').is_file():
            raise ValueError('Incomplete inference bundle')
        root.rename(output)
    return manifest
