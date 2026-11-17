"""Metadata-only held-out audits, separate from training inventories."""
from pathlib import Path

from .adaptive_data import read_json, read_v5_dataset
from .dataset_generation import digest
from .decoded_cache import verify_decoded_frame


def heldout_inventory(root, groups, trained_inventory):
    """Bind test frames and reject identity/content overlap without reading pixels."""
    root = Path(root)
    descriptor = read_v5_dataset(root)
    if (descriptor.get('storage') != 'decoded'
            or descriptor['partitions'] != trained_inventory['partitions']
            or digest(root/'decoded_cache_lock.json') != trained_inventory['dataset_manifest_sha256']):
        raise ValueError('Held-out data differs from trained snapshot')
    if (not groups or len(set(groups)) != len(groups)
            or set(groups)-set(descriptor['partitions']['test'])):
        raise ValueError('Held-out selection must contain reserved test clips only')
    used = trained_inventory['train']+trained_inventory['validation']
    identities = {r['scene_fingerprint'] for r in used}
    images = {r[k] for r in used for k in ('source_image_sha256', 'layer_image_sha256')}
    reserved = {(r['sequence_id'], r['frame_index']): r['manifest_sha256']
                for r in trained_inventory['reserved_test_metadata']}
    records = []
    for group in groups:
        identity = None
        for index in range(descriptor['groups'][group]['frame_count']):
            manifest = root/group/f'frame_{index:06d}'/'manifest.json'
            if (group, index) in reserved and digest(manifest) != reserved[group, index]:
                raise ValueError('Reserved test manifest changed since training snapshot')
            base, decoded = verify_decoded_frame(root, group, index, verify_hashes=False, allow_missing=True)
            layer = read_json(root/'sample_layers'/group/f'frame_{index:06d}'/'manifest.json')
            scene = base.get('scene_fingerprint')
            if (not isinstance(scene, str) or not scene or any(c not in '0123456789abcdef' for c in scene)
                    or (identity is not None and identity != scene)):
                raise ValueError('Invalid or changing held-out scene identity')
            identity = scene
            if scene in identities or base['image_sha256'] in images or layer['image_sha256'] in images:
                raise ValueError('Reserved test identity/content crosses trained partitions')
            records.append({'sequence_id': group, 'frame_index': index, 'scene_fingerprint': scene,
                            'completion_sha256': digest(root/'decoded'/group/f'frame_{index:06d}'/'manifest.json'),
                            'source_image_sha256': base['image_sha256'],
                            'layer_image_sha256': layer['image_sha256'], 'arrays': decoded['arrays']})
    return {'purpose': 'heldout_test', 'selected': {'test': groups}, 'frames': records,
            'dataset_manifest_sha256': trained_inventory['dataset_manifest_sha256'],
            'test_pixels_loaded': False}
