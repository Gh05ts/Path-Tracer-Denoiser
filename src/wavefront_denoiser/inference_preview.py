"""Scene selection, directory isolation and fixed-scale allocation displays."""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


SCENE_PARTITIONS = {'bathroom': 'validation', 'classroom': 'validation', 'sponza': 'test'}


def separate_directories(*paths):
    """Reject equal/nested paths, including aliases through existing symlinks."""
    resolved = [Path(path).resolve() for path in paths]
    for index, first in enumerate(resolved):
        for second in resolved[index+1:]:
            if first == second or first in second.parents or second in first.parents:
                raise ValueError('Model, download and preview directories must be separate (not nested)')


def select_scene_groups(descriptor, selected_validation, scenes):
    """One reproducible clip per requested scene, with the fixed Sponza camera."""
    if not scenes or len(set(scenes)) != len(scenes):
        raise ValueError('Choose unique scenes: bathroom, classroom, sponza')
    groups = []
    for scene in scenes:
        if scene not in SCENE_PARTITIONS: raise ValueError(f'Unknown preview scene: {scene}')
        partition = SCENE_PARTITIONS[scene]
        allowed = selected_validation if partition == 'validation' else descriptor['partitions']['test']
        candidates = sorted(g for g in allowed if descriptor['groups'][g]['scene'] == scene)
        if scene == 'sponza': candidates = [g for g in candidates if g == 'sponza_view_00_circle_slow']
        if not candidates: raise ValueError(f'No compatible {partition} clip for scene {scene}')
        groups.append(candidates[0])
    return groups


def allocation_images(counts, max_spp):
    """Raw linear grayscale plus white-to-blue log-scale actual integer SPP."""
    values = np.asarray(counts)
    if (values.ndim != 2 or values.size == 0 or max_spp < 1 or
            not np.issubdtype(values.dtype, np.integer) or (values < 0).any() or (values > max_spp).any()):
        raise ValueError('Sample counts must be a nonempty integer HW map in [0, max_spp]')
    gray = np.rint(values.astype(np.float32)*255/max_spp).astype(np.uint8)
    position = np.log2(1+values.astype(np.float32))/np.log2(1+max_spp)
    anchors = np.array([[247, 251, 255], [186, 216, 238], [92, 164, 212],
                        [33, 113, 181], [8, 48, 107]], dtype=np.float32)
    colors = np.stack([np.interp(position, np.linspace(0, 1, len(anchors)), anchors[:, channel])
                       for channel in range(3)], axis=-1)
    return {'sample-count': Image.fromarray(gray),
            'allocation-heatmap': Image.fromarray(np.rint(colors).astype(np.uint8))}


def allocation_legend(max_spp):
    """Same logarithmic normalization as every frame; never auto-normalize."""
    width = 640
    counts = np.rint(np.expm1(np.linspace(0, np.log1p(max_spp), width))).astype(np.int64)[None]
    bar = allocation_images(counts, max_spp)['allocation-heatmap'].resize((width, 24))
    canvas = Image.new('RGB', (width+32, 84), 'white'); canvas.paste(bar, (16, 26))
    draw = ImageDraw.Draw(canvas)
    draw.text((16, 6), 'Actual samples/pixel | darker blue = more | fixed log2(1 + SPP) scale', fill='black')
    for count in sorted({0, 1, 2, 4, 8, 16, 32, max_spp}):
        if count > max_spp: continue
        x = 16+round(np.log1p(count)/np.log1p(max_spp)*(width-1))
        draw.line((x, 50, x, 55), fill='black')
        draw.text((min(x, width), 59), str(count), fill='black')
    return canvas
