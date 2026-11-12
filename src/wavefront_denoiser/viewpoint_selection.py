"""Cheap candidate cameras and deterministic visual/pose diversity selection."""
from copy import deepcopy

import numpy as np

from .dataset_generation import trajectory, unit


def candidate_anchors(scene):
    """Keep original cameras and add modest translations around each one."""
    anchors = deepcopy(scene["anchors"])
    for anchor in scene["anchors"]:
        # pan_tilt's first frame has an off-axis target; recover the centre look.
        centre = trajectory(dict(scene, motion_scale=0), anchor, "pan_tilt")[0]
        forward = unit(np.asarray(centre["target"])-centre["position"])
        right = unit(np.cross(forward, centre["up"]))
        step = min(scene["focus_distance"]*.06, scene["motion_scale"]*2)
        for suffix, shift in (("left", -step*right), ("right", step*right),
                              ("forward", step*.7*forward)):
            camera = deepcopy(centre)
            for key in ("position", "target"):
                camera[key] = (np.asarray(camera[key])+shift).tolist()
            anchors.append({"name": f"{anchor['name']}_{suffix}", "camera": camera})
    return anchors


def descriptor(frame):
    """Use noise-free geometry/material guides, avoiding noisy radiance ranking."""
    buffers = frame.buffers
    valid = buffers["guide_valid"][..., 0] == 1
    depth = buffers["depth"][..., 0]
    positive = depth[valid & (depth > 0)]
    near = float(np.mean(positive < .1)) if positive.size else 1.
    # Broad spatial layout, rather than individual samples/texture pixels.
    parts = []
    for key in ("albedo", "normal"):
        value = buffers[key][::6, ::6].astype(float)
        parts.append(value.ravel())
    scaled_depth = np.where(valid, np.log1p(np.maximum(depth, 0)), 0)[::6, ::6]
    parts.append((scaled_depth/10).ravel())
    return np.concatenate(parts), {"near_surface_fraction": near,
                                  "guide_valid_fraction": float(valid.mean()),
                                  "eligible": bool(positive.size and near <= .25 and valid.mean() >= .15)}


def select_diverse(anchors, descriptors, scene, *, count=6, pinned=()):
    """Greedy farthest-point coverage with pinned cameras for existing captures."""
    if len(anchors) < count or len(descriptors) != len(anchors):
        raise ValueError("Not enough eligible candidate viewpoints")
    visual = np.asarray(descriptors)
    visual_distance = np.sqrt(np.mean((visual[:, None]-visual[None, :])**2, axis=-1))
    cameras = [trajectory(dict(scene, motion_scale=0), a, "pan_tilt")[0] for a in anchors]
    positions = np.array([c["position"] for c in cameras]) / scene["focus_distance"]
    directions = np.array([unit(np.asarray(c["target"])-c["position"]) for c in cameras])
    pose_distance = np.linalg.norm(positions[:, None]-positions[None, :], axis=-1)
    pose_distance += np.linalg.norm(directions[:, None]-directions[None, :], axis=-1)
    def normalized(matrix):
        scale = np.quantile(matrix[matrix > 0], .75) if np.any(matrix > 0) else 1.
        return np.minimum(matrix/max(scale, 1e-12), 2.)
    distance = .7*normalized(visual_distance) + .3*normalized(pose_distance)
    names = [a["name"] for a in anchors]
    if set(pinned)-set(names) or len(pinned) > count or len(set(pinned)) != len(pinned):
        raise ValueError("Pinned viewpoints must be unique eligible candidates")
    selected = [names.index(name) for name in pinned]
    if not selected:
        # Begin with a camera having broad aggregate distance from the pool.
        selected = [int(np.argmax(distance.mean(axis=1)))]
    while len(selected) < count:
        nearest = distance[:, selected].min(axis=1)
        nearest[selected] = -1
        selected.append(int(np.argmax(nearest)))
    return selected, {"method": "greedy_farthest_point", "visual_weight": .7, "pose_weight": .3,
                      "minimum_pair_distance": float(min(distance[i, j] for i in selected for j in selected if i != j))}
