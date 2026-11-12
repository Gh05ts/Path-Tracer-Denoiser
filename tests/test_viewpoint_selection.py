from collections import Counter
from copy import deepcopy

import numpy as np
import pytest

from wavefront_denoiser.dataset_generation import DEFAULT_PLAN, MOTIONS, build_configs, read_plan
from wavefront_denoiser.viewpoint_selection import candidate_anchors, select_diverse


def test_reduced_plan_balances_all_motion_speed_pairs_and_preserves_captured_configs():
    original = build_configs(DEFAULT_PLAN.with_name("plan-v4.json"))
    configs = build_configs(DEFAULT_PLAN)
    reports = build_configs(DEFAULT_PLAN, reference_samples=4096)
    plan = read_plan(DEFAULT_PLAN)
    assert len(configs) == 162  # 144 target clips plus 18 already captured BMW extras.
    assert sum(len(c["frames"]) for c in configs.values()) == 1296
    for scene in plan["scenes"]:
        groups = {g: c for g, c in configs.items() if c["scene"] == scene["id"]}
        assert len(groups) == (36 if scene["id"] == "bmw" else 18)
        if scene["partition"] == "train" and scene["id"] != "bmw":
            assert len(scene["anchors"]) == 6
            pairs = Counter(tuple(pair) for a in scene["anchors"] for pair in a["motion_speeds"])
            assert pairs == Counter({(motion, speed): 1 for motion in MOTIONS for speed in ("slow", "medium", "fast")})
            assert all(len(a["motion_speeds"]) == 3 for a in scene["anchors"])
    for group, config in configs.items():
        if config["scene"] == "bmw" or group in ("bistro_view_00_linear_slow", "bistro_view_00_dolly_medium"):
            assert config == original[group]
        assert len(config["frames"]) == 8
        assert config["frames"] == reports[group]["frames"]
        assert config["sample_start"] == reports[group]["sample_start"]
        assert not 1_000_000_000 <= config["sample_start"] < 1_200_000_000
    spans = sorted((c["sample_start"], c["sample_start"]+8*c["sample_stride"]) for c in configs.values())
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:]))


def test_candidate_pool_is_larger_preserves_original_anchors_and_selection_is_deterministic():
    scene = read_plan(DEFAULT_PLAN.with_name("plan-v4.json"))["scenes"][1]
    before = deepcopy(scene)
    anchors = candidate_anchors(scene)
    assert scene == before
    assert len(anchors) == 24 and anchors[:6] == scene["anchors"]
    assert len({a["name"] for a in anchors}) == 24
    features = np.arange(24*10).reshape(24, 10)
    first, metrics = select_diverse(anchors, features, scene, pinned=["view_00"])
    assert first == select_diverse(anchors, features, scene, pinned=["view_00"])[0]
    assert first[0] == 0 and len(first) == len(set(first)) == 6
    assert metrics["minimum_pair_distance"] > 0
    with pytest.raises(ValueError, match="Pinned"):
        select_diverse(anchors, features, scene, pinned=["missing"])
    with pytest.raises(ValueError, match="Not enough"):
        select_diverse(anchors[:5], features[:5], scene)
