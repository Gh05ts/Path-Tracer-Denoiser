"""Training sidecar for uniform-SPP NTASAD sequence datasets."""


def dataset_training_config() -> dict:
    return {"split_manifest": "split.json", "epochs": 1000,
            "patch_size": 128, "patches_per_window": 4, "window_length": 5,
            "batch_size": 4, "base_channels": 48, "learning_rate": .001,
            "validation_patch_size": 256, "validation_crops_per_window": 5, "seed": 7}


def joint_training_config(data_root=".") -> dict:
    """A separate executable v5 joint experiment; raw paper policy is opt-in."""
    return {"format":"wavefront-ntasad-joint-training","version":1,"data_root":data_root,
        "epochs":1000,"batch_size":4,"patch_size":128,"patches_per_window":4,"window_length":5,
        "learning_rate":.001,"seed":7,"denoiser_channels":48,"sampler_channels":48,
        "average_spp":4,"max_spp":63,"allocation_policy":"capped_softmax_v1",
        "zero_policy":"unit_denominator","validation_patch_size":256,"validation_crops_per_window":5,
        "reconstruction_version":"pixel_mean_v2",
        "num_workers":0,"cpu_threads":4,"train_groups":None,"validation_groups":None}
