"""Training sidecar for uniform-SPP NTASAD sequence datasets."""


def dataset_training_config() -> dict:
    return {"split_manifest": "split.json", "epochs": 1000,
            "patch_size": 128, "patches_per_window": 4, "window_length": 5,
            "batch_size": 4, "base_channels": 48, "learning_rate": .001,
            "validation_patch_size": 256, "validation_crops_per_window": 5, "seed": 7}
