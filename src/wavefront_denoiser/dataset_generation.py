"""Deterministic camera plans and frame-resumable, provenance-locked captures."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import time
from urllib.parse import unquote

import numpy as np

from .dataset_splits import audit_dataset_split
from .frames import load_frame, read_manifest

MOTIONS = ("linear", "dolly", "wave", "circle", "orbit", "pan_tilt")
PARTITIONS = ("train", "validation", "test")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = PROJECT_ROOT.parent / "dataset"
DEFAULT_PLAN = PROJECT_ROOT / "dataset_generation" / "plan-v2.json"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def unit(vector):
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all() or np.linalg.norm(value) < 1e-10:
        raise ValueError("Camera vectors must be finite, nonzero triples")
    return value / np.linalg.norm(value)


def yaw(vector, degrees):
    angle = math.radians(degrees)
    x, y, z = vector
    return np.array([math.cos(angle)*x + math.sin(angle)*z, y,
                     -math.sin(angle)*x + math.cos(angle)*z])


def trajectory(scene: dict, anchor: dict, motion: str, count: int = 8, *, motion_multiplier: float = 1.) -> list[dict]:
    """Small local loops/arcs, never a 360-degree orbit around an interior."""
    if motion not in MOTIONS or type(count) is not int or count < 5:
        raise ValueError("Need a supported motion and at least five frames")
    if type(motion_multiplier) not in (int, float) or not math.isfinite(motion_multiplier) or motion_multiplier <= 0:
        raise ValueError("Motion multiplier must be finite and positive")
    camera = scene["camera"]
    origin = np.asarray(camera["position"], dtype=np.float64)
    forward = unit(np.asarray(camera["target"]) - origin)
    right = unit(np.cross(forward, unit(camera["up"])))
    up = unit(np.cross(right, forward))
    offset = anchor["offset"]
    origin = origin + offset[0]*right + offset[1]*up + offset[2]*forward
    forward = unit(yaw(forward, anchor.get("yaw_degrees", 0)))
    right = unit(np.cross(forward, up))
    up = unit(np.cross(right, forward))
    focus, amplitude = scene["focus_distance"], scene["motion_scale"]*motion_multiplier
    target = origin + focus*forward
    frames = []
    for t in np.linspace(0, 1, count):
        s, phase = 2*t-1, 2*math.pi*t
        position, look = origin.copy(), target.copy()
        if motion == "linear":
            delta = amplitude*s*right
            position += delta
            look += delta
        elif motion == "dolly":
            delta = amplitude*s*forward
            position += delta
            look += delta
        elif motion == "wave":
            delta = amplitude*(s*right + .4*math.sin(phase)*up + .25*math.sin(math.pi*t)*forward)
            position += delta
            look += delta
        elif motion == "circle":
            position += amplitude*(math.cos(phase)*right + .4*math.sin(phase)*up)
        elif motion == "orbit":
            # Arc length is limited by the same scene-specific motion scale.
            position = target + yaw(origin-target, math.degrees(amplitude/focus)*s)
        elif motion == "pan_tilt":
            look += amplitude*(s*right + .35*math.sin(math.pi*s)*up)
        frames.append({"position": position.tolist(), "target": look.tolist(),
                       "up": up.tolist(), "vertical_fov_degrees": camera["vertical_fov_degrees"]})
    return frames


def read_plan(path: Path) -> dict:
    plan = json.loads(path.read_text())
    if plan.get("format") != "wavefront-denoiser-capture-plan" or plan.get("version") != 1:
        raise ValueError("Unsupported capture plan")
    for key in ("width", "height", "frames_per_clip", "samples", "reference_samples",
                "report_reference_samples", "sample_stride", "clip_sample_stride", "max_depth",
                "russian_roulette_start_depth", "tile_width", "tile_height"):
        if type(plan.get(key)) is not int or plan[key] < 1:
            raise ValueError(f"Invalid positive plan integer: {key}")
    if (type(plan.get("sample_start")) is not int or plan["sample_start"] < 0
            or plan.get("caustics") is not False or plan["frames_per_clip"] != 8):
        raise ValueError("This dataset requires eight-frame clips, nonnegative sample_start and caustics off")
    if plan["sample_stride"] < plan["samples"] + max(plan["reference_samples"], plan["report_reference_samples"]):
        raise ValueError("Sample stride must accommodate the report reference budget")
    if plan["clip_sample_stride"] < plan["frames_per_clip"] * plan["sample_stride"]:
        raise ValueError("Clip sample ranges overlap")
    if not plan["motions"] or len(set(plan["motions"])) != len(plan["motions"]) or set(plan["motions"])-set(MOTIONS):
        raise ValueError("Invalid motion selection")
    if "motion_profiles" in plan:
        profiles = plan["motion_profiles"]
        if (not isinstance(profiles, dict) or set(profiles) != {"slow", "medium", "fast"}
                or any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in profiles.values())
                or not profiles["slow"] < profiles["medium"] < profiles["fast"]):
            raise ValueError("Need increasing positive slow/medium/fast motion profiles")
    seen, partitions = set(), set()
    for scene in plan["scenes"]:
        name = scene["id"]
        if not isinstance(name, str) or not name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789_" for c in name) or name in seen:
            raise ValueError("Scene IDs must be unique lowercase identifiers")
        seen.add(name)
        if scene["partition"] not in PARTITIONS or not scene["anchors"]:
            raise ValueError("Each scene needs a partition and camera anchors")
        partitions.add(scene["partition"])
        manifest = json.loads((path.parent / scene["scene_file"]).read_text())
        if manifest.get("name") != name:
            raise ValueError("Scene manifest name must match the plan")
        if (not 0 < scene["motion_scale"] < scene["focus_distance"]
                or not 0 < scene["camera"]["vertical_fov_degrees"] < 179):
            raise ValueError("Invalid camera scale or FOV")
        if len({a["name"] for a in scene["anchors"]}) != len(scene["anchors"]):
            raise ValueError("Camera anchors must have unique names")
        for anchor in scene["anchors"]:
            frames = trajectory(scene, anchor, plan["motions"][0])
            if not np.isfinite([f["position"] for f in frames]).all():
                raise ValueError("Non-finite camera trajectory")
    if partitions != set(PARTITIONS):
        raise ValueError("Declare independent train, validation and test scenes")
    return plan


def build_configs(plan_path: Path, *, width=None, height=None, reference_samples=None) -> dict[str, dict]:
    plan = read_plan(plan_path)
    configs = {}
    for scene in plan["scenes"]:
        for anchor_index, anchor in enumerate(scene["anchors"]):
            for motion_index, motion in enumerate(plan["motions"]):
                group = f"{scene['id']}_{anchor['name']}_{motion}"
                multiplier = 1.
                if "motion_profiles" in plan:
                    profile = ("slow", "medium", "fast")[(anchor_index+motion_index) % 3]
                    multiplier = plan["motion_profiles"][profile]
                    group += "_"+profile
                config = {k: plan[k] for k in ("max_depth", "russian_roulette_start_depth", "samples",
                          "sample_stride", "tile_width", "tile_height", "caustics")}
                config.update(sequence_id=group, scene=scene["id"],
                    scene_file=str((plan_path.parent / scene["scene_file"]).resolve()),
                    width=width or plan["width"], height=height or plan["height"],
                    reference_samples=reference_samples or plan["reference_samples"],
                    sample_start=plan["sample_start"] + len(configs)*plan["clip_sample_stride"],
                    frames=trajectory(scene, anchor, motion, plan["frames_per_clip"], motion_multiplier=multiplier))
                if config["reference_samples"] + config["samples"] > config["sample_stride"]:
                    raise ValueError("Reference budget exceeds the locked sample stride")
                if config["sample_start"] + len(config["frames"])*config["sample_stride"] > 0xffffffff:
                    raise ValueError("Capture sample range exceeds uint32")
                configs[group] = config
    return configs


def asset_files(scene_file: Path) -> set[Path]:
    """All referenced bytes, including external buffers/textures omitted by scene fingerprints."""
    files = {scene_file.resolve()}
    scene = json.loads(scene_file.read_text())
    source = scene.get("object", {}).get("file")
    if source:
        source = (scene_file.parent / source).resolve()
        files.add(source)
        if source.suffix.lower() == ".gltf":
            model = json.loads(source.read_text())
        elif source.suffix.lower() == ".glb":
            with source.open("rb") as stream:
                if stream.read(4) != b"glTF":
                    raise ValueError(f"Invalid GLB: {source}")
                stream.read(8)
                size, kind = struct.unpack("<II", stream.read(8))
                if kind != 0x4e4f534a:
                    raise ValueError("GLB must begin with JSON")
                model = json.loads(stream.read(size))
        else:
            raise ValueError("This dataset supports glTF/GLB geometry")
        for item in model.get("buffers", []) + model.get("images", []):
            uri = item.get("uri", "")
            if uri and not uri.startswith("data:"):
                if "://" in uri:
                    raise ValueError("Download remote asset dependencies before capture")
                files.add((source.parent / unquote(uri)).resolve())
    environment = scene.get("environment", {}).get("file")
    if environment:
        files.add((scene_file.parent / environment).resolve())
    for file in files:
        if not file.is_file():
            raise FileNotFoundError(file)
        with file.open("rb") as stream:
            if stream.read(44).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError(f"Unresolved Git LFS asset: {file}")
    return files


def split_spec(plan: dict, configs: dict, output: Path) -> dict:
    membership = {scene["id"]: scene["partition"] for scene in plan["scenes"]}
    return {"format": "wavefront-dataset-split", "version": 1, "data_root": ".",
            "require_reference_policy": False,
            "declared_reference_samples": next(iter(configs.values()))["reference_samples"],
            "reference_status": "fixed_budget_not_convergence_qualified",
            "partitions": {p: [g for g, c in configs.items() if membership[c["scene"]] == p] for p in PARTITIONS},
            "groups": {g: {"scene": c["scene"], "family": c["scene"], "frame_count": len(c["frames"]),
                            "capture_config": f"capture_configs/{g}.json"} for g, c in configs.items()}}


def prepare(plan_path: Path, output: Path, configs: dict, executable: Path | None) -> dict:
    plan = read_plan(plan_path)
    assets = set().union(*(asset_files(Path(p)) for p in {c["scene_file"] for c in configs.values()}))
    print(f"Checking {len(assets)} referenced asset files...", flush=True)
    lock = {"format": "wavefront-capture-lock", "version": 1, "plan": plan,
            "configs": configs, "assets": {str(p): digest(p) for p in sorted(assets)},
            "renderer_sha256": digest(executable) if executable else None}
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / "capture_lock.json"
    if lock_path.exists():
        previous = json.loads(lock_path.read_text())
        if executable is None:
            lock["renderer_sha256"] = previous["renderer_sha256"]
        elif previous["renderer_sha256"] is None and not list(output.glob("*/frame_*")):
            previous["renderer_sha256"] = lock["renderer_sha256"]
        if previous != lock:
            raise ValueError("Capture plan, assets or renderer changed. Use a new output directory.")
        write_json(lock_path, lock)
    else:
        if list(output.glob("*/frame_*")):
            raise ValueError("Refusing to adopt captures without a provenance lock")
        write_json(lock_path, lock)
    for group, config in configs.items():
        path = output / "capture_configs" / f"{group}.json"
        if path.exists() and json.loads(path.read_text()) != config:
            raise ValueError(f"Stored capture config was modified: {path}")
        write_json(path, config)
    spec = split_spec(plan, configs, output)
    split_path = output / "split.json"
    if split_path.exists() and json.loads(split_path.read_text()) != spec:
        raise ValueError("Stored split manifest was modified")
    write_json(split_path, spec)
    training_configs = {"train-spatial.json": {"split_manifest": "split.json", "require_reference_policy": False,
        "epochs": 40, "patch_size": 128, "patches_per_frame": 8, "batch_size": 4,
        "base_channels": 16, "seed": 7}, "train-recurrent.json": {"split_manifest": "split.json", "epochs": 1000,
        "patch_size": 128, "patches_per_window": 4, "window_length": 5, "batch_size": 4,
        "base_channels": 48, "validation_patch_size": 256, "validation_crops_per_window": 5, "seed": 7}}
    for name, config in training_configs.items():
        if not (output / name).exists():
            write_json(output / name, config)
    if all(len(c["frames"]) == 8 for c in configs.values()):
        cases = []
        for scene in plan["scenes"]:
            if scene["partition"] == "test":
                continue
            group = next(g for g, c in configs.items() if c["scene"] == scene["id"])
            for index in (0, 7):
                capture = dict(configs[group])
                capture.pop("sequence_id")
                capture.pop("sample_start")
                capture["frames"] = [configs[group]["frames"][index]]
                cases.append({"id": f"{group}_f{index}", "group": group, "capture": capture})
        write_json(output / "reference-check-plan.json", {"format": "wavefront-reference-plan", "version": 1,
            "split_manifest": "split.json", "sample_counts": [256, 1024, 4096], "replicates": 2,
            "sample_start": 1000000000, "max_noise_fraction": .1, "cases": cases})
    counts = {p: {"clips": len(spec["partitions"][p]),
                  "frames": sum(len(configs[g]["frames"]) for g in spec["partitions"][p])} for p in PARTITIONS}
    summary = {"partitions": counts, "width": next(iter(configs.values()))["width"],
               "height": next(iter(configs.values()))["height"], "reference_samples": spec["declared_reference_samples"],
               "reference_status": spec["reference_status"], "raw_bytes_per_frame": next(iter(configs.values()))["width"] * next(iter(configs.values()))["height"] * 88,
               "guides": ["albedo", "normal", "geometric_normal", "depth", "motion", "guide_valid", "history_valid", "sample_count", "reference_count"]}
    write_json(output / "plan-summary.json", summary)
    return spec


@contextmanager
def capture_lock(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".capture.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another capture process is using this output directory") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def expected_camera(frame: dict, config: dict) -> dict:
    forward = unit(np.asarray(frame["target"])-np.asarray(frame["position"]))
    right = unit(np.cross(forward, frame["up"]))
    return {"position": frame["position"], "forward": forward, "right": right,
            "up": np.cross(right, forward), "vertical_fov_radians": math.radians(frame["vertical_fov_degrees"]),
            "aspect_ratio": config["width"]/config["height"]}


def verify_frame(directory: Path, config: dict, index: int, *, pixels=False) -> dict:
    metadata = read_manifest(directory)
    render = metadata["render"]
    start = config["sample_start"] + index*config["sample_stride"]
    expected = {"sample_start": start, "reference_sample_start": start+config["samples"],
                "samples": config["samples"], "reference_samples": config["reference_samples"],
                "sample_stride": config["sample_stride"], "max_depth": config["max_depth"],
                "russian_roulette_start_depth": config["russian_roulette_start_depth"], "caustics": False}
    if (metadata["sequence_id"] != config["sequence_id"] or metadata["scene"] != config["scene"]
            or metadata["frame_index"] != index or metadata["previous_frame_index"] != (index-1 if index else None)
            or metadata["width"] != config["width"] or metadata["height"] != config["height"]
            or any(render.get(key) != value for key, value in expected.items())):
        raise ValueError(f"Frame metadata disagrees with the locked capture: {directory}")
    for name, frame_index in (("camera", index), ("previous_camera", index-1)):
        if frame_index < 0:
            continue
        camera = expected_camera(config["frames"][frame_index], config)
        if any(not np.allclose(metadata[name][key], value, rtol=2e-5, atol=2e-5) for key, value in camera.items()):
            raise ValueError(f"Camera disagrees with capture plan: {directory}")
    if pixels:
        load_frame(directory)
    elif digest(directory / metadata["image"]) != metadata["image_sha256"]:
        raise ValueError(f"Completed EXR checksum mismatch: {directory}")
    return metadata


def missing_ranges(indices: list[int]) -> list[tuple[int, int]]:
    ranges = []
    for i in indices:
        if ranges and ranges[-1][1] == i:
            ranges[-1] = ranges[-1][0], i+1
        else:
            ranges.append((i, i+1))
    return ranges


def run_captures(configs: dict, groups: list[str], output: Path, executable: Path, renderer_root: Path,
                 *, first_frame: int = 0, end_frame: int | None = None) -> dict:
    captured, skipped = 0, 0
    records = []
    for ordinal, group in enumerate(groups, 1):
        config = configs[group]
        directory = output / group
        missing = []
        stop = len(config["frames"]) if end_frame is None else end_frame
        if not 0 <= first_frame < stop <= len(config["frames"]):
            raise ValueError("Frame range must lie inside the planned clip")
        for i in range(first_frame, stop):
            frame = directory / f"frame_{i:06d}"
            if frame.exists():
                verify_frame(frame, config, i)
                skipped += 1
            else:
                missing.append(i)
                partial = frame.with_name(frame.name+".partial")
                if partial.exists():
                    recovery = output / "recovery.partial" / group
                    recovery.mkdir(parents=True, exist_ok=True)
                    partial.rename(recovery / f"{partial.name}-{time.time_ns()}")
        print(f"[{ordinal}/{len(groups)}] {group}: {len(missing)} frames to capture", flush=True)
        for first, end in missing_ranges(missing):
            logs = output / "logs"
            logs.mkdir(exist_ok=True)
            log_path = logs / f"{group}.log"
            before = time.monotonic()
            with log_path.open("a") as log:
                command = [str(executable), str(output / "capture_configs" / f"{group}.json"),
                           str(directory), "--start-frame", str(first), "--end-frame", str(end)]
                log.write(json.dumps({"command": command, "started_unix": time.time()})+"\n")
                log.flush()
                try:
                    subprocess.run(command, cwd=renderer_root, stdout=log, stderr=subprocess.STDOUT, check=True)
                except subprocess.CalledProcessError as error:
                    raise RuntimeError(f"Capture failed; completed frames are retained. See {log_path}") from error
            for i in range(first, end):
                metadata = verify_frame(directory / f"frame_{i:06d}", config, i, pixels=True)
                captured += 1
                records.append({"group": group, "frame": i, "image_sha256": metadata["image_sha256"]})
            print(f"  Completed {end-first} frames in {time.monotonic()-before:.1f}s; log: {log_path}", flush=True)
    result = {"captured_frames": captured, "skipped_frames": skipped, "groups": groups,
              "frame_range": [first_frame, end_frame],
              "new_frames": records, "reference_status": "fixed_budget_not_convergence_qualified"}
    write_json(output / "last-capture.json", result)
    return result


def contact_sheet(output: Path, configs: dict, groups: list[str], *, first_frame=0, end_frame=None):
    from PIL import Image, ImageDraw
    thumb_w, thumb_h = 320, 180
    selections = []
    for group in groups:
        stop = end_frame or len(configs[group]["frames"])
        selections.append(list(dict.fromkeys((first_frame, (first_frame+stop)//2, stop-1))))
    sheet = Image.new("RGB", (thumb_w*max(map(len, selections)), (thumb_h+24)*len(groups)))
    draw = ImageDraw.Draw(sheet)
    comparison = Image.new("RGB", (thumb_w*2, (thumb_h+40)*sum(map(len, selections))))
    labels = ImageDraw.Draw(comparison)
    comparison_row = 0
    for row, group in enumerate(groups):
        for col, i in enumerate(selections[row]):
            frame = load_frame(output / group / f"frame_{i:06d}")
            rgb = frame.buffers["reference"].clip(0)
            mapped = np.power(rgb/(1+rgb), 1/2.2)
            image = Image.fromarray(np.uint8(np.clip(mapped*255, 0, 255))).resize((thumb_w, thumb_h))
            sheet.paste(image, (col*thumb_w, row*(thumb_h+24)))
            reference_spp = frame.metadata["render"]["reference_samples"]
            input_spp = frame.metadata["render"]["samples"]
            draw.text((col*thumb_w+4, row*(thumb_h+24)+thumb_h+3),
                      f"Reference {reference_spp} SPP | frame {i}", fill="white")
            y = comparison_row*(thumb_h+40)
            comparison_row += 1
            labels.text((4, y+3), f"{group} | frame {i}", fill="white")
            for side, name, spp in ((0, "radiance", input_spp), (1, "reference", reference_spp)):
                linear = frame.buffers[name].clip(0)
                display = np.power(linear/(1+linear), 1/2.2)
                full = Image.fromarray(np.uint8(np.clip(display*255, 0, 255)))
                comparison.paste(full.resize((thumb_w, thumb_h)), (side*thumb_w, y+20))
                label = f"Noisy input: {spp} SPP" if side == 0 else f"Reference: {spp} SPP"
                labels.text((side*thumb_w+4, y+thumb_h+23), label, fill="white")
                preview_dir = output / "previews" / group
                preview_dir.mkdir(parents=True, exist_ok=True)
                full.save(preview_dir / f"frame_{i:06d}-{name}-{spp}spp.png")
    sheet.save(output / "contact-sheet.png")
    comparison.save(output / "comparison.png")


def main(argv=None, *, reports=False) -> int:
    default_dir = DEFAULT_PLAN.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--stage", choices=("plan", "preview", "pilot", "capture"), default="capture")
    parser.add_argument("--renderer", type=Path, default=default_dir.parent.parent / "Wavefront-Path-Tracer/build/dataset_capture")
    parser.add_argument("--renderer-root", type=Path, default=default_dir.parent.parent / "Wavefront-Path-Tracer")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scenes", nargs="+")
    parser.add_argument("--groups", nargs="+")
    parser.add_argument("--frame-range", nargs=2, type=int, metavar=("START", "END"),
                        help="Capture only [START,END) per clip, preserving the full plan for later resume")
    parser.add_argument("--partitions", nargs="+", choices=PARTITIONS, default=["validation", "test"] if reports else list(PARTITIONS))
    args = parser.parse_args(argv)
    args.plan = args.plan.resolve()
    plan = read_plan(args.plan)
    preview = args.stage == "preview"
    if reports and preview:
        parser.error("Report captures use the full report budget; preview with generate.py")
    config_options = {"width": 320, "height": 180, "reference_samples": 16} if preview else {}
    if reports:
        config_options["reference_samples"] = plan["report_reference_samples"]
    configs = build_configs(args.plan, **config_options)
    membership = {s["id"]: s["partition"] for s in plan["scenes"]}
    if args.scenes and set(args.scenes)-membership.keys():
        parser.error(f"Unknown scenes: {sorted(set(args.scenes)-membership.keys())}")
    if args.groups and set(args.groups)-configs.keys():
        parser.error("Unknown capture group")
    chosen_scenes = args.scenes or (["bmw", "breakfast_room"] if args.stage == "pilot" else list(membership))
    groups = [g for g, c in configs.items() if c["scene"] in chosen_scenes
              and membership[c["scene"]] in args.partitions and (not args.groups or g in args.groups)]
    if args.stage in ("preview", "pilot") and not args.groups:
        groups = list({configs[g]["scene"]: g for g in reversed(groups)}.values())[::-1]
    if not groups:
        parser.error("No capture groups selected")
    first, end = args.frame_range or (0, plan["frames_per_clip"])
    if not 0 <= first < end <= plan["frames_per_clip"]:
        parser.error("Frame range must lie inside the eight-frame clip")
    suffix = "-reports4096" if reports else ""
    if args.stage in ("preview", "pilot"):
        suffix += "-"+args.stage
    output = (args.output or DEFAULT_DATASET_ROOT / (plan["dataset_id"]+suffix)).resolve()
    executable = args.renderer.resolve() if args.stage != "plan" else None
    if executable and not executable.is_file():
        parser.error(f"Build dataset_capture and pass --renderer: {executable}")
    with capture_lock(output):
        spec = prepare(args.plan, output, configs, executable)
        print(json.dumps({"output": str(output), "selected_clips": len(groups),
                          "selected_frames": len(groups)*(end-first),
                          "dimensions": [next(iter(configs.values()))["width"], next(iter(configs.values()))["height"]],
                          "reference_samples": spec["declared_reference_samples"]}, indent=2), flush=True)
        if args.stage != "plan":
            result = run_captures(configs, groups, output, executable, args.renderer_root.resolve(),
                                  first_frame=first, end_frame=end)
            print(json.dumps({k: v for k, v in result.items() if k != "new_frames"}, indent=2), flush=True)
            if preview or args.stage == "pilot":
                contact_sheet(output, configs, groups, first_frame=first, end_frame=end)
            if set(groups) == set(configs) and first == 0 and end == plan["frames_per_clip"]:
                write_json(output / "split_audit.json", audit_dataset_split(output / "split.json"))
    return 0
