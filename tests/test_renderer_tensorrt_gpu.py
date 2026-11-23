"""Opt-in end-to-end renderer tests using a verified dense spatial engine.

Set PATHTRACER_EXECUTABLE and PATHTRACER_TEST_ENGINE, and run with GPU access.
The adjacent engine .json report supplies its fixed resolution.
"""

import json
import os
from pathlib import Path
import socket
import struct
import subprocess
import time

import pytest


@pytest.fixture(scope="module")
def renderer():
    executable = os.environ.get("PATHTRACER_EXECUTABLE")
    engine = os.environ.get("PATHTRACER_TEST_ENGINE")
    if not executable or not engine:
        pytest.skip("Set PATHTRACER_EXECUTABLE and PATHTRACER_TEST_ENGINE for GPU integration")
    executable, engine = Path(executable).resolve(), Path(engine).resolve()
    assert executable.is_file() and engine.is_file()
    report = json.loads(engine.with_suffix(engine.suffix + ".json").read_text())
    _, _, height, width = report["input"]["shape"]
    cwd = Path(__file__).resolve().parents[2] / "Wavefront-Path-Tracer"
    command = [str(executable), "--scene", "prism", "--width", str(width),
               "--height", str(height), "--spp", "4"]
    return command, cwd, str(engine)


def run(command, cwd):
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.mark.parametrize("persistent", [False, True])
def test_denoising_preserves_raw_accumulation_and_resume(renderer, tmp_path, persistent):
    command, cwd, engine = renderer
    if persistent:
        command = command + ["--persistent-wavefront"]
    baseline, inferred, clean = [tmp_path / name for name in ("baseline.ppm", "inferred.ppm", "clean.ppm")]
    first, second = tmp_path / "baseline.checkpoint", tmp_path / "inferred.checkpoint"
    baseline_command = command + ["--output", str(baseline), "--checkpoint", str(first)]
    run(baseline_command, cwd)
    inference = command + ["--output", str(inferred), "--checkpoint", str(second),
                           "--denoiser-engine", engine, "--denoised-output", str(clean)]
    run(inference, cwd)
    assert baseline.read_bytes() == inferred.read_bytes()
    # The existing writer deliberately skips the final sample, and busy copy
    # slots can skip intermediate snapshots. Compare equal progress, then resume
    # both to the same last progress snapshot (3 SPP for this 4-SPP render).
    def saved_count(path):
        data = path.read_bytes()
        assert data[:8] == b"PTCHKPT1"
        assert struct.unpack_from("<I", data, 8)[0] == 3
        return struct.unpack_from("<I", data, 20)[0]
    baseline_count, inference_count = saved_count(first), saved_count(second)
    assert 1 <= baseline_count <= 3 and 1 <= inference_count <= 3
    if baseline_count == inference_count:
        assert first.read_bytes() == second.read_bytes()
    assert clean.read_bytes().startswith(b"P3\n")
    assert clean.read_bytes() != inferred.read_bytes()
    before = clean.read_bytes()
    result = run(inference + ["--resume"], cwd)
    assert f"Resuming from {inference_count} completed samples" in result.stdout
    assert "Completed sample 4 of 4" in result.stdout
    assert clean.read_bytes() == before
    run(baseline_command + ["--resume"], cwd)
    assert baseline.read_bytes() == inferred.read_bytes()
    assert saved_count(first) == saved_count(second) == 3
    assert first.read_bytes() == second.read_bytes()


def test_engine_resolution_mismatch_fails_before_tracing(renderer, tmp_path):
    command, cwd, engine = renderer
    command = list(command)
    index = command.index("--width") + 1
    command[index] = str(int(command[index]) + 1)
    result = subprocess.run(command + ["--denoiser-engine", engine,
        "--denoised-output", str(tmp_path / "clean.ppm")], cwd=cwd, text=True,
        capture_output=True, timeout=30)
    assert result.returncode != 0
    assert "Incompatible dense-spatial-v1 tensor" in result.stderr
    assert "Trace time" not in result.stdout
    assert not (tmp_path / "clean.ppm").exists()


def test_distributed_denoising_and_completed_checkpoint_resume(renderer, tmp_path):
    command, cwd, engine = renderer
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = str(probe.getsockname()[1])
    raw, clean = tmp_path / "raw.ppm", tmp_path / "clean.ppm"
    checkpoint = tmp_path / "distributed.checkpoint"
    coordinator_command = command + ["--coordinator", "--listen-host", "127.0.0.1",
        "--listen-port", port, "--samples-per-task", "2", "--distributed-checkpoint", str(checkpoint),
        "--output", str(raw), "--denoiser-engine", engine, "--denoised-output", str(clean)]
    with (tmp_path / "coordinator.log").open("w+") as log:
        coordinator = subprocess.Popen(coordinator_command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 30
            while True:
                assert coordinator.poll() is None, "Coordinator exited before worker registration"
                try:
                    with socket.create_connection(("127.0.0.1", int(port)), timeout=.2):
                        break
                except OSError:
                    if time.monotonic() >= deadline:
                        pytest.fail("Coordinator did not open its listening socket")
                    time.sleep(.1)
            run(command + ["--worker", "--worker-id", "integration-0",
                "--coordinator-host", "127.0.0.1", "--coordinator-port", port], cwd)
            coordinator.wait(timeout=120)
            log.seek(0)
            assert coordinator.returncode == 0, log.read()
        finally:
            if coordinator.poll() is None:
                coordinator.terminate()
                coordinator.wait(timeout=10)
    assert checkpoint.is_file() and raw.is_file() and clean.is_file()
    expected_raw, expected_clean = raw.read_bytes(), clean.read_bytes()
    run(coordinator_command + ["--resume"], cwd)
    assert raw.read_bytes() == expected_raw
    assert clean.read_bytes() == expected_clean
