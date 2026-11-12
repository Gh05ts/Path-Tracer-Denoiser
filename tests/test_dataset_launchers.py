import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def test_background_r2_mode_bypasses_b2_launcher_and_preserves_arguments(tmp_path):
    tools = tmp_path / "workspace/Path-Tracer-Denoiser/tools"
    tools.mkdir(parents=True)
    launcher = tools / "run_dataset_background.sh"
    shutil.copyfile(Path(__file__).parents[1] / "tools/run_dataset_background.sh", launcher)
    marker = tmp_path / "r2-arguments.json"
    (tools / "generate_r2_dataset.py").write_text(
        "import json, os, sys\nfrom pathlib import Path\n"
        f"Path({str(marker)!r}).write_text(json.dumps(sys.argv[1:]))\n"
        f"Path({str(marker.with_suffix('.session'))!r}).write_text(str(os.getsid(0)==os.getpid()))\n")
    (tools / "generate_dataset.py").write_text("raise RuntimeError('Wrong launcher: B2')\n")
    environment = dict(os.environ, DATASET_PYTHON=sys.executable)
    result = subprocess.run(["bash", str(launcher), "r2", "--scenes", "bmw", "--output", "custom output"],
                            env=environment, capture_output=True, text=True, check=True)
    deadline = time.monotonic() + 5
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(.01)
    assert marker.exists()
    assert json.loads(marker.read_text()) == ["--scenes", "bmw", "--output", "custom output"]
    if shutil.which("setsid"):
        deadline = time.monotonic()+5
        while not marker.with_suffix(".session").exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert marker.with_suffix(".session").read_text() == "True"
    assert "Started PID" in result.stdout and "Log:" in result.stdout
