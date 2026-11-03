"""Execute model stages in fresh, spawn-safe Python processes."""
import os
from pathlib import Path
import subprocess
import sys


def run_cli(arguments):
    # Also supports a source checkout without requiring pip install -e .
    env = os.environ.copy()
    source = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = source + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        subprocess.run([sys.executable, "-m", "star_dro", *map(str, arguments)], env=env, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"{arguments[0]} stage exited with status {exc.returncode}; see its output above") from exc
