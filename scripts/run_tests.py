"""Run STaR-DRO tests directly from a source checkout."""
import argparse
import compileall
import importlib.util
import platform
from pathlib import Path
import sys
import unittest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/"src"))
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--require-torch", action="store_true")
p.add_argument("--require-data", action="store_true")
p.add_argument("--require-model", action="store_true", help="require offline model integration dependencies")
p.add_argument("--report", help="write a machine-readable test summary")
args = p.parse_args()
if args.require_torch and importlib.util.find_spec("torch") is None:
    p.error("PyTorch is required for this validation run; install requirements.txt")
if args.require_data and importlib.util.find_spec("datasets") is None:
    p.error("datasets is required for this run; install requirements.txt or star-dro[data]")
if args.require_model and not all(importlib.util.find_spec(m) for m in ["torch", "transformers", "peft", "tokenizers"]):
    p.error("model integration requires torch, transformers, peft, tokenizers")
for name in ["src", "scripts", "tests"]:
    if not compileall.compile_dir(root/name, quiet=1):
        raise SystemExit(1)
result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.discover(str(root/"tests")))
if args.report:
    from star_dro.utils import write_json
    write_json(args.report, {"python": platform.python_version(), "platform": platform.system(), "tests": result.testsRun,
        "passed": result.testsRun-len(result.skipped)-len(result.failures)-len(result.errors),
        "skipped": len(result.skipped), "failures": len(result.failures), "errors": len(result.errors),
        "torch_available": importlib.util.find_spec("torch") is not None, "datasets_available": importlib.util.find_spec("datasets") is not None, "successful": result.wasSuccessful()})
raise SystemExit(not result.wasSuccessful())

