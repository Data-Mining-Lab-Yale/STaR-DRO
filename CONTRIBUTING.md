# Contributing to STaR-DRO

Use a Python 3.11 environment, install requirements-dev.txt and the package in editable mode, and run:

```bash
python scripts/run_tests.py --require-torch --require-data --require-model
ruff check src scripts tests
python -m build
python scripts/package_release.py
```

Algorithm changes should preserve the documented group inventory, EMA persistence, count-weighted scale, sparse mapback and multiplier/objective semantics unless a method change is explicitly intended. Include numerical or interface tests for those changes.

Issue reports should identify the failing command, environment versions, observed behavior and expected behavior. Follow the documented input contract and the institutional requirements governing any data used. Review terms are in LICENSE; third-party dependencies retain their own licenses.
