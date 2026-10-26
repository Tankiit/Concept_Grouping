# cmcert

Read/write-path admission tests for concept bottleneck models (CLeaR 2027).

The repository is intentionally preregistration-first.  Scientific choices are
explicit `TODO(tag)` stubs; the lock script and every experiment refuse to run
until those choices have been resolved and the thresholds file has been locked.

```bash
pip install -e ".[dev]"
python scripts/prereg_lock.py
python scripts/run.py e0 --config configs/e0.yaml
pytest
```

`waterbirds` has no native intervention mechanism and is rejected by the
certification gate by construction.
