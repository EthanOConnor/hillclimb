# SCRUTINY.md — notes

Critical review and risk tracking. Record concerns about correctness, performance, API contracts, and UX, plus mitigation/validation ideas.

## 2025‑12‑11 – Repo audit risks

### Dependency / reproducibility
- **High:** `requirements.txt` listed `fitparse` (PyPI 1.2.0, unmaintained) but the project intends `python-fitparse`. Risk: users install wrong parser and CLI fails or mis‑parses.
  - Status: dependency renamed to `python-fitparse`, version bounds added, and a runtime warning now flags old `fitparse` installs.
- **Medium:** No version pins for Python deps; numpy 2.x and Typer 0.20 may introduce subtle API/behavior changes.
  - Status: version ranges added to `requirements.txt`; consider CI smoke run later.

### Maintainability
- **High:** `hc_curve.py` was ~6700 lines; hard for juniors to navigate and for seniors to safely extend.
  - Status: plotting and CLI moved into `hc_plotting.py` / `hc_cli.py`; `hc_curve.py` trimmed to core library with stable CLI shim.

### Rust/web drift
- **Medium:** Rust uses older `fitparser` (0.7 vs 0.9) and `leptos` (0.6 vs 0.8). Upgrades may bring bug fixes but have breaking changes.
  - Mitigation: schedule a controlled upgrade branch with parity tests vs Python outputs.

### Caching correctness
- **Low:** FIT cache keys on mtime/size only; parsing‑logic changes won’t invalidate cached records.
  - Status: cache payloads now include a schema version and both CLIs expose `--clear-cache`.
