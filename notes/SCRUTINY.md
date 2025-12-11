# SCRUTINY.md — notes

Critical review and risk tracking. Record concerns about correctness, performance, API contracts, and UX, plus mitigation/validation ideas.

## 2025‑12‑11 – Repo audit risks

### Dependency / reproducibility
- **High:** `requirements.txt` lists `fitparse` (PyPI 1.2.0, unmaintained) but the project intends `python-fitparse` (2.0.4). Risk: users install wrong parser and CLI fails or mis‑parses.
  - Mitigation: rename dep, pin lower/upper bounds, add a runtime version check in `_require_dependency`.
- **Medium:** No version pins for Python deps; numpy 2.x and Typer 0.20 may introduce subtle API/behavior changes.
  - Mitigation: pin or bound, add `requirements-dev.txt`, CI smoke run.

### Maintainability
- **High:** `hc_curve.py` is ~6700 lines; hard for juniors to navigate and for seniors to safely extend.
  - Mitigation: modularize into concern‑scoped files; keep public API stable.

### Rust/web drift
- **Medium:** Rust uses older `fitparser` (0.7 vs 0.9) and `leptos` (0.6 vs 0.8). Upgrades may bring bug fixes but have breaking changes.
  - Mitigation: schedule a controlled upgrade branch with parity tests vs Python outputs.

### Caching correctness
- **Low:** FIT cache keys on mtime/size only; parsing‑logic changes won’t invalidate cached records.
  - Mitigation: embed schema/hash in cache payload; add `--clear-cache` CLI.

