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
- **Medium:** Rust/web dependencies can drift and break WASM builds unexpectedly (especially around `getrandom` + WASM targets).
  - Status: upgraded to `fitparser` 0.9.x and `leptos` 0.8.x; web now includes a WASM-only `getrandom` override (`features = ["js"]`) and builds under `wasm32-unknown-unknown`.
  - Mitigation: add CI smoke checks (Rust CLI + WASM compile, plus Python unit tests) so future drift is caught early.

### Caching correctness
- **Low:** FIT cache keys on mtime/size only; parsing‑logic changes won’t invalidate cached records.
  - Status: cache payloads now include a schema version and both CLIs expose `--clear-cache`.

## 2025‑12‑18 – Algorithm correctness and scale risks

### Python vs Rust semantic drift
- **High:** Two full “truth” implementations exist (Python + Rust). Even with parity intent, semantics can diverge (gap handling, idle gating meaning, QC removal reporting, and concave envelope rules), silently changing results.
  - Mitigation: make Rust `hc_curve` canonical and migrate Python to wrapper/plot role; add cross-impl verification tests while both exist.

### “Inactivity gaps” definition mismatch
- **High:** Rust can represent inactivity as idle segments; Python auto mode currently infers inactivity gaps primarily from timestamp gaps. This affects window skipping and envelope shaping, especially around breaks.
  - Mitigation: write and enforce a single spec for gaps/idle/sessions; add targeted tests and JSON diagnostics.

### Resampling blow-ups on large gaps
- **High:** Python `--resample-1hz` currently fills the entire time span, which can explode memory/time on multi-file multi-day timelines with large gaps.
  - Mitigation: segment-aware resampling or max-gap guardrails; tests that simulate worst-case gaps.

### FIT parser maintenance risk (Python)
- **Medium:** `python-fitparse` upstream warns about maintainer bandwidth and recommends `fitdecode`. Staying pinned without a plan risks forced migration later.
  - Mitigation: migrate Python to call Rust core (preferred), or add a parser abstraction and validate `fitdecode` compatibility on fixtures.

### Web dependency drift / CDN reliance
- **Medium:** Web app relies on Plotly via CDN; upgrades can break charts or introduce subtle behavior changes.
  - Mitigation: pin versions intentionally, add “known good” smoke checks (build + basic run instructions), and consider an offline/local-dev fallback plan if needed.
