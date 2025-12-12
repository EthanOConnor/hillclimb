# BACKLOG.md — notes

Detailed technical backlog / roadmap. Prefer adding items here over TODOs in code. Keep items small, testable, and prioritized.

## P0 (next work)
- Fix Python dependency naming: `requirements.txt` should depend on `python-fitparse` (module `fitparse`) instead of the dormant `fitparse` 1.2.0 package; update README accordingly.
- Add explicit version bounds/pins for Python deps to avoid numpy/typer/matplotlib breakage; document optional extras (`numba`, `scipy`).
- Break `hc_curve.py` into modules (`fit_parsing.py`, `curve_math.py`, `wr.py`, `plotting.py`, `cli.py`) without changing behavior; add a minimal import shim for backwards CLI parity.

## P1
- [x] Add JSON export with metadata (inputs, selected source, sampling/QC stats, engine) alongside CSV.
- [ ] Add unit tests for parser and curve math using small synthetic FIT‑like fixtures.
- [ ] Improve cache invalidation: include schema/version in FIT cache entries and expose `--clear-cache`.

## P2
- Optional smoothing/interpolation for nicer curves on sparse data (configurable, off by default).
- Web UI polish: progress indicator, theme toggle, and better mobile plot layout; keep WASM bundle small.
- Consider upgrading Rust crates (`fitparser` 0.9, `leptos` 0.8) after assessing breaking changes.
