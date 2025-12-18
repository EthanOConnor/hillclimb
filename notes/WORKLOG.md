# WORKLOG.md — notes

Chronological log of meaningful work. Add a short dated entry for any substantive change.

## 2025‑09‑19 – Initial implementation
- Scaffolded `hc_curve.py` CLI with Typer; added `requirements.txt` and `README.md`.
- Implemented FIT parsing with fitparse; computed curve via two‑pointer sweep.
- Added CSV output and logging; ensured monotonic non‑increasing rate.

## 2025‑09‑21 – Diagnostics and parsing improvements
- Added `diagnose` subcommand to summarize field keys and candidate ascent fields.
- Merged multiple FIT files across overlapping timelines; coalesced near‑duplicate timestamps.
- Supported source selection: `runn_total_gain`, `runn_incline` from `incline×distance`, and `altitude` fallback.
- Preferred developer fields: `total_ascent` / `total_gain` for ascent; `total_distance` for distance.

## 2025‑09‑21 – Visualization
- Added matplotlib plot generation (PNG) alongside CSV with dual‑axis design.

## 2025‑10‑08 – Performance improvements
- Reworked `compute_curve` to cache cumulative gain values per sample, vectorize start/end window gains via NumPy `searchsorted`, and skip end‑aligned windows entirely inside inactivity gaps.
- Added a multi‑resolution duration grid so `--exhaustive` scales to multi‑day spans.
- Added engine selection with parallel Numba kernel and stride mode; auto‑detect falls back to NumPy when Numba unavailable.
- Reduced pipeline overhead with threaded FIT parsing, disk caching, per‑stage profiling, cached WR envelopes, and fast‑plot mode.

## 2025‑10‑08 – Rust CLI parity & enhancements
- Brought Rust workspace (`hc_curve_rs`) to feature parity with Python CLI, adding mixed‑source stitching, WR modeling, scoring overlays, and Plotters charts.
- Added disk‑backed FIT/GPX cache keyed by path, size, and mtime; cache hits remix `file_id` so reordered runs stay correct.
- Parallelized parsing with Rayon; surfaced stage timings via `--profile`.
- Restored combined PNG as default; split plots opt‑in; clamped log‑scale minima for flat activities.
- Updated docs to describe Rust workflow, caching, profiling, and split‑plot defaults.

## 2025‑12‑11 – Inter‑session notes system
- Added `notes/` directory with MEMORY/WORKLOG/BACKLOG/CHAT/SCRUTINY to mirror the `streamvis` coordination pattern.
- Updated `AGENTS.md` to require maintaining these notes and marked legacy `AGENT_NOTES.md`/`WORKLOG.md` as pointers.

## 2025‑12‑11 – Fix FIT parser dependency naming
- Updated `requirements.txt` to depend on `python-fitparse` (the maintained FIT parser) and adjusted install hints/docs accordingly.

## 2025‑12‑11 – Pin Python dependency ranges
- Added explicit version bounds for Python runtime deps and documented optional `numba`/`scipy` extras.
- Added a runtime warning if an old `fitparse` (<2.x) package is detected.

## 2025‑12‑11 – Modularize Python implementation
- Split Matplotlib plotting into `hc_plotting.py` and Typer CLI orchestration into `hc_cli.py`.
- Trimmed `hc_curve.py` to core parsing/math/WR logic and left a thin CLI shim (`python hc_curve.py ...` unchanged).

## 2025‑12‑11 – Add JSON sidecar exports
- Added `--json/--no-json` to Python and Rust CLIs to emit a `.json` report next to CSV outputs for `curve`, `time`/`gain-time`, and `export-series`.
- JSON includes metadata (inputs, selected source, sampling/QC stats, engine) plus the computed curves/series.

## 2025‑12‑12 – Improve FIT cache invalidation
- Added a cache schema version to both Python pickle caches and Rust JSON caches; old caches now safely invalidate.
- Added `--clear-cache` to Python and Rust CLIs to delete `.cache/parsed_fit` before parsing inputs.

## 2025‑12‑12 – Add Python unit tests
- Added a minimal `unittest` suite under `tests/` covering core curve math and FIT-record merge heuristics.
- Run with: `.venv/bin/python -m unittest discover -s tests -v`.

## 2025‑12‑13 – Altitude smoothing (`--smooth`) and epsilon fix
- Wired the existing Rust CLI `--smooth` flag through the core altitude pipeline and added a matching Python `--smooth` option (off by default).
- `--smooth` applies an additional rolling-median window (seconds) after the effective altitude path to reduce staircase artifacts on sparse/noisy data.
- Fixed Python altitude ascent integration to respect `--gain-eps` instead of a hardcoded epsilon.

## 2025‑12‑13 – Web UI polish (progress, theme, mobile)
- Added a light/dark theme toggle with persistent preference (`localStorage`) and applied theme colors to Plotly layouts.
- Added a compute progress indicator (spinner + progress bar) and inserted small async yields so the UI updates before long WASM compute steps.
- Improved responsiveness: Plotly `responsive` config, safer plot sizing, and a two-column layout on wide screens.

## 2025‑12‑13 – Rust crate upgrades (fitparser/leptos)
- Upgraded Rust FIT parsing crate `fitparser` to 0.9.x and web UI framework `leptos` to 0.8.x; updated `Cargo.lock`.
- Added a WASM-only `getrandom` override (`features = ["js"]`) required by the newer dependency graph.
- Updated the web app to current Leptos idioms (`signal`, `Effect::new`, and new import paths) and validated builds for CLI + WASM target.

## 2025‑12‑14 – Notes/docs currency sweep
- Updated `AGENTS.md` to reflect the current Python module layout and dependency naming.
- Refreshed `docs/web_mvp_spec.md` and `docs/handoff_prompt_next_agent.md` so they reflect the post‑MVP web app.
- Captured remaining follow‑ups (CI smoke checks, Rust warnings, repo hygiene) in `notes/BACKLOG.md` and updated `notes/SCRUTINY.md`.

## 2025‑12‑18 – Comprehensive repo review + ticketization
- Added a comprehensive, cross-functional review handoff doc: `docs/REVIEW_2025-12-18.md`.
- Reworked `notes/BACKLOG.md` into a prioritized, owner-assigned ticket list with acceptance criteria.
- Recorded strategic direction: Rust `hc_curve` should become the canonical “source of truth”, with Python moving toward a wrapper/plotting role.
- Documented parser strategy guidance (python-fitparse vs fitdecode) and the recommended migration path.
- Added a CI smoke workflow (`.github/workflows/ci-smoke.yml`) to run Python unit tests, Rust workspace tests, and a WASM compile check on PR/push.
- Cleaned up Rust build warnings so `cargo check --workspace` is quiet (cfg-gated `mount_to_body`, removed unused `mut`, simplified unused `MagicPoint` fields).
- Added `docs/compute_semantics.md` and aligned Rust gap/idle semantics to match the documented behavior (session gaps are timestamp gaps; gap-only windows are skipped; idle segments gate altitude ascent but don’t define session gaps).

## 2025‑12‑18 – Resample guardrails (large gaps)
- Added guardrails for 1 Hz resampling to prevent huge allocations when timelines contain large timestamp gaps.
- Exposed overrides in both CLIs: `--resample-max-gap-sec` and `--resample-max-points`; JSON sidecars now record these settings.
- Rust altitude pipeline now skips resampling (with a warning) when guardrails trigger; all-windows modes return a clear error instead.
- Added unit tests in Python and Rust covering guardrail behavior and overrides.

## 2025‑12‑18 – Repo hygiene
- Added explicit ignores for `outputs/`, `AGENTS.md.old`, and Office temp files (`~$*.xlsx`).

## 2025‑12‑18 – FIT parsing strategy + JSON output versioning
- Added `docs/fit_parsing_strategy.md` documenting the Rust-only canonical parsing decision and the Python `python-fitparse` vs `fitdecode` recommendation.
- Added `schema_version` and `parser` metadata to JSON sidecars (Python + Rust) to prevent silent ingestion/output drift as dependencies evolve.

## 2025‑12‑18 – Personas + defaults
- Added `docs/personas_defaults.md` defining primary personas (treadmill vs outdoor), recommended defaults, and user-facing copy for session gaps and idle detection.

## 2025‑12‑18 – Algorithm Lab UX spec
- Added `docs/algorithm_lab_ux.md` with wireframe-level layout + copy + explainability requirements for multi-algorithm ascent comparison.
