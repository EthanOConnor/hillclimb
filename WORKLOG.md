Worklog (legacy)
================

This file is kept for backwards compatibility. Canonical work history now lives in `notes/WORKLOG.md`.

Initial Implementation
----------------------
- Scaffolded `hc_curve.py` CLI with Typer; added `requirements.txt` and `README.md`.
- Implemented FIT parsing with fitparse; computed curve via two-pointer sweep.
- Added CSV output and logging; ensured monotonic non-increasing rate.

Diagnostics and Parsing Improvements
------------------------------------
- Added `diagnose` subcommand to summarize field keys and candidate ascent fields.
- Merged multiple FIT files across overlapping timelines; coalesced near-duplicate timestamps.
- Supported source selection: `runn_total_gain`, `runn_incline` from `incline`×`distance`, and `altitude` fallback.
- Preferred developer fields: `total_ascent` / `total_gain` for ascent; `total_distance` for distance.

Visualization
-------------
- Added matplotlib plot generation (PNG) alongside CSV with dual-axis design.

Performance Improvements
------------------------
- Reworked `compute_curve` to cache cumulative gain values per sample, vectorize start/end window gains via NumPy `searchsorted`, and skip end-aligned windows that live entirely inside inactivity gaps.
- Added a multi-resolution duration grid (dense seconds up to ~2h, geometric above with curated anchors) so the `--exhaustive` sweep remains exact where it matters but scales to multi-day spans.
- Documented the new behaviour in `README.md` and `AGENT_NOTES.md` for future maintainers.
- Added engine selection with a parallel Numba kernel (pointer-based envelope sweep) and a stride mode for 1 Hz data; auto-detect falls back to NumPy when Numba is unavailable.
- Reduced pipeline overhead with threaded FIT parsing, parsed-FIT disk caching, per-stage profiling, cached WR envelopes, and a fast-plot mode that trims heavy annotations.

Rust CLI Parity & Enhancements (2025-10-08)
-------------------------------------------
- Brought the Rust workspace (`hc_curve_rs`) to feature parity with the Python CLI: canonical mixed-source stitching, engine semantics (Auto/Numpy/Stride), WR envelope modelling, scoring overlays, and Plotters-based combined/split charts.
- Added disk-backed FIT/GPX cache keyed by path, size, and mtime; cache hits now remix `file_id` so reordered runs stay correct.
- Parallelised FIT parsing with Rayon and surfaced end-to-end stage timings via the new `--profile` flag.
- Restored the default combined PNG output (split plots remain opt-in via `--split-plots`) and clamped log-scale minima so flat activities render safely when `--ylog-rate` or `--ylog-climb` is requested.
- Updated top-level documentation (`README.md`, `hc_curve_rs/README.md`) to describe the Rust CLI workflow, caching behaviour, profiling flag, and split-plot defaults.

Repo Setup
----------
- To be initialized as a git repository with initial commit including docs.

Environment Setup
-----------------
- Created a local `.venv` virtual environment with `python3 -m venv .venv` and installed requirements; documented the workflow in the README so commands run via `.venv/bin/python`.
- Added automatic Matplotlib cache configuration to use a repo-local `.mplconfig/` directory, preventing font/cache warnings during plotting.
- Tweaked rate/climb plot annotations to alternate offsets and moved legends away from the data so overlays stay legible.

Upcoming Stabilization Work
---------------------------
- Harden mixed-source stitching by fixing dwell fallback and preserving contributing `file_id` during timestamp coalescing.
- Connect the CLI `--source` option to the actual computation path so CSV metadata reflects the selected stream.
- Swap the exhaustive/all-windows sweep to the linear-time concave-envelope implementation for better scaling on long sessions.
- Enable cross-session windows by building a global canonical timeseries and using prefix sums so the curve can report multi-day spans across breaks.
- Introduce a `stats` report CLI that exports detailed tables (curve summary, best windows, timeline overview, gap report, session rollups), derived metrics (peak streaks, consistency scores, inactivity insights, multi-day gains), and supporting visuals (gap-aware curves, heatmaps, calendars) in Markdown/HTML/JSON so the new inactivity data becomes actionable.
- Add a QC layer that scans from short to long durations (e.g., 5s/10s/30s/1m/5m/etc.), flags ascent bursts exceeding realistic thresholds, and censors the affected cumulative gain segments by flattening them before the curve computation.
- Resolve remaining plot label overlap at shared durations (e.g., 30 min) by improving the de-duplication/offset strategy for base-point and magic annotations in dense and non-dense modes.
