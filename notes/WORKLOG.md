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
