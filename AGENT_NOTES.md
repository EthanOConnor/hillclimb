Design Notes
============

Source Selection (Auto)
-----------------------
Priority is chosen per merged timeline based on available fields:
1. runn_total_gain: cumulative ascent from developer fields (e.g., `total_ascent`, `total_gain`). Handles counter resets across files.
2. runn_incline: derive vertical from treadmill `incline` (%) and `distance` (m). Only positive incline contributes.
3. altitude: sum positive deltas from `enhanced_altitude`/`altitude`.

Distance Preference
-------------------
- Prefer `total_distance` developer field when present; else `enhanced_distance`; else `distance`.
- When coalescing overlapping samples, keep the higher-priority distance.

Merging
-------
- Parse each file into records: epoch time `t`, optional `alt`, `tg` (total gain), `inc` (incline), `dist` (distance), plus `dist_prio`.
- Merge across files by time (ascending). Deduplicate points within ~0.5s; prefer non-null and higher-priority sources.

Curve Algorithm
---------------
- Build monotonically non-decreasing cumulative gain series G(t) and store it as piecewise-linear segments (ex, ey, slope).
- Precompute U(times[i]) once, then vectorize start- and end-aligned windows while evaluating the opposite boundary via NumPy `searchsorted` interpolation.
- Skip end-aligned candidates that fall wholly inside inactivity gaps when D is shorter than the gap, keeping scans focused on real activity.
- Durations use a multi-resolution grid: 1s steps to 2h, then ~1% geometric increments rounded to “nice” minutes/hours/days plus curated anchors (3h, 4h, daily, weekly, etc.).
- Complexity stays linear in samples per duration even for multi-day spans, and the dense short-duration behaviour is unaffected.
- Optional Numba engine jit-compiles the exhaustive sweep (parallel `prange`, pointer-based envelope lookups, gap skipping) and drops wall-clock by another ~5×; a stride engine reuses uniform 1 Hz cumulative sums when available.
- Parsing can fan out across threads via `--parse-workers` (fitparse releases the GIL); use `--profile` to log parse/merge/curve/plot timings when chasing regressions.
- Parsed FIT files are cached under `.cache/parsed_fit` keyed by mtime/size so repeated runs skip base decoding; remove that directory to invalidate the cache.

Diagnostics
-----------
- `diagnose` lists the most frequent keys, time span, and candidate gain fields to help adapt to new developer fields.

Plotting
--------
- Dual-axis plot: climb rate (m/h) and max climb (m) vs duration (log x-axis). Annotated points for quick reading.

Potential Next Steps
--------------------
- Add JSON export with metadata (files, selected source, sampling stats).
- Optional smoothing/interpolation for prettier curves on sparse data.
- Unit tests for parser and curve math using synthetic FIT-like inputs.

Roadmap (Post-Export + Gain-Time)
---------------------------------
This roadmap aligns with the requested sequence: CLI polish → Performance/Robustness → Reference tooling → Documentation.

1) CLI Polish
- Accept comma-separated gain tokens in Python: allow `-g 50,100,200` in addition to space-separated forms. Implementation sketch: split tokens on commas before `_parse_gain_list` in `_run_gain_time`, mirroring Rust’s `value_delimiter=','`.
- Add `--gains-from <path>`: read one token per line (supports `m|ft` suffix); merge with `--gains`. Fail fast on unreadable file. Update README examples accordingly.
- Improve error messages for empty/invalid targets: if no positive gains after parsing, print a clear hint and show default set fallback.
- Keep parity with Rust flags and behavior for `time`/`export-series`.

2) Performance & Robustness
- Optional numba acceleration for `min_time_for_gains` (two-pointer sweep): guard behind `--engine numba` (already wired) or auto-detect. Target large inputs or many targets (K). Ensure identical results to numpy path.
- Early exits and guards: if total ascent ≤ 0 after QC, exit with a concise message (both `time` and `curve`).
- QC invariants: expose an internal debug mode to assert monotonicity of cumulative ascent, and record censored windows in logs when `--verbose`.
- Memory and stride: re-use 1 Hz cumulative buffers where available; avoid resampling twice.

3) Reference Tooling Improvements
- Mathematica (`docs/mathematica_gain_time.wl`): add a small plotting helper and `VerificationTest` examples based on synthetic datasets. Provide `CompareWithCsv[data, csvPath]` to diff results from CLI `gain_time.csv`.
- Add a lightweight Python verification script under `docs/` to compare Python CLI vs Rust CLI gain-time outputs for given inputs, producing a CSV diff and summary stats (max delta seconds/rate).
- Author `docs/verification.md` with a repeatable procedure (commands + expected outputs, treadmill/outdoor samples).

4) Documentation
- README: add a complete treadmill example (commands + embedded PNG thumbnails) and a short outdoor example; document `export-series` → Mathematica loop.
- Add `docs/usage_recipes.md` for common tasks (WR overlays, unit toggles, session gap interpretation).
- Ensure artifacts go under `outputs/` and are excluded from commits; advise cleaning before PRs.

Dev Notes for Next Agent
------------------------
- Respect existing options and naming; keep Python/Rust feature parity.
- When adding flags: update both CLIs and reflect changes in README and `hc_curve_rs/README.md`.
- Avoid breaking the exported CSV schemas; add columns only with versioning or clear documentation.
- Validate changes via the sample commands in AGENTS.md and record deltas in `outputs/` during review.

Acceptance Checklist
--------------------
- `time` accepts `-g 50 100,200 500ft` and `--gains-from` simultaneously; README examples run without errors.
- Numba path produces identical windows to the numpy path on provided samples (within 1s tolerance) and is faster on large inputs.
- Mathematica helpers can reproduce CLI `gain_time.csv` within 1s for several targets; `docs/verification.md` steps pass.
- README and docs updated; new content is concise and actionable.
