# MEMORY.md — hillclimb

Long‑lived architectural memory, design decisions, and rationale. Treat this as ADR‑lite: add short, dated entries when you make or rely on a decision that future agents/devs should know *why* it exists.

## 2025‑10‑10 – Baseline architecture & data model

### Source Selection (Auto)
Priority is chosen per merged timeline based on available fields:
1. `runn_total_gain`: cumulative ascent from developer fields (e.g., `total_ascent`, `total_gain`). Handles counter resets across files.
2. `runn_incline`: derive vertical from treadmill `incline` (%) and `distance` (m). Only positive incline contributes.
3. `altitude`: sum positive deltas from `enhanced_altitude`/`altitude`.

### Distance Preference
- Prefer `total_distance` developer field when present; else `enhanced_distance`; else `distance`.
- When coalescing overlapping samples, keep the higher‑priority distance.

### Merging
- Parse each file into records: epoch time `t`, optional `alt`, `tg` (total gain), `inc` (incline), `dist` (distance), plus `dist_prio`.
- Merge across files by time (ascending). Deduplicate points within ~0.5s; prefer non‑null and higher‑priority sources.

### Curve Algorithm
- Build monotonically non‑decreasing cumulative gain series `G(t)` and store it as piecewise‑linear segments (ex, ey, slope).
- Precompute `U(times[i])` once, then vectorize start‑ and end‑aligned windows while evaluating the opposite boundary via NumPy `searchsorted` interpolation.
- Skip end‑aligned candidates that live entirely inside inactivity gaps when D is shorter than the gap, keeping scans focused on real activity.
- Durations use a multi‑resolution grid: 1s steps to 2h, then ~1% geometric increments rounded to “nice” minutes/hours/days plus curated anchors (3h, 4h, daily, weekly, etc.).
- Complexity stays linear in samples per duration even for multi‑day spans; dense short‑duration behavior is unaffected.
- Optional Numba engine jit‑compiles the exhaustive sweep (parallel `prange`, pointer‑based envelope lookups, gap skipping) and drops wall‑clock by another ~5×; a stride engine reuses uniform 1 Hz cumulative sums when available.
- Parsing can fan out across threads via `--parse-workers` (fitparse releases the GIL); use `--profile` to log parse/merge/curve/plot timings when chasing regressions.
- Parsed FIT files are cached under `.cache/parsed_fit` keyed by mtime/size so repeated runs skip base decoding; remove that directory to invalidate the cache.

### Diagnostics
- `diagnose` lists the most frequent keys, time span, and candidate gain fields to help adapt to new developer fields.

### Plotting
- Default: split plots for Python CLI (`_rate.png`, `_climb.png`), combined plot optional.
- Dual‑axis combined view: climb rate (m/h) and max climb (m) vs duration (log x‑axis). Annotated points for quick reading.

## 2025‑12‑11 – Python module layout

- `hc_curve.py` is now the core library (FIT parsing, curve math, gain‑time, WR envelope). It still exposes a `main_cli()` shim so `python hc_curve.py …` works as before.
- `hc_plotting.py` owns all Matplotlib rendering and plot‑specific constants.
- `hc_cli.py` owns the Typer CLI app and orchestration (`curve`, `time`, `export-series`, `diagnose`).

Rationale: reduce monolith size while keeping CLI parity and import stability.

## 2025‑12‑13 – Altitude smoothing semantics

- `--smooth` is an optional, off-by-default extra smoothing pass for altitude-derived ascent only.
- Implementation: apply a rolling-median window (seconds) to the *effective altitude path* before idle detection and ascent integration. This lives alongside (not replacing) the existing spike repair, morphological closing, and local-polynomial smoothing.
- `--gain-eps` remains the ascent hysteresis threshold (meters) used when converting altitude deltas into cumulative ascent.
