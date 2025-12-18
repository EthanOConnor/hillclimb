Gain-Centric Report: Best Time for Target Ascent
================================================

Purpose
-------
Invert the existing “critical hill climb rate curve” (best gain for a given time) to produce a gain‑centric view: the minimum time required to achieve at least a target gain. This enables questions such as “How quickly can I climb 100 m?” across targets, with textual, tabular, and graphical outputs.

Scope
-----
- Builds on existing FIT parsing, ascent derivation, QC filtering, and session merging.
- No new runtime dependencies; reuses `numpy`, `matplotlib`, `fitparse`, and Typer.
- Implemented initially in the Python CLI (`hc_curve.py`), with a mirrored design for the Rust CLI (`hc_curve_rs`) as follow‑up.

User Experience & Human Factors
-------------------------------
- Primary mental model: “N meters in T time.” Users think in common targets (e.g., 50 m, 100 m, 200 m, 500 m) and want both an at‑a‑glance summary and a CSV for further analysis.
- Targets may be supplied explicitly (`-g 50 100 200`) or generated as a sensible default set.
- Results must communicate feasibility (e.g., target > max session gain → unachievable), show equivalent climb rate, and reference the window location in the activity timeline.
- Plots should be readable at a glance: gain on x‑axis, time on y‑axis, with faint iso‑rate guide lines and optional overlays for world‑record (WR) and personal‑scaled curves.

CLI Design
----------
Add a new command alongside `curve` and `diagnose`:

```
hc_curve.py time [OPTIONS] FIT_FILES...
```

Key options:
- `--gains, -g <m...>`: One or more gain targets in meters. Accepts suffixes `m` or `ft` (e.g., `200m`, `500ft`). If absent, use a default target set: `50, 100, 150, 200, 300, 500, 750, 1000` m (pruned by session max gain).
- `--output, -o <csv>`: Output CSV path (default: `gain_time.csv`). The PNG is written alongside by default unless `--no-plot`.
- `--png <file>`: Optional explicit PNG path.
- `--no-plot`: Disable PNG generation.
- `--all`: Use the exact per‑second duration curve for inversion (recommended for accuracy and speed when feasible).
- `--exhaustive`: Evaluate a multi‑resolution duration grid and invert (good compromise for multi‑hour activities).
- `--step <s>`: Step size (seconds) for exhaustive durations (default 1).
- `--max-duration <s>`: Cap the maximum duration considered by the grid.
- `--source`: `auto|runn|altitude` ascent source (inherited from `curve`).
- `--verbose, -v`: Verbose logging (as today).
- `--plot-wr/--no-plot-wr`: Overlay WR envelope (inverted) on the plot.
- `--plot-personal/--no-plot-personal`: Overlay personal scaled WR envelope (inverted).
- `--magic-gains <list>`: Comma‑separated gains to annotate on the plot and textual summary (default: `50m,100m,200m,300m,500m,1000m`).
- `--gain-units <m|ft>`: Display/parse convenience; CSV always uses meters.
- `--profile`: Emit timings for parse/compute/CSV/plot.

Usage examples:
- `.venv/bin/python hc_curve.py time Tracklogs/Treadmill/*.fit -g 50 100 200 500 -o outputs/gain_time.csv --all --plot-wr --magic-gains 100m,200m,500m`
- `.venv/bin/python hc_curve.py time activity.fit --exhaustive --step 2 --max-duration 7200 -o outputs/gain_time.csv`

Outputs
-------
Textual (stdout unless `--quiet` is later added):

```
Gain Time Report (source: runn_total_gain)
-  50 m:   1:12  (rate 2500 m/h)  window 00:23:18–00:24:30
- 100 m:   2:59  (rate 2010 m/h)  window 00:51:07–00:54:06
- 200 m:   7:45  (rate 1548 m/h)  window 01:42:22–01:50:07
- 500 m:  24:18  (rate 1234 m/h)  window 02:03:10–02:27:28
```

CSV (written to `--output`, default `gain_time.csv`):
- Columns:
  - `gain_m`: Target gain in meters.
  - `min_time_s`: Minimum time (seconds) to achieve at least `gain_m`.
  - `avg_rate_m_per_hr`: Equivalent climb rate for that window (`gain_m / min_time_s * 3600`).
  - `start_offset_s`, `end_offset_s`: Window offsets (seconds) from activity start for the representative achieving window.
  - `source`: `runn_total_gain` or `altitude`.
  - `note`: Optional text (`unachievable`, `bounded_by_grid`, etc.).

Graphical (PNG next to CSV unless `--no-plot`):
- X‑axis: Gain (m); Y‑axis: Time (min). Optional `--ylog-time` for log‑scaled time.
- Rendered series: `T(g)` (minimum time vs gain), optional WR and personal overlays (inverted envelopes), plus light grey iso‑rate guide lines (e.g., 800, 1000, 1200, 1500, 2000, 2500 m/h).
- Annotations: markers/labels at `--magic-gains`, with rate and time callouts; optional “goals” akin to current magic points (weakest annotations first).
- Split plots: optional `--split-plots` creates (a) time vs gain and (b) derived rate vs gain (monotone non‑increasing), matching the existing split behavior for the duration‑centric plot.

Algorithms & Internal Implementation
------------------------------------
We provide two complementary pathways; the CLI chooses the best based on flags:

1) Invert the duration→gain curve (primary)
   - Compute `G(D)` as today via `--all` (per‑second) or `--exhaustive` duration sets.
   - Treat `T(g)` as the generalized inverse: `T(g) = min{ D : G(D) ≥ g }`.
   - Implementation details:
     - Ensure `G(D)` is non‑decreasing (already enforced by the concave envelope step).
     - For each queried `g`, binary‑search `G(D)` to find the smallest index where `G ≥ g` and take the associated `D`.
     - Representative window: reuse the stored arg‑max window at that `D` (it must achieve `≥ g`).
     - Complexity: O(K log N) for K targets and N durations; O(N) if targets are sorted and we sweep once.
     - Accuracy: With `--all` (1 s resolution), inversion is exact; with a coarser grid, returned `T(g)` is an upper bound, marked in CSV via `note=bounded_by_grid` when `G(D*)` only just exceeds `g` and neighbouring `D` differs by more than 1 s.

2) Direct two‑pointer search on cumulative ascent (for large K or dynamic targets)
   - Maintain merged time `t[i]` and cumulative ascent `a[i]` (non‑decreasing).
   - For a single target `g`, classic sliding window finds the minimal time:
     ```
     best = +∞; best_win = None
     L = 0
     for R in range(N):
         while L < R and a[R] - a[L] >= g:
             dt = t[R] - t[L]
             if dt < best:
                 best, best_win = dt, (L, R)
             L += 1
     ```
   - For multiple targets, sort them ascending; as `best` improves for a target, emit and move to the next. Complexity is O(N + K) in practice; can be numba‑accelerated.
   - Use this path when user provides many targets without `--all`/`--exhaustive`, or when inverting a coarse grid would be too quantized.

Data & QC Considerations
------------------------
- Source selection and QC mirror `curve`: prefer developer cumulative gain (e.g., NPE Runn), otherwise derive ascent from altitude with hysteresis (`--gain-eps`).
- Session gaps are ignored for windowing (windows can span gaps), but gap statistics are still reported in logs.
- Flat sessions (no positive ascent): return empty results; textual report states “no ascent recorded”.
- Targets above `max_total_gain`: mark `unachievable` with empty window offsets.

Plotting Details
----------------
- Share the same Matplotlib theme and `--fast-plot` default.
- Iso‑rate guide lines: draw line families `t = g / r` for reference rates; label unobtrusively on the right edge.
- Overlay WR envelopes: invert the existing WR curve or anchors the same way, respecting `--wr-min-seconds` and short‑duration caps; show personal scaling when enabled.
- Label suppression settings mirror current flags: `--goal-min-seconds` and `--personal-min-seconds` apply to annotations translated from durations to gains via `g = r * T(g)` where needed.

File Naming
-----------
- CSV default: `gain_time.csv`; PNG default: `gain_time.png` next to CSV.
- When inputs include multiple FIT files, derive a context suffix similar to the current plot logic (e.g., `gain_time_<firstbasename>_and_<n-1>more.csv`).

Python Implementation Plan (hc_curve.py)
----------------------------------------
- Dataclasses:
  - `GainTimePoint { gain_m: float, min_time_s: float, avg_rate_m_per_hr: float, start_offset_s: float|None, end_offset_s: float|None, note: str|None }`
  - `GainTimeCurve { points: List[GainTimePoint], source: str, total_span_s: float }`
- Helpers (curve math group):
  - `invert_duration_curve_to_gain_time(curve: Curve, targets_m: Sequence[float]) -> GainTimeCurve`
  - `min_time_for_gains(cum_gain: np.ndarray, time_s: np.ndarray, targets_m: Sequence[float]) -> GainTimeCurve` (two‑pointer, optional numba kernel)
- CLI:
  - New Typer command `time` with options above; reuse parsing/merging/QC pipeline from `curve`.
  - Write CSV and optionally PNG; print textual summary to stdout.
- Plotting:
  - `_plot_gain_time(curve: GainTimeCurve, png_path: str, ..., wr_curve_inv: Optional[GainTimeCurve])`
  - Reuse figure layout utilities; add iso‑rate painter.
- Logging:
  - `--profile` laps: parse, compute, csv, plot.

Rust Parity (hc_curve_rs)
-------------------------
- Mirror the command as `gain-time` or reuse a `--by gain` flag on the existing subcommand.
- Reuse parsed cache under `.cache/parsed_fit/` and the same envelope inversion.
- Plot via Plotters: analogous time vs gain with iso‑rate lines.

Performance Notes
-----------------
- Inversion path: linear or near‑linear after the base curve (`--all`) is computed; no extra FIT passes.
- Two‑pointer path: O(N + K) with cache‑friendly scans; accelerate with numba when available.
- Memory: reuses existing cumulative vectors and envelope arrays.

Validation Plan
---------------
- Run both `curve --all` and `time --all` on the same inputs.
  - Check that `T(G(D)) ≤ D` and that `G(T(g)) ≥ g` for sampled pairs.
- Compare CSV and plots on treadmill examples under `Tracklogs/`.
- Edge cases: flat sessions, sessions with only altitude‑derived ascent, mixed files with overlaps, and very long activities (multi‑hour).

Acceptance Criteria
-------------------
- New `time` command produces:
  - Correct textual summary for requested targets.
  - CSV with all specified columns and representative windows.
  - PNG with legible time‑vs‑gain curve, optional WR/personal overlays, and iso‑rate guides.
- No regressions to existing `curve`/`diagnose` commands; no new dependencies.

