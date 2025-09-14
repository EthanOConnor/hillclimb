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
- Build monotonically non-decreasing cumulative gain series G(t).
- For each duration D, slide over sample starts i; binary-advance j pointer to the first t[j] ≥ t[i]+D; linearly interpolate G at window end; track max G_end - G_start.
- Complexity: O(n) per duration; fast for typical activities.

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

