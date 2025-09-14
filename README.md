Critical Hill Climb Rate Curve (FIT)
====================================

Small CLI to compute a “critical hill climb rate curve” from a Garmin FIT file. It scans time windows of fixed durations and finds the maximum gross vertical climb in each, then reports both the climb (m) and rate (m/h). The curve is non-increasing with duration.

Data sources
------------
- Prefers NorthPoleEngineering Runn Data Field developer data if present, using a cumulative total gain field.
- Falls back to Garmin altitude (`enhanced_altitude`/`altitude`) by summing positive elevation deltas.

Usage
-----

Requirements: `fitparse`, `typer` (see `requirements.txt`).

Examples:

```
# Single file
python hc_curve.py curve activity.fit --output curve.csv --durations 60 120 300 600 1200 1800 3600 --source auto

# Multiple files merged (overlaps handled)
python hc_curve.py curve Tracklogs/Treadmill/file1.fit Tracklogs/Treadmill/file2.fit -o curve.csv -d 60 120 300 600 1200 1800 3600

# Diagnostics (summarize fields and candidate gain keys)
python hc_curve.py diagnose Tracklogs/Treadmill/*.fit --out fit_diagnostics.txt
```

Options:
- `--output, -o`: Output CSV path (default: `curve.csv`).
- `--durations, -d`: Durations in seconds (default: 60, 120, 300, 600, 1200, 1800, 3600).
- `--source`: `auto` (default), `runn`, or `altitude`.
- `--verbose, -v`: Verbose logging.

CSV columns
-----------
- `duration_s`: Window length in seconds.
- `max_climb_m`: Max gross vertical climb (meters) found over any window of that length.
- `climb_rate_m_per_hr`: Best average climb rate (m/h) for that duration.
- `start_offset_s`, `end_offset_s`: Window offsets (seconds) from activity start.
- `source`: Data source used (`runn_total_gain` or `altitude`).

Notes
-----
- You can pass multiple FIT files; timelines are merged by timestamp. If files overlap, windows extend across files as one continuous activity.
- If developer total gain is present in any file (e.g., NPE Runn), it is preferred and stitched across files, handling counter resets at file boundaries.
- If no total gain is present but treadmill `incline` and `distance` exist (e.g., `inclineRunn` + `distance`), ascent is derived by integrating positive vertical: `delta_vertical = max(incline,0)% * delta_distance`.
- Otherwise altitude-derived ascent is computed as the sum of positive elevation deltas.

Developer field names
---------------------
Common developer-named fields are auto-detected when present, including:
- `total_ascent`, `total_gain`, and variants (treated as cumulative ascent)
- `total_distance` (preferred distance source, used over `enhanced_distance`/`distance` when present)
- `inclineRunn` (incline percent)
- The algorithm uses a two-pointer sweep with interpolation at window end for efficiency (O(n) per duration). For the provided seven durations, this is plenty fast for typical activities.

Version discovery
-----------------
If you want me to discover and pin exact library versions, I can run a package query. Alternatively, use this query with a web-enabled agent:

"Find current stable PyPI versions and install instructions for: python-fitparse (aka fitparse), Typer, and any recommended progress/logging helpers for Python CLIs in 2025. Include notes on Python 3.10+ compatibility."
