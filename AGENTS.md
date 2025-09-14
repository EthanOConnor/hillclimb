Project: Critical Hill Climb Rate Curve (FIT)
============================================

Purpose
-------
Compute a “critical hill climb rate curve” from one or more Garmin FIT files, merging timelines and evaluating maximum gross vertical climb over fixed durations. Outputs a CSV and a PNG visualization.

CLI Overview
------------
- `hc_curve.py curve <files...>`: Compute curve CSV (+PNG by default)
- `hc_curve.py diagnose <files...>`: Summarize field keys and candidate ascent fields

Key Behaviors
-------------
- Sources (auto): prefer `runn_total_gain` → `runn_incline` (incline × distance) → `altitude`.
- Distance priority: developer `total_distance` > `enhanced_distance` > `distance`.
- Merge across files by timestamp; windows extend across overlaps; coalesce near-duplicate timestamps.
- Curve computed via two‑pointer sliding windows with linear interpolation at window end.
- Post-process to enforce non-increasing rate with increasing duration.

Dev Setup
---------
- Python 3.10+
- Install: `pip install -r requirements.txt`
- Run curve: `python hc_curve.py curve Tracklogs/Treadmill/*.fit -o curve.csv`
- Diagnostics: `python hc_curve.py diagnose Tracklogs/Treadmill/*.fit --out fit_diagnostics.txt`

Contributing Notes
------------------
- Use type hints; keep functions small and testable.
- Log informative INFO messages; use `--verbose` for DEBUG.
- Avoid changing file formats or CLI flags without updating README.
- Keep dependencies minimal: fitparse, typer, matplotlib.

