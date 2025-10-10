# Verification Workflow

This checklist keeps the Python and Rust CLIs, plus the Mathematica helpers, in agreement.

## 1. Generate canonical inputs

```bash
# Python CLI (virtualenv assumed)
.venv/bin/python hc_curve.py export-series Tracklogs/Treadmill/20387570593_ACTIVITY.fit -o outputs/series_py.csv
.venv/bin/python hc_curve.py time Tracklogs/Treadmill/20387570593_ACTIVITY.fit -g 50,100 200ft --gains-from docs/targets_example.txt \
    -o outputs/gain_time_py.csv --all --plot-wr

# Rust CLI (from hc_curve_rs workspace)
cargo run -p hc_curve_cli -- export-series Tracklogs/Treadmill/20387570593_ACTIVITY.fit -o outputs/series_rs.csv
cargo run -p hc_curve_cli -- gain-time Tracklogs/Treadmill/20387570593_ACTIVITY.fit -g 50,100 200ft --gains-from docs/targets_example.txt \
    -o outputs/gain_time_rs.csv --all --plot-wr
```

## 2. Compare Python vs Rust outputs

```bash
python docs/verify_gain_time.py outputs/gain_time_py.csv outputs/gain_time_rs.csv --tolerance 1.0
```

The script prints per-gain deltas and warns if any time delta exceeds the tolerance (default 1 s).

## 3. Mathematica cross-check

```wolfram
Get["/full/path/to/docs/mathematica_gain_time.wl"];
series = LoadTimeseries["/full/path/to/outputs/series_py.csv"];
report = MinTimeReport[series, {50, 100, 200}];
comparison = CompareWithCsv[series, "/full/path/to/outputs/gain_time_py.csv"]; (* or gain_time_rs.csv *)
```

- `PlotGainTimeSeries[series]` renders the cumulative ascent curve.
- `PlotGainTimeCurve[series]` plots the gain vs minimum-time curve using the default target set.
- `comparison["Rows"]` contains per-target deltas; `comparison["WithinTolerance"]` should be `True` when differences are within 1 s.
- `GainTimeSelfTest[]` returns built-in `VerificationTest` objects covering the helper functions.

## 4. Document the run

Capture the command lines, key deltas, and any noteworthy QC removals in your PR or review notes. Attach refreshed PNGs/CSVs under `outputs/` for reviewers.
