# Ascent Bench Harness (CLI)

`hc_curve_cli ascent bench` is the regression/benchmark entrypoint for comparing ascent algorithms across **many** activities while storing **summary metrics only** (no raw series).

## Quick start

Run across a directory of FIT files (shell globbing works):

```bash
cargo run -p hc_curve_cli -- ascent bench Tracklogs/**/*.fit -o outputs/ascent_bench.json
```

Notes:
- Each input file is treated as one activity.
- Parsing is cached under `.cache/parsed_fit/` (use `--clear-cache` when needed).

## Treadmill vs altitude comparison

To compare treadmill incline integration against altitude-derived ascent, pick an incline-based baseline:

```bash
cargo run -p hc_curve_cli -- ascent bench Tracklogs/Treadmill/*.fit \
  --baseline hc.source.runn_incline.v1 \
  --algorithms hc.altitude.canonical.v1,strava.altitude.threshold.v1,goldencheetah.altitude.hysteresis.v1,twonav.altitude.min_altitude_increase.v1 \
  -o outputs/treadmill_vs_altitude.json
```

The console output includes per-algorithm delta distribution summaries (mean/median/p95) vs the chosen baseline.

## Output shape

The JSON file contains:
- `meta`: inputs + parameters + schema + tool version
- `bench`:
  - `activities`: per-file totals and deltas per algorithm (errors recorded; no raw `gain(t)` series)
  - `summary`: aggregated delta statistics per algorithm (and baseline total gain stats)

## CI / strict mode

Use `--strict` to return a non-zero exit code if any activity fails to parse or compute:

```bash
cargo run -p hc_curve_cli -- ascent bench Tracklogs/**/*.fit --strict -o outputs/ascent_bench.json
```

## Next step (recommended)

Add a committed snapshot of the bench JSON (or a reduced “summary-only” snapshot) under a dedicated folder (e.g., `benchmarks/`) and compare new runs against it in CI with tolerances.

