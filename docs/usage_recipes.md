# Usage Recipes

## Overlay world-record benchmarks

```bash
.venv/bin/python hc_curve.py curve Tracklogs/Treadmill/*.fit -o outputs/curve_wr.csv --plot-wr --split-plots --magic 30m,100m,300m
```

- `--plot-wr` overlays the world-record envelope.
- Adjust `--wr-profile` (e.g. `stairs`, `female_overall`) to change the reference model.
- Use `--goal-min-seconds` / `--personal-min-seconds` to control annotation clutter.

## Gain-time deep dive with custom targets

```bash
.venv/bin/python hc_curve.py time activity.fit --all --gains 75,150 --gains-from docs/targets_example.txt \
  --gain-units ft --magic-gains 250ft,500ft --split-plots -o outputs/gain_time_custom.csv
```

- Comma and space separated values are accepted; `--gains-from` merges file-based targets.
- Switch units for display/parsing with `--gain-units ft` (CSV remains meters).
- Add `--plot-wr --plot-personal` to see world-record and personal overlays on the gain-time plots.

## Zero-ascent diagnostics

When the CLI exits with status `3` and `"No positive ascent recorded"`, rerun diagnostics to inspect available ascent sources:

```bash
.venv/bin/python hc_curve.py diagnose activity.fit --out outputs/diagnose.txt --verbose
```

Look for cumulative gain fields (e.g., `total_gain`) or confirm that altitude/Incline is present. Consider adjusting `--source` or relaxing QC thresholds.

## Comparing Python and Rust outputs

```bash
python docs/verify_gain_time.py outputs/gain_time_py.csv outputs/gain_time_rs.csv --tolerance 1.0
```

Pair this with Mathematica `CompareWithCsv` for a third-party check (see `docs/verification.md`).
