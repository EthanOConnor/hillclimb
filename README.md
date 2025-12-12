Critical Hill Climb Rate Curve (FIT)
====================================

Small CLI to compute a “critical hill climb rate curve” from a Garmin FIT file. It scans time windows of fixed durations and finds the maximum gross vertical climb in each, then reports both the climb (m) and rate (m/h). The curve is non-increasing with duration.

Data sources
------------
- Prefers NorthPoleEngineering Runn Data Field developer data if present, using a cumulative total gain field.
- Falls back to Garmin altitude (`enhanced_altitude`/`altitude`) by summing positive elevation deltas.

Usage
-----

Requirements: `python-fitparse` (imported as `fitparse`), `typer`, `matplotlib`, `numpy` (see `requirements.txt` for supported ranges).

Optional extras:
- `numba` for ~5× faster exhaustive/all‑windows kernels (auto‑detected).
- `scipy` for improved WR parameter fitting (falls back to grid search if absent).
Install with: `pip install numba scipy`.

Recommended environment setup:

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

All examples below assume either an activated virtualenv or invoking the tooling via `.venv/bin/python`. Plot generation automatically writes Matplotlib cache data to a local `.mplconfig/` directory so you don’t need system-level write access.

Examples (recommended):

```
# Single file (fixed durations)
.venv/bin/python hc_curve.py curve activity.fit -o curve.csv -d 60 120 300 600 1200 1800 3600 --split-plots --goal-min-seconds 120 --personal-min-seconds 60

# Multiple files merged (overlaps handled)
.venv/bin/python hc_curve.py curve Tracklogs/Treadmill/file1.fit Tracklogs/Treadmill/file2.fit -o curve.csv -d 60 120 300 600 1200 1800 3600 --split-plots --goal-min-seconds 120 --personal-min-seconds 60

# Diagnostics (summarize fields and candidate gain keys)
.venv/bin/python hc_curve.py diagnose Tracklogs/Treadmill/*.fit --out fit_diagnostics.txt

# Export canonical time/gain series for external tools
.venv/bin/python hc_curve.py export-series Tracklogs/Treadmill/20387570593_ACTIVITY.fit -o outputs/timeseries.csv

# Exhaustive curve (every second up to activity length)
.venv/bin/python hc_curve.py curve activity.fit -o exhaustive_curve.csv --all --resample-1hz --split-plots --goal-min-seconds 120 --personal-min-seconds 60

# Exhaustive up to 2 hours with 5s steps
.venv/bin/python hc_curve.py curve activity.fit -o exhaustive_curve.csv --exhaustive --max-duration 7200 --step 5 --split-plots --goal-min-seconds 120 --personal-min-seconds 60

# Gain-centric report (best time for target gains)
.venv/bin/python hc_curve.py time activity.fit -g 50,100 200ft --gains-from docs/targets_example.txt -o gain_time.csv --all --plot-wr --gain-units m --magic-gains 50m,100m,200m
```

Options:
- `--output, -o`: Output CSV path (default: `curve.csv`).
- `--json/--no-json`: Write a JSON sidecar next to the CSV (same basename, `.json`) containing metadata and the computed curves.
- `--durations, -d`: Durations in seconds (default: 60, 120, 300, 600, 1200, 1800, 3600).
- `--all`: Exact per-second curve across the whole activity (recommended).
- `--exhaustive`: Evaluate a multi-resolution duration grid (dense 1s up to ~2h, geometric above; tweak short-range density with `--step`).
- `--step`: Step size in seconds for exhaustive durations (default 1s).
- `--max-duration`: Limit maximum duration (seconds) when using `--exhaustive`.
- `--source`: `auto` (default), `runn`, or `altitude`.
- `--verbose, -v`: Verbose logging.
- Plotting defaults: split plots on for the Python CLI; WR overlay off; goal curve hidden below 120s. Use `--plot-wr` to show WR. The Rust CLI keeps a single combined PNG by default—use `--split-plots` there to opt in.
- `--goal-min-seconds`: Hide goal curve/labels below this duration (default 120s).
- `--personal-min-seconds`: Anchor personal curve scaling at best relative point ≥ this duration (default 60s).
- `--engine`: Curve engine `auto|numpy|numba|stride` (auto prefers the Numba kernel when available; `stride` expects resampled 1 Hz data).
- `--parse-workers`: Thread pool size for FIT parsing (0 = auto, 1 = serial).
- `--fast-plot/--no-fast-plot`: Skip heavy annotations on plots for faster rendering (default fast).
- `--profile`: Emit per-stage timing to the log for quick performance investigations.
- `--gains-from`: Load gain targets from a file (one per line, accepts `m|ft` suffix); values merge with `--gains`.
- `time` command extras: `--gains/-g` (comma and space separated), `--gains-from`, `--gain-units`, `--magic-gains`, `--ylog-time`, and `--split-plots` control the gain-centric report; CSV columns remain meters while plots reflect the requested display units.
- `export-series` writes the canonical cumulative gain series after QC/resampling; combine with the Mathematica helper in `docs/mathematica_gain_time.wl` for external validation pipelines.

Rust CLI (hc_curve_rs)
----------------------
The Rust rewrite (`hc_curve_rs/hc_curve_cli`) mirrors the Python features while adding a few defaults tailored to the new Plotters-based renderer:

- Build with `cargo build` (workspace root) and run via `cargo run -p hc_curve_cli -- curve …`.
- Combined PNG output remains the default; add `--split-plots` (or `--split-plots/--no-split-plots`) to write separate `_rate`/`_climb` images. When split plots are requested, the CLI still honours the explicit `--png` path for the combined figure.
- `--ylog-rate` / `--ylog-climb` now clamp the lower bound so flat sessions (zero or negative ascent) render without errors.
- FIT/GPX inputs are parsed in parallel and cached on disk under `.cache/parsed_fit/`, keyed by path, size, and mtime; cached entries are re-keyed to the current file ordering so multi-run workflows stay correct.
- `--profile` emits parse/compute/CSV/plot timing just like the Python CLI. Combine with `--verbose` for more granular tracing.
- All other flags mirror the Python names; see `hc_curve_rs/README.md` for a concise summary of the Rust-specific options and behaviour.

CSV columns
-----------
- `duration_s`: Window length in seconds.
- `max_climb_m`: Max gross vertical climb (meters) found over any window of that length.
- `climb_rate_m_per_hr`: Best average climb rate (m/h) for that duration.
- `start_offset_s`, `end_offset_s`: Window offsets (seconds) from activity start.
- `source`: Data source used (`runn_total_gain` or `altitude`).
- `gain_time.csv` adds `gain_m`, `min_time_s`, `avg_rate_m_per_hr`, `start_offset_s`, `end_offset_s`, `source`, and `note` (e.g., `bounded_by_grid`, `unachievable`).

Notes
-----
- You can pass multiple FIT files; timelines are merged by timestamp. If files overlap, windows extend across files as one continuous activity.
- `--exhaustive` caches cumulative gain samples, prunes gap-only windows, and uses the multi-resolution grid so multi-day sweeps finish quickly without changing results.
- Install `numba` to enable the just-in-time engine that brings another ~5× speedup for exhaustive grids (`pip install numba`).
- `--parse-workers` lets you overlap FIT decoding across files; `--profile` logs parse/merge/curve/plot timings so you can spot slow stages quickly.
- Keep `--fast-plot` enabled for day-to-day use; disable it when you need the detailed annotations on the PNGs.
- Parsed FIT records are cached under `.cache/parsed_fit`; delete the cache if you want to force a re-parse after editing source files.
- If developer total gain is present in any file (e.g., NPE Runn), it is preferred and stitched across files, handling counter resets at file boundaries.
- If no total gain is present but treadmill `incline` and `distance` exist (e.g., `inclineRunn` + `distance`), ascent is derived by integrating positive vertical: `delta_vertical = max(incline,0)% * delta_distance`.
- Otherwise altitude-derived ascent is computed as the sum of positive elevation deltas.
- Additional references: `docs/usage_recipes.md` (common CLI scenarios) and `docs/verification.md` (cross-check workflow for Python/Rust/Mathematica).

Developer field names
---------------------
Common developer-named fields are auto-detected when present, including:
- `total_ascent`, `total_gain`, and variants (treated as cumulative ascent)
- `total_distance` (preferred distance source, used over `enhanced_distance`/`distance` when present)
- `inclineRunn` (incline percent)
- The curve search vectorizes start/end-aligned windows with cached envelope lookups (NumPy `searchsorted`) and gap-aware skipping, keeping per-duration work linear even on multi-day spans.

Web App (WASM)
---------------
A minimal client‑side web app is scaffolded under `hc_curve_rs/hc_curve_web`. It lets you upload FIT/GPX files in the browser, computes the same curves locally (no server), renders interactive Plotly charts, and offers CSV downloads.

Build locally:
- `rustup target add wasm32-unknown-unknown`
- `cargo install trunk`
- `cd hc_curve_rs/hc_curve_web && trunk serve`

Deploy to GitHub Pages:
- Push to `main`; the included workflow `.github/workflows/deploy-web.yml` publishes to `gh-pages`.
- Enable Pages in repo settings (branch: `gh-pages`).
- Visit `https://<owner>.github.io/<repo>/`.

See `docs/web_mvp_spec.md` for architecture and details, and `docs/WEB_APP_USAGE.md` for an end‑user guide.
