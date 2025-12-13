# hc_curve_rs

Rust workspace implementing hillclimb curve analysis, CLI tooling, and a Leptos-based web front-end.

## Workspace Layout

- `hc_curve`: Core library providing FIT/GPX ingestion and curve computation APIs.
- `hc_curve_cli`: Command-line wrapper for computing curves and exporting CSV/plots.
- `hc_curve_web`: Web UI built with Leptos (WASM via Trunk).

## Getting Started

### Prerequisites

- Rust toolchain (1.75+ recommended)
- Trunk (`cargo install trunk`) for the web front-end
- `wasm32-unknown-unknown` target for the web build (`rustup target add wasm32-unknown-unknown`)

### Building

```
cargo build
```

### Running the CLI

Basic usage:

```
cargo run -p hc_curve_cli -- curve Tracklogs/Treadmill/*.fit -o outputs/curve.csv --png outputs/curve.png
cargo run -p hc_curve_cli -- gain-time Tracklogs/Treadmill/*.fit -g 50,100 200ft --gains-from docs/targets_example.txt -o outputs/gain_time.csv --png outputs/gain_time.png
```

Key flags:

- `--split-plots` to write separate `_rate` and `_climb` figures (combined PNG remains the default)
- `--fast-plot` to render a lightweight chart (default)
- `--ylog-rate` or `--ylog-climb` for log-scaled axes on split plots
- `--plot-wr` to overlay the world‑record envelope
- `--durations 60,300,600` or `--all --step 1` to control evaluation grid
- `--source auto|runn|altitude` to select data source preference
- `--gain-eps` (meters) and `--smooth` (seconds) to tune altitude-derived ascent (`--smooth` is an optional extra rolling median window, off by default)
- `--raw-sampling` to skip 1 Hz resampling
- `--no-qc` to disable QC censoring; `--qc-spec path.json` to override limits
- `--json` to write a JSON sidecar next to the CSV with metadata and computed curves
- `--clear-cache` to delete `.cache/parsed_fit` before parsing inputs
- `--score-output outputs/scores.csv` to write the scoring table
- `--profile` to log timings for parse/compute/CSV/plot stages
- `--gains-from path.txt` to merge gain targets from a file (one per line, accepts `m|ft` suffix)
- Gain-time command adds: `--gains/-g` (comma and space separated), `--gains-from`, `--gain-units`, `--magic-gains`, `--ylog-time`, and reuses WR/personal plotting toggles. CSV columns mirror the Python CLI (`gain_m`, `min_time_s`, `avg_rate_m_per_hr`, `start_offset_s`, `end_offset_s`, `source`, `note`).

Performance features:

- Parsed FIT records are cached under `.cache/parsed_fit/` keyed by path + mtime + size
- FIT/GPX files are parsed in parallel when multiple inputs are supplied

### Web (Trunk)

```
cd hc_curve_web
trunk serve --open
```

### Deploying to GitHub Pages

The provided workflow (`.github/workflows/deploy.yml`) builds the WASM bundle and deploys `dist/` to GitHub Pages.

```
trunk build --release
```

## Testing

```
cargo test
```
