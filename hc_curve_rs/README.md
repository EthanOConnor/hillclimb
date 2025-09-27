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

```
cargo run -p hc_curve_cli -- Tracklogs/*.fit --output curve.csv --png curve.png
```

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
