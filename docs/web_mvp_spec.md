Hillclimb Web MVP — Static Site Spec
====================================

Objective
---------
Publish a static, client‑side web app (GitHub Pages) to upload FIT/GPX files, compute hillclimb duration curves and gain‑time metrics entirely in the browser, and render interactive plots with CSV downloads.

Architecture
------------
- Frontend: Leptos CSR app in `hc_curve_rs/hc_curve_web` compiled to WebAssembly.
- Core compute: Rust crate `hc_curve` (FIT/GPX parsing, QC, curve/gain‑time) compiled to wasm with `default-features = false` and `features = ["wasm"]` via the web crate.
- Charts: Plotly via CDN. Rust renders by calling global `Plotly.newPlot` through `web-sys` interop.
- Hosting: GitHub Pages (gh-pages branch) built by Trunk.

User Flow
---------
1. User uploads one or more FIT/GPX files (local only; no upload to server).
2. App parses records per file (`parse_records`), merges, applies QC.
3. App computes duration→gain curve (`compute_curves`) and gain→min‑time (`compute_gain_time`).
4. App renders plots and exposes CSV downloads.

MVP UI
------
- Header with title and a one‑liner about local processing.
- Controls: file input (multiple), compute button, status line.
- Plots: two div containers (curve and gain‑time) rendered via Plotly.
- Downloads: two links for `curve.csv` and `gain_time.csv` (revealed after compute).

Files Added/Modified
--------------------
- `hc_curve_rs/hc_curve_web/index.html`: Trunk entry, loads Plotly, styles, and mounts the app.
- `hc_curve_rs/hc_curve_web/Trunk.toml`: Trunk config (`dist = "dist"`).
- `hc_curve_rs/hc_curve_web/src/lib.rs`: Extends Leptos app to read files, compute, plot, and export CSVs.
- `docs/web_mvp_spec.md`: This spec.
- `.github/workflows/deploy-web.yml`: CI to build with Trunk and deploy to `gh-pages` (see below).

Build & Run Locally
-------------------
Prereqs: Rust stable, wasm target, Trunk.

- `rustup target add wasm32-unknown-unknown`
- `cargo install trunk`
- `cd hc_curve_rs/hc_curve_web`
- `trunk serve` (for local dev), or `trunk build -d dist -r`

GitHub Pages Deployment
-----------------------
- Action publishes `hc_curve_rs/hc_curve_web/dist` to `gh-pages`.
- For repo‑scoped pages, build with `--public-url /<repo>/` to fix asset paths.

Notes & Defaults
----------------
- Defaults mirror CLI: `exhaustive` mode, 1 Hz resample, QC enabled, concave envelope.
- Gain‑time default targets (meters): 50,100,150,200,300,500,750,1000.
- World‑record anchors use built‑in profiles; no anchor file path on web.

Future Enhancements
-------------------
- UI controls for WR overlays, personal/goal curves, and split plots.
- Idle detection display and session segmentation summaries.
- Larger file support with progress feedback.

