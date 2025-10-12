You are the coding agent picking up the Hillclimb Web MVP.

Context
-------
- Repo root: this project computes hillclimb duration curves and a gain-centric “min time for gain” report from FIT/GPX data.
- Python CLI: `hc_curve.py:1` (reference design and plotting pipeline).
- Rust core: `hc_curve_rs/hc_curve/src/lib.rs:1` (FIT/GPX parsing, QC, curve/gain-time, WR envelope).
- Rust CLIs: `hc_curve_rs/hc_curve_cli/src/main.rs:1` (mirrors Python features via Plotters).
- Web entry: `hc_curve_rs/hc_curve_web` (Leptos CSR + wasm). We’ve added an MVP scaffold: file upload, compute, minimal plots, CSV export, and a Pages workflow.
- Spec: docs/web_mvp_spec.md:1 (architecture, UX, build/deploy, defaults).

Your Goals
----------
1) Make the web MVP fully functional and polished while staying minimal.
2) Ensure the Trunk build works locally and on GitHub Pages.
3) Keep the changes surgical; don’t regress existing CLIs.

Where to Start
--------------
1. Read the spec: docs/web_mvp_spec.md:1.
2. Open web app: hc_curve_rs/hc_curve_web/src/lib.rs:1 and hc_curve_rs/hc_curve_web/index.html:1.
3. Confirm function imports from core (`parse_records`, `compute_curves`, `compute_gain_time`) compile in wasm.

Implementation Tasks
--------------------
- File ingestion
  - Verify `read_files_from_input` properly reads both FIT and GPX; handle unknown extensions with a clear status.
  - Add drag-and-drop if time permits; otherwise keep input-click workflow.

- Compute
  - Confirm Params defaults are acceptable for browser; optionally expose toggles for `source`, `exhaustive` vs `all_windows`, `step`, and `max_duration`.
  - Handle compute failures (show error; keep UI responsive).

- Plots
  - Current plots: climb vs duration and min time vs gain. Ensure proper axes labels and units.
  - Add iso-rate guide lines on gain-time if possible.
  - Optionally add WR overlay using built-in profiles; skip anchors path support on web.

- CSV downloads
  - Validate CSV content aligns with CLI column semantics.
  - Keep URLs fresh (revoke old object URLs if you iterate on memory hygiene).

- UX polish
  - Disable compute while running; show progress updates.
  - Render selected source, total span, total gain (from GainTimeResult) in a small diagnostics area.
  - Add a brief “Nothing leaves your device” note.

- CI/CD (GitHub Pages)
  - The workflow `.github/workflows/deploy-web.yml:1` builds with `--public-url /<repo>/` and publishes `hc_curve_rs/hc_curve_web/dist` to `gh-pages`.
  - After merge to main, enable Pages (from branch `gh-pages`) in repo settings once.

Local Dev Instructions (first-time friendly)
-------------------------------------------
Prereqs: Rust stable toolchain

1) Install tools:
   - `rustup target add wasm32-unknown-unknown`
   - `cargo install trunk`
2) Build & serve the web app:
   - `cd hc_curve_rs/hc_curve_web`
   - `trunk serve` (open the local URL shown in the console)
3) Try it:
   - Use sample files in `Tracklogs/` (treadmill or outdoor) by choosing files in the UI.
   - Click Compute. You should see two plots and two Download links.
4) Release build:
   - `trunk build --release --dist dist`

Pages Deploy Instructions
------------------------
1) Push to `main`; GitHub Actions will build and publish to `gh-pages` via the included workflow.
2) In the GitHub repo settings, enable Pages from branch `gh-pages` (root).
3) Visit `https://<owner>.github.io/<repo>/`.

Notes
-----
- Keep dependencies unchanged unless essential; the web crate already includes optional `wasm` dependencies via feature `chart_plotly`.
- If FIT parsing fails in wasm on some browser, gate FIT behind a status message and accept GPX for MVP; do not block the release — document any limitation.
- Follow AGENTS.md coding style and repo guidelines when touching files.

Deliverables
------------
- Working web app with plots and CSV downloads.
- Verified local trunk build and a passing Pages deployment.
- Brief README section addition describing the web app and linking to the live site.

