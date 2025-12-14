You are the coding agent picking up the Hillclimb Web App (post‑MVP).

Context
-------
- Repo root: this project computes hillclimb duration curves and a gain-centric “min time for gain” report from FIT/GPX data.
- Python CLI: `hc_curve.py:1` (reference design and plotting pipeline).
- Rust core: `hc_curve_rs/hc_curve/src/lib.rs:1` (FIT/GPX parsing, QC, curve/gain-time, WR envelope).
- Rust CLIs: `hc_curve_rs/hc_curve_cli/src/main.rs:1` (mirrors Python features via Plotters).
- Web entry: `hc_curve_rs/hc_curve_web` (Leptos CSR + wasm). The app is functional and includes: drag‑and‑drop upload, compute controls, WR/personal/session overlays, iso‑rate guides, progress UI, a theme toggle, responsive Plotly charts, and CSV downloads.
- Spec: docs/web_mvp_spec.md:1 (architecture, UX, build/deploy, defaults).

Your Goals
----------
1) Keep the web app working across dependency upgrades and browser quirks.
2) Improve robustness/perf for large files without bloating the bundle.
3) Add lightweight CI smoke checks so regressions are caught early.

Where to Start
--------------
1. Read the spec: docs/web_mvp_spec.md:1.
2. Open web app: hc_curve_rs/hc_curve_web/src/lib.rs:1 and hc_curve_rs/hc_curve_web/index.html:1.
3. Confirm function imports from core (`parse_records`, `compute_curves`, `compute_gain_time`) compile in wasm and that Plotly is present on the page.

Implementation Tasks (next)
---------------------------
- CI smoke checks
  - Add a minimal GitHub Actions workflow to run:
    - Python unit tests (`.venv/bin/python -m unittest …` or `python -m unittest …` with `pip install -r requirements.txt`)
    - `cargo check -p hc_curve_cli`
    - `cargo check -p hc_curve_web --target wasm32-unknown-unknown`
  - Keep it fast and dependency‑light; don’t add heavy test infra.

- Large‑file resilience
  - Improve progress fidelity (more granular phases, yield more often, or chunked parsing/compute).
  - Consider optional Web Worker offload if the UI becomes unresponsive for multi‑hour files.

- UX clarity
  - Make error states actionable (e.g., “FIT unsupported by browser; try GPX”).
  - Consider exposing `gain_eps`/`smooth` and QC toggles if user reports noisy altitude plots.

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
- If FIT parsing fails in wasm on some browser, degrade gracefully (status message, accept GPX), and document any limitation.
- Follow AGENTS.md coding style and repo guidelines when touching files.

Deliverables
------------
- CI smoke workflow that guards CLI + WASM compilation.
- Any perf/UX improvement shipped with updated `docs/WEB_APP_USAGE.md`.
