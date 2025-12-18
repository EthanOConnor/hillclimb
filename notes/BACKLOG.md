# BACKLOG.md — notes

Detailed technical backlog / roadmap. Prefer adding items here over TODOs in code. Keep items small, testable, and prioritized.

See also: `docs/REVIEW_2025-12-18.md` (comprehensive handoff + decisions).

## P0 (Now — next 1–2 sprints)
- [x] **(ENG/Senior)** ADR: Rust is canonical core; Python is wrapper
  - Acceptance: `notes/MEMORY.md` updated with the decision, migration milestones, and deprecation plan for Python “truth”.
  - Acceptance: one “thin wrapper” path chosen and documented (PyO3/maturin preferred; subprocess acceptable interim).

- [x] **(ENG/Junior)** CI smoke checks for Python + Rust + WASM
  - Acceptance: new workflow under `.github/workflows/` runs on PR/push:
    - Python: install requirements; run unit tests.
    - Rust: `cargo test --workspace` (or `cargo check` as minimum).
    - WASM: `cargo check -p hc_curve_web --target wasm32-unknown-unknown`.
  - Acceptance: repo badges/docs updated only if needed (keep minimal).

- [x] **(ENG/Senior)** Semantics spec: gaps, idle, sessions, overlaps (single definition)
  - Acceptance: new spec doc in `docs/` defining:
    - whether windows may span idle time and/or session gaps
    - what “inactivity gaps” means (timestamp gaps vs idle segments vs both)
    - overlap policy semantics and guarantees
  - Acceptance: Python and Rust match this spec (tests cover the edge cases).

- [x] **(ENG/Junior)** Guardrails for `--resample-1hz` on large gaps
  - Acceptance: resampling no longer fills multi-hour/day gaps by default (segment-aware or max-gap guard).
  - Acceptance: tests reproduce the previous failure mode (huge gap) and confirm bounded behavior.

- [x] **(ENG/Junior)** Rust warnings cleanup (“cargo check is quiet”)
  - Acceptance: `cargo check --workspace` produces no warnings in default configuration.
  - Acceptance: if feasible, CI enforces `RUSTFLAGS="-Dwarnings"` for core crates.

- [x] **(ENG/Junior)** Repo hygiene (untracked local files)
  - Acceptance: decide fate of `AGENTS.md.old` and `docs/gain_time_spec.md` (delete, ignore, or commit as canonical docs).
  - Acceptance: `.gitignore` remains correct (never track `outputs/`, `.cache/`, `.mplconfig/`, etc.).

- [ ] **(ENG/Senior)** FIT parsing dependency strategy (python-fitparse vs fitdecode vs Rust-only)
  - Acceptance: decision recorded with rationale and timeline:
    - (preferred) Rust parsing exposed to Python via bindings, OR
    - Python parser abstraction added and `fitdecode` evaluated as replacement default.
  - Acceptance: no silent behavior changes without a versioned output note in JSON.

- [ ] **(PM/Product)** Personas + defaults + “what counts as ascent”
  - Acceptance: short doc defining primary personas (treadmill incline vs outdoor) and default settings per persona.
  - Acceptance: explicitly define the product stance on “session gaps” and “idle” in UI copy.

- [ ] **(Design/UX)** “Algorithm Lab” UX spec (comparison + explainability)
  - Acceptance: wireframes + copy for:
    - selecting algorithms
    - showing deltas and “why different” explanations
    - exposing advanced settings without overwhelming default users

## P1 (Next)
- [ ] **(ENG/Senior)** `AscentAlgorithm` interface + JSON schema (versioned)
  - Acceptance: Rust core exposes a registry of ascent algorithms with stable IDs and parameter hashing in JSON sidecars.
  - Acceptance: outputs include per-algorithm diagnostics (idle %, smoothing applied, thresholds, removed ascent from QC).

- [ ] **(ENG/Senior)** Add “Algorithm Lab” CLI commands
  - Acceptance: `hc_curve_cli ascent compare … --json` (or similar) produces a multi-algorithm report for a single activity.
  - Acceptance: report includes algorithm-by-algorithm totals + curve deltas + summary table.

- [ ] **(ENG/Junior)** Web UI: multi-algorithm comparison mode
  - Acceptance: multi-select algorithms, recompute, and render overlays + a results table.
  - Acceptance: UI communicates “local processing” and “definitions” clearly; error states are actionable.

- [ ] **(ENG/Senior)** Implement initial external/reference ascent behaviors
  - Acceptance: add at least 3 named profiles with documented parameterization:
    - Strava-style thresholding (parameterized; behavior documented in `docs/`).
    - GoldenCheetah-style hysteresis thresholding.
    - TwoNav minimum altitude increase thresholding.
  - Acceptance: each algorithm produces deterministic output and is included in `ascent compare`.

- [ ] **(ENG/Senior)** Benchmark harness + regression suite
  - Acceptance: harness runs algorithms on `Tracklogs/` and stores summary metrics (not raw data) for regressions.
  - Acceptance: can compare “treadmill incline integration” vs altitude-derived totals and report delta distributions.

- [ ] **(ENG/Senior)** Python wrapper over Rust core (first milestone)
  - Acceptance: Python CLI can run curves/gain-time via Rust core and still produce the same CSV/plots as before (within tolerance), with a feature flag to fall back.

## P2 (Later / Strategic)
- [ ] **(ENG/Senior)** Packaging: publish Python wheels that include Rust core (no Rust toolchain required)
  - Acceptance: pip install gives a working CLI + plotting on macOS/Linux; pin/lock strategy documented.

- [ ] **(ENG/Senior + Design)** “Why platforms disagree” productization
  - Acceptance: web UI can export a shareable report explaining differences (algorithm, smoothing, thresholds, idle gating, baro drift).

## P0 (completed)
- [x] Fix Python dependency naming: `requirements.txt` should depend on `python-fitparse` (module `fitparse`) instead of the dormant `fitparse` 1.2.0 package; update README accordingly.
- [x] Add explicit version bounds/pins for Python deps to avoid numpy/typer/matplotlib breakage; document optional extras (`numba`, `scipy`).
- [x] Split the Python CLI and plotting out of `hc_curve.py` into `hc_cli.py` / `hc_plotting.py` while keeping `python hc_curve.py …` working via a shim.

## P1
- [x] Add JSON export with metadata (inputs, selected source, sampling/QC stats, engine) alongside CSV.
- [x] Add unit tests for parser and curve math using small synthetic FIT‑like fixtures.
- [x] Improve cache invalidation: include schema/version in FIT cache entries and expose `--clear-cache`.

## P2
- [x] Optional smoothing/interpolation for nicer curves on sparse data (configurable, off by default).
- [x] Web UI polish: progress indicator, theme toggle, and better mobile plot layout; keep WASM bundle small.
- [x] Consider upgrading Rust crates (`fitparser` 0.9, `leptos` 0.8) after assessing breaking changes.

## P3 (parking lot / ideas)
- Add richer per-algorithm plots (distribution of step sizes, idle segment overlays, drift estimates).
- Add a “calibration” workflow (e.g., treadmill speed/incline sanity checks; baro drift estimation).
