# FIT Parsing Strategy (Python `python-fitparse` vs `fitdecode` vs Rust-only)

**Last updated:** 2025-12-18  
**Scope:** FIT ingestion/parsing for the Hillclimb project (Python CLI, Rust CLI, Web/WASM, future Python↔Rust bindings).

## Summary decision (executive)

1. **Canonical parsing path: Rust-only.** All “truth” parsing + ascent computation should live in the Rust core (`hc_curve_rs/hc_curve`) and be reused by:
   - Rust CLI (`hc_curve_cli`)
   - Web/WASM (`hc_curve_web`)
   - Python CLI via a thin wrapper (preferred: PyO3/maturin; acceptable interim: subprocess JSON bridge)

2. **Python parsing is a temporary compatibility layer.**
   - Keep `python-fitparse` for the short term (already integrated).
   - Treat `fitdecode` as the preferred Python fallback if Python parsing must remain first-class longer than expected.

This keeps cross-surface outputs comparable and avoids re-implementing FIT edge cases twice.

## Current repo state

- Python reads FIT via `fitparse.FitFile` (package: `python-fitparse`; module: `fitparse`).
- Rust reads FIT via the `fitparser` crate.
- Both then normalize/merge records into a canonical time series, and compute curve + gain-time.

## Why “Rust-only canonical” is the right long-term shape

**Product requirement:** the project’s core value is *comparative ascent algorithms* (“Algorithm Lab”), so we need:
- deterministic, versioned, explainable outputs across CLI + web + notebooks
- minimal surface-to-surface drift from ingestion differences

Rust-only parsing supports this by:
- enabling a single parsing + normalization implementation across all surfaces
- enabling robust fuzzing/bench/regression harnesses against `Tracklogs/`
- avoiding Python GIL/CPU overhead for heavy parsing/normalization

## Python `python-fitparse` vs `fitdecode` (what’s true as of 2025‑12‑18)

### Maintenance / currency
- `python-fitparse` is current on PyPI (2.0.4, Jul 2025) but the project description notes maintainer bandwidth constraints and suggests `fitdecode` as an alternative.
- The PyPI package name `fitparse` is *not* the same thing (it is older/dormant); this repo intentionally depends on `python-fitparse`.
- `fitdecode` is current on PyPI (0.11.0, Aug 2025) and explicitly tracks recent FIT SDK profile versions, including developer-data handling switches.

### API / performance characteristics (high-level)
- `fitdecode` is a streaming decoder and is designed to be performant and thread-safe; its README calls out performance advantages over `fitparse`.
- `python-fitparse` has a simpler object model (`FitFile`, messages, fields) and is widely used, but it tends to encourage “load then iterate” patterns which are less ideal for very large FIT files or multi-file merges.

### Developer fields (critical for this project)
The project relies on developer/custom fields (e.g., treadmill/Runn-derived total gain and incline streams). `fitdecode` exposes a `developer_fields` option; we must ensure it is enabled for parity when evaluating it.

## Recommendation: how we should proceed (timeline)

### P0 (now)
- This doc exists (decision recorded).
- **Output versioning:** every JSON sidecar should include a stable `schema_version` and a `parser` identifier (name + version) so parser swaps can’t be “silent”.

### P1 (next)
- Implement a “thin wrapper” path so Python can call Rust core for:
  - parsing + merge/overlap resolution
  - canonical series export
  - curve + gain-time + (future) ascent algorithm registry
- Add a feature flag in Python to choose:
  - `--backend rust` (default once stable), or
  - `--backend python` (compat mode)

### P2 (later)
- Ship Python wheels bundling the Rust core (maturin) so users don’t need a Rust toolchain.
- Deprecate direct Python parsing by default; keep only as troubleshooting fallback.

## If we *must* keep a Python parser long-term

If constraints force us to keep Python parsing as a primary path:
- Introduce a small parser abstraction (`parse_fit_records(files) -> List[Record]`) and implement:
  - `python-fitparse` backend (current)
  - `fitdecode` backend (candidate default)
- Build a regression suite that compares:
  - parsed record counts + key availability
  - canonical series shape/stats
  - curve/gain-time outputs within tolerances

## Non-goals

- We are **not** adopting Garmin’s official SDK code as a primary dependency in the repo right now; Rust + `fitparser` already meets our near-term needs.

## References
- `python-fitparse` (PyPI): <https://pypi.org/project/python-fitparse/>
- `fitparse` (PyPI, legacy): <https://pypi.org/project/fitparse/>
- `fitdecode` (PyPI): <https://pypi.org/project/fitdecode/>
- `fitdecode` (GitHub): <https://github.com/polyvertex/fitdecode>
