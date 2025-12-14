# Repository Guidelines

## Project Structure & Module Organization
- `hc_cli.py` contains the Typer CLI and orchestration for the Python implementation.
- `hc_curve.py` contains core parsing/curve math/WR logic and a small CLI shim (`python hc_curve.py …` remains supported).
- `hc_plotting.py` contains the Matplotlib plotting pipeline for the Python CLI.
- Keep helpers grouped by concern (FIT parsing, curve math, plotting) for readability.
- `Tracklogs/` stores example FIT data for local diagnostics; treat as read-only fixtures when writing tests or repro scripts.
- `outputs/` is the recommended scratch space for generated CSV/PNG artifacts; clean it before committing.
- `requirements.txt` pins the minimum runtime packages (`python-fitparse`, `typer`, `matplotlib`, `numpy`); update the README if you add dependencies.
- The script creates `.mplconfig/` and `.cache/` automatically for Matplotlib; no manual setup is required.

## Build, Test, and Development Commands
- `python3 -m venv .venv` then `.venv/bin/pip install -r requirements.txt` to provision the environment.
- `.venv/bin/python hc_curve.py curve Tracklogs/Treadmill/*.fit -o outputs/curve.csv` runs the primary curve calculation and writes the PNG alongside the CSV.
- `.venv/bin/python hc_curve.py diagnose Tracklogs/Treadmill/*.fit --out outputs/fit_diagnostics.txt` inspects field availability before curve tuning.
- Append `--verbose` for debug logging or `--plot-wr` to overlay world-record benchmarks when troubleshooting plots.

## Coding Style & Naming Conventions
- Use Python 3.10+ with 4-space indentation, `snake_case` identifiers, and explicit type hints for public functions and dataclasses.
- Keep functions focused; prefer pure helpers that accept iterables rather than touching globals. Place shared constants near their usage blocks.
- Log with the standard `logging` module at INFO by default; gate noisy details behind `--verbose` and avoid printing directly.
- Match file naming to their role (e.g., `curve_<context>.csv`, `diagnostics_<date>.txt`) for artifacts saved under `outputs/`.
- Maintain feature parity between the Python and Rust CLI implementations; ship new features in both unless explicitly agreed otherwise.

## Testing Guidelines
- A lightweight `unittest` suite lives under `tests/`; run with `.venv/bin/python -m unittest discover -s tests -v`.
- Validate major changes by replaying the sample commands above against multiple FIT inputs.
- When altering data sourcing logic, compare the resulting `curve.csv` and plot to a known-good baseline and note deltas in your PR.
- Capture edge cases such as mixed `runn_total_gain` and altitude fallback by crafting short notebooks or scripts under `outputs/` and referencing them in reviews.

## Notes system (`notes/`)

`notes/` is for inter‑session, inter‑agent memory and coordination. Files:

- `notes/MEMORY.md`
  - Long‑lived architectural memory and “why” behind choices.
- `notes/WORKLOG.md`
  - Chronological log of meaningful work.
- `notes/BACKLOG.md`
  - Detailed technical backlog/roadmap.
- `notes/CHAT.md`
  - Scratchpad for ideas that may become work.
- `notes/SCRUTINY.md`
  - Risk tracking and critical review items.

When editing code in this repo:

- Update `notes/WORKLOG.md` for any substantive change.
- Update `notes/MEMORY.md` when you make or rely on a design decision.
- Prefer adding tasks to `notes/BACKLOG.md` instead of TODOs in code.
- If you spot a risk or subtle behavior, add it to `notes/SCRUTINY.md`.

## Commit & Pull Request Guidelines
- Follow the existing history: short, imperative commit subjects ("Implement ascent QC filtering"); include a focused body when context is non-obvious.
- Each PR should summarize the behavior change, list the commands used for verification, and attach before/after artifacts when plots or CSV outputs change.
- Link related issues, update documentation (README, `AGENTS.md`) for CLI or format changes, and request review from maintainers familiar with FIT parsing when touching ingestion logic.
