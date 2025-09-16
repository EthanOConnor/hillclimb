Worklog
=======

Initial Implementation
----------------------
- Scaffolded `hc_curve.py` CLI with Typer; added `requirements.txt` and `README.md`.
- Implemented FIT parsing with fitparse; computed curve via two-pointer sweep.
- Added CSV output and logging; ensured monotonic non-increasing rate.

Diagnostics and Parsing Improvements
------------------------------------
- Added `diagnose` subcommand to summarize field keys and candidate ascent fields.
- Merged multiple FIT files across overlapping timelines; coalesced near-duplicate timestamps.
- Supported source selection: `runn_total_gain`, `runn_incline` from `incline`×`distance`, and `altitude` fallback.
- Preferred developer fields: `total_ascent` / `total_gain` for ascent; `total_distance` for distance.

Visualization
-------------
- Added matplotlib plot generation (PNG) alongside CSV with dual-axis design.

Repo Setup
----------
- To be initialized as a git repository with initial commit including docs.

Environment Setup
-----------------
- Created a local `.venv` virtual environment with `python3 -m venv .venv` and installed requirements; documented the workflow in the README so commands run via `.venv/bin/python`.
- Added automatic Matplotlib cache configuration to use a repo-local `.mplconfig/` directory, preventing font/cache warnings during plotting.
- Tweaked rate/climb plot annotations to alternate offsets and moved legends away from the data so overlays stay legible.

Upcoming Stabilization Work
---------------------------
- Harden mixed-source stitching by fixing dwell fallback and preserving contributing `file_id` during timestamp coalescing.
- Connect the CLI `--source` option to the actual computation path so CSV metadata reflects the selected stream.
- Swap the exhaustive/all-windows sweep to the linear-time concave-envelope implementation for better scaling on long sessions.
- Enable cross-session windows by building a global canonical timeseries and using prefix sums so the curve can report multi-day spans across breaks.
- Introduce a `stats` report CLI that exports detailed tables (curve summary, best windows, timeline overview, gap report, session rollups), derived metrics (peak streaks, consistency scores, inactivity insights, multi-day gains), and supporting visuals (gap-aware curves, heatmaps, calendars) in Markdown/HTML/JSON so the new inactivity data becomes actionable.
