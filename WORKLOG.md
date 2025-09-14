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

