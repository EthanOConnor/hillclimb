Hillclimb Web App — Usage Guide
================================

Overview
--------
The web app computes hillclimb duration curves and gain‑centric minimum‑time curves directly in your browser. No data leaves your device.

Supported inputs: Garmin FIT or GPX files. You may select multiple files; timelines are merged by timestamp with sensible overlap handling.

Quick Start
-----------
1) Open the web app (GitHub Pages deployment).
2) Drag & drop or click to select one or more FIT/GPX files.
3) (Optional) Use the Theme toggle (top right) for light/dark mode.
4) (Optional) Adjust controls:
   - Source: auto | runn | altitude
   - All windows: exact per‑second durations (can be slower on long activities)
   - Step (s): step size for the exhaustive grid (when All windows is off)
   - Max duration (s): upper bound when exploring long activities
   - WR overlay: show/hide world‑record envelope overlay
5) Click Compute. Two plots will render:
   - Max Climb vs Duration (with optional WR overlay)
   - Minimum Time vs Gain (with iso‑rate guide lines and optional WR overlay)
6) Use the download links to save `curve.csv` and `gain_time.csv`.

Notes
-----
- Status line and progress bar show progress and any skipped/unsupported files.
- Diagnostics display the selected source, total span, and total gain after QC.
- For very long sessions, prefer the exhaustive mode (All windows off) with a modest Step and Max duration cap for responsiveness.
- WR overlays use built‑in profiles on the client; custom anchors are not used on the web.
- Plots are responsive: they resize to the screen and render side-by-side on wider displays.

CSV Columns
-----------
- curve.csv: `duration_s,max_climb_m,climb_rate_m_per_hr,start_offset_s,end_offset_s,source`
- gain_time.csv: `gain_m,min_time_s,avg_rate_m_per_hr,start_offset_s,end_offset_s,note,source`

Troubleshooting
---------------
- “No valid records”: Check file types (.fit/.gpx) and try another browser; if FIT fails, try a GPX export of the same activity.
- Slow or stuck: Reduce Max duration, increase Step, or disable All windows for very long activities.
- Plot looks empty: Ensure your files contain ascent or distance/altitude fields; check the Diagnostics line for selected source.

Developer Links
---------------
- Architecture/spec: `docs/web_mvp_spec.md`
- Core logic: `hc_curve_rs/hc_curve/src/lib.rs`, `hc_curve_rs/hc_curve/src/wr.rs`
- CLI reference: `README.md`
