# Personas, Defaults, and “What Counts as Ascent”

**Last updated:** 2025-12-18  
**Audience:** Product, Design/UX, Engineering  
**Goal:** Make defaults and UI copy explicit so results are predictable and comparisons are meaningful.

## Core definitions (product copy)

### What is “ascent” in this product?
**Ascent** is a monotone non-decreasing cumulative total climb `G(t)` (meters) computed from one of:
- a device-provided cumulative total ascent stream (preferred when present), or
- treadmill incline integration (incline × distance), or
- altitude-derived ascent from an “effective altitude path” (smoothing + drift/idle gating + hysteresis).

The product always computes ascent *from the selected definition*; it does not attempt to “correct” other platforms.

### Session gaps (timestamp gaps)
**Session gap** = a period of missing samples where consecutive timestamps differ by more than `session_gap_sec`.

**UI stance (recommended copy):**
> “Data gaps are used for session segmentation and diagnostics. Curve windows may still span gaps, because time still passed even if samples are missing.”

### Idle (altitude-derived only)
**Idle** is detected from motion features (distance/speed/cadence/vertical speed). When idle is detected:
- altitude-derived ascent is gated to **zero** during idle (prevents barometric drift from becoming “climb”)
- treadmill/Runn sources are not affected (they are already cumulative ascent streams)

**UI stance (recommended copy):**
> “Idle detection only applies to altitude-derived ascent. It prevents ‘climb’ from accumulating while stationary.”

## Primary personas

### Persona A: Treadmill incline (Runn / indoor)
**User goal:** “How hard did I climb on the treadmill today?”  
**Primary risks:** missing/incorrect distance, incline spikes, mis-detected indoor altitude drift.

**Default source preference:** Auto (prefers treadmill/Runn sources over altitude).

**Default settings (recommended)**
- `source`: `auto`
- `qc`: enabled
- `gain_eps`: 0.5 m (only matters if altitude is used)
- `smooth`: 0 s (only matters if altitude is used)
- `session_gap_sec`: 600 s (segment + diagnostics only)
- `resample_1hz`: off (enable only for “all windows” modes; guardrails on)
- `engine`: `auto`

**UI defaults**
- Show “Source used” prominently (e.g., “Runn total gain”, “Runn incline”, “Altitude”).
- If altitude is selected in auto mode, show a warning: “No treadmill ascent stream found; using altitude (may drift indoors).”

### Persona B: Outdoor run/hike (GPS + barometer)
**User goal:** “What was the real climbing in this activity?”  
**Primary risks:** barometric drift, GPS spikes, pauses/stops, tunnels/coverage gaps.

**Default source preference:** Auto (falls back to altitude if treadmill sources absent).

**Default settings (recommended)**
- `source`: `auto`
- `qc`: enabled
- `gain_eps`: 0.5 m (reduces noise-chatter ascent)
- `smooth`: 0–3 s (optional; keep off by default until validated)
- `session_gap_sec`: 600 s
- `engine`: `auto`

**UI defaults**
- Expose an “Altitude processing” info panel (“Idle gating”, “Smoothing”, “Hysteresis”) as a collapsible advanced section.
- Provide a one-click “Compare algorithms” entry into Algorithm Lab for power users.

## Secondary personas (useful for UX/testing)

### Persona C: Analyst / coach (comparison power user)
**User goal:** “Why do Strava/Garmin/GC disagree, and which definition should I trust?”  
**Need:** multi-algorithm comparison, exportable report, stable versioned outputs.

### Persona D: Competitive stair climber
**User goal:** “Compare my stair climbs to WR envelopes and personal targets.”

## Defaults enforcement rules (engineering notes)

- Always write JSON sidecars with:
  - `schema_version`
  - `selected_source`
  - key parameters (`gain_eps`, `smooth_sec`, `qc_enabled`, `session_gap_sec`, resample guardrails)
- When defaults change, bump JSON `schema_version` and note in release notes.

