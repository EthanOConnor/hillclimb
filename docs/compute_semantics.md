# Hillclimb Compute Semantics (Gaps, Idle, Sessions, Overlaps)

**Status:** Canonical behavior spec (2025-12-18)  
**Goal:** Keep Python + Rust in semantic lockstep so comparisons are meaningful.

This document defines intended behavior for:
- how multiple files are merged
- how ascent sources are selected and stitched
- what “gaps” mean, and how they affect window scans
- what “idle” means, and how it affects altitude-derived ascent

---

## 1) Timeline & Records

### Canonical record fields
Each input file yields per-record samples with:
- `t`: timestamp (seconds since epoch)
- `tg`: optional cumulative total gain / total ascent (developer field)
- `inc`: optional incline percent (treadmill)
- `dist`: optional distance meters (prefer developer `total_distance`, else `enhanced_distance`, else `distance`)
- `alt`: optional altitude meters (`enhanced_altitude`/`altitude`)

### Merge ordering and de-duplication
- Records from all inputs are merged by timestamp ascending.
- Samples within `merge_eps_sec` coalesce into one point (preserving non-null data and preferring higher-priority distance fields).

### Overlap policy for cumulative total gain (`tg`)
When two files overlap in time and both contain `tg`:
- `overlap_policy=file:last`: the later file index is authoritative for `tg` in overlap spans.
- `overlap_policy=file:first`: the earlier file index is authoritative for `tg` in overlap spans.

Non-winning `tg` samples are suppressed during overlap spans so the stitched cumulative `tg` stream is well-defined.

---

## 2) Source Selection (Auto / Runn / Altitude)

### Sources
- `runn_total_gain`: use `tg` as the ascent signal (cumulative).
- `runn_incline`: integrate positive vertical from incline percent and distance deltas:  
  `Δgain = max(inc, 0)% * Δdist`
- `altitude`: derive ascent from an effective altitude path.

### Auto selection
Auto mode selects the best available source per sample with a dwell time to reduce flapping:
`runn_total_gain` > `runn_incline` > `altitude`.

When switching sources, the canonical cumulative series is offset so overall `G(t)` remains continuous and non-decreasing.

---

## 3) `G(t)` (Cumulative Gain) Semantics

### Fundamental rule
All ascent computations produce a **monotone non-decreasing** cumulative gain series `G(t)` (meters).

Between samples, `G(t)` is treated as piecewise-linear for window evaluation/interpolation.

### Altitude-derived ascent (only)
Altitude-derived ascent is computed from an “effective altitude path” with:
- spike/outlier repair
- smoothing / morphological closing
- idle detection + ascent gating (below)
- hysteresis (`gain_eps`) applied once per contiguous positive ascent run (“uprun epsilon”)

---

## 4) Gaps vs Idle (and why they’re different)

### Session gaps (“timestamp gaps”)
**Definition:** a session gap exists when consecutive canonical samples have `dt > session_gap_sec`.

Session gaps are represented as intervals `[t_i, t_{i+1}]` where there are no samples.

### Idle segments (altitude-derived only)
**Definition:** idle segments are detected from motion features (distance advance, speed, cadence, vertical speed) on the altitude path.

Idle affects altitude-derived ascent by:
- holding/clamping the effective altitude path during idle to prevent baro drift accumulation
- gating ascent integration so idle segments contribute **zero** to cumulative gain

Idle is *not* a “session gap” for window skipping semantics; it simply yields `G(t)` flat over idle.

---

## 5) Window Scan Semantics (Curve + Gain-Time)

### Window definition
For a duration `D`, a window is `[t, t + D]`.

The curve point for `D` is:
- `max_climb_m = max_t ( G(t + D) - G(t) )`
- `climb_rate_m_per_hr = max_climb_m / D * 3600`

### Gaps: allowed to span, but “gap-only windows” are skipped
Windows are allowed to span session gaps (they include “no-sample / no-gain” time).

For performance (and to avoid pointless candidates), implementations may skip windows that are **fully inside a session gap** when `D` is shorter than that gap:
- A “gap-only window” is one whose start time lies in `[gap.start, gap.end - D]` for a gap interval `[gap.start, gap.end]`.
- Skipping applies primarily to end-aligned candidates where start times are not limited to sample timestamps.

Python and Rust are expected to match this behavior.

### Concave envelope shaping and gaps
Curve post-processing enforces:
- monotone non-decreasing `max_climb_m` with duration
- monotone non-increasing rate with duration (outside special cases)
- optional concave envelope smoothing

When a curve point’s best window **fully spans a session gap** (start before the gap starts and end after the gap ends), curve-shape constraints may be applied segment-wise to avoid over-constraining behavior across a discontinuity.

---

## 6) Resampling (1 Hz)

Some engines/modes require uniform sampling:
- “all windows” exact sweeps
- “stride” engine

Resampling to 1 Hz produces a uniform grid and interpolates cumulative gain to each second.

**Guardrail requirement:** resampling must not blindly fill multi-hour/day gaps by default. The intent is segment-aware resampling or a max-gap-to-fill limit to prevent huge allocations (see backlog P0).

---

## 7) Quality Control (QC) censoring

QC censors implausible ascent spikes by flattening cumulative gain segments when `G(t + W) - G(t)` exceeds a duration-dependent limit.

QC is applied before window scanning so curve/gain-time are computed on the censored `G(t)`.

---

## 8) Where this is implemented

### Python
- Merge + overlap policy: `hc_curve.py` (`_merge_records`, `_compute_tg_overlap_spans`)
- Auto canonical stitching: `hc_curve.py` (`_build_canonical_timeseries`)
- Gap detection: `hc_curve.py` (`_find_gaps`)
- Gap-only window skipping: `hc_curve.py` (`_compute_curve_numpy`, `_numba_curve_kernel`)
- Altitude idle gating: `hc_curve.py` (`_apply_idle_detection`, `_apply_idle_hold`, `_cum_ascent_from_alt`)

### Rust
- Core compute: `hc_curve_rs/hc_curve/src/lib.rs`
- Gap detection: `detect_gaps`
- Gap-only window skipping: `spans_gap` (used by curve engines)
- Altitude preprocessing/idle gating: `effective_altitude_path`, `apply_idle_detection`, `cumulative_ascent_from_altitude`

