# Ascent Algorithm Registry + Report Schema (v1)

**Last updated:** 2025-12-18  
**Goal:** Define stable, versioned JSON shapes for Algorithm Lab so comparisons are reproducible and tooling can rely on IDs/params.

## 1) Stable IDs

Algorithms have stable string IDs (examples):
- `hc.source.runn_total_gain.v1`
- `hc.source.runn_incline.v1`
- `hc.altitude.canonical.v1`
- `strava.altitude.threshold.v1`
- `goldencheetah.altitude.hysteresis.v1`
- `twonav.altitude.min_altitude_increase.v1`

IDs are:
- stable across releases (new behavior → new ID suffix `.v2`, etc.)
- used in CLI/UI selection, JSON outputs, and regression harnesses

## 2) Algorithm config object

Serialized form (`AscentAlgorithmConfig`) uses:
- `id`: stable algorithm ID
- `params`: algorithm parameters

Example:
```json
{
  "id": "hc.altitude.canonical.v1",
  "params": {
    "gain_eps_m": 0.5,
    "smooth_sec": 0.0
  }
}
```

## 3) Parameter hashing

Every algorithm output includes:
- `params`: the serialized config object
- `params_hash`: SHA‑256 hex of the canonical JSON bytes for `params`

Rationale:
- prevents “silent” parameter drift in reports
- makes regression comparisons stable even when UI defaults change

## 4) Single‑algorithm result object

`AscentAlgorithmResult` (Rust core) represents a computed ascent series + diagnostics for one algorithm:
```json
{
  "algorithm_id": "hc.altitude.canonical.v1",
  "params": { "id": "...", "params": { "...": 0 } },
  "params_hash": "…sha256hex…",
  "diagnostics": {
    "qc_segments_removed": 0,
    "qc_gain_removed_m": 0.0,
    "idle_time_pct": 0.12,
    "gain_eps_m": 0.5,
    "smooth_sec": 0.0,
    "resample_1hz_requested": true,
    "resample_applied": true,
    "resample_skipped_reason": null
  },
  "series": {
    "times_s": [0.0, 1.0, 2.0],
    "gain_m": [0.0, 0.0, 0.5]
  },
  "total_span_s": 3600.0,
  "total_gain_m": 500.0,
  "gaps": [
    { "start": 1200.0, "end": 1900.0, "length": 700.0 }
  ]
}
```

Notes:
- Units: meters (`*_m`) and seconds (`*_s`).
- `idle_time_pct` is altitude‑algorithms only; treadmill algorithms may emit `null`.
- `resample_applied=false` with a non-null `resample_skipped_reason` indicates guardrail skipping.

## 5) Compare report object (planned)

Algorithm Lab “compare” reports will wrap multiple `AscentAlgorithmResult` objects with:
- `baseline_algorithm_id`
- a results table summary (totals + deltas)
- optional downsampled series and curve overlays

This will remain `schema_version = 1` until the top-level shape changes.
