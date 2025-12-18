# Reference Ascent Profiles (Initial Set)

**Last updated:** 2025-12-18  
**Goal:** Provide named, documented algorithm profiles for broad comparison (Algorithm Lab).

This doc describes the first three “external/reference” profiles implemented in Rust core:
- Strava-style thresholding
- GoldenCheetah-style hysteresis
- TwoNav minimum altitude increase thresholding

These profiles are **parameterized** and intentionally explicit about what is modeled vs what is unknown.

## Shared preprocessing (Hillclimb v1)

Unless otherwise noted, altitude-based profiles in this repo currently share a common preprocessing stage:
- optional 1 Hz resample (guarded)
- an “effective altitude path” denoiser (outlier repair + smoothing primitives)
- optional additional rolling-median smoothing (`smooth_sec`)
- idle detection is computed for diagnostics; profiles may or may not gate ascent on idle

This keeps comparisons stable and makes it clear what differs: the *integration rule* and thresholds.

## 1) Strava-style thresholding (altitude)

**Algorithm ID:** `strava.altitude.threshold.v1`  
**Parameters:** `threshold_m`, `smooth_sec`

### Source of truth
Strava states that elevation is smoothed and that ascent is only counted once “climbing needs to occur consistently” beyond a threshold:
- 2 m threshold for activities with strong barometric data
- 10 m threshold for activities without strong barometric data

Reference: Strava Support → “Elevation on Strava FAQs”.  
<https://support.strava.com/hc/en-us/articles/216919427-Elevation-on-Strava-FAQs>

### Modeled behavior in this repo
We model the “consistent climb threshold” as a minimum continuous uphill run:
- sum positive deltas across a contiguous uphill segment
- only commit that segment to total ascent if the segment’s total ≥ `threshold_m`

Default in registry: `threshold_m = 2.0` (barometric-style).

## 2) GoldenCheetah-style hysteresis (altitude)

**Algorithm ID:** `goldencheetah.altitude.hysteresis.v1`  
**Parameters:** `hysteresis_m`, `smooth_sec`

### Source of truth
GoldenCheetah’s docs describe using an elevation hysteresis threshold; the FAQ notes a hysteresis value of 3 m.

Reference: GoldenCheetah Wiki → FAQ Metrics.  
<https://github.com/GoldenCheetah/GoldenCheetah/wiki/FAQ---METRICS>

### Modeled behavior in this repo
We model hysteresis as a Schmitt-trigger style threshold filter on altitude:
- maintain a reference altitude
- add ascent only when altitude rises at least `hysteresis_m` above the reference
- reset reference when altitude drops at least `hysteresis_m` below the reference

Default in registry: `hysteresis_m = 3.0`.

## 3) TwoNav minimum altitude increase (altitude)

**Algorithm ID:** `twonav.altitude.min_altitude_increase.v1`  
**Parameters:** `min_increase_m`, `smooth_sec`

### Source of truth
TwoNav documents a “Minimum Altitude Increase” setting and explains it is used to prevent small oscillations from being counted as elevation gain; example values include a default of 5 m and suggestions to raise it.

References:
- TwoNav Support: “How to configure your elevation gain”.  
  <https://support.twonav.com/hc/en-us/articles/360018610339-How-to-configure-your-elevation-gain>
- TwoNav Support: “Configure in the GO app the minimum elevation gain”.  
  <https://support.twonav.com/hc/en-us/articles/17839864623132-Configure-in-the-GO-app-the-minimum-elevation-gain>

### Modeled behavior in this repo
We model this as the same hysteresis-threshold mechanism as above, with a different default:
Default in registry: `min_increase_m = 5.0`.

## Notes / caveats

- These are “style” profiles: they aim to capture the *published* threshold behavior, but platforms also apply proprietary smoothing, map corrections, device heuristics, and sensor fusion that we do not replicate exactly.
- Algorithm Lab is designed to make these differences explicit via per-algorithm diagnostics and parameter hashing.

