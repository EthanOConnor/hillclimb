# Algorithm Lab (UX Spec): Multi‑Algorithm Ascent Comparison + Explainability

**Last updated:** 2025-12-18  
**Audience:** Design/UX, Product, Engineering  
**Goal:** Provide a clear, low-friction UI to compare ascent algorithms, see deltas, and understand *why* results differ.

## 1) Product promise (copy)

> “Different platforms compute ascent differently. Algorithm Lab lets you compare definitions side‑by‑side and see why they disagree.”

Non-negotiables:
- Local processing (CLI + Web): user data stays local by default.
- Deterministic output: results are versioned (`schema_version`) and reproducible.
- Explainability: every algorithm shows what it did (thresholds, smoothing, idle gating, spikes removed).

## 2) Primary user flows

### Flow A: “Just tell me my ascent”
1. User chooses a persona preset (Treadmill / Outdoor).
2. User hits “Compute”.
3. UI shows the default algorithm and total ascent + curve.
4. UI offers “Compare algorithms” (Algorithm Lab mode).

### Flow B: Algorithm comparison (core)
1. User uploads/selects an activity.
2. User selects multiple algorithms (multi-select).
3. User chooses a baseline algorithm (for deltas).
4. UI computes and displays:
   - totals table
   - overlays (curves)
   - “why different” explanations
5. User exports a JSON report (shareable with a coach/dev).

### Flow C: Troubleshooting (advanced)
1. User opens “Diagnostics”.
2. UI shows what data exists (altitude, total_gain, incline, distance).
3. UI recommends which algorithms are valid and flags missing requirements.

## 3) Screen layout (wireframe-ish)

### Top-level layout
```
┌─────────────────────────────────────────────────────────────────────┐
│ Activity: [choose file(s)]   Persona: [Treadmill ▼]   [Compute]      │
│ Status: local processing • schema v1 • parser: fitparser             │
└─────────────────────────────────────────────────────────────────────┘

┌───────────────┬─────────────────────────────────────────────────────┐
│ Algorithms     │ Results                                             │
│ (multi-select) │                                                     │
│               │  [Totals table]  [Export JSON]                       │
│  Baseline: ▼   │  [Overlay plot: climb vs time / curve vs duration]  │
│               │  [Delta view: vs baseline]                           │
│  Presets:      │  [Why different? (collapsible)]                     │
│   • Canonical  │                                                     │
│   • Strava     │                                                     │
│   • GoldenC…   │                                                     │
│   • TwoNav     │                                                     │
│               │                                                     │
│  Advanced ▼    │                                                     │
└───────────────┴─────────────────────────────────────────────────────┘
```

## 4) Algorithm selection model

### Concepts
- **Algorithm**: a named profile with stable ID (e.g., `hc.canonical.v1`, `strava.threshold.v1`).
- **Preset**: a curated set of algorithms + baseline + UI defaults.
- **Baseline**: one algorithm chosen as “reference” for delta columns and delta plots.

### Algorithm list item (copy)
Each algorithm entry shows:
- Name (human): “Canonical (Hillclimb)”
- Short tag: “Altitude hysteresis + idle gating”
- Requirements badges: `ALT`, `DIST`, `INCLINE`, `TG` (greyed out if missing in data)
- Toggle checkbox

## 5) Results table spec

### Columns (minimum)
- Algorithm (name)
- Total ascent (m)
- Δ vs baseline (m, %)
- Samples used (n)
- Time span (h:mm:ss)
- Notes (compact flags)

### Notes flags (examples)
- “QC removed 12.5 m”
- “Idle gated 18% of time”
- “No altitude (skipped)”
- “Resample skipped (gap > 2h)”

Rows are sortable; baseline row is pinned at top.

## 6) Plot spec

### Plots (recommended)
1. **Curve overlay:** best climb (m) vs duration (s) for selected algorithms.
2. **Cumulative gain overlay:** `G(t)` vs time (for explainability and QC/idle intuition).
3. **Delta plot:** `G_alg(t) - G_baseline(t)` vs time (optional, powerful for “why”).

### Default view
- Show curve overlay by default (fits existing hillclimb mental model).
- Add tabs for “Cumulative gain” and “Delta”.

## 7) “Why different?” explainability panel

This panel is the product differentiator. For each algorithm, show:
- **Definition summary** (one sentence)
- **Key parameters** (thresholds, smoothing window, hysteresis, drift/idle rules)
- **Diagnostics** (computed during run):
  - total ascent removed by QC
  - fraction of time gated as idle (altitude algorithms)
  - number of spikes repaired (if applicable)
  - resample applied/skipped + reason

### UX pattern
- Collapsible per algorithm (accordion).
- Provide a “Compare to baseline” view that highlights parameter differences.

## 8) Advanced settings (without overwhelming)

Rules:
- Keep defaults safe; hide advanced behind a single “Advanced” disclosure.
- Group by user intent, not by implementation detail.

Suggested groups:
1. **Data & gaps**
   - Session gap threshold (`session_gap_sec`) tooltip: “Used for session segmentation/diagnostics; windows may span gaps.”
2. **Altitude processing** (only if altitude algorithms selected)
   - Hysteresis (`gain_eps`)
   - Smoothing (`smooth_sec`)
   - Idle gating toggle + thresholds
3. **Quality control**
   - QC on/off
   - QC spec preset selector (Conservative/Default/Aggressive)
4. **Performance**
   - Resample 1 Hz toggle + guardrails (max gap / max points)
   - Engine selector (auto/stride/all-windows) (power users only)

## 9) Error states (actionable)

- Missing required fields:
  - “TwoNav profile needs altitude; this file has none.”
  - Suggest: “Try treadmill algorithms” or “Upload a file with altitude.”
- Guardrail triggered:
  - “Resample refused: gap 9h > 2h guard.”
  - Suggest: “Increase resample-max-gap-sec” or “Disable resampling.”
- Parse failures:
  - “Unable to parse FIT (unsupported message/CRC).”
  - Suggest: “Run `diagnose`” and attach report.

## 10) Engineering hooks (so UX is implementable)

Algorithm Lab requires a versioned “compare” API that returns:
- algorithm registry (IDs, names, requirements, default params)
- for each algorithm:
  - totals + curve points + `G(t)` (optionally downsampled)
  - diagnostics object (for notes + explainability panel)
- `schema_version`, `parser`, and algorithm parameter hash in every report

