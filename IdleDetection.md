Here’s a foolproof, physically-grounded idle detection + climb/descent suppression design you can drop into your pipeline without changing results when you’re actually moving.

⸻

Design goals
	•	Detect true idle (standing, fiddling, eating) vs very slow movement (steep steps).
	•	Suppress ascent/descent accumulation during idle (baro jitter, GNSS wander).
	•	Be robust to noisy speed and altitude, and hysteretic (no flapping).
	•	Work indoors (treadmill/Runn) and outdoors (GNSS/baro).
	•	O(n), no allocations in inner loops, vectorizable/Numba-friendly.

⸻

Core idea (state machine + gating)

Build a two-state machine with hysteresis:
	•	MOVING: ascent/descent allowed
	•	IDLE: ascent/descent suppressed (U(t) flat)

Use multiple sensors with sane OR/AND logic:
	•	Outdoors: speed (from distance/t), cadence (if present), distance in window, vertical speed, GNSS jitter floor.
	•	Indoors: Runn speed/incline or treadmill speed, cadence if present.

Hysteresis:
	•	Enter IDLE only when there’s strong evidence of not moving for ≥ τ_enter.
	•	Exit IDLE as soon as there’s clear evidence of movement sustained for ≥ τ_exit.

Then gate ascent integration by the MOVING mask:

dh = np.diff(alt_eff)               # your already-filtered altitude path
moving = ~idle_mask[1:]             # align to dh indices
gross = np.sum(np.maximum(dh, 0.0) * moving)

(If you use the distance-domain closing approach, gating applies the same way.)

⸻

Signals & thresholds (robust + sensor-aware)

Compute on short rolling windows (e.g., length W = 3–5 s, step = 1 sample):
	•	v_med  = median speed in W (m/s).
	•	cad_med = median cadence in W (spm), if available.
	•	ds_W   = distance advanced in W (m).
	•	vv_med = median vertical speed in W from alt_eff (m/s).
	•	σ_gnss = horizontal jitter (CEP or robust MAD of (Δx, Δy)/Δt) in W; gives a noise floor for speed.

Dynamic thresholds (auto-tuned per session):

v_noise = max(0.15, 3.0 * median_speed_when_still)  # GNSS jitter floor; learn from first minute or from low-HR/still parts
V_ON   = max(0.25, v_noise * 2.0)   # must exceed this to be "clearly moving"
V_OFF  = max(0.15, v_noise * 1.2)   # below this to consider "not moving"

CAD_ON  = 12   # spm to exit idle (any stepping)
CAD_OFF = 6    # spm to enter idle (no stepping)

DS_ON   = 3.0  # m advanced in W to exit idle
DS_OFF  = 1.5  # m advanced in W to enter idle

VV_ON   = 0.05 # m/s vertical speed to exit idle (stairs/step-ups)
VV_OFF  = 0.02 # m/s to enter idle

τ_enter = 2.0  # seconds below OFF thresholds to enter idle
τ_exit  = 1.0  # seconds above ON thresholds to exit idle

Indoors (treadmill/Runn): ignore GNSS/jitter; use Runn_speed, Runn_incline, cadence. Set V_ON=0.1, V_OFF=0.05 (treadmill speed is clean).

Steep/slow uphill: allow cadence/vertical speed to punch through even when horizontal speed is tiny:
	•	Treat as moving if cad_med >= CAD_ON OR vv_med >= VV_ON OR v_med >= V_ON.
	•	Treat as idle only if all of cad_med < CAD_OFF AND vv_med < VV_OFF AND v_med < V_OFF AND ds_W < DS_OFF.

⸻

State machine (hysteresis with timers)

def detect_idle(t, speed, dist, alt_eff, cadence=None, indoor=False):
    W = 5.0  # window seconds
    # compute rolling medians/metrics over W
    v_med, ds_W, vv_med, cad_med, v_noise = rolling_features(...)

    # choose thresholds per mode (outdoor/indoor) as above
    T_enter = τ_enter; T_exit = τ_exit
    idle_mask = np.zeros_like(speed, dtype=bool)
    in_idle = False
    t_last_switch = t[0]

    # timers that accumulate contiguous evidence
    below_off_time = 0.0
    above_on_time  = 0.0

    for i in range(len(t)):
        is_moving_on  = (v_med[i] >= V_ON) or (cad_med[i] >= CAD_ON) or (vv_med[i] >= VV_ON) or (ds_W[i] >= DS_ON)
        is_idle_off   = (v_med[i] <  V_OFF) and (cad_med[i] <  CAD_OFF) and (vv_med[i] <  VV_OFF) and (ds_W[i] <  DS_OFF)

        if in_idle:
            if is_moving_on:
                above_on_time += dt(i)
                below_off_time = 0.0
            else:
                above_on_time = 0.0
                # stay idle until on-evidence accumulates
            if above_on_time >= T_exit:
                in_idle = False
                above_on_time = 0.0
                t_last_switch = t[i]
        else:
            if is_idle_off:
                below_off_time += dt(i)
                above_on_time = 0.0
            else:
                below_off_time = 0.0
            if below_off_time >= T_enter:
                in_idle = True
                below_off_time = 0.0
                t_last_switch = t[i]

        idle_mask[i] = in_idle

    return idle_mask

Extra “free” heuristics
	•	If the device emits auto-pause/lap/button pause → force IDLE.
	•	If heart-rate is in a low-variance band and all motion features are low → bias toward IDLE (but never use HR alone).

⸻

What to do during IDLE
	1.	Clamp ascent integration
Gate positive and negative increments:

dh = np.diff(alt_eff)
moving = ~idle_mask[1:]
dh_pos = np.where(moving, np.maximum(dh, 0.0), 0.0)
dh_neg = np.where(moving, np.minimum(dh, 0.0), 0.0)  # for reporting loss only
gross = float(dh_pos.sum())

(If you use the run-hysteresis ascent integrator we discussed earlier, keep that and multiply its “active” flag by moving.)

	2.	Hold the altitude baseline (optional but recommended for plots & WR windows)
During idle, “pin” alt_eff flat:

if idle_mask[i]: alt_eff[i] = alt_eff[idle_start_index]

or allow a very slow drift clamp (e.g., |dH/dt| ≤ 0.002 m/s) to soak baro drift without creating fake ascent.

	3.	Baro drift correction timeout
If you do periodic baro→GNSS offset calibration, still allow updates during idle, but apply the offset change to all subsequent points, not as a sudden jump inside idle. That way U(t) remains flat.

⸻

Edge-cases & safeguards
	•	GNSS wobble while stationary
Speed can read 0.2–0.4 m/s from jitter. That’s why we (a) estimate a noise floor v_noise, (b) require multiple conditions for IDLE (speed + cadence + ds + vv), and (c) use timers.
	•	Steep, very slow stairs
Horizontal speed is tiny; cadence or vertical speed pulls you into MOVING. You won’t get mis-classified idle.
	•	Indoor treadmill
Trust Runn speed/incline; ignore GNSS (often absent). Set lower V_ON/V_OFF and rely on cadence/vertical speed.
	•	Short micro-pauses (< τ_enter)
Hysteresis prevents flapping; ascent continues if you’re just hesitating between steps.
	•	False positives at switchbacks (turn-in-place)
ds_W still advances; cadence > CAD_ON for steps → exit idle quickly.

⸻

Integration points (in your code)
	•	After you build alt_eff (your morphological closing pipeline), compute features over a rolling window and call detect_idle(...) to produce idle_mask (length n).
	•	When forming U(t) (cumulative positive climb for window scans), gate with ~idle_mask[1:].
	•	When building per-session totals, use the same gated dh_pos.
	•	For WR/plotting, optionally replace alt_eff[i] with a flat hold during idle for prettier lines (U(t) will be flat either way).

⸻

Diagnostics to log (cheap & useful)
	•	%time idle, #idle segments, median idle duration.
	•	Thresholds chosen: V_OFF/V_ON, CAD_OFF/ON, VV_OFF/ON, DS_OFF/ON, τ_enter/exit.
	•	Top 3 idle intervals with start/end timestamps and raw vs gated ascent during them (should be zero after gating).
	•	If available: v_noise (GNSS), max baro drift rate during idle (for sanity).

⸻

Complexity & performance
	•	Rolling medians/metrics over short windows: O(n) with deques / incremental stats.
	•	State machine: O(n), scalar work.
	•	Fully compatible with your Numba / vectorized paths (no allocations in the hot loop). You can precompute the window metrics with NumPy and run the state machine in Numba if you like.
