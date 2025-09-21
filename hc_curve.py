import sys
import os
import bisect
import csv
import logging
import math
import copy
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict, Any, Set, NamedTuple, Union
import json

import numpy as np
try:
    from scipy.optimize import minimize
except Exception:  # pragma: no cover
    minimize = None  # type: ignore

try:
    import typer
except Exception:  # pragma: no cover
    typer = None  # type: ignore

try:
    from fitparse import FitFile
except Exception:  # pragma: no cover
    FitFile = None  # type: ignore


_BASE_DIR = os.path.dirname(__file__)
_LOCAL_MPL_DIR = os.path.join(_BASE_DIR, ".mplconfig")
if "MPLCONFIGDIR" not in os.environ:
    try:
        os.makedirs(_LOCAL_MPL_DIR, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = _LOCAL_MPL_DIR
    except Exception:
        pass

if "XDG_CACHE_HOME" not in os.environ:
    _local_cache = os.path.join(_BASE_DIR, ".cache")
    try:
        os.makedirs(os.path.join(_local_cache, "fontconfig"), exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = _local_cache
    except Exception:
        pass


# -----------------
# Data structures
# -----------------

@dataclass
class CurvePoint:
    duration_s: int
    max_climb_m: float
    climb_rate_m_per_hr: float
    start_offset_s: float
    end_offset_s: float


SOURCE_NAME_MAP = {
    "tg": "runn_total_gain",
    "incline": "runn_incline",
    "alt_enh": "altitude",
    "alt": "altitude",
}

USER_COLOR = "C0"
GOAL_COLOR = "tab:green"
PERSONAL_STYLE = (0, (3, 3, 1.5, 3))
WR_STYLE = (0, (6, 4))
WR_SAMPLE_SECONDS_MIN = 1e-3

SOURCE_DISPLAY_MAP = {
    "runn_total_gain": "runn (total_gain)",
    "runn_incline": "runn (incline)",
    "altitude": "altitude",
    "mixed": "mixed",
    "auto": "auto",
    "runn": "runn",
}


def _normalize_source_label(label: str) -> str:
    return SOURCE_DISPLAY_MAP.get(label, label)


QC_DEFAULT_SPEC: Dict[float, float] = {
    5.0: 8.0,
    10.0: 12.0,
    30.0: 25.0,
    60.0: 40.0,
    300.0: 150.0,
    600.0: 250.0,
    1800.0: 500.0,
    3600.0: 900.0,
}


def _load_qc_spec(path: str) -> Dict[float, float]:
    with open(path, "r") as f:
        data = json.load(f)
    spec: Dict[float, float] = {}
    for key, value in data.items():
        try:
            window = float(key)
            limit = float(value)
        except (TypeError, ValueError):
            continue
        if window > 0 and limit > 0:
            spec[window] = limit
    return spec


# -----------------
# FIT parsing utils
# -----------------

def _setup_logging(verbose: bool, log_file: Optional[str] = None) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)
    if log_file:
        fh = logging.FileHandler(log_file, mode="w")
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(fmt, datefmt))
        logging.getLogger().addHandler(fh)


def _require_dependency(dep, name: str, install_hint: Optional[str] = None) -> None:
    if dep is None:
        hint = f"\nInstall with: {install_hint}" if install_hint else ""
        raise RuntimeError(
            f"Missing dependency: {name}. {hint}".strip()
        )


def _pick_total_gain_key(sample_values: dict) -> Optional[str]:
    # Heuristic: prefer a developer field that looks like cumulative total gain
    candidates: List[str] = []
    for k, v in sample_values.items():
        kl = str(k).lower().replace(" ", "_")
        if not isinstance(v, (int, float)):
            continue
        if ("gain" in kl or "ascent" in kl or "climb" in kl) and (
            "total" in kl or "cum" in kl or "cumulative" in kl
        ):
            candidates.append(k)
    # Stable preference: keys containing 'total_gain' explicitly first
    for key in candidates:
        if "total_gain" in str(key).lower().replace(" ", "_"):
            return key
    return candidates[0] if candidates else None


def _parse_single_fit_records(fit_path: str, file_id: int) -> List[Dict[str, Any]]:
    _require_dependency(FitFile, "fitparse", "pip install fitparse")
    fit = FitFile(fit_path)
    fit.parse()
    records = list(fit.get_messages("record"))
    out: List[Dict[str, Any]] = []
    # Detect a preferred total gain key for this file
    total_gain_key: Optional[str] = None
    for msg in records[:100]:
        vals = msg.get_values()
        k = _pick_total_gain_key(vals)
        if k is not None:
            total_gain_key = k
            break

    for msg in records:
        vals = msg.get_values()
        ts = vals.get("timestamp")
        if ts is None:
            continue
        if hasattr(ts, "timestamp"):
            t = float(ts.timestamp())
        else:
            try:
                t = float(ts)
            except Exception:
                continue
        alt = vals.get("enhanced_altitude")
        if alt is None:
            alt = vals.get("altitude")
        alt_val: Optional[float] = None
        if alt is not None:
            try:
                alt_val = float(alt)
            except Exception:
                alt_val = None

        tg_val: Optional[float] = None
        # Try preferred key first, else probe this message for any plausible key
        keys_to_try: List[Any] = []
        if total_gain_key is not None:
            keys_to_try.append(total_gain_key)
        probe_key = _pick_total_gain_key(vals)
        if probe_key is not None and probe_key not in keys_to_try:
            keys_to_try.append(probe_key)
        for k in keys_to_try:
            if k in vals and vals.get(k) is not None:
                try:
                    tg_val = float(vals.get(k))
                except Exception:
                    tg_val = None
                break
        if tg_val is None:
            # Common developer field names for cumulative ascent
            for name in (
                "total_ascent",
                "total_ascent_m",
                "total_gain",
                "total_gain_m",
                "total ascent",
                "Total Ascent",
                "Total Gain",
            ):
                if name in vals and vals.get(name) is not None:
                    try:
                        tg_val = float(vals.get(name))
                    except Exception:
                        tg_val = None
                    break

        # Incline (percent) - prefer explicit developer field like 'inclineRunn'
        inc_val: Optional[float] = None
        for k in ("inclineRunn", "incline", "grade"):
            if k in vals and vals.get(k) is not None:
                try:
                    inc_val = float(vals.get(k))
                except Exception:
                    inc_val = None
                break
        if inc_val is None:
            # any key containing 'incline'
            for k, v in vals.items():
                try:
                    if "incline" in str(k).lower() and v is not None:
                        inc_val = float(v)
                        break
                except Exception:
                    pass

        # Distance in meters with preference: developer total_distance > enhanced_distance > distance
        dist_val: Optional[float] = None
        dist_prio: int = 0
        for name in ("total_distance", "total distance", "Total Distance"):
            if vals.get(name) is not None:
                try:
                    dist_val = float(vals.get(name))
                    dist_prio = 3
                except Exception:
                    dist_val = None
                break
        if dist_val is None and vals.get("enhanced_distance") is not None:
            try:
                dist_val = float(vals.get("enhanced_distance"))
                dist_prio = 2
            except Exception:
                dist_val = None
        if dist_val is None and vals.get("distance") is not None:
            try:
                dist_val = float(vals.get("distance"))
                dist_prio = 1
            except Exception:
                dist_val = None

        out.append({
            "t": t,
            "file_id": file_id,
            "alt": alt_val,
            "tg": tg_val,
            "inc": inc_val,
            "dist": dist_val,
            "dist_prio": dist_prio,
        })
    return out


def _merge_records(
    records_by_file: List[List[Dict[str, Any]]],
    merge_eps_sec: float = 0.5,
    overlap_policy: str = "file:last",
) -> List[Dict[str, Any]]:
    # Flatten and sort by time
    allrecs = [r for lst in records_by_file for r in lst]
    if not allrecs:
        return []
    allrecs.sort(key=lambda r: r["t"]) 

    # Precompute overlap spans where >=2 files have tg present
    spans = _compute_tg_overlap_spans(records_by_file, policy=overlap_policy)
    span_starts = [s for s, _, _ in spans]

    merged: List[Dict[str, Any]] = []
    eps = float(merge_eps_sec)
    for rec in allrecs:
        # Suppress losing tg within overlap spans
        if rec.get("tg") is not None:
            winner = _lookup_overlap_winner(spans, span_starts, rec["t"])
            if winner is not None and rec.get("file_id") != winner:
                rec = dict(rec)
                rec["tg"] = None

        if not merged:
            merged.append(rec)
            continue
        last = merged[-1]
        if rec["t"] <= last["t"] + eps:
            # Merge preference: keep tg if present; prefer non-None fields
            chosen = dict(last)
            if rec.get("tg") is not None:
                chosen["tg"] = rec["tg"]
                # Ensure downstream stitching sees the contributing file id
                chosen["file_id"] = rec.get("file_id")
            if chosen.get("alt") is None and rec.get("alt") is not None:
                chosen["alt"] = rec["alt"]
            if rec.get("inc") is not None and chosen.get("inc") is None:
                chosen["inc"] = rec["inc"]
            # Prefer higher-priority distance source (developer total_distance highest)
            rec_dp = rec.get("dist_prio", 0) or 0
            ch_dp = chosen.get("dist_prio", 0) or 0
            if rec.get("dist") is not None and (chosen.get("dist") is None or rec_dp > ch_dp):
                chosen["dist"] = rec.get("dist")
                chosen["dist_prio"] = rec_dp
            # keep earliest time to preserve order
            merged[-1] = chosen
        else:
            merged.append(rec)
    return merged


def _compute_tg_overlap_spans(records_by_file: List[List[Dict[str, Any]]], policy: str) -> List[Tuple[float, float, int]]:
    intervals: List[Tuple[float, float, int]] = []
    for fid, recs in enumerate(records_by_file):
        ts = [r["t"] for r in recs if r.get("tg") is not None]
        if not ts:
            continue
        intervals.append((min(ts), max(ts), fid))
    if not intervals:
        return []
    events: List[Tuple[float, int, int]] = []
    for s, e, fid in intervals:
        events.append((s, +1, fid))
        events.append((e, -1, fid))
    events.sort()
    active: Dict[int, int] = {}
    spans: List[Tuple[float, float, int]] = []
    last_t: Optional[float] = None
    for t, typ, fid in events:
        if last_t is not None and t > last_t:
            files = [f for f, c in active.items() if c > 0]
            if len(files) >= 2:
                winner = min(files) if policy == "file:first" else max(files)
                spans.append((last_t, t, winner))
        active[fid] = active.get(fid, 0) + (1 if typ > 0 else -1)
        last_t = t
    return spans


def _lookup_overlap_winner(spans: List[Tuple[float, float, int]], starts: List[float], t: float) -> Optional[int]:
    if not spans:
        return None
    idx = bisect.bisect_right(starts, t) - 1
    if idx >= 0:
        s, e, w = spans[idx]
        if s <= t <= e:
            return w
    return None


def _build_timeseries(
    merged: List[Dict[str, Any]],
    source: str = "auto",
    gain_eps: float = 0.5,
) -> Tuple[List[float], List[float], str]:
    # Decide source
    any_tg = any(r.get("tg") is not None for r in merged)
    any_incline_dist = any(r.get("inc") is not None and r.get("dist") is not None for r in merged)
    any_alt = any(r.get("alt") is not None for r in merged)

    selected: str
    if source == "runn":
        if any_tg:
            selected = "runn_total_gain"
        elif any_incline_dist:
            selected = "runn_incline"
        else:
            raise RuntimeError("Requested source 'runn' but found neither total gain nor incline+distance.")
    elif source == "altitude":
        if not any_alt:
            raise RuntimeError("Requested source 'altitude' but found no altitude fields.")
        selected = "altitude"
    else:  # auto
        if any_tg:
            selected = "runn_total_gain"
        elif any_incline_dist:
            selected = "runn_incline"
        elif any_alt:
            selected = "altitude"
        else:
            raise RuntimeError("No suitable fields found (need total gain, incline+distance, or altitude).")

    times: List[float] = []
    G: List[float] = []

    if not merged:
        return times, G, "altitude"

    t0 = merged[0]["t"]
    if selected == "runn_total_gain":
        # Build continuous cumulative gain using offsets to handle resets
        base = 0.0
        last_tg: Optional[float] = None
        last_file: Optional[int] = None
        stitched_transition: Dict[Tuple[Optional[int], Optional[int]], bool] = {}
        for r in merged:
            t_rel = r["t"] - t0
            tg = r.get("tg")
            fid = r.get("file_id")
            if tg is None:
                # carry last value forward
                if G:
                    times.append(t_rel)
                    G.append(G[-1])
                else:
                    # No tg yet; just skip until we see one
                    continue
            else:
                if last_tg is not None and tg + 1.0 < last_tg:
                    key = (last_file, fid)
                    if last_file == fid:
                        base += last_tg
                    elif not stitched_transition.get(key, False):
                        base += last_tg
                        stitched_transition[key] = True
                last_tg = tg
                last_file = fid
                cum = base + tg
                # Enforce non-decreasing
                if G and cum < G[-1]:
                    cum = G[-1]
                times.append(t_rel)
                G.append(cum)
    elif selected == "runn_incline":
        # Integrate positive vertical from incline (%) and distance (m)
        last_dist: Optional[float] = None
        last_inc: Optional[float] = None
        cum = 0.0
        last_time: Optional[float] = None
        for r in merged:
            dist = r.get("dist")
            inc = r.get("inc")
            if dist is None and last_dist is None:
                continue
            # carry forward incline if missing
            if inc is None:
                inc = last_inc
            t_rel = r["t"] - t0
            if last_time is not None and t_rel < last_time:
                continue
            if dist is not None and last_dist is not None:
                dd = dist - last_dist
                if dd < 0:
                    # distance reset
                    dd = 0.0
                if inc is not None and inc > 0:
                    cum += dd * (inc / 100.0)
            if dist is not None:
                last_dist = dist
            last_inc = inc if inc is not None else last_inc
            last_time = t_rel
            times.append(t_rel)
            G.append(cum)
    else:
        # Sum positive altitude deltas across merged files
        last_alt: Optional[float] = None
        cum = 0.0
        pending_gain = 0.0
        last_time: Optional[float] = None
        for r in merged:
            alt = r.get("alt")
            if alt is None:
                continue
            t_rel = r["t"] - t0
            if last_time is not None and t_rel < last_time:
                # should not happen due to sorting, but guard
                continue
            if last_alt is not None:
                delta = alt - last_alt
                if delta > 0:
                    pending_gain += delta
                    if pending_gain >= gain_eps:
                        cum += pending_gain
                        pending_gain = 0.0
                elif delta < 0:
                    pending_gain = max(0.0, pending_gain + delta)
            last_alt = alt
            last_time = t_rel
            times.append(t_rel)
            G.append(cum)

    if len(times) < 2:
        raise RuntimeError("Insufficient data to compute curve after merging.")

    return times, G, selected


# -----------------
# Curve computation
# -----------------

def _interp_cum_gain(
    t_target: float, times: List[float], G: List[float], idx_after: int
) -> float:
    """
    Interpolate cumulative gain at an arbitrary time. idx_after is the smallest index
    such that times[idx_after] >= t_target. Returns G(t_target).
    Assumes G is piecewise-linear between samples.
    """
    if idx_after <= 0:
        return G[0]
    if idx_after >= len(times):
        return G[-1]
    t1 = times[idx_after - 1]
    t2 = times[idx_after]
    g1 = G[idx_after - 1]
    g2 = G[idx_after]
    if t2 == t1:
        return g2
    # Linear interpolation
    frac = (t_target - t1) / (t2 - t1)
    frac = 0.0 if frac < 0 else (1.0 if frac > 1 else frac)
    return g1 + frac * (g2 - g1)


def U_eval_many(
    ex: np.ndarray, ey: np.ndarray, slopes: np.ndarray, t: Union[np.ndarray, float]
) -> np.ndarray:
    """Vectorized evaluation of the cumulative gain envelope at times ``t``.

    Outside the knot domain the value is clamped to the endpoint gains to match the
    behaviour of the legacy interpolator.
    """

    if ex.size == 0:
        return np.zeros_like(np.asarray(t, dtype=np.float64))

    t_arr = np.asarray(t, dtype=np.float64)
    j = np.searchsorted(ex, t_arr, side="left")

    result = np.empty_like(t_arr, dtype=np.float64)

    mask_low = j <= 0
    mask_high = j >= ex.size
    mask_core = (~mask_low) & (~mask_high)

    if np.any(mask_low):
        result[mask_low] = ey[0]
    if np.any(mask_high):
        result[mask_high] = ey[-1]
    if np.any(mask_core):
        core_indices = j[mask_core]
        k = core_indices - 1
        result[mask_core] = ey[k] + slopes[k] * (t_arr[mask_core] - ex[k])

    return result


def U_eval_np(ex: np.ndarray, ey: np.ndarray, slopes: np.ndarray, t: float) -> float:
    return float(U_eval_many(ex, ey, slopes, np.asarray([t], dtype=np.float64))[0])


class Gap(NamedTuple):
    start: float
    end: float
    length: float


def _find_gaps(times: List[float], gap_sec: float) -> List[Gap]:
    gaps: List[Gap] = []
    if gap_sec <= 0:
        return gaps
    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        if dt > gap_sec:
            gaps.append(Gap(times[i - 1], times[i], dt))
    return gaps


def _split_sessions_from_gaps(
    times: List[float],
    values: List[float],
    gaps: List[Gap],
) -> List[Dict[str, Any]]:
    if not times:
        return []
    sessions: List[Dict[str, Any]] = []
    n = len(times)
    import bisect

    sorted_gaps = sorted(gaps, key=lambda g: g.start) if gaps else []
    start_idx = 0

    def _append_session(end_idx: int) -> None:
        nonlocal start_idx
        if end_idx <= start_idx or end_idx > n:
            return
        session_times_raw = times[start_idx:end_idx]
        session_values_raw = values[start_idx:end_idx]
        if not session_times_raw:
            return
        base_time = session_times_raw[0]
        base_value = session_values_raw[0]
        session_times = [t - base_time for t in session_times_raw]
        session_values = [v - base_value for v in session_values_raw]
        if session_times[-1] <= 0:
            return
        sessions.append(
            {
                "times": session_times,
                "values": session_values,
                "span": session_times[-1],
                "start_index": start_idx,
                "end_index": end_idx,
            }
        )
        start_idx = end_idx

    for gap in sorted_gaps:
        if gap.start <= times[0]:
            continue
        end_idx = bisect.bisect_left(times, gap.start)
        _append_session(end_idx)
        start_idx = bisect.bisect_left(times, gap.end)
        # ensure we start at first point after gap
    _append_session(n)
    return sessions


def compute_curve(
    times: List[float],
    cumulative_gain: List[float],
    durations: Iterable[int],
    gaps: Optional[List[Gap]] = None,
) -> List[CurvePoint]:
    results: List[CurvePoint] = []

    n = len(times)
    if n == 0:
        return results

    times_np = np.asarray(times, dtype=np.float64)
    gains_np = np.asarray(cumulative_gain, dtype=np.float64)
    if times_np.shape != gains_np.shape:
        raise ValueError("times and cumulative_gain must have the same length")

    ex = times_np
    ey = gains_np

    if ex.size >= 2:
        dt = np.diff(ex)
        dg = np.diff(ey)
        with np.errstate(divide="ignore", invalid="ignore"):
            slopes = np.divide(
                dg,
                dt,
                out=np.zeros_like(dg),
                where=np.abs(dt) > 1e-12,
            )
    else:
        slopes = np.zeros(1, dtype=np.float64)

    U_at_sample = U_eval_many(ex, ey, slopes, times_np)

    gaps_list = gaps or []
    start_time_min = float(ex[0])
    end_time_max = float(ex[-1])

    eps = 1e-9

    for D in durations:
        D = int(D)
        if D <= 0 or start_time_min + D > end_time_max + eps:
            results.append(CurvePoint(D, 0.0, 0.0, start_time_min, start_time_min))
            continue

        best_gain = -math.inf
        best_start = start_time_min

        # Start-aligned windows (t = times_np)
        t_plus = times_np + D
        valid_start = t_plus <= end_time_max + eps
        if np.any(valid_start):
            starts = times_np[valid_start]
            u_start = U_at_sample[valid_start]
            u_end = U_eval_many(ex, ey, slopes, t_plus[valid_start])
            gains_start = u_end - u_start
            if gains_start.size:
                idx = int(np.argmax(gains_start))
                gain_val = float(gains_start[idx])
                if gain_val > best_gain:
                    best_gain = gain_val
                    best_start = float(starts[idx])

        # End-aligned windows (t + D = times_np)
        t_minus = times_np - D
        valid_end = t_minus >= start_time_min - eps
        if np.any(valid_end):
            ends = times_np[valid_end]
            starts_end = t_minus[valid_end]
            u_end = U_at_sample[valid_end]
            u_start = U_eval_many(ex, ey, slopes, starts_end)
            gains_end = u_end - u_start

            if gaps_list and gains_end.size:
                invalid = np.zeros_like(gains_end, dtype=bool)
                for gap in gaps_list:
                    if D <= gap.length + eps:
                        gap_lo = gap.start
                        gap_hi = gap.end - D
                        if gap_hi < gap_lo:
                            continue
                        mask = (starts_end >= gap_lo - eps) & (starts_end <= gap_hi + eps)
                        if np.any(mask):
                            invalid |= mask
                if np.any(invalid):
                    keep = ~invalid
                    gains_end = gains_end[keep]
                    starts_end = starts_end[keep]
            if gains_end.size:
                idx = int(np.argmax(gains_end))
                gain_val = float(gains_end[idx])
                if gain_val > best_gain:
                    best_gain = gain_val
                    best_start = float(starts_end[idx])

        if not math.isfinite(best_gain) or best_gain < 0:
            best_gain = 0.0
            best_start = start_time_min

        window_start_cap = max(start_time_min, end_time_max - D)
        if window_start_cap < start_time_min:
            window_start_cap = start_time_min
        best_start = min(max(best_start, start_time_min), window_start_cap)

        end_offset = best_start + D
        rate_m_per_hr = best_gain / D * 3600.0 if D > 0 else 0.0
        results.append(
            CurvePoint(
                duration_s=D,
                max_climb_m=best_gain,
                climb_rate_m_per_hr=rate_m_per_hr,
                start_offset_s=best_start,
                end_offset_s=end_offset,
            )
        )

    return results


def nice_durations_for_span(
    total_span_s: int,
    fine_until_s: int = 2 * 3600,
    fine_step_s: int = 1,
    pct_step: float = 0.01,
) -> List[int]:
    """Build a multi-resolution duration grid for exhaustive scanning."""
    if total_span_s <= 0:
        return []

    fine_step = max(1, int(fine_step_s))
    fine_limit = min(total_span_s, max(fine_step, int(fine_until_s)))

    out: List[int] = list(range(fine_step, fine_limit + 1, fine_step))
    if 1 <= total_span_s and 1 not in out:
        out.append(1)

    curated: List[int] = []
    for hours in (
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        14,
        16,
        18,
        20,
        22,
        24,
        28,
        32,
        36,
        40,
        44,
        48,
        54,
        60,
        66,
        72,
        84,
        96,
        120,
        132,
        144,
        168,
    ):
        secs = int(hours * 3600)
        if secs <= total_span_s:
            curated.append(secs)

    curated.extend([int(days * 86400) for days in range(1, 15) if days * 86400 <= total_span_s])
    curated.extend(
        [
            int(2 * 86400 * k)
            for k in range(1, 16)
            if 2 * 86400 * k <= total_span_s
        ]
    )
    curated.extend(
        [
            int(7 * 86400 * k)
            for k in range(5, 53)
            if 7 * 86400 * k <= total_span_s
        ]
    )

    if pct_step <= 0:
        pct_step = 0.01
    if total_span_s > fine_limit:
        x = float(fine_limit)

        def _round_nice_seconds(value: float) -> int:
            if value < 3 * 3600:
                return int(round(value / 30.0)) * 30
            if value < 24 * 3600:
                return int(round(value / 60.0)) * 60
            if value < 7 * 86400:
                return int(round(value / 600.0)) * 600
            return int(round(value / 3600.0)) * 3600

        while True:
            x *= 1.0 + pct_step
            d = _round_nice_seconds(x)
            if d > total_span_s:
                break
            curated.append(d)

    grid: Set[int] = {
        d
        for d in out
        if d > 0 and d <= total_span_s
    }
    grid.update(d for d in curated if 0 < d <= total_span_s)
    grid.add(total_span_s)
    durations = sorted(grid)
    return durations


def _diagnose_curve_monotonicity(
    curve: List[CurvePoint],
    epsilon: float = 1e-9,
    inactivity_gaps: Optional[List[Tuple[float, float]]] = None,
) -> None:
    if not curve:
        return
    gap_flags: List[bool] = []
    if inactivity_gaps:
        eps = 1e-6
        for cp in curve:
            spans = False
            for start, end in inactivity_gaps:
                if cp.start_offset_s <= start + eps and cp.end_offset_s >= end - eps:
                    spans = True
                    break
            gap_flags.append(spans)
    else:
        gap_flags = [False] * len(curve)
    last_climb = -1.0
    last_rate = float("inf")
    last_d = None
    last_start = None
    last_end = None
    for idx, cp in enumerate(curve):
        if last_d is not None:
            if cp.max_climb_m + epsilon < last_climb:
                logging.warning(
                    "Non-monotonic max climb: D=%ss climb=%.6f < prev D=%ss climb=%.6f",
                    cp.duration_s,
                    cp.max_climb_m,
                    last_d,
                    last_climb,
                )
                logging.info(
                    "  prev window: start=%0.3fs end=%0.3fs | curr window: start=%0.3fs end=%0.3fs",
                    last_start if last_start is not None else -1,
                    last_end if last_end is not None else -1,
                    cp.start_offset_s,
                    cp.end_offset_s,
                )
        last_climb = max(last_climb, cp.max_climb_m)
        spans_gap = gap_flags[idx]
        if cp.climb_rate_m_per_hr > last_rate + 1e-9 and not (spans_gap or gap_flags[idx - 1]):
            logging.warning(
                "Non-monotonic rate: D=%ss rate=%.6f > prev rate=%.6f (fix may apply)",
                cp.duration_s,
                cp.climb_rate_m_per_hr,
                last_rate,
            )
        if cp.climb_rate_m_per_hr < last_rate:
            last_rate = cp.climb_rate_m_per_hr
        last_d = cp.duration_s
        last_start = cp.start_offset_s
        last_end = cp.end_offset_s


def _upper_concave_envelope(durations: List[int], climbs: List[float]) -> List[float]:
    if not climbs:
        return []
    if len(climbs) == 1:
        return list(climbs)

    hull: List[int] = []
    for idx, (d, c) in enumerate(zip(durations, climbs)):
        hull.append(idx)
        while len(hull) >= 3:
            i1, i2, i3 = hull[-3], hull[-2], hull[-1]
            x1, x2, x3 = durations[i1], durations[i2], durations[i3]
            y1, y2, y3 = climbs[i1], climbs[i2], climbs[i3]
            dx12 = x2 - x1
            dx23 = x3 - x2
            if dx12 <= 0 or dx23 <= 0:
                hull.pop(-2)
                continue
            slope12 = (y2 - y1) / dx12
            slope23 = (y3 - y2) / dx23
            if slope23 > slope12 + 1e-9:
                hull.pop(-2)
            else:
                break

    adjusted = list(climbs)
    for a, b in zip(hull, hull[1:]):
        x1, x2 = durations[a], durations[b]
        y1, y2 = climbs[a], climbs[b]
        if x2 == x1:
            continue
        slope = (y2 - y1) / (x2 - x1)
        for j in range(a + 1, b):
            x = durations[j]
            adjusted[j] = y1 + slope * (x - x1)
    return adjusted


def _enforce_curve_shape(
    curve: List[CurvePoint],
    inactivity_gaps: Optional[List[Tuple[float, float]]] = None,
    apply_concave: bool = True,
) -> None:
    if not curve:
        return
    durs = [cp.duration_s for cp in curve]
    climbs = [cp.max_climb_m for cp in curve]

    # Ensure cumulative gain is non-decreasing with duration
    best = 0.0
    for idx, val in enumerate(climbs):
        if val < best:
            climbs[idx] = best
        else:
            best = val

    span_gaps: List[bool] = []
    if inactivity_gaps:
        eps = 1e-6
        for cp in curve:
            spans = False
            for start, end in inactivity_gaps:
                if cp.start_offset_s <= start + eps and cp.end_offset_s >= end - eps:
                    spans = True
                    break
            span_gaps.append(spans)
    else:
        span_gaps = [False] * len(curve)

    concave_source = list(climbs)
    if apply_concave:
        if any(span_gaps):
            n = len(curve)
            idx = 0
            while idx < n:
                flag = span_gaps[idx]
                j = idx
                while j < n and span_gaps[j] == flag:
                    j += 1
                if not flag:
                    segment_d = durs[idx:j]
                    segment_c = concave_source[idx:j]
                    segment_adj = _upper_concave_envelope(segment_d, segment_c)
                    for offset, value in enumerate(segment_adj):
                        concave_source[idx + offset] = value
                idx = j
        else:
            concave_source = _upper_concave_envelope(durs, concave_source)

    # Final pass to apply values and recompute rates monotonically
    best_climb = 0.0
    best_rate = float("inf")
    prev_flag = span_gaps[0] if span_gaps else False
    for idx, cp in enumerate(curve):
        target_gain = concave_source[idx] if apply_concave else climbs[idx]
        current_flag = span_gaps[idx] if span_gaps else False
        if idx == 0:
            prev_flag = current_flag
        elif current_flag != prev_flag:
            if current_flag:
                best_rate = float("inf")
            prev_flag = current_flag
        gain = max(target_gain, best_climb)
        rate = (gain / cp.duration_s * 3600.0) if cp.duration_s > 0 else 0.0
        if not current_flag and rate > best_rate:
            rate = best_rate
            gain = rate / 3600.0 * cp.duration_s
        else:
            best_rate = rate
        cp.max_climb_m = gain
        cp.climb_rate_m_per_hr = rate
        if gain > best_climb:
            best_climb = gain

# -----------------
# Canonical timeline and all-windows sweep
# -----------------

def _build_per_source_cum(session: List[Dict[str, Any]], gain_eps: float) -> Tuple[List[float], List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    if not session:
        return [], [], [], []
    t0 = session[0]["t"]
    times: List[float] = []
    # TG cumulative (with single-stream resets handled)
    tg_cum: List[Optional[float]] = []
    base_tg = 0.0
    last_tg: Optional[float] = None
    # Incline cumulative
    inc_cum: List[Optional[float]] = []
    cum_inc = 0.0
    last_dist: Optional[float] = None
    last_inc: Optional[float] = None
    # Alt cumulative with hysteresis
    alt_cum: List[Optional[float]] = []
    cum_alt = 0.0
    pending_alt = 0.0
    last_alt: Optional[float] = None
    last_time: Optional[float] = None

    for r in session:
        t_rel = r["t"] - t0
        times.append(t_rel)

        # TG
        tg = r.get("tg")
        if tg is not None:
            if last_tg is not None and tg + 1.0 < last_tg:
                base_tg += last_tg
            last_tg = tg
            tg_val = base_tg + tg
            tg_cum.append(tg_val)
        else:
            tg_cum.append(None)

        # Incline
        dist = r.get("dist")
        raw_inc = r.get("inc")
        inc = raw_inc if raw_inc is not None else last_inc
        if dist is not None and last_dist is not None and inc is not None:
            dd = dist - last_dist
            if dd < 0:
                dd = 0.0
            if inc > 0:
                cum_inc += dd * (inc / 100.0)
        if dist is not None:
            last_dist = dist
        if raw_inc is not None:
            last_inc = raw_inc
        inc_available = last_inc is not None and last_dist is not None
        inc_cum.append(cum_inc if inc_available else None)

        # Altitude
        alt = r.get("alt")
        if alt is not None:
            if last_alt is not None:
                delta = alt - last_alt
                if delta > 0:
                    pending_alt += delta
                    if pending_alt >= gain_eps:
                        cum_alt += pending_alt
                        pending_alt = 0.0
                elif delta < 0:
                    pending_alt = max(0.0, pending_alt + delta)
            last_alt = alt
            alt_cum.append(cum_alt)
        else:
            alt_cum.append(None)

        last_time = t_rel

    # Enforce monotonicity within each source
    def enforce_monotone(arr: List[Optional[float]]):
        lastv = None
        for i, v in enumerate(arr):
            if v is None:
                continue
            if lastv is not None and v < lastv:
                arr[i] = lastv
                lastv = arr[i]
            else:
                lastv = v
    enforce_monotone(tg_cum)
    enforce_monotone(inc_cum)
    enforce_monotone(alt_cum)
    return times, tg_cum, inc_cum, alt_cum


def _build_canonical_timeseries(
    session: List[Dict[str, Any]],
    gain_eps: float,
    dwell_sec: float = 5.0,
    gap_threshold: float = 600.0,
) -> Tuple[List[float], List[float], Set[str], List[Tuple[float, float]]]:
    times, tg_cum, inc_cum, alt_cum = _build_per_source_cum(session, gain_eps)
    if not times:
        return [], [], set(), []
    rank = {"tg": 3, "incline": 2, "alt_enh": 1, "alt": 0}
    # In our arrays, alt_cum includes both enhanced_altitude/altitude merged already; we won't distinguish here
    # Availability selection
    canonical: List[float] = []
    cur_src = None
    base = 0.0
    last = None
    last_switch_t = None
    used_sources: Set[str] = set()
    gaps: List[Tuple[float, float]] = []
    last_time = times[0]
    for idx, t in enumerate(times):
        if idx > 0:
            delta = t - last_time
            if delta > gap_threshold:
                gaps.append((last_time, t))
            last_time = t
        avail: Dict[str, Optional[float]] = {}
        if tg_cum[idx] is not None:
            avail["tg"] = tg_cum[idx]
        if inc_cum[idx] is not None:
            avail["incline"] = inc_cum[idx]
        if alt_cum[idx] is not None:
            avail["alt_enh"] = alt_cum[idx]
        if not avail:
            # carry last
            if canonical:
                canonical.append(canonical[-1])
                continue
            else:
                # no data yet
                canonical.append(0.0)
                continue
        # choose best available source; prefer continuity when possible
        preferred = max(avail.keys(), key=lambda k: rank[k])
        active = cur_src if cur_src is not None and cur_src in avail else None
        if active is None:
            # First selection or current stream vanished -> adopt preferred immediately
            cur_src = preferred
            vnew = avail[cur_src] or 0.0
            last_val = canonical[-1] if canonical else 0.0
            base = last_val - vnew
            last_switch_t = t
        else:
            if preferred != cur_src:
                # allow immediate switch only if rank improves or dwell time elapsed
                if rank[preferred] > rank[cur_src] or (last_switch_t is not None and (t - last_switch_t) >= dwell_sec):
                    cur_src = preferred
                    vnew = avail[cur_src] or 0.0
                    last_val = canonical[-1] if canonical else 0.0
                    base = last_val - vnew
                    last_switch_t = t
        if cur_src is not None:
            used_sources.add(cur_src)
        val = avail[cur_src] if cur_src is not None else 0.0
        val = val or 0.0
        c = base + val
        if last is not None and c < last:
            c = last
        canonical.append(c)
        last = c
    return times, canonical, used_sources, gaps


def _resample_to_1hz(times: List[float], values: List[float]) -> Tuple[List[float], List[float]]:
    if not times:
        return [], []
    import bisect
    t_start = int(times[0])
    t_end = int(times[-1])
    new_times = [float(t) for t in range(t_start, t_end + 1)]
    new_vals: List[float] = []
    for t in new_times:
        j = bisect.bisect_left(times, t)
        new_vals.append(_interp_cum_gain(t, times, values, j))
    # Shift to start at zero
    t0 = new_times[0]
    new_times = [t - t0 for t in new_times]
    return new_times, new_vals


def _apply_qc_censor(
    times: List[float],
    values: List[float],
    qc_spec: Dict[float, float],
) -> Tuple[List[float], int, float]:
    if not qc_spec:
        return list(values), 0, 0.0
    vals = list(values)
    n = len(vals)
    segments_removed = 0
    gain_removed = 0.0
    eps = 1e-9
    for window_sec, limit_gain in sorted(qc_spec.items(), key=lambda kv: kv[0]):
        if window_sec <= 0 or limit_gain <= 0:
            continue
        i = 0
        j = 0
        while i < n:
            if j <= i:
                j = i + 1
            while j < n and times[j] - times[i] <= window_sec + eps:
                gain = vals[j] - vals[i]
                if gain > limit_gain + eps:
                    base = vals[i]
                    delta = gain
                    if delta <= 0:
                        j += 1
                        continue
                    for k in range(i + 1, j + 1):
                        vals[k] = base
                    for k in range(j + 1, n):
                        vals[k] -= delta
                    segments_removed += 1
                    gain_removed += delta
                    continue  # Re-evaluate same j with flattened values
                j += 1
            i += 1
    # Restore monotonicity in case of numerical drift
    last = vals[0]
    for idx in range(1, n):
        if vals[idx] < last:
            vals[idx] = last
        else:
            last = vals[idx]
    return vals, segments_removed, gain_removed
def all_windows_curve(T: List[float], P: List[float], step: int = 1) -> List[CurvePoint]:
    if not T or len(T) < 2:
        return []
    if step <= 0:
        step = 1

    base_dt = T[1] - T[0]
    # Verify uniform spacing (required for the sweep implementation)
    for idx in range(2, len(T)):
        if abs((T[idx] - T[0]) - idx * base_dt) > 1e-6:
            raise RuntimeError("all_windows_curve requires uniform sampling; enable --resample-1hz")

    try:
        import numpy as np
    except Exception:
        np = None

    max_duration = int(round(T[-1]))
    if max_duration < step:
        return []

    results: List[CurvePoint] = []
    if np is not None:
        times = np.asarray(T, dtype=float)
        gains = np.asarray(P, dtype=float)
        for duration in range(step, max_duration + 1, step):
            stride = int(round(duration / base_dt))
            if stride <= 0 or stride >= gains.size:
                break
            deltas = gains[stride:] - gains[:-stride]
            idx = int(np.argmax(deltas))
            gain = float(deltas[idx])
            start_t = float(times[idx])
            end_t = float(times[idx + stride])
            dur_sec = end_t - start_t
            rate = gain / dur_sec * 3600.0 if dur_sec > 0 else 0.0
            results.append(
                CurvePoint(
                    duration_s=int(round(dur_sec)),
                    max_climb_m=gain,
                    climb_rate_m_per_hr=rate,
                    start_offset_s=start_t,
                    end_offset_s=end_t,
                )
            )
    else:
        n = len(P)
        for duration in range(step, max_duration + 1, step):
            stride = int(round(duration / base_dt))
            if stride <= 0 or stride >= n:
                break
            best_gain = -1.0
            best_idx = 0
            for idx in range(n - stride):
                gain = P[idx + stride] - P[idx]
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            start_t = T[best_idx]
            end_t = T[best_idx + stride]
            dur_sec = end_t - start_t
            rate = best_gain / dur_sec * 3600.0 if dur_sec > 0 else 0.0
            results.append(
                CurvePoint(
                    duration_s=int(round(dur_sec)),
                    max_climb_m=best_gain if best_gain > 0 else 0.0,
                    climb_rate_m_per_hr=rate if best_gain > 0 else 0.0,
                    start_offset_s=start_t,
                    end_offset_s=end_t,
                )
            )
    return results


# -----------------
# WR envelope, scoring, goals
# -----------------

def _parse_duration_token(token: str) -> float:
    t = token.strip().lower()
    try:
        if t.endswith('h'):
            return float(t[:-1]) * 3600.0
        if t.endswith('m'):
            return float(t[:-1]) * 60.0
        if t.endswith('s'):
            return float(t[:-1])
        return float(t)
    except Exception:
        return 0.0


WR_PROFILE_PRESETS: Dict[str, Dict[str, Any]] = {
    "overall": {
        "model": "sbpl2",
        "anchors": [
            (0.481 * 3600.0, 1000.0),        # Vertical kilometre
            (1.0 * 3600.0, 1616.0),          # 1h stairs
            (12.0 * 3600.0, 13145.65),       # 12h stairs
            (24.0 * 3600.0, 21720.0),        # 24h overall
        ],
        "treadmill": {"grade": 0.40, "f_max": 3.4, "L_max": 0.85},
        "stairs": {"riser": 0.17, "f_max": 3.4},
        "energy": {"v200_mps": 10.4, "cost_per_m": 4.3, "efficiency": 0.25},
        "bounds": {
            "s_inf": (0.55, 0.90),
            "t_star": (35.0, 7200.0),
            "k": (1.0, 6.0),
        },
        "bounds_sbpl2": {
            "s1": (0.70, 0.96),
            "s2": (0.45, 0.78),
            "t1": (35.0, 1800.0),
            "t2": (3600.0, 90000.0),
        },
        "k1": 4.0,
        "k2": 4.0,
        "coarse_grid": (15, 15, 12),
        "refine_grid": (9, 9, 9),
        "coarse_grid_sbpl2": (12, 12, 10, 10),
        "refine_grid_sbpl2": (9, 9, 7, 7),
        "anchor_weights": {
            0.481 * 3600.0: 45.0,
            1.0 * 3600.0: 60.0,
            12.0 * 3600.0: 20.0,
            24.0 * 3600.0: 10.0,
        },
        "short_anchor_weight": 15.0,
        "default_anchor_weight": 1.0,
    },
    "stairs": {
        "model": "sbpl2",
        "anchors": [
            (0.481 * 3600.0, 1000.0),
            (1.0 * 3600.0, 1616.0),
            (12.0 * 3600.0, 13145.65),
            (24.0 * 3600.0, 18713.0),
        ],
    },
    "female_overall": {
        "anchor_scale": 0.87,
        "treadmill": {"grade": 0.40, "f_max": 3.2, "L_max": 0.80},
        "stairs": {"riser": 0.17, "f_max": 3.2},
        "energy": {"v200_mps": 9.2, "cost_per_m": 4.2, "efficiency": 0.25},
    },
    "female_stairs": {
        "anchor_scale": 0.87,
        "treadmill": {"grade": 0.40, "f_max": 3.2, "L_max": 0.80},
        "stairs": {"riser": 0.17, "f_max": 3.2},
        "energy": {"v200_mps": 9.2, "cost_per_m": 4.2, "efficiency": 0.25},
    },
}

CAP_MODE_SCALE = {
    "conservative": 0.95,
    "standard": 1.0,
    "aggressive": 1.05,
}


def _wr_profile_config(profile: str) -> Dict[str, Any]:
    base = copy.deepcopy(WR_PROFILE_PRESETS["overall"])
    override = WR_PROFILE_PRESETS.get(profile)
    if override is None:
        return base
    for key, value in override.items():
        if isinstance(value, dict):
            base.setdefault(key, {})
            base[key].update(value)
        else:
            base[key] = value
    return base


def _treadmill_cap(grade: float, f_max: float, L_max: float) -> float:
    return float(max(0.0, grade) * f_max * L_max)


def _stairs_cap_double(riser: float, f_max: float) -> float:
    return float(f_max * (2.0 * riser))


def _energy_cap_from_200m(v200_mps: float, cost_per_m: float, efficiency: float) -> float:
    g = 9.81
    return float(efficiency * cost_per_m * v200_mps / g)


WR_VCAP_HARD_MAX = 1.05  # m/s temporary cap pending evidence beyond 30s


def _resolve_vertical_cap(config: Dict[str, Any]) -> Dict[str, float]:
    tm_cfg = config.get("treadmill", {})
    stairs_cfg = config.get("stairs", {})
    energy_cfg = config.get("energy", {})

    caps: Dict[str, float] = {}
    if tm_cfg:
        caps["treadmill"] = _treadmill_cap(
            tm_cfg.get("grade", 0.40),
            tm_cfg.get("f_max", 3.4),
            tm_cfg.get("L_max", 0.85),
        )
    if stairs_cfg:
        caps["stairs"] = _stairs_cap_double(
            stairs_cfg.get("riser", 0.17),
            stairs_cfg.get("f_max", 3.4),
        )
    if energy_cfg:
        caps["energy"] = _energy_cap_from_200m(
            energy_cfg.get("v200_mps", 10.4),
            energy_cfg.get("cost_per_m", 4.3),
            energy_cfg.get("efficiency", 0.25),
        )
    caps = {k: v for k, v in caps.items() if v > 0}
    if not caps:
        caps = {"default": 1.0}
    caps["v_cap"] = min(min(caps.values()), WR_VCAP_HARD_MAX)
    return caps


def _H_sbpl_cap(t_s: np.ndarray, v_cap: float, s_inf: float, t_star: float, k: float) -> np.ndarray:
    t = np.asarray(t_s, dtype=float)
    t = np.maximum(1e-6, t)
    z = np.power(t / t_star, k)
    return v_cap * t * np.power(1.0 + z, (s_inf - 1.0) / k)


def _H_sbpl_cap_scalar(t_s: float, v_cap: float, s_inf: float, t_star: float, k: float) -> float:
    t = max(1e-6, float(t_s))
    z = (t / t_star) ** k
    return v_cap * t * (1.0 + z) ** ((s_inf - 1.0) / k)


def _dH_sbpl_cap_scalar(t_s: float, v_cap: float, s_inf: float, t_star: float, k: float) -> float:
    # Analytic derivative d/dt of H_sbpl_cap: instantaneous rate curve v(t)
    t = max(1e-6, float(t_s))
    z = (t / t_star) ** k
    A = (1.0 + z) ** ((s_inf - 1.0) / k)
    if t_star <= 0:
        return v_cap
    # dz/dt = k * (t/t_star)^(k-1) / t_star
    dz_dt = (k / t_star) * (t / t_star) ** (k - 1.0)
    term = 1.0 + (s_inf - 1.0) * t * dz_dt / (1.0 + z)
    return v_cap * A * term


def _dH_sbpl_cap(t_s: np.ndarray, v_cap: float, s_inf: float, t_star: float, k: float) -> np.ndarray:
    t = np.asarray(t_s, dtype=float)
    return np.array([_dH_sbpl_cap_scalar(x, v_cap, s_inf, t_star, k) for x in t], dtype=float)


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1.0 - p))


def _sbpl_loss(
    t: np.ndarray,
    H: np.ndarray,
    weights: np.ndarray,
    v_cap: float,
    s_inf: float,
    t_star: float,
    k: float,
    penalty_lambda: float = 5000.0,
    overshoot_lambda: float = 5000.0,
) -> float:
    pred = _H_sbpl_cap(t, v_cap, s_inf, t_star, k)
    pred = np.maximum(pred, 1e-9)
    denom = np.maximum(H, 1.0)
    residual = (pred - H) / denom
    under = np.maximum(0.0, -residual)
    over = np.maximum(0.0, residual)
    return float(
        np.sum(weights * residual ** 2)
        + penalty_lambda * np.sum(weights * under ** 2)
        + overshoot_lambda * np.sum(weights * over ** 2)
    )


def _H_sbpl_two_break(
    t_s: np.ndarray,
    v_cap: float,
    s_mid: float,
    s_long: float,
    t_break1: float,
    t_break2: float,
    k1: float,
    k2: float,
) -> np.ndarray:
    t = np.asarray(t_s, dtype=float)
    t = np.maximum(1e-6, t)
    z1 = np.power(t / t_break1, k1)
    z2 = np.power(t / t_break2, k2)
    return (
        v_cap
        * t
        * np.power(1.0 + z1, (s_mid - 1.0) / k1)
        * np.power(1.0 + z2, (s_long - s_mid) / k2)
    )


def _H_sbpl_two_break_scalar(
    t_s: float,
    v_cap: float,
    s_mid: float,
    s_long: float,
    t_break1: float,
    t_break2: float,
    k1: float,
    k2: float,
) -> float:
    t = max(1e-6, float(t_s))
    z1 = (t / t_break1) ** k1
    z2 = (t / t_break2) ** k2
    return v_cap * t * (1.0 + z1) ** ((s_mid - 1.0) / k1) * (1.0 + z2) ** ((s_long - s_mid) / k2)


def _sbpl2_loss(
    t: np.ndarray,
    H: np.ndarray,
    weights: np.ndarray,
    v_cap: float,
    s_mid: float,
    s_long: float,
    t_break1: float,
    t_break2: float,
    k1: float,
    k2: float,
    penalty_lambda: float = 5000.0,
    overshoot_lambda: float = 5000.0,
) -> float:
    pred = _H_sbpl_two_break(t, v_cap, s_mid, s_long, t_break1, t_break2, k1, k2)
    pred = np.maximum(pred, 1e-9)
    denom = np.maximum(H, 1.0)
    residual = (pred - H) / denom
    under = np.maximum(0.0, -residual)
    over = np.maximum(0.0, residual)
    return float(
        np.sum(weights * residual ** 2)
        + penalty_lambda * np.sum(weights * under ** 2)
        + overshoot_lambda * np.sum(weights * over ** 2)
    )


def _fit_sbpl2_parameters(
    t: np.ndarray,
    H: np.ndarray,
    weights: np.ndarray,
    v_cap: float,
    config: Dict[str, Any],
    wr_min_seconds: float,
    penalty_lambda: float = 5000.0,
) -> Dict[str, float]:
    bounds = config.get(
        "bounds_sbpl2",
        {
            "s1": (0.75, 0.95),
            "s2": (0.55, 0.80),
            "t1": (900.0, 5400.0),
            "t2": (5400.0, 86400.0),
        },
    )
    s1_low, s1_high = bounds.get("s1", (0.75, 0.95))
    s2_low, s2_high = bounds.get("s2", (0.55, 0.80))
    t1_low, t1_high = bounds.get("t1", (900.0, 5400.0))
    t2_low, t2_high = bounds.get("t2", (5400.0, 86400.0))

    k1 = config.get("k1", 4.0)
    k2 = config.get("k2", 4.0)

    coarse = config.get("coarse_grid_sbpl2", (12, 12, 10, 10))
    refine = config.get("refine_grid_sbpl2", (8, 8, 7, 7))

    weights = np.asarray(weights, dtype=float)
    if not np.any(weights > 0):
        weights = np.ones_like(t, dtype=float)
    weights = weights / np.sum(weights)

    def loss(s1: float, s2: float, t1: float, t2: float) -> float:
        if not (s1_low < s1 < min(s1_high, 0.9999)):
            return float("inf")
        if not (s2_low < s2 < min(s2_high, s1 - 0.02)):
            return float("inf")
        if not (t1_low <= t1 <= t1_high):
            return float("inf")
        if not (t2_low <= t2 <= t2_high) or t2 <= t1 * 1.05:
            return float("inf")
        return _sbpl2_loss(
            t,
            H,
            weights,
            v_cap,
            s1,
            s2,
            t1,
            t2,
            k1,
            k2,
            penalty_lambda=penalty_lambda,
        )

    best_params: Optional[Tuple[float, float, float, float]] = None
    best_loss = float("inf")

    s1_grid = np.linspace(s1_low, s1_high, coarse[0])
    s2_grid = np.linspace(s2_low, s2_high, coarse[1])
    t1_grid = np.logspace(math.log10(t1_low), math.log10(t1_high), coarse[2])
    t2_grid = np.logspace(math.log10(t2_low), math.log10(t2_high), coarse[3])

    for s1 in s1_grid:
        for s2 in s2_grid:
            if s2 >= s1 - 0.02:
                continue
            for t1_val in t1_grid:
                for t2_val in t2_grid:
                    if t2_val <= t1_val * 1.05:
                        continue
                    current_loss = loss(s1, s2, t1_val, t2_val)
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_params = (s1, s2, t1_val, t2_val)

    if best_params is None:
        raise RuntimeError("Failed to fit WR curve parameters (coarse sbpl2 search).")

    s1_best, s2_best, t1_best, t2_best = best_params

    def refine_range(value: float, lower: float, upper: float, count: int, scale: float = 0.25) -> np.ndarray:
        span = max(scale * value, 0.05 * (upper - lower))
        lo = max(lower, value - span)
        hi = min(upper, value + span)
        if hi <= lo:
            lo, hi = lower, upper
        return np.linspace(lo, hi, max(3, count))

    s1_fine = refine_range(s1_best, s1_low, s1_high, refine[0])
    s2_fine = refine_range(s2_best, s2_low, min(s2_high, s1_best - 0.01), refine[1])
    t1_fine = np.logspace(
        math.log10(max(t1_low, t1_best * 0.5)),
        math.log10(min(t1_high, t1_best * 1.8)),
        max(3, refine[2]),
    )
    t2_fine = np.logspace(
        math.log10(max(t2_low, t2_best * 0.6)),
        math.log10(min(t2_high, t2_best * 1.6)),
        max(3, refine[3]),
    )

    for s1 in s1_fine:
        for s2 in s2_fine:
            if s2 >= s1 - 0.01:
                continue
            for t1_val in t1_fine:
                for t2_val in t2_fine:
                    if t2_val <= t1_val * 1.05:
                        continue
                    current_loss = loss(s1, s2, t1_val, t2_val)
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_params = (s1, s2, t1_val, t2_val)

    if best_params is None:
        raise RuntimeError("Failed to fit WR curve parameters (refine sbpl2).")

    s1_best, s2_best, t1_best, t2_best = best_params

    if minimize is not None:
        log_t1_low = math.log(t1_low)
        log_t1_high = math.log(t1_high)

        def _pack_sbpl2(x_vec: np.ndarray) -> Tuple[float, float, float, float]:
            s1_val = s1_low + (s1_high - s1_low) * _sigmoid(float(x_vec[0]))
            s2_upper = min(s2_high, s1_val - 0.01)
            s2_upper = max(s2_upper, s2_low + 1e-3)
            s2_val = s2_low + (s2_upper - s2_low) * _sigmoid(float(x_vec[1]))
            log_t1 = log_t1_low + (log_t1_high - log_t1_low) * _sigmoid(float(x_vec[2]))
            t1_val_local = math.exp(log_t1)
            delta = math.exp(float(x_vec[3]))
            t2_candidate = t1_val_local + delta
            t2_candidate = max(t2_candidate, max(t2_low, t1_val_local + 1.0))
            t2_val = min(t2_candidate, t2_high)
            if t2_val <= t1_val_local:
                t2_val = t1_val_local * 1.0001
            return s1_val, s2_val, t1_val_local, t2_val

        def objective(x_vec: np.ndarray) -> float:
            s1_val, s2_val, t1_val_local, t2_val = _pack_sbpl2(x_vec)
            return loss(s1_val, s2_val, t1_val_local, t2_val)

        x0 = np.array([
            _logit((s1_best - s1_low) / (s1_high - s1_low + 1e-12)),
            _logit((s2_best - s2_low) / (s2_high - s2_low + 1e-12)),
            _logit((math.log(t1_best) - log_t1_low) / (log_t1_high - log_t1_low + 1e-12)),
            math.log(max(t2_best - t1_best, 1.0)),
        ])

        res = minimize(objective, x0, method="L-BFGS-B")
        if res.success:
            s1_opt, s2_opt, t1_opt, t2_opt = _pack_sbpl2(res.x)
            opt_loss = loss(s1_opt, s2_opt, t1_opt, t2_opt)
            if opt_loss < best_loss:
                best_loss = opt_loss
                s1_best, s2_best, t1_best, t2_best = s1_opt, s2_opt, t1_opt, t2_opt

    return {
        "model": "sbpl2",
        "v_cap": v_cap,
        "s1": s1_best,
        "s2": s2_best,
        "t1": t1_best,
        "t2": t2_best,
        "k1": k1,
        "k2": k2,
        "loss": best_loss,
        "wr_min_seconds": wr_min_seconds,
    }


def _fit_wr_parameters(
    t: np.ndarray,
    H: np.ndarray,
    weights: np.ndarray,
    v_cap: float,
    profile_config: Dict[str, Any],
    wr_min_seconds: float,
) -> Dict[str, float]:
    model = profile_config.get("model", "sbpl1").lower()
    if model == "sbpl2":
        return _fit_sbpl2_parameters(t, H, weights, v_cap, profile_config, wr_min_seconds)
    return _fit_sbpl1_parameters(t, H, weights, v_cap, profile_config, wr_min_seconds)


def _fit_sbpl1_parameters(
    t: np.ndarray,
    H: np.ndarray,
    weights: np.ndarray,
    v_cap: float,
    config: Dict[str, Any],
    wr_min_seconds: float,
    penalty_lambda: float = 500.0,
) -> Dict[str, float]:
    bounds = config.get("bounds", {})
    s_low, s_high = bounds.get("s_inf", (0.5, 0.9))
    t_low, t_high = bounds.get("t_star", (600.0, 7200.0))
    k_low, k_high = bounds.get("k", (1.0, 6.0))

    coarse = config.get("coarse_grid", (15, 15, 12))
    refine = config.get("refine_grid", (9, 9, 9))

    weights = np.asarray(weights, dtype=float)
    if not np.any(weights > 0):
        weights = np.ones_like(t, dtype=float)
    weights = weights / np.sum(weights)

    log_t_low = math.log(t_low)
    log_t_high = math.log(t_high)

    def _pack_vars(x_vec: np.ndarray) -> Tuple[float, float, float]:
        s_val = s_low + (s_high - s_low) * _sigmoid(float(x_vec[0]))
        log_t = log_t_low + (log_t_high - log_t_low) * _sigmoid(float(x_vec[1]))
        t_val = math.exp(log_t)
        k_val = k_low + (k_high - k_low) * _sigmoid(float(x_vec[2]))
        return s_val, t_val, k_val

    def loss(s_inf: float, t_star: float, k: float) -> float:
        if not (s_low < s_inf < min(s_high, 0.9999)):
            return float("inf")
        if not (t_low <= t_star <= t_high):
            return float("inf")
        if not (k_low <= k <= k_high):
            return float("inf")
        return _sbpl_loss(t, H, weights, v_cap, s_inf, t_star, k, penalty_lambda)

    best_params: Optional[Tuple[float, float, float]] = None
    best_loss = float("inf")

    s_grid = np.linspace(s_low, s_high, coarse[0])
    t_grid = np.logspace(math.log10(t_low), math.log10(t_high), coarse[1])
    k_grid = np.linspace(k_low, k_high, coarse[2])

    for s_inf in s_grid:
        for t_star in t_grid:
            for k_val in k_grid:
                current_loss = loss(s_inf, t_star, k_val)
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_params = (s_inf, t_star, k_val)

    if best_params is None:
        raise RuntimeError("Failed to fit WR curve parameters (coarse search).")

    s_best, t_best, k_best = best_params

    def _refine_range(value: float, lower: float, upper: float, count: int, scale: float = 0.3) -> np.ndarray:
        span = max(scale * value, 0.05 * (upper - lower))
        lo = max(lower, value - span)
        hi = min(upper, value + span)
        if hi <= lo:
            lo, hi = lower, upper
        return np.linspace(lo, hi, count)

    s_fine = _refine_range(s_best, s_low, s_high, refine[0])
    t_fine = np.logspace(
        math.log10(max(t_low, t_best * 0.4)),
        math.log10(min(t_high, t_best * 2.5)),
        refine[1],
    )
    k_fine = _refine_range(k_best, k_low, k_high, refine[2], scale=0.5)

    for s_inf in s_fine:
        for t_star in t_fine:
            for k_val in k_fine:
                current_loss = loss(s_inf, t_star, k_val)
                if current_loss < best_loss:
                    best_loss = current_loss
                    best_params = (s_inf, t_star, k_val)

    if best_params is None:
        raise RuntimeError("Failed to fit WR curve parameters (refinement).")

    s_best, t_best, k_best = best_params

    if minimize is not None:
        x0 = np.array([
            _logit((s_best - s_low) / (s_high - s_low + 1e-12)),
            _logit((math.log(t_best) - log_t_low) / (log_t_high - log_t_low + 1e-12)),
            _logit((k_best - k_low) / (k_high - k_low + 1e-12)),
        ])

        def objective(x_vec: np.ndarray) -> float:
            s_val, t_val, k_val = _pack_vars(x_vec)
            return loss(s_val, t_val, k_val)

        res = minimize(objective, x0, method="L-BFGS-B") if minimize is not None else None
        if res is not None and res.success:
            s_opt, t_opt, k_opt = _pack_vars(res.x)
            opt_loss = loss(s_opt, t_opt, k_opt)
            if opt_loss < best_loss:
                best_loss = opt_loss
                s_best, t_best, k_best = s_opt, t_opt, k_opt

    return {
        "model": "sbpl1",
        "v_cap": v_cap,
        "s_inf": s_best,
        "t_star": t_best,
        "k": k_best,
        "loss": best_loss,
        "wr_min_seconds": wr_min_seconds,
    }


def _build_wr_anchors(
    profile_config: Dict[str, Any],
    wr_min_seconds: float,
    v_cap: float,
    custom_anchors: Optional[List[Dict[str, float]]] = None,
) -> List[Dict[str, float]]:
    weight_map = {
        float(k): float(v)
        for k, v in profile_config.get("anchor_weights", {}).items()
    }
    default_weight = float(profile_config.get("default_anchor_weight", 1.0))
    short_weight = float(profile_config.get("short_anchor_weight", default_weight))
    if custom_anchors:
        anchors = [
            {
                "w_s": float(a["w_s"] if isinstance(a, dict) else a[0]),
                "gain_m": float(a.get("gain_m") if isinstance(a, dict) else a[1]),
                "weight": float(a.get("weight", default_weight)) if isinstance(a, dict) else default_weight,
            }
            for a in custom_anchors
        ]
    else:
        scale = profile_config.get("anchor_scale", 1.0)
        anchors = [
            {
                "w_s": float(w),
                "gain_m": float(g * scale),
                "weight": float(weight_map.get(float(w), default_weight)),
            }
            for w, g in profile_config["anchors"]
        ]

    def _ensure_anchor(duration_s: float):
        if duration_s < wr_min_seconds:
            return
        if any(abs(a["w_s"] - duration_s) < 1e-6 for a in anchors):
            return
        anchors.append({
            "w_s": duration_s,
            "gain_m": v_cap * duration_s,
            "weight": short_weight,
        })

    # Do not force near-cap short anchors; allow fit to approach cap naturally

    anchors.sort(key=lambda a: a["w_s"])
    for anchor in anchors:
        if "weight" not in anchor or anchor["weight"] <= 0:
            anchor["weight"] = float(weight_map.get(anchor["w_s"], default_weight))
    return anchors


def _load_wr_anchor_override(path: str) -> Tuple[List[Dict[str, float]], Optional[float]]:
    with open(path, "r") as f:
        data = json.load(f)

    anchors: List[Dict[str, float]] = []
    v_cap_override: Optional[float] = None

    if isinstance(data, dict):
        anchor_items = data.get("anchors") or data.get("points") or []
        v_cap_override = data.get("v_cap")
    else:
        anchor_items = data

    for item in anchor_items:
        if isinstance(item, dict):
            if "w_s" in item and "gain_m" in item:
                anchors.append({"w_s": float(item["w_s"]), "gain_m": float(item["gain_m"])})
            elif "seconds" in item and "gain" in item:
                anchors.append({"w_s": float(item["seconds"]), "gain_m": float(item["gain"])})
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            anchors.append({"w_s": float(item[0]), "gain_m": float(item[1])})

    return anchors, v_cap_override


def _build_wr_envelope(
    profile_config: Dict[str, Any],
    wr_min_seconds: float,
    wr_anchors_path: Optional[str] = None,
    cap_mode: str = "standard",
):
    override_anchors: Optional[List[Dict[str, float]]] = None
    override_cap: Optional[float] = None
    if wr_anchors_path:
        override_anchors, override_cap = _load_wr_anchor_override(wr_anchors_path)

    cap_profile = copy.deepcopy(profile_config)
    multiplier = CAP_MODE_SCALE.get(cap_mode, 1.0)
    if abs(multiplier - 1.0) > 1e-6:
        if "treadmill" in cap_profile:
            cap_profile.setdefault("treadmill", {})
            cap_profile["treadmill"]["f_max"] = cap_profile["treadmill"].get("f_max", 3.4) * multiplier
            cap_profile["treadmill"]["L_max"] = cap_profile["treadmill"].get("L_max", 0.85) * multiplier
        if "stairs" in cap_profile:
            cap_profile.setdefault("stairs", {})
            cap_profile["stairs"]["f_max"] = cap_profile["stairs"].get("f_max", 3.4) * multiplier
        if "energy" in cap_profile:
            cap_profile.setdefault("energy", {})
            cap_profile["energy"]["v200_mps"] = cap_profile["energy"].get("v200_mps", 10.4) * multiplier

    cap_info = _resolve_vertical_cap(cap_profile)
    if override_cap is not None:
        cap_info["v_cap"] = override_cap

    v_cap = cap_info["v_cap"]
    anchors = _build_wr_anchors(profile_config, wr_min_seconds, v_cap, override_anchors)
    if not anchors:
        raise RuntimeError("No WR anchors available for fitting.")

    anchors = [a for a in anchors if a["w_s"] >= wr_min_seconds]
    anchors.sort(key=lambda a: a["w_s"])

    t_vals = np.array([a["w_s"] for a in anchors], dtype=float)
    H_vals = np.array([a["gain_m"] for a in anchors], dtype=float)
    weight_vals = np.array([a.get("weight", 1.0) for a in anchors], dtype=float)
    if not np.any(weight_vals > 0):
        weight_vals = np.ones_like(weight_vals)
    weights = weight_vals / np.sum(weight_vals)

    params = _fit_wr_parameters(t_vals, H_vals, weights, v_cap, profile_config, wr_min_seconds)

    if params["model"] == "sbpl2":

        def H_WR(w_s: float) -> float:
            return _H_sbpl_two_break_scalar(
                w_s,
                params["v_cap"],
                params["s1"],
                params["s2"],
                params["t1"],
                params["t2"],
                params.get("k1", 4.0),
                params.get("k2", 4.0),
            )

        def _sample_curve(samples: np.ndarray) -> np.ndarray:
            return _H_sbpl_two_break(
                samples,
                params["v_cap"],
                params["s1"],
                params["s2"],
                params["t1"],
                params["t2"],
                params.get("k1", 4.0),
                params.get("k2", 4.0),
            )
        def _rate_curve(samples: np.ndarray) -> np.ndarray:
            # Use a stable logspace derivative for sbpl2 (no closed form)
            vals = _sample_curve(samples)
            rates = np.zeros_like(vals)
            # central differences in log domain
            for i in range(len(samples)):
                if i == 0:
                    dt = samples[i+1] - samples[i]
                    rates[i] = (vals[i+1] - vals[i]) / max(dt, 1e-9)
                elif i == len(samples)-1:
                    dt = samples[i] - samples[i-1]
                    rates[i] = (vals[i] - vals[i-1]) / max(dt, 1e-9)
                else:
                    dt = samples[i+1] - samples[i-1]
                    rates[i] = (vals[i+1] - vals[i-1]) / max(dt, 1e-9)
            return rates
    else:

        def H_WR(w_s: float) -> float:
            return _H_sbpl_cap_scalar(
                w_s,
                params["v_cap"],
                params["s_inf"],
                params["t_star"],
                params["k"],
            )

        def _sample_curve(samples: np.ndarray) -> np.ndarray:
            return _H_sbpl_cap(
                samples,
                params["v_cap"],
                params["s_inf"],
                params["t_star"],
                params["k"],
            )
        def _rate_curve(samples: np.ndarray) -> np.ndarray:
            return _dH_sbpl_cap(
                samples,
                params["v_cap"],
                params["s_inf"],
                params["t_star"],
                params["k"],
            )

    anchor_min = float(np.min(t_vals))
    min_candidates = [anchor_min]
    if wr_min_seconds > 0:
        min_candidates.append(wr_min_seconds)
    w_min = max(min(min_candidates), WR_SAMPLE_SECONDS_MIN)

    anchor_max = float(np.max(t_vals))
    if wr_min_seconds > 0:
        w_max = max(anchor_max, wr_min_seconds * 4.0)
    else:
        w_max = max(anchor_max, anchor_min * 4.0)
    if w_max <= w_min:
        w_max = w_min * 1.01

    sample = np.logspace(math.log10(w_min), math.log10(w_max), 200)
    wr_values = _sample_curve(sample)
    wr_rates = _rate_curve(sample) * 3600.0

    return H_WR, {
        "durations": sample.tolist(),
        "climbs": wr_values.tolist(),
        "rates": wr_rates.tolist(),
        "anchors": anchors,
        "cap_info": cap_info,
        "params": params,
    }


def _interp_monotone(x: float, xs: List[int], ys: List[float]) -> float:
    import bisect
    if not xs:
        return 0.0
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    j = bisect.bisect_left(xs, x)
    x1, x2 = xs[j-1], xs[j]
    y1, y2 = ys[j-1], ys[j]
    if x2 == x1:
        return y2
    a = (x - x1) / (x2 - x1)
    return y1 + a * (y2 - y1)


def _scoring_tables(
    W_s: List[int],
    H_user: List[float],
    H_WR_func,
    magic_ws: List[int],
    min_anchor_s: float = 60.0,
    topk: int = 3,
) -> Tuple[List[float], List[float], List[Dict[str, Any]], List[Dict[str, Any]]]:
    import numpy as np
    H_wr = [H_WR_func(w) for w in W_s]
    rat = [ (u / wr) if wr > 0 else 0.0 for u, wr in zip(H_user, H_wr) ]
    # best relative point >= min_anchor_s
    r_star = 0.0
    idx_best = None
    for i, w in enumerate(W_s):
        if w >= min_anchor_s and H_wr[i] > 0:
            if idx_best is None or rat[i] > rat[idx_best]:
                idx_best = i
    if idx_best is not None:
        r_star = min(rat[idx_best], 1.0)
    H_pers = [r_star * wr for wr in H_wr]

    rows: List[Dict[str, Any]] = []
    for w in magic_ws:
        u = _interp_monotone(w, W_s, H_user)
        wr = H_WR_func(w)
        pct = (u / wr * 100.0) if wr > 0 else None
        pers = r_star * wr
        goal = u + (2.0/3.0) * (pers - u)
        rows.append({
            'duration_s': w,
            'user_gain_m': u,
            'wr_gain_m': wr,
            'score_pct': pct,
            'personal_gain_m': pers,
            'goal_gain_m': goal,
        })
    valid = [r for r in rows if isinstance(r.get('score_pct'), (int, float))]
    weakest = sorted(valid, key=lambda r: r['score_pct'])[:topk]
    return H_wr, H_pers, rows, weakest


# -----------------
# CLI
# -----------------

def _fmt_duration_label(value: int) -> str:
    if value < 120:
        return f"{int(value)}s"
    if value < 3600:
        return f"{int(round(value/60))}m"
    return f"{round(value/3600,1)}h"


def _setup_duration_axis(ax, durations: List[int]) -> None:
    import numpy as np
    import math

    u0 = math.log10(60.0)
    compress = 0.4

    def _fwd(xx):
        x = np.asarray(xx, dtype=float)
        x = np.where(x <= 0, 1e-6, x)
        u = np.log10(x)
        y = np.where(u <= u0, u0 + compress * (u - u0), u)
        return y

    def _inv(yy):
        y = np.asarray(yy, dtype=float)
        u = np.where(y <= u0, u0 + (y - u0) / compress, y)
        return 10.0 ** u

    ax.set_xscale("function", functions=(_fwd, _inv))
    cleaned = [int(max(1, round(d))) for d in durations]
    if not cleaned:
        cleaned = [60]
    dmin = max(60, min(cleaned))
    dmax = max(dmin, max(cleaned))
    tick_candidates = [
        60,
        120,
        300,
        600,
        1200,
        1800,
        3600,
        5400,
        7200,
        10800,
        14400,
        21600,
        43200,
        86400,
        2 * 86400,
        3 * 86400,
        5 * 86400,
        7 * 86400,
        10 * 86400,
        14 * 86400,
        21 * 86400,
        28 * 86400,
        42 * 86400,
        56 * 86400,
        84 * 86400,
    ]
    if dmin < 60:
        tick_candidates.extend([10, 15, 20, 30, 45])
    ticks = [t for t in tick_candidates if dmin <= t <= dmax]
    ticks.extend([d for d in (dmin, dmax) if dmin <= d <= dmax])
    ticks = sorted(set(ticks))
    if not ticks:
        ticks = [dmin, dmax]
    ax.set_xticks(ticks)
    ax.set_xticklabels([_fmt_duration_label(t) for t in ticks])
    ax.grid(True, which="both", axis="both", linestyle=":", alpha=0.6)


def _prepare_duration_axis_values(
    durations: Iterable[int],
    full_span_seconds: Optional[float],
    extra_durations: Optional[Iterable[float]] = None,
) -> Tuple[List[int], float, float]:
    axis_values = [int(max(1, round(d))) for d in durations if d is not None]
    if extra_durations:
        axis_values.extend(int(max(1, round(d))) for d in extra_durations if d not in (None, float("nan")))
    if full_span_seconds is not None and full_span_seconds > 0:
        axis_values.append(int(math.ceil(full_span_seconds)))
    axis_values.append(60)
    axis_values.append(30)
    axis_values = sorted(set(axis_values))
    x_min = float(axis_values[0]) if axis_values else 60.0
    x_min = max(1.0, x_min)
    x_max = float(axis_values[-1]) if axis_values else 60.0
    x_max = max(x_max, x_min * 1.001)
    return axis_values, x_min, x_max


def _summarize_magic(magic_rows: Optional[List[Dict[str, Any]]], goals_topk: int) -> Tuple[List[Dict[str, Any]], Set[int]]:
    if not magic_rows:
        return [], set()
    valid_rows = [r for r in magic_rows if isinstance(r.get('score_pct'), (int, float))]
    weakest = sorted(valid_rows, key=lambda r: r['score_pct'])[:goals_topk]
    return weakest, {int(r['duration_s']) for r in weakest}


def _plot_curve(
    curve: List[CurvePoint],
    out_png: str,
    source_label: str,
    wr_curve: Optional[Tuple[List[int], List[float]]] = None,
    wr_rates: Optional[List[float]] = None,
    personal_curve: Optional[Tuple[List[int], List[float]]] = None,
    magic_rows: Optional[List[Dict[str, Any]]] = None,
    goals_topk: int = 3,
    show_wr: bool = True,
    show_personal: bool = True,
    inactivity_gaps: Optional[List[Tuple[float, float]]] = None,
    full_span_seconds: Optional[float] = None,
    session_curves: Optional[List[Dict[str, Any]]] = None,
    envelope_curve: Optional[Tuple[List[int], List[float]]] = None,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting. Install with: pip install matplotlib") from e

    durs = [cp.duration_s for cp in curve]
    rates = [cp.climb_rate_m_per_hr for cp in curve]
    climbs = [cp.max_climb_m for cp in curve]
    base_durations = sorted({cp.duration_s for cp in curve})
    def _near_base_duration(w: int) -> bool:
        for b in base_durations:
            if abs(w - b) <= max(30, 0.05 * max(b, 1)):
                return True
        return False
    dense = len(curve) > 100

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(durs, rates, marker="" if dense else "o", linewidth=1.8, color=USER_COLOR, label="Climb rate (m/h)")

    session_data = session_curves or []

    extra_durations: List[float] = []
    if wr_curve:
        extra_durations.extend(wr_curve[0])
    if session_data:
        for session in session_data:
            extra_durations.extend(session.get("durations") or [])
    if envelope_curve:
        extra_durations.extend(envelope_curve[0])
    axis_durations, xmin, xmax = _prepare_duration_axis_values(durs, full_span_seconds, extra_durations)
    _setup_duration_axis(ax, axis_durations)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel("Duration")
    ax.set_ylabel("Climb rate (m/h)", color=USER_COLOR)

    ax2 = ax.twinx()
    ax2.plot(durs, climbs, marker="" if dense else "s", linestyle="--", linewidth=1.5, color=USER_COLOR, label="Max climb (m)")
    ax2.set_xlim(xmin, xmax)

    if session_data:
        from matplotlib import colors as mcolors, colormaps as mcolormaps
        cmap = mcolormaps.get_cmap("RdYlGn")
        n_sessions = len(session_data)
        for idx, session in enumerate(session_data):
            dur_list = session.get("durations", [])
            climb_list = session.get("climbs", [])
            if not dur_list or not climb_list:
                continue
            if n_sessions == 1:
                frac = 0.0
            else:
                frac = 1.0 - idx / (n_sessions - 1)
            rgba = cmap(frac)
            color = mcolors.to_hex(rgba, keep_alpha=False)
            rates_session = [c / d * 3600.0 if d > 0 else 0.0 for d, c in zip(dur_list, climb_list)]
            label = "Session curves (oldest→newest)" if idx == 0 else None
            ax.plot(
                dur_list,
                rates_session,
                linestyle=(0, (4, 6)),
                linewidth=1.1,
                color=color,
                alpha=0.45,
                label=label,
                zorder=1.2,
            )
            ax2.plot(
                dur_list,
                climb_list,
                linestyle=(0, (4, 6)),
                linewidth=1.1,
                color=color,
                alpha=0.45,
                zorder=1.2,
            )

    if envelope_curve:
        env_durs, env_climbs = envelope_curve
        if env_durs and env_climbs:
            env_rates = [c / d * 3600.0 if d > 0 else 0.0 for d, c in zip(env_durs, env_climbs)]
            ax.plot(
                env_durs,
                env_rates,
                linestyle="-",
                linewidth=1.0,
                color="0.2",
                alpha=0.35,
                label="Concave envelope",
                zorder=1.1,
            )
            ax2.plot(
                env_durs,
                env_climbs,
                linestyle="-",
                linewidth=1.0,
                color="0.2",
                alpha=0.35,
                zorder=1.1,
            )

    weak_rows, weak_ws = _summarize_magic(magic_rows, goals_topk)

    def _is_near_magic(d: int, magic_durs: Set[int]) -> bool:
        for w in magic_durs:
            if abs(d - w) <= max(30, 0.05 * max(d, 1)):
                return True
        return False

    if not dense:
        for x, y in zip(durs, climbs):
            if _is_near_magic(x, weak_ws):
                continue
            ax2.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=9, color=USER_COLOR)
    ax2.set_ylabel("Max climb (m)", color=USER_COLOR)

    ax.set_title(f"Ascent Rate Curve — {source_label}")

    # WR and personal overlays on max-climb axis
    if show_wr and wr_curve is not None:
        wr_durs, wr_vals = wr_curve
        wr_rates_arr = wr_rates if wr_rates else [v / w * 3600.0 if w > 0 else 0.0 for w, v in zip(wr_durs, wr_vals)]
        ax.plot(wr_durs, wr_rates_arr, linestyle=WR_STYLE, color="0.3", alpha=0.9, label="WR rate")
        ax2.plot(wr_durs, wr_vals, linestyle=WR_STYLE, color="0.3", alpha=0.9, label="WR climb")

    if show_personal and personal_curve is not None:
        p_durs, p_vals = personal_curve
        p_rates = [v / w * 3600.0 if w > 0 else 0.0 for w, v in zip(p_durs, p_vals)]
        ax.plot(p_durs, p_rates, linestyle=PERSONAL_STYLE, linewidth=1.2, color=USER_COLOR, alpha=0.6, label="Personal rate")
        ax2.plot(p_durs, p_vals, linestyle=PERSONAL_STYLE, linewidth=1.2, color=USER_COLOR, alpha=0.6, label="Personal climb")

    # Magic connectors and goals
    if magic_rows:
        # Only annotate up to goals_topk weakest to avoid clutter, but connect all
        label_toggle = 1
        for row in weak_rows:
            w = int(row['duration_s'])
            usr = row.get('user_gain_m')
            wrv = row.get('wr_gain_m')
            if usr is None or wrv is None:
                continue
            if show_wr and wrv is not None:
                target = wrv
            elif show_personal and isinstance(row.get('personal_gain_m'), (int, float)):
                target = row['personal_gain_m']
            else:
                target = None
            if target is not None:
                ax2.plot([w, w], [usr, target], linestyle=":", color="gray", alpha=0.8)
            ax2.plot([w], [usr], marker="o", color=USER_COLOR)
            if show_wr and wrv is not None:
                ax2.plot([w], [wrv], marker="o", color="0.3", alpha=0.9)
            pct = row.get('score_pct')
            if isinstance(pct, (int, float)):
                offset_y = -14 if label_toggle > 0 else -28
                if not dense and _near_base_duration(w):
                    offset_y -= 12
                ax2.annotate(
                    f"{usr:.0f} m • {pct:.0f}%",
                    (w, usr),
                    textcoords="offset points",
                    xytext=(0, offset_y),
                    ha="center",
                    fontsize=8,
                    color="black",
                )
            if isinstance(row.get('goal_gain_m'), (int, float, float)):
                goal = row.get('goal_gain_m')
                ax2.plot([w], [goal], marker=">", color=GOAL_COLOR)
                wrv2 = row.get('wr_gain_m')
                gpct = (goal / wrv2 * 100.0) if isinstance(wrv2, (int, float)) and wrv2 > 0 else None
                label = f"goal {goal:.0f} m" + (f" • {gpct:.0f}%" if gpct is not None else "")
                goal_offset = 10
                if not dense and _near_base_duration(w):
                    goal_offset += 6
                ax2.annotate(
                    label,
                    (w, goal),
                    textcoords="offset points",
                    xytext=(-6, goal_offset),
                    ha="right",
                    fontsize=8,
                    color=GOAL_COLOR,
                )
            label_toggle *= -1
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_split(
    curve: List[CurvePoint],
    out_png_base: str,
    source_label: str,
    wr_curve: Optional[Tuple[List[int], List[float]]] = None,
    wr_rates: Optional[List[float]] = None,
    personal_curve: Optional[Tuple[List[int], List[float]]] = None,
    goal_curve: Optional[Tuple[List[int], List[float]]] = None,
    magic_rows: Optional[List[Dict[str, Any]]] = None,
    goals_topk: int = 3,
    show_wr: bool = True,
    show_personal: bool = True,
    ylog_rate: bool = False,
    ylog_climb: bool = False,
    goal_min_seconds: float = 120.0,
    inactivity_gaps: Optional[List[Tuple[float, float]]] = None,
    full_span_seconds: Optional[float] = None,
    session_curves: Optional[List[Dict[str, Any]]] = None,
    envelope_curve: Optional[Tuple[List[int], List[float]]] = None,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting. Install with: pip install matplotlib") from e

    durs = [cp.duration_s for cp in curve]
    rates = [cp.climb_rate_m_per_hr for cp in curve]
    climbs = [cp.max_climb_m for cp in curve]
    base_durations = sorted({cp.duration_s for cp in curve})
    def _near_base_duration(w: int) -> bool:
        for b in base_durations:
            if abs(w - b) <= max(30, 0.05 * max(b, 1)):
                return True
        return False
    dense = len(curve) > 100

    session_data = session_curves or []

    extra_durations: List[float] = []
    if wr_curve:
        extra_durations.extend(wr_curve[0])
    if session_data:
        for session in session_data:
            extra_durations.extend(session.get("durations") or [])
    if envelope_curve:
        extra_durations.extend(envelope_curve[0])
    axis_durations, xmin, xmax = _prepare_duration_axis_values(durs, full_span_seconds, extra_durations)

    # Rate plot
    plt.style.use("ggplot")
    fig1, ax1 = plt.subplots(figsize=(12, 7))

    ax1.plot(durs, rates, marker="" if dense else "o", linewidth=1.8, color=USER_COLOR, label="Climb rate (m/h)")

    annotated_magic_rate_durs: Set[int] = set()
    if magic_rows:
        seen_durations: Set[int] = set()
        for row in magic_rows:
            w = int(row['duration_s'])
            if w in seen_durations:
                continue
            seen_durations.add(w)
            u = row.get('user_gain_m')
            goal = row.get('goal_gain_m')
            if not isinstance(u, (int, float)) or not isinstance(goal, (int, float)):
                continue
            annotated_magic_rate_durs.add(w)

    def _is_near_magic_rate(d: int) -> bool:
        for w in annotated_magic_rate_durs:
            if abs(d - w) <= max(30, 0.05 * max(d, 1)):
                return True
        return False

    if not dense:
        label_toggle = 1
        for cp in curve:
            if _is_near_magic_rate(cp.duration_s):
                continue
            offset_y = -12 if label_toggle > 0 else -26
            ax1.annotate(
                f"{cp.climb_rate_m_per_hr:.0f} m/h",
                (cp.duration_s, cp.climb_rate_m_per_hr),
                textcoords="offset points",
                xytext=(0, offset_y),
                ha="center",
                fontsize=8,
                color=USER_COLOR,
            )
            label_toggle *= -1
    if session_data:
        from matplotlib import colors as mcolors, colormaps as mcolormaps
        cmap = mcolormaps.get_cmap("RdYlGn")
        n_sessions = len(session_data)
        for idx, session in enumerate(session_data):
            dur_list = session.get("durations", [])
            climb_list = session.get("climbs", [])
            if not dur_list or not climb_list:
                continue
            if n_sessions == 1:
                frac = 0.0
            else:
                frac = 1.0 - idx / (n_sessions - 1)
            rgba = cmap(frac)
            color = mcolors.to_hex(rgba, keep_alpha=False)
            rates_session = [c / d * 3600.0 if d > 0 else 0.0 for d, c in zip(dur_list, climb_list)]
            label = "Session curves (oldest→newest)" if idx == 0 else None
            ax1.plot(
                dur_list,
                rates_session,
                linestyle=(0, (4, 6)),
                linewidth=1.1,
                color=color,
                alpha=0.45,
                label=label,
                zorder=1.2,
            )

    if envelope_curve:
        env_durs, env_climbs = envelope_curve
        if env_durs and env_climbs:
            env_rates = [c / d * 3600.0 if d > 0 else 0.0 for d, c in zip(env_durs, env_climbs)]
            ax1.plot(
                env_durs,
                env_rates,
                linestyle="-",
                linewidth=1.0,
                color="0.2",
                alpha=0.35,
                label="Concave envelope",
                zorder=1.1,
            )

    if show_wr and wr_curve is not None:
        wr_durs, wr_vals = wr_curve
        wr_rates_arr = wr_rates if wr_rates else [v / w * 3600.0 if w > 0 else 0.0 for w, v in zip(wr_durs, wr_vals)]
        ax1.plot(wr_durs, wr_rates_arr, linestyle=WR_STYLE, color="0.3", alpha=0.9, label="WR rate (≥min)")
    if show_personal and personal_curve is not None:
        p_durs, p_vals = personal_curve
        p_rates = [v / w * 3600.0 if w > 0 else 0.0 for w, v in zip(p_durs, p_vals)]
        ax1.plot(p_durs, p_rates, linestyle=PERSONAL_STYLE, linewidth=1.2, color=USER_COLOR, alpha=0.6, label="Personal rate")
    if goal_curve is not None:
        g_durs, g_vals = goal_curve
        gd = [(w, v) for w, v in zip(g_durs, g_vals) if w >= goal_min_seconds]
        if gd:
            gx, gy = zip(*gd)
            g_rates = [v / w * 3600.0 if w > 0 else 0.0 for w, v in gd]
            ax1.plot(gx, g_rates, linestyle="-", linewidth=1.2, color=GOAL_COLOR, alpha=0.8, label="Goal rate")
    _setup_duration_axis(ax1, axis_durations)
    ax1.set_xlim(xmin, xmax)
    ax1.set_xlabel("Duration")
    ax1.set_ylabel("Climb rate (m/h)")
    if ylog_rate:
        ymin = max(min(r for r in rates if r > 0) * 0.8, 1e-3)
        ax1.set_yscale('log')
        ax1.set_ylim(bottom=ymin)
    # Magic annotations for rate
    weak_rows, weak_ws = _summarize_magic(magic_rows, goals_topk)

    if magic_rows:
        label_toggle = 1
        seen_durations: Set[int] = set()
        for row in magic_rows:
            w = int(row['duration_s'])
            u = row.get('user_gain_m')
            goal = row.get('goal_gain_m')
            if not isinstance(u, (int, float)) or not isinstance(goal, (int, float)):
                continue
            if w in seen_durations:
                continue
            seen_durations.add(w)
            u_rate = u / w * 3600.0 if w > 0 else 0.0
            g_rate = goal / w * 3600.0 if w > 0 else 0.0
            pct = row.get('score_pct')
            text = f"{u_rate:.0f} m/h"
            if isinstance(pct, (int, float)) and w in weak_ws:
                text += f" • {pct:.0f}%"
            offset_y = -12 if label_toggle > 0 else -26
            if not dense and _near_base_duration(w):
                offset_y -= 12
            ax1.annotate(
                text,
                (w, u_rate),
                textcoords="offset points",
                xytext=(0, offset_y),
                ha="center",
                fontsize=8,
                color="black",
            )
            # approximate goal % from goal value and WR if present
            wrv = row.get('wr_gain_m')
            if w in weak_ws and w >= goal_min_seconds and isinstance(wrv, (int, float)) and wrv > 0:
                gpct = goal / wrv * 100.0
                goal_offset = 12
                if not dense and _near_base_duration(w):
                    goal_offset += 8
                ax1.annotate(
                    f"goal {g_rate:.0f} • {gpct:.0f}%",
                    (w, g_rate),
                    textcoords="offset points",
                    xytext=(0, goal_offset),
                    ha="center",
                    fontsize=8,
                    color=GOAL_COLOR,
                )
            label_toggle *= -1

    ax1.set_title(f"Ascent Rate Curve — {source_label}")
    ax1.legend(loc="upper right")
    fig1.tight_layout()
    rate_path = out_png_base.replace('.png', '_rate.png') if out_png_base.endswith('.png') else out_png_base + '_rate.png'
    fig1.savefig(rate_path, dpi=200)
    plt.close(fig1)

    # Climb plot
    fig2, axc = plt.subplots(figsize=(12, 7))
    axc.plot(durs, climbs, marker="" if dense else "s", linewidth=1.8, color=USER_COLOR, label="Max climb (m)")
    if session_data:
        from matplotlib import colors as mcolors, colormaps as mcolormaps
        cmap = mcolormaps.get_cmap("RdYlGn")
        n_sessions = len(session_data)
        for idx, session in enumerate(session_data):
            dur_list = session.get("durations", [])
            climb_list = session.get("climbs", [])
            if not dur_list or not climb_list:
                continue
            if n_sessions == 1:
                frac = 0.0
            else:
                frac = 1.0 - idx / (n_sessions - 1)
            rgba = cmap(frac)
            color = mcolors.to_hex(rgba, keep_alpha=False)
            axc.plot(
                dur_list,
                climb_list,
                linestyle=(0, (4, 6)),
                linewidth=1.1,
                color=color,
                alpha=0.45,
                zorder=1.2,
            )

    if envelope_curve:
        env_durs, env_climbs = envelope_curve
        if env_durs and env_climbs:
            axc.plot(
                env_durs,
                env_climbs,
                linestyle="-",
                linewidth=1.0,
                color="0.2",
                alpha=0.35,
                zorder=1.1,
            )

    if show_wr and wr_curve is not None:
        wr_durs, wr_vals = wr_curve
        axc.plot(wr_durs, wr_vals, linestyle=WR_STYLE, color="0.3", alpha=0.9, label="WR climb (≥min)")
    if show_personal and personal_curve is not None:
        p_durs, p_vals = personal_curve
        axc.plot(p_durs, p_vals, linestyle=PERSONAL_STYLE, linewidth=1.2, color=USER_COLOR, alpha=0.6, label="Personal climb")
    if goal_curve is not None:
        g_durs, g_vals = goal_curve
        gd = [(w, v) for w, v in zip(g_durs, g_vals) if w >= goal_min_seconds]
        if gd:
            gx, gy = zip(*gd)
            axc.plot(gx, gy, linestyle="-", linewidth=1.2, color=GOAL_COLOR, alpha=0.8, label="Goal climb")
    _setup_duration_axis(axc, axis_durations)
    axc.set_xlim(xmin, xmax)
    axc.set_xlabel("Duration")
    axc.set_ylabel("Max climb (m)")
    if ylog_climb:
        ymin = max(min(c for c in climbs if c > 0) * 0.8, 1e-6)
        axc.set_yscale('log')
        axc.set_ylim(bottom=ymin)
    axc.set_title(f"Ascent Max Climb — {source_label}")

    # Magic connectors/labels on climb plot
    weak_rows, weak_ws = _summarize_magic(magic_rows, goals_topk)
    if weak_rows:
        label_toggle = 1
        for row in weak_rows:
            w = int(row['duration_s'])
            usr = row.get('user_gain_m')
            wrv = row.get('wr_gain_m')
            if usr is None or wrv is None:
                continue
            if show_wr and wrv is not None:
                target = wrv
            elif show_personal and isinstance(row.get('personal_gain_m'), (int, float)):
                target = row['personal_gain_m']
            else:
                target = None
            if target is not None:
                axc.plot([w, w], [usr, target], linestyle=":", color="gray", alpha=0.8)
            axc.plot([w], [usr], marker="o", color=USER_COLOR)
            if show_wr and wrv is not None:
                axc.plot([w], [wrv], marker="o", color="0.3", alpha=0.9)
            pct = row.get('score_pct')
            if isinstance(pct, (int, float)):
                offset_y = -14 if label_toggle > 0 else -28
                if not dense and _near_base_duration(w):
                    offset_y -= 12
                axc.annotate(
                    f"{usr:.0f} m • {pct:.0f}%",
                    (w, usr),
                    textcoords="offset points",
                    xytext=(0, offset_y),
                    ha="center",
                    fontsize=8,
                    color="black",
                )
            if w in weak_ws and w >= goal_min_seconds and isinstance(row.get('goal_gain_m'), (int, float, float)):
                goal = row.get('goal_gain_m')
                axc.plot([w], [goal], marker=">", color=GOAL_COLOR)
                wrv2 = row.get('wr_gain_m')
                gpct = (goal / wrv2 * 100.0) if isinstance(wrv2, (int, float)) and wrv2 > 0 else None
                label = f"goal {goal:.0f} m" + (f" • {gpct:.0f}%" if gpct is not None else "")
                goal_offset = 10
                if not dense and _near_base_duration(w):
                    goal_offset += 6
                axc.annotate(
                    label,
                    (w, goal),
                    textcoords="offset points",
                    xytext=(-6, goal_offset),
                    ha="right",
                    fontsize=8,
                    color=GOAL_COLOR,
                )
            label_toggle *= -1

    ymax = max(
        max(climbs or [0]),
        max(goal_curve[1]) if goal_curve and goal_curve[1] else 0,
        max(personal_curve[1]) if (show_personal and personal_curve and personal_curve[1]) else 0,
    )
    if show_wr and wr_curve is not None and wr_curve[1]:
        ymax = max(ymax, max(wr_curve[1]))
    axc.set_ylim(top=ymax * 1.15 if ymax > 0 else axc.get_ylim()[1])

    axc.legend(loc="lower right")
    fig2.tight_layout()
    climb_path = out_png_base.replace('.png', '_climb.png') if out_png_base.endswith('.png') else out_png_base + '_climb.png'
    fig2.savefig(climb_path, dpi=200)
    plt.close(fig2)

DEFAULT_DURATIONS = [60, 120, 300, 600, 1200, 1800, 3600]


def _run(
    fit_files: List[str],
    output: str,
    durations: List[int],
    source: str,
    verbose: bool,
    png: Optional[str] = None,
    no_plot: bool = False,
    exhaustive: bool = False,
    step_s: int = 1,
    max_duration_s: Optional[int] = None,
    log_file: Optional[str] = None,
    merge_eps_sec: float = 0.5,
    overlap_policy: str = "file:last",
    resample_1hz: bool = False,
    gain_eps: float = 0.5,
    session_gap_sec: float = 600.0,
    all_windows: bool = False,
    wr_profile: str = "overall",
    wr_anchors_path: Optional[str] = None,
    wr_min_seconds: float = 30.0,
    wr_short_cap: str = "standard",
    magic: Optional[str] = None,
    plot_wr: bool = False,
    plot_personal: bool = True,
    goals_topk: int = 3,
    score_output: Optional[str] = None,
    split_plots: bool = True,
    ylog_rate: bool = False,
    ylog_climb: bool = False,
    goal_min_seconds: float = 120.0,
    personal_min_seconds: float = 60.0,
    qc_enabled: bool = True,
    qc_spec_path: Optional[str] = None,
    concave_envelope: bool = True,
) -> int:
    _setup_logging(verbose, log_file=log_file)

    selected = "mixed"
    curve: List[CurvePoint] = []
    try:
        logging.info("Reading %d FIT file(s)...", len(fit_files))
        records_by_file = []
        for i, fp in enumerate(fit_files):
            logging.info("Parsing: %s", fp)
            records_by_file.append(_parse_single_fit_records(fp, file_id=i))
        merged = _merge_records(records_by_file, merge_eps_sec=merge_eps_sec, overlap_policy=overlap_policy)
        logging.info("Merged samples: %d", len(merged))
        times: List[float]
        values: List[float]
        overall_sources_raw: Set[str] = set()
        inactivity_gaps: List[Tuple[float, float]] = []
        if source == "auto":
            times, values, used, gaps = _build_canonical_timeseries(
                merged,
                gain_eps=gain_eps,
                dwell_sec=5.0,
                gap_threshold=session_gap_sec,
            )
            overall_sources_raw = {SOURCE_NAME_MAP.get(u, u) for u in used if u}
            inactivity_gaps = gaps
        else:
            times, values, label = _build_timeseries(merged, source=source, gain_eps=gain_eps)
            overall_sources_raw = {label}

        if not times:
            logging.error("No data available to compute curve after merging inputs.")
            return 2

        qc_limits: Optional[Dict[float, float]] = None
        if qc_enabled:
            qc_limits = dict(QC_DEFAULT_SPEC)
            if qc_spec_path:
                try:
                    qc_limits.update(_load_qc_spec(qc_spec_path))
                except Exception as exc:
                    logging.warning("Failed to load QC spec %s: %s", qc_spec_path, exc)
        qc_segments = 0
        if qc_limits:
            values, qc_segments, qc_removed = _apply_qc_censor(times, values, qc_limits)
            if qc_segments:
                logging.info(
                    "QC censored %d spike segment(s); removed %.1f m of ascent.",
                    qc_segments,
                    qc_removed,
                )

        if resample_1hz:
            times, values = _resample_to_1hz(times, values)

        session_gap_list = _find_gaps(times, session_gap_sec)
        full_span_seconds = max(0.0, times[-1] - times[0]) if len(times) > 1 else 0.0
        if session_gap_list:
            gap_preview = ", ".join(
                _fmt_duration_label(int(round(g.length))) for g in session_gap_list[:5]
            )
            if len(session_gap_list) > 5:
                gap_preview += ", ..."
            logging.info(
                "Detected %d gap(s) longer than %.0fs (skip applied): %s",
                len(session_gap_list),
                session_gap_sec,
                gap_preview,
            )

        duration_grid: List[int] = []

        if all_windows:
            step_for_all = step_s if step_s > 0 else 1
            if not resample_1hz:
                times, values = _resample_to_1hz(times, values)
            curve = all_windows_curve(times, values, step=step_for_all)
            duration_grid = [cp.duration_s for cp in curve]
        else:
            if exhaustive:
                total_span = max(0.0, times[-1] - times[0])
                if max_duration_s is not None:
                    total_span = min(total_span, float(max_duration_s))
                span_int = int(math.floor(total_span))
                if span_int <= 0:
                    logging.warning("Total span too small for exhaustive sweep (%ss)", total_span)
                    durs = []
                else:
                    fine_until = min(span_int, 2 * 3600)
                    step_eval = step_s if step_s > 0 else 1
                    durs = nice_durations_for_span(
                        total_span_s=span_int,
                        fine_until_s=fine_until,
                        fine_step_s=step_eval,
                        pct_step=0.01,
                    )
                    logging.info(
                        "Exhaustive duration grid: %d candidates up to %s",
                        len(durs),
                        _fmt_duration_label(span_int),
                    )
            else:
                base_durations = [int(d) for d in durations if d is not None]
                if not base_durations:
                    base_durations = list(DEFAULT_DURATIONS)
                max_base = max(base_durations) if base_durations else 0
                durs_set: Set[int] = set(d for d in base_durations if d > 0)
                target_span = int(math.ceil(full_span_seconds)) if full_span_seconds > 0 else 0
                if target_span > max_base:
                    fine_until = min(target_span, max(max_base, 2 * 3600))
                    step_eval = step_s if step_s > 0 else 60
                    step_eval = max(60, step_eval)
                    extra = nice_durations_for_span(
                        total_span_s=target_span,
                        fine_until_s=fine_until,
                        fine_step_s=step_eval,
                        pct_step=0.02,
                    )
                    for d in extra:
                        if d >= 60:
                            durs_set.add(int(d))
                    durs_set.add(target_span)
                    logging.info(
                        "Augmented durations with %d additional candidates up to %s",
                        max(0, len(durs_set) - len(set(base_durations))),
                        _fmt_duration_label(target_span),
                    )
                durs = sorted(durs_set) if durs_set else list(DEFAULT_DURATIONS)

            curve = compute_curve(times, values, durs, gaps=session_gap_list)
            duration_grid = list(durs)

        if not curve:
            logging.error("Unable to compute curve for requested durations.")
            return 2

        if not overall_sources_raw:
            selected_raw = "mixed" if source == "auto" else source
        elif len(overall_sources_raw) == 1:
            selected_raw = next(iter(overall_sources_raw))
        else:
            selected_raw = "mixed"
        selected = _normalize_source_label(selected_raw)
    except Exception as e:
        logging.error(str(e))
        return 2

    # Enforce monotonic climb and non-increasing rate behaviour
    curve_original = copy.deepcopy(curve)
    apply_concave_main = concave_envelope and not exhaustive
    _enforce_curve_shape(curve, inactivity_gaps=inactivity_gaps, apply_concave=apply_concave_main)

    envelope_curve: Optional[Tuple[List[int], List[float]]] = None
    if concave_envelope and exhaustive:
        curve_env = copy.deepcopy(curve_original)
        _enforce_curve_shape(curve_env, inactivity_gaps=inactivity_gaps, apply_concave=True)
        envelope_curve = (
            [cp.duration_s for cp in curve_env],
            [cp.max_climb_m for cp in curve_env],
        )

    # Diagnose monotonicity issues (should be non-decreasing max climb with duration)
    _diagnose_curve_monotonicity(curve, epsilon=1e-6, inactivity_gaps=inactivity_gaps)

    # Scoring and WR envelope
    wr_curve: Optional[Tuple[List[int], List[float]]] = None
    personal_curve: Optional[Tuple[List[int], List[float]]] = None
    goal_curve: Optional[Tuple[List[int], List[float]]] = None
    magic_rows: Optional[List[Dict[str, Any]]] = None
    try:
        profile_config = _wr_profile_config(wr_profile)
        H_WR, wr_env = _build_wr_envelope(
            profile_config,
            wr_min_seconds=wr_min_seconds,
            wr_anchors_path=wr_anchors_path,
            cap_mode=wr_short_cap,
        )
        cap_info = wr_env.get("cap_info", {})
        if cap_info:
            logging.info(
                "WR cap %.2f m/s (treadmill=%.2f, stairs=%.2f, energy=%.2f)",
                cap_info.get("v_cap", float("nan")),
                cap_info.get("treadmill", float("nan")),
                cap_info.get("stairs", float("nan")),
                cap_info.get("energy", float("nan")),
            )
        anchor_report = []
        for anchor in wr_env.get("anchors", []):
            predicted = H_WR(anchor["w_s"])
            slack = predicted - anchor["gain_m"]
            anchor_report.append((anchor["w_s"], anchor["gain_m"], slack))
        if anchor_report:
            slack_text = ", ".join(
                f"{w/3600.0:.2f}h: +{slack:.1f} m" for w, _, slack in anchor_report
            )
            logging.info("WR anchor slack: %s", slack_text)
        try:
            if wr_min_seconds <= 3600.0:
                wr_1h = H_WR(3600.0)
                assert 1500.0 < wr_1h < 2600.0, f"WR(1h) sanity failed: {wr_1h}"
            if wr_min_seconds <= 12 * 3600.0:
                wr_12h = H_WR(12 * 3600.0)
                assert 11000.0 < wr_12h < 15000.0, f"WR(12h) sanity failed: {wr_12h}"
        except AssertionError as err:
            raise RuntimeError(str(err))
        W_s = [cp.duration_s for cp in curve]
        H_user = [cp.max_climb_m for cp in curve]
        wr_durations = wr_env.get("durations", [])
        wr_climbs = wr_env.get("climbs", [])
        wr_rates_env: Optional[List[float]] = wr_env.get("rates") if isinstance(wr_env.get("rates"), list) else None
        if wr_durations and wr_climbs:
            wr_curve = (wr_durations, wr_climbs)
            start_label = _fmt_duration_label(max(int(round(wr_durations[0])), 1))
            end_label = _fmt_duration_label(max(int(round(wr_durations[-1])), 1))
            logging.info(
                "WR envelope sample span: %s→%s (%d points)",
                start_label,
                end_label,
                len(wr_durations),
            )
        else:
            wr_curve = None
            logging.warning("WR envelope returned no sample points; disabling WR overlay.")

        # Filter WR overlay to w >= wr_min_seconds for scoring/personal scaling
        wr_indices = [idx for idx, w in enumerate(W_s) if w >= wr_min_seconds]
        W_wr = [W_s[idx] for idx in wr_indices]
        parsed_magic = [ _parse_duration_token(t) for t in (magic.split(',') if magic else []) ] if magic else []
        magic_ws = [int(w) for w in parsed_magic if w >= wr_min_seconds]
        H_wr_arr, H_pers_arr, rows, weakest = _scoring_tables(W_s, H_user, H_WR, magic_ws, min_anchor_s=personal_min_seconds, topk=goals_topk)
        personal_curve = (W_wr, [H_pers_arr[idx] for idx in wr_indices]) if W_wr else None
        # Goal curve across full grid: user + 2/3*(personal - user)
        H_goal_full = [u + (2.0/3.0) * (p - u) for u, p in zip(H_user, H_pers_arr)]
        goal_curve = (W_s, H_goal_full)
        magic_rows = rows
        if rows:
            checkpoint_logs: List[str] = []
            for row in rows:
                w_val = row.get("duration_s")
                wr_gain = row.get("wr_gain_m")
                user_gain = row.get("user_gain_m")
                score_pct = row.get("score_pct")
                if not isinstance(w_val, (int, float)) or not isinstance(wr_gain, (int, float)):
                    continue
                if w_val <= 0:
                    continue
                wr_rate = wr_gain / w_val * 3600.0
                label = _fmt_duration_label(int(round(w_val)))
                if isinstance(user_gain, (int, float)):
                    user_rate = user_gain / w_val * 3600.0
                    if isinstance(score_pct, (int, float)):
                        checkpoint_logs.append(
                            f"{label}: WR {wr_gain:.0f} m ({wr_rate:.0f} m/h), user {user_gain:.0f} m ({user_rate:.0f} m/h, {score_pct:.0f}%)"
                        )
                    else:
                        checkpoint_logs.append(
                            f"{label}: WR {wr_gain:.0f} m ({wr_rate:.0f} m/h), user {user_gain:.0f} m ({user_rate:.0f} m/h)"
                        )
                else:
                    checkpoint_logs.append(
                        f"{label}: WR {wr_gain:.0f} m ({wr_rate:.0f} m/h)"
                    )
            if checkpoint_logs:
                logging.info("WR checkpoints: %s", "; ".join(checkpoint_logs))
        if score_output:
            with open(score_output, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=["duration_s","user_gain_m","wr_gain_m","score_pct","personal_gain_m","goal_gain_m"])
                w.writeheader()
                for r in rows:
                    w.writerow(r)
    except Exception as e:
        logging.warning(f"WR/scoring computation failed: {e}")

    session_curves: Optional[List[Dict[str, Any]]] = None
    try:
        base_durations_for_sessions = duration_grid if duration_grid else [cp.duration_s for cp in curve]
        sessions = _split_sessions_from_gaps(times, values, session_gap_list)
        if sessions and base_durations_for_sessions:
            session_curves = []
            for idx, segment in enumerate(sessions):
                span = float(segment.get("span", 0.0))
                if span <= 0:
                    continue
                valid_durs = [d for d in base_durations_for_sessions if 0 < d <= span]
                if not valid_durs:
                    continue
                seg_curve = compute_curve(segment["times"], segment["values"], valid_durs, gaps=None)
                if not seg_curve:
                    continue
                session_curves.append(
                    {
                        "durations": [cp.duration_s for cp in seg_curve],
                        "climbs": [cp.max_climb_m for cp in seg_curve],
                        "span": span,
                        "order": idx,
                    }
                )
        if session_curves and len(session_curves) <= 1:
            session_curves = None
    except Exception as exc:
        logging.debug("Session overlay computation failed: %s", exc)
        session_curves = None

    # Write CSV (include WR columns when available)
    fieldnames = [
        "duration_s",
        "max_climb_m",
        "climb_rate_m_per_hr",
        "start_offset_s",
        "end_offset_s",
        "source",
    ]
    fieldnames += ["wr_climb_m", "wr_rate_m_per_hr"]
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        # Build on-the-fly WR values at our grid for consistency
        def _wr_at(w: int) -> Optional[float]:
            try:
                return float(H_WR(w)) if 'H_WR' in locals() and callable(H_WR) else None
            except Exception:
                return None
        for cp in curve:
            wr_gain = _wr_at(cp.duration_s)
            wr_rate = (wr_gain / cp.duration_s * 3600.0) if (wr_gain is not None and cp.duration_s > 0) else None
            row = {
                "duration_s": cp.duration_s,
                "max_climb_m": round(cp.max_climb_m, 3),
                "climb_rate_m_per_hr": round(cp.climb_rate_m_per_hr, 3),
                "start_offset_s": round(cp.start_offset_s, 3),
                "end_offset_s": round(cp.end_offset_s, 3),
                "source": selected_raw,
                "wr_climb_m": round(wr_gain, 3) if isinstance(wr_gain, (int, float)) else None,
                "wr_rate_m_per_hr": round(wr_rate, 3) if isinstance(wr_rate, (int, float)) else None,
            }
            writer.writerow(row)

    logging.info("Wrote: %s", output)

    if not no_plot:
        png_path = png
        if png_path is None:
            if output.lower().endswith(".csv"):
                png_path = output[:-4] + ".png"
            else:
                png_path = output + ".png"
        try:
            if split_plots:
                _plot_split(
                    curve,
                    png_path,
                    selected,
                    wr_curve=wr_curve if plot_wr else None,
                    wr_rates=wr_rates_env if plot_wr else None,
                    personal_curve=personal_curve if plot_personal else None,
                    goal_curve=goal_curve,
                    magic_rows=magic_rows,
                    goals_topk=goals_topk,
                    show_wr=plot_wr,
                    show_personal=plot_personal,
                    ylog_rate=ylog_rate,
                    ylog_climb=ylog_climb,
                    goal_min_seconds=goal_min_seconds,
                    inactivity_gaps=inactivity_gaps,
                    full_span_seconds=full_span_seconds,
                    session_curves=session_curves,
                    envelope_curve=envelope_curve,
                )
                logging.info("Wrote plots: %s", png_path)
            else:
                _plot_curve(
                    curve,
                    png_path,
                    selected,
                    wr_curve=wr_curve if plot_wr else None,
                    wr_rates=wr_rates_env if plot_wr else None,
                    personal_curve=personal_curve if plot_personal else None,
                    magic_rows=magic_rows,
                    goals_topk=goals_topk,
                    show_wr=plot_wr,
                    show_personal=plot_personal,
                    inactivity_gaps=inactivity_gaps,
                    full_span_seconds=full_span_seconds,
                    session_curves=session_curves,
                    envelope_curve=envelope_curve,
                )
                logging.info("Wrote plot: %s", png_path)
        except Exception as e:
            logging.error(f"Plotting failed: {e}")

    return 0


def _build_typer_app():  # pragma: no cover
    app = typer.Typer(add_completion=False, help="Critical hill climb rate curve from FIT files.")

    @app.command(name="curve")
    def curve(
        fit_files: List[str] = typer.Argument(..., help="One or more input .fit files"),
        output: str = typer.Option(
            "curve.csv",
            "--output",
            "-o",
            help="Output CSV path",
        ),
        durations: List[int] = typer.Option(
            DEFAULT_DURATIONS,
            "--durations",
            "-d",
            help="Durations in seconds to evaluate",
        ),
        source: str = typer.Option(
            "auto",
            "--source",
            help="Data source: auto|runn|altitude",
        ),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
        png: Optional[str] = typer.Option(None, "--png", help="Optional output PNG path (defaults next to CSV)"),
        no_plot: bool = typer.Option(False, "--no-plot", help="Disable PNG generation"),
        exhaustive: bool = typer.Option(False, "--exhaustive", help="Evaluate every duration window (1s steps or --step) up to total activity length or --max-duration"),
        step_s: int = typer.Option(1, "--step", help="Step size in seconds for exhaustive durations"),
        max_duration_s: Optional[int] = typer.Option(None, "--max-duration", help="Maximum duration in seconds for exhaustive evaluation"),
        log_file: Optional[str] = typer.Option(None, "--log-file", help="Optional log file path for diagnostics"),
        merge_eps_sec: float = typer.Option(0.5, "--merge-eps-sec", help="Coalesce tolerance (seconds) for overlapping timestamps"),
        overlap_policy: str = typer.Option("file:last", "--overlap-policy", help="Overlap precedence for tg: file:first|file:last"),
        resample_1hz: bool = typer.Option(False, "--resample-1hz", help="Resample cumulative series to 1 Hz"),
        gain_eps: float = typer.Option(0.5, "--gain-eps", help="Altitude hysteresis (m) for ascent from altitude"),
        session_gap_sec: float = typer.Option(600.0, "--session-gap-sec", help="Gap (s) considered inactivity when summarising and reporting; windows span all gaps"),
        qc: bool = typer.Option(True, "--qc/--no-qc", help="Censor implausible ascent spikes before computing the curve"),
        qc_spec: Optional[str] = typer.Option(None, "--qc-spec", help="Path to JSON overriding QC windows {window_s: max_gain_m}"),
        all_windows: bool = typer.Option(False, "--all", help="Compute per-second curve via concave envelope sweep (near-linear)"),
        wr_profile: str = typer.Option("overall", "--wr-profile", help="WR envelope profile: overall|stairs|female_overall|female_stairs"),
        wr_anchors: Optional[str] = typer.Option(None, "--wr-anchors", help="Path to JSON defining WR anchors [{w_s,gain_m}]"),
        wr_min_seconds: float = typer.Option(30.0, "--wr-min-seconds", help="Minimum duration (s) for WR envelope; below this WR is undefined"),
        wr_short_cap: str = typer.Option("standard", "--wr-short-cap", help="Short-duration cap rates: conservative|standard|aggressive"),
        magic: str = typer.Option("60s,300s,600s,1800s,3600s,0.481h,7200s,21600s,43200s", "--magic", help="Comma-separated durations (s|m|h) for scoring labels"),
        plot_wr: bool = typer.Option(False, "--plot-wr/--no-plot-wr", help="Show WR envelope on plot"),
        plot_personal: bool = typer.Option(True, "--plot-personal/--no-plot-personal", help="Show personal scaled WR curve"),
        goals_topk: int = typer.Option(3, "--goals-topk", help="How many weakest magic points to annotate with goals"),
        score_output: Optional[str] = typer.Option(None, "--score-output", help="Optional CSV to write per-magic scoring table"),
        split_plots: bool = typer.Option(True, "--split-plots/--no-split-plots", help="Write separate _rate.png and _climb.png instead of a combined plot"),
        ylog_rate: bool = typer.Option(False, "--ylog-rate/--no-ylog-rate", help="Use log scale on rate Y axis (split plots only)"),
        ylog_climb: bool = typer.Option(False, "--ylog-climb/--no-ylog-climb", help="Use log scale on climb Y axis (split plots only)"),
        goal_min_seconds: float = typer.Option(120.0, "--goal-min-seconds", help="Hide goal curve and labels below this duration (s) to avoid blow-ups"),
        personal_min_seconds: float = typer.Option(60.0, "--personal-min-seconds", help="Minimum duration (s) to anchor personal curve scaling"),
        concave_envelope: bool = typer.Option(True, "--concave-envelope/--no-concave-envelope", help="Apply concave envelope smoothing to the aggregate curve (auto-disabled when --exhaustive)", show_default=True),
    ) -> None:
        """Compute the critical hill climb rate curve and save to CSV."""
        code = _run(
            fit_files,
            output,
            durations,
            source,
            verbose,
            png=png,
            no_plot=no_plot,
            exhaustive=exhaustive,
            step_s=step_s,
            max_duration_s=max_duration_s,
            log_file=log_file,
            merge_eps_sec=merge_eps_sec,
            overlap_policy=overlap_policy,
            resample_1hz=resample_1hz,
            gain_eps=gain_eps,
            session_gap_sec=session_gap_sec,
            all_windows=all_windows,
            wr_profile=wr_profile,
            wr_anchors_path=wr_anchors,
            wr_min_seconds=wr_min_seconds,
            wr_short_cap=wr_short_cap,
            magic=magic,
            plot_wr=plot_wr,
            plot_personal=plot_personal,
            goals_topk=goals_topk,
            score_output=score_output,
            split_plots=split_plots,
            ylog_rate=ylog_rate,
            ylog_climb=ylog_climb,
            goal_min_seconds=goal_min_seconds,
            personal_min_seconds=personal_min_seconds,
            qc_enabled=qc,
            qc_spec_path=qc_spec,
            concave_envelope=concave_envelope,
        )
        if code != 0:
            raise typer.Exit(code)

    @app.command()
    def diagnose(
        fit_files: List[str] = typer.Argument(..., help="One or more input .fit files"),
        out: str = typer.Option("fit_diagnostics.txt", "--out", "-o", help="Output diagnostic report path"),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
    ) -> None:
        """Summarize record field keys and candidate total-gain fields for debugging."""
        _setup_logging(verbose)
        _require_dependency(FitFile, "fitparse", "pip install fitparse")
        report_lines: List[str] = []
        for path in fit_files:
            try:
                logging.info("Scanning: %s", path)
                fit = FitFile(path)
                fit.parse()
                recs = list(fit.get_messages("record"))
                if not recs:
                    report_lines.append(f"FILE: {path}\n  records: 0\n")
                    continue
                # Collect key stats
                key_stats: Dict[str, Dict[str, Any]] = {}
                t_first = None
                t_last = None
                cand_keys: Dict[str, int] = {}
                def upd_key(k: str, v: Any):
                    st = key_stats.setdefault(k, {"count": 0, "numeric": 0, "min": None, "max": None, "first": None, "last": None})
                    st["count"] += 1
                    if st["first"] is None:
                        st["first"] = v
                    st["last"] = v
                    if isinstance(v, (int, float)):
                        st["numeric"] += 1
                        st["min"] = v if st["min"] is None else (v if v < st["min"] else st["min"])
                        st["max"] = v if st["max"] is None else (v if v > st["max"] else st["max"])

                for msg in recs:
                    vals = msg.get_values()
                    ts = vals.get("timestamp")
                    if ts is not None:
                        try:
                            t = float(ts.timestamp()) if hasattr(ts, "timestamp") else float(ts)
                            t_first = t if t_first is None else t_first
                            t_last = t
                        except Exception:
                            pass
                    for k,v in vals.items():
                        kstr = str(k)
                        upd_key(kstr, v)
                        kl = kstr.lower().replace(" ", "_")
                        if ("gain" in kl or "ascent" in kl or "climb" in kl) and ("total" in kl or "cum" in kl or "cumulative" in kl):
                            cand_keys[kstr] = cand_keys.get(kstr, 0) + 1

                # Build short report
                report_lines.append(f"FILE: {path}")
                report_lines.append(f"  records: {len(recs)}")
                if t_first is not None and t_last is not None:
                    dur = t_last - t_first
                    report_lines.append(f"  timespan_s: {dur:.1f}")
                # Candidate keys
                if cand_keys:
                    top = sorted(cand_keys.items(), key=lambda x: -x[1])[:5]
                    report_lines.append("  candidate_total_gain_keys:")
                    for k,c in top:
                        st = key_stats.get(k, {})
                        f = st.get("first")
                        l = st.get("last")
                        # keep values short
                        def short(v):
                            s = str(v)
                            return s if len(s) < 40 else s[:37] + '...'
                        report_lines.append(f"    - {k} (count={c}) first={short(f)} last={short(l)}")
                # List a concise set of keys
                keys_sorted = sorted(key_stats.items(), key=lambda kv: (-kv[1]["count"], kv[0]))
                report_lines.append("  keys:")
                for k, st in keys_sorted[:20]:
                    minv = st["min"]; maxv = st["max"]
                    numeric = st["numeric"]
                    cnt = st["count"]
                    report_lines.append(
                        f"    - {k}: count={cnt}, numeric={numeric}, min={minv}, max={maxv}"
                    )
                report_lines.append("")
            except Exception as e:
                report_lines.append(f"FILE: {path}\n  error: {e}\n")

        with open(out, "w") as f:
            f.write("\n".join(report_lines))
        logging.info("Diagnostic report written: %s", out)

    return app


def main_cli() -> int:
    if typer is None:
        print(
            "Typer is not installed. Install with: pip install typer",
            file=sys.stderr,
        )
        return 2
    app = _build_typer_app()
    app()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main_cli())
