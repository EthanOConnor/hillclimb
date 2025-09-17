import sys
import os
import bisect
import csv
import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict, Any, Set
import json

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
                d = alt - last_alt
                if d > gain_eps:
                    cum += d
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


def compute_curve(
    times: List[float],
    cumulative_gain: List[float],
    durations: Iterable[int],
) -> List[CurvePoint]:
    results: List[CurvePoint] = []

    n = len(times)
    for D in durations:
        D = int(D)
        if D <= 0 or times[0] + D > times[-1] + 1e-9:
            results.append(CurvePoint(D, 0.0, 0.0, 0.0, 0.0))
            continue

        t_min = times[0]
        t_max = times[-1] - D

        # Pointers for interpolation at start (p) and end (q)
        # p: smallest index with times[p] >= t
        # q: smallest index with times[q] >= t + D
        # Initialize indices
        # ai: index into start-candidate list S = times
        # k: index into end-candidate list E where e = times[k] - D
        # Start ai at first S >= t_min; k at first times[k] >= t_min + D
        import bisect
        ai = bisect.bisect_left(times, t_min)
        k = bisect.bisect_left(times, t_min + D)
        p = ai
        q = k

        best_gain = -1.0
        best_t = t_min

        def next_s():
            return times[ai] if ai < n else float("inf")

        def next_e():
            return (times[k] - D) if k < n else float("inf")

        while True:
            ts = next_s()
            te = next_e()
            t = ts if ts <= te else te
            if t > t_max + 1e-12:
                break
            # Advance candidate indices if consumed
            if ts <= te:
                ai += 1
            if te <= ts:
                k += 1

            # Advance p so that times[p] >= t
            while p < n and times[p] < t:
                p += 1
            # Advance q so that times[q] >= t + D
            t_end = t + D
            while q < n and times[q] < t_end:
                q += 1

            G_start = _interp_cum_gain(t, times, cumulative_gain, p)
            G_end = _interp_cum_gain(t_end, times, cumulative_gain, q)
            gain = G_end - G_start
            if gain > best_gain:
                best_gain = gain
                best_t = t

        if best_gain < 0:
            best_gain = 0.0
        rate_m_per_hr = best_gain / D * 3600.0 if D > 0 else 0.0
        results.append(
            CurvePoint(
                duration_s=D,
                max_climb_m=best_gain,
                climb_rate_m_per_hr=rate_m_per_hr,
                start_offset_s=best_t,
                end_offset_s=best_t + D,
            )
        )
    return results


def _diagnose_curve_monotonicity(curve: List[CurvePoint], epsilon: float = 1e-9) -> None:
    if not curve:
        return
    last_climb = -1.0
    last_rate = float("inf")
    last_d = None
    last_start = None
    last_end = None
    for cp in curve:
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
        if cp.climb_rate_m_per_hr > last_rate + 1e-9:
            logging.warning(
                "Non-monotonic rate: D=%ss rate=%.6f > prev rate=%.6f (fix may apply)",
                cp.duration_s,
                cp.climb_rate_m_per_hr,
                last_rate,
            )
        last_rate = min(last_rate, cp.climb_rate_m_per_hr)
        last_d = cp.duration_s
        last_start = cp.start_offset_s
        last_end = cp.end_offset_s


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
        inc = r.get("inc") if r.get("inc") is not None else last_inc
        if dist is not None and last_dist is not None and inc is not None:
            dd = dist - last_dist
            if dd < 0:
                dd = 0.0
            if inc > 0:
                cum_inc += dd * (inc / 100.0)
        if dist is not None:
            last_dist = dist
        last_inc = inc
        inc_cum.append(cum_inc if last_dist is not None else None)

        # Altitude
        alt = r.get("alt")
        if alt is not None:
            if last_alt is not None:
                d = alt - last_alt
                if d > gain_eps:
                    cum_alt += d
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


def _build_lcm_envelope(T: List[float], P: List[float]) -> Tuple[List[float], List[float], List[float]]:
    hull: List[int] = []
    for i in range(len(T)):
        hull.append(i)
        while len(hull) >= 3:
            i1, i2, i3 = hull[-3], hull[-2], hull[-1]
            s12 = (P[i2]-P[i1]) / max(T[i2]-T[i1], 1e-12)
            s23 = (P[i3]-P[i2]) / max(T[i3]-T[i2], 1e-12)
            if s12 < s23:
                hull.pop(-2)
            else:
                break
    ex = [T[i] for i in hull]
    ey = [P[i] for i in hull]
    s: List[float] = []
    for k in range(len(ex)-1):
        s.append((ey[k+1]-ey[k]) / max(ex[k+1]-ex[k], 1e-12))
    return ex, ey, s


def _U_eval(ex: List[float], ey: List[float], s: List[float], t: float) -> float:
    import bisect
    j = bisect.bisect_left(ex, t)
    if j <= 0:
        return ey[0]
    if j >= len(ex):
        return ey[-1]
    k = j - 1
    return ey[k] + s[k] * (t - ex[k])


def all_windows_curve(T: List[float], P: List[float], step: int = 1) -> List[CurvePoint]:
    if not T or len(T) < 2:
        return []
    ex, ey, s = _build_lcm_envelope(T, P)
    t0 = T[0]
    t1 = T[-1]
    Wmax = int((t1 - t0) // step)
    # Initialize x at domain start
    x = t0
    import bisect
    i = max(bisect.bisect_left(ex, x) - 1, 0)
    results: List[CurvePoint] = []
    for w_idx in range(1, Wmax + 1):
        w = w_idx * step
        end = x + w
        j = max(min(bisect.bisect_left(ex, end) - 1, len(s)-1), 0)
        while True:
            deriv = s[j] - s[i]
            if deriv <= 1e-15:
                break
            next_start = ex[i+1] if i+1 < len(ex) else float('inf')
            next_end = ex[j+1] if j+1 < len(ex) else float('inf')
            dx1 = next_start - x
            dx2 = next_end - (x + w)
            dx = dx1 if dx1 <= dx2 else dx2
            if dx <= 1e-12:
                dx = 1e-12
            x += dx
            if i+1 < len(s) and abs(x - next_start) <= 1e-12:
                i += 1
            if j+1 < len(s) and abs(x + w - next_end) <= 1e-12:
                j += 1
            if j >= len(s):
                j = len(s) - 1
                break
        g = _U_eval(ex, ey, s, x + w) - _U_eval(ex, ey, s, x)
        rate = g / w * 3600.0
        results.append(CurvePoint(duration_s=int(w), max_climb_m=g, climb_rate_m_per_hr=rate, start_offset_s=x - t0, end_offset_s=x - t0 + w))
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


def _short_caps(profile: str, cap_mode: str) -> List[Dict[str, float]]:
    # Average rates in m/h at short durations; women scaled ~0.875
    female = 'female' in profile
    scale = 0.875 if female else 1.0
    table = {
        'conservative': (2700.0, 2500.0),
        'standard': (2800.0, 2600.0),
        'aggressive': (3000.0, 2800.0),
    }
    if cap_mode not in table:
        cap_mode = 'standard'
    r30, r60 = table[cap_mode]
    r30 *= scale
    r60 *= scale
    a30 = {'w_s': 30.0, 'gain_m': r30 * 30.0 / 3600.0}
    a60 = {'w_s': 60.0, 'gain_m': r60 * 60.0 / 3600.0}
    return [a30, a60]


def _default_wr_anchors(profile: str = 'overall', cap_mode: str = 'standard', wr_min_seconds: float = 30.0) -> List[Dict[str, float]]:
    anchors: List[Dict[str, float]] = []
    # Short caps at 30s/60s if within min seconds
    if wr_min_seconds <= 30.0:
        anchors += _short_caps(profile, cap_mode)
    elif wr_min_seconds <= 60.0:
        anchors += _short_caps(profile, cap_mode)[1:]
    # Classic anchors (men overall by default; profiles can adjust values later as needed)
    anchors.append({'w_s': 0.481 * 3600.0, 'gain_m': 1000.0})       # VK
    anchors.append({'w_s': 1.0 * 3600.0,   'gain_m': 1616.0})       # 1h stairs
    anchors.append({'w_s': 12.0 * 3600.0,  'gain_m': 13145.65})     # 12h stairs
    anchors.append({'w_s': 24.0 * 3600.0,  'gain_m': 21720.0 if 'overall' in profile else 18713.0})
    return anchors


def _concave_envelope_xy(W: List[float], H: List[float]) -> Tuple[List[float], List[float], List[float]]:
    hull: List[int] = []
    for i in range(len(W)):
        hull.append(i)
        while len(hull) >= 3:
            i1, i2, i3 = hull[-3], hull[-2], hull[-1]
            s12 = (H[i2] - H[i1]) / max(W[i2] - W[i1], 1e-12)
            s23 = (H[i3] - H[i2]) / max(W[i3] - W[i2], 1e-12)
            if s12 < s23:
                hull.pop(-2)
            else:
                break
    ex = [W[i] for i in hull]
    ey = [H[i] for i in hull]
    s: List[float] = []
    for k in range(len(ex) - 1):
        s.append((ey[k + 1] - ey[k]) / max(ex[k + 1] - ex[k], 1e-12))
    return ex, ey, s


def _build_wr_envelope(anchors: List[Dict[str, float]], wr_min_seconds: float = 30.0):
    # Build concave envelope of anchors in (w,H) space; clamp to wr_min_seconds
    pts = sorted(((a['w_s'], a['gain_m']) for a in anchors if a.get('w_s', 0) >= wr_min_seconds), key=lambda x: x[0])
    if not pts:
        # Fallback: build from any anchors ignoring min, then clamp
        pts = sorted(((a['w_s'], a['gain_m']) for a in anchors), key=lambda x: x[0])
    W = [p[0] for p in pts]
    H = [p[1] for p in pts]
    ex, ey, s = _concave_envelope_xy(W, H)

    def H_WR(w_s: float) -> float:
        import bisect
        w = max(wr_min_seconds, float(w_s))
        j = bisect.bisect_left(ex, w)
        if j <= 0:
            return ey[0]
        if j >= len(ex):
            return ey[-1]
        k = j - 1
        return ey[k] + s[k] * (w - ex[k])

    return H_WR, (ex, ey, s)


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
    dmin = max(1, int(min(durations))) if durations else 1
    dmax = int(max(durations)) if durations else 1
    sec_ticks: List[int] = []
    for k in range(0, 2):
        for m in (1, 2, 5):
            v = int(m * (10 ** k))
            if v < 60 and dmin <= v <= dmax:
                sec_ticks.append(v)
    min_ticks_all = [60, 120, 300, 600, 1200, 1800, 3600]
    min_ticks = [t for t in min_ticks_all if dmin <= t <= dmax]
    ticks = sec_ticks + min_ticks
    if not ticks:
        ticks = [dmin, dmax]
    ax.set_xticks(ticks)
    ax.set_xticklabels([_fmt_duration_label(t) for t in ticks])
    ax.grid(True, which="both", axis="both", linestyle=":", alpha=0.6)


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
    personal_curve: Optional[Tuple[List[int], List[float]]] = None,
    magic_rows: Optional[List[Dict[str, Any]]] = None,
    goals_topk: int = 3,
    show_wr: bool = True,
    show_personal: bool = True,
    inactivity_gaps: Optional[List[Tuple[float, float]]] = None,
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
    dense = len(curve) > 100

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(durs, rates, marker="" if dense else "o", linewidth=1.8, color=USER_COLOR, label="Climb rate (m/h)")

    # Use a custom compressed-log x-scale that compresses < 60s
    # so that more horizontal space is devoted to >= 1 minute.
    from matplotlib import scale as mscale
    import numpy as np
    import math
    u0 = math.log10(60.0)
    compress = 0.4  # 40% compression for sub-minute region

    def _fwd(xx):
        x = np.asarray(xx, dtype=float)
        u = np.log10(x)
        y = np.where(u <= u0, u0 + compress * (u - u0), u)
        return y

    def _inv(yy):
        y = np.asarray(yy, dtype=float)
        u = np.where(y <= u0, u0 + (y - u0) / compress, y)
        return 10.0 ** u

    ax.set_xscale("function", functions=(_fwd, _inv))
    # Ticks: 1/2/5 seconds below 60s, then specific minute marks
    dmin = max(1, int(min(durs)))
    dmax = int(max(durs))
    sec_ticks: List[int] = []
    for k in range(0, 2):  # seconds decades: 1..9, 10..99
        for m in (1, 2, 5):
            v = int(m * (10 ** k))
            if v < 60 and dmin <= v <= dmax:
                sec_ticks.append(v)
    min_ticks_all = [60, 120, 300, 600, 1200, 1800, 3600]
    min_ticks = [t for t in min_ticks_all if dmin <= t <= dmax]
    ticks = sec_ticks + min_ticks
    if not ticks:
        ticks = [dmin, dmax]
    ax.set_xticks(ticks)
    ax.set_xticklabels([fmt_dur(s) for s in ticks])
    _setup_duration_axis(ax, durs)
    ax.set_xlabel("Duration")
    ax.set_ylabel("Climb rate (m/h)", color=USER_COLOR)

    ax2 = ax.twinx()
    ax2.plot(durs, climbs, marker="" if dense else "s", linestyle="--", linewidth=1.5, color=USER_COLOR, label="Max climb (m)")
    if not dense:
        for x, y in zip(durs, climbs):
            ax2.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=9, color=USER_COLOR)
    ax2.set_ylabel("Max climb (m)", color=USER_COLOR)

    ax.set_title(f"Ascent Rate Curve — {source_label}")

    # WR and personal overlays on max-climb axis
    if show_wr and wr_curve is not None:
        wr_durs, wr_vals = wr_curve
        wr_rates = [v / w * 3600.0 if w > 0 else 0.0 for w, v in zip(wr_durs, wr_vals)]
        ax.plot(wr_durs, wr_rates, linestyle=WR_STYLE, color="0.3", alpha=0.9, label="WR rate")
        ax2.plot(wr_durs, wr_vals, linestyle=WR_STYLE, color="0.3", alpha=0.9, label="WR climb")

    if show_personal and personal_curve is not None:
        p_durs, p_vals = personal_curve
        p_rates = [v / w * 3600.0 if w > 0 else 0.0 for w, v in zip(p_durs, p_vals)]
        ax.plot(p_durs, p_rates, linestyle=PERSONAL_STYLE, linewidth=1.2, color=USER_COLOR, alpha=0.6, label="Personal rate")
        ax2.plot(p_durs, p_vals, linestyle=PERSONAL_STYLE, linewidth=1.2, color=USER_COLOR, alpha=0.6, label="Personal climb")

    # Magic connectors and goals
    if magic_rows:
        # Only annotate up to goals_topk weakest to avoid clutter, but connect all
        # Determine weakest
        valid = [r for r in magic_rows if isinstance(r.get('score_pct'), (int, float))]
        weak = sorted(valid, key=lambda r: r['score_pct'])[:goals_topk]
        label_toggle = 1
        for row in weak:
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
                ax2.annotate(
                    label,
                    (w, goal),
                    textcoords="offset points",
                    xytext=(-6, 10),
                    ha="right",
                    fontsize=8,
                    color=GOAL_COLOR,
                )
            label_toggle *= -1
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="lower left")

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_split(
    curve: List[CurvePoint],
    out_png_base: str,
    source_label: str,
    wr_curve: Optional[Tuple[List[int], List[float]]] = None,
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
    dense = len(curve) > 100

    import numpy as np
    import math

    # Rate plot
    plt.style.use("ggplot")
    fig1, ax1 = plt.subplots(figsize=(12, 7))

    ax1.plot(durs, rates, marker="" if dense else "o", linewidth=1.8, color=USER_COLOR, label="Climb rate (m/h)")
    if show_wr and wr_curve is not None:
        wr_durs, wr_vals = wr_curve
        wr_rates = [v / w * 3600.0 if w > 0 else 0.0 for w, v in zip(wr_durs, wr_vals)]
        ax1.plot(wr_durs, wr_rates, linestyle=WR_STYLE, color="0.3", alpha=0.9, label="WR rate (≥min)")
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
    _setup_duration_axis(ax1, durs)
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
                ax1.annotate(
                    f"goal {g_rate:.0f} • {gpct:.0f}%",
                    (w, g_rate),
                    textcoords="offset points",
                    xytext=(0, 12),
                    ha="center",
                    fontsize=8,
                    color=GOAL_COLOR,
                )
            label_toggle *= -1

    ax1.set_title(f"Ascent Rate Curve — {source_label}")
    ax1.legend(loc="lower left")
    fig1.tight_layout()
    rate_path = out_png_base.replace('.png', '_rate.png') if out_png_base.endswith('.png') else out_png_base + '_rate.png'
    fig1.savefig(rate_path, dpi=200)
    plt.close(fig1)

    # Climb plot
    fig2, axc = plt.subplots(figsize=(12, 7))
    axc.plot(durs, climbs, marker="" if dense else "s", linewidth=1.8, color=USER_COLOR, label="Max climb (m)")
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
    _setup_duration_axis(axc, durs)
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
                axc.annotate(
                    label,
                    (w, goal),
                    textcoords="offset points",
                    xytext=(-6, 10),
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

    axc.legend(loc="upper left")
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
        if session_gap_sec > 0 and len(merged) > 1:
            gap_count = sum(1 for i in range(1, len(merged)) if merged[i]["t"] - merged[i-1]["t"] > session_gap_sec)
            if gap_count:
                logging.info("Detected %d gap(s) longer than %.0fs (windows span all gaps).", gap_count, session_gap_sec)

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

        if all_windows:
            step_for_all = step_s if step_s > 0 else 1
            curve = all_windows_curve(times, values, step=step_for_all)
        else:
            durs = durations
            if exhaustive:
                total = int(times[-1])
                limit = min(max_duration_s or total, total)
                step_eval = step_s if step_s > 0 else 1
                durs = list(range(step_eval, limit + 1, step_eval))
            curve = compute_curve(times, values, durs)

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

    # Ensure non-decreasing climb with increasing duration (monotonic fix)
    best_climb = 0.0
    for cp in curve:
        if cp.max_climb_m < best_climb:
            cp.max_climb_m = best_climb
        else:
            best_climb = cp.max_climb_m
        # Recompute rate from climb to avoid any stale/clamped values
        cp.climb_rate_m_per_hr = (cp.max_climb_m / cp.duration_s * 3600.0) if cp.duration_s > 0 else 0.0

    # Diagnose monotonicity issues (should be non-decreasing max climb with duration)
    _diagnose_curve_monotonicity(curve, epsilon=1e-6)

    # Scoring and WR envelope
    wr_curve: Optional[Tuple[List[int], List[float]]] = None
    personal_curve: Optional[Tuple[List[int], List[float]]] = None
    goal_curve: Optional[Tuple[List[int], List[float]]] = None
    magic_rows: Optional[List[Dict[str, Any]]] = None
    try:
        import json
        anchors = _default_wr_anchors(wr_profile, cap_mode=wr_short_cap, wr_min_seconds=wr_min_seconds)
        if wr_anchors_path:
            with open(wr_anchors_path, 'r') as jf:
                anchors = json.load(jf)
        H_WR, wr_env = _build_wr_envelope(anchors, wr_min_seconds=wr_min_seconds)
        try:
            if wr_min_seconds <= 3600.0:
                wr_1h = H_WR(3600.0)
                assert 1500.0 < wr_1h < 1800.0, f"WR(1h) sanity failed: {wr_1h}"
            if wr_min_seconds <= 12 * 3600.0:
                wr_12h = H_WR(12 * 3600.0)
                assert 11000.0 < wr_12h < 15000.0, f"WR(12h) sanity failed: {wr_12h}"
        except AssertionError as err:
            raise RuntimeError(str(err))
        W_s = [cp.duration_s for cp in curve]
        H_user = [cp.max_climb_m for cp in curve]
        # Filter WR overlay to w >= wr_min_seconds for plotting
        W_wr = [w for w in W_s if w >= wr_min_seconds]
        H_wr_vals = [H_WR(w) for w in W_wr]
        wr_curve = (W_wr, H_wr_vals)
        # Personal scaled curve
        parsed_magic = [ _parse_duration_token(t) for t in (magic.split(',') if magic else []) ] if magic else []
        magic_ws = [int(w) for w in parsed_magic if w >= wr_min_seconds]
        H_wr_arr, H_pers_arr, rows, weakest = _scoring_tables(W_s, H_user, H_WR, magic_ws, min_anchor_s=personal_min_seconds, topk=goals_topk)
        personal_curve = (W_wr, [H_pers_arr[W_s.index(w)] for w in W_wr]) if W_wr else None
        # Goal curve across full grid: user + 2/3*(personal - user)
        H_goal_full = [u + (2.0/3.0) * (p - u) for u, p in zip(H_user, H_pers_arr)]
        goal_curve = (W_s, H_goal_full)
        magic_rows = rows
        if score_output:
            with open(score_output, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=["duration_s","user_gain_m","wr_gain_m","score_pct","personal_gain_m","goal_gain_m"])
                w.writeheader()
                for r in rows:
                    w.writerow(r)
    except Exception as e:
        logging.warning(f"WR/scoring computation failed: {e}")

    # Write CSV
    fieldnames = [
        "duration_s",
        "max_climb_m",
        "climb_rate_m_per_hr",
        "start_offset_s",
        "end_offset_s",
        "source",
    ]
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cp in curve:
            writer.writerow(
                {
                    "duration_s": cp.duration_s,
                    "max_climb_m": round(cp.max_climb_m, 3),
                    "climb_rate_m_per_hr": round(cp.climb_rate_m_per_hr, 3),
                    "start_offset_s": round(cp.start_offset_s, 3),
                    "end_offset_s": round(cp.end_offset_s, 3),
                    "source": selected_raw,
                }
            )

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
                )
                logging.info("Wrote plots: %s", png_path)
            else:
                _plot_curve(
                    curve,
                    png_path,
                    selected,
                    wr_curve=wr_curve if plot_wr else None,
                    personal_curve=personal_curve if plot_personal else None,
                    magic_rows=magic_rows,
                    goals_topk=goals_topk,
                    show_wr=plot_wr,
                    show_personal=plot_personal,
                    inactivity_gaps=inactivity_gaps,
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
