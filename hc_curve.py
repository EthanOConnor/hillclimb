import sys
import os
import bisect
import csv
import logging
import math
import copy
import time
import heapq
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict, Any, Set, NamedTuple, Union, Literal, Sequence
import json
import pickle

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

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    njit = None  # type: ignore
    prange = range  # type: ignore
    HAVE_NUMBA = False


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

_PARSED_FIT_CACHE_DIR = os.path.join(_BASE_DIR, ".cache", "parsed_fit")

_MATPLOTLIB_STYLE_READY = False



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


@dataclass
class GainTimePoint:
    gain_m: float
    min_time_s: float
    avg_rate_m_per_hr: float
    start_offset_s: Optional[float]
    end_offset_s: Optional[float]
    note: Optional[str] = None


@dataclass
class GainTimeCurve:
    points: List[GainTimePoint]
    source_label: str
    total_span_s: float


@dataclass
class ActivitySeries:
    times: List[float]
    values: List[float]
    selected_raw: str
    selected_label: str
    inactivity_gaps: List[Tuple[float, float]]
    session_gaps: List['Gap']
    full_span_seconds: float
    used_sources: Set[str]


@dataclass
class WREnvelopeResult:
    wr_curve: Optional[Tuple[List[int], List[float]]]
    wr_rates: Optional[List[float]]
    personal_curve: Optional[Tuple[List[int], List[float]]]
    goal_curve: Optional[Tuple[List[int], List[float]]]
    magic_rows: Optional[List[Dict[str, Any]]]
    H_WR_func: Optional[Any]
    wr_sample_arrays: Optional[Tuple[np.ndarray, np.ndarray]]


DEFAULT_GAIN_TARGETS: Tuple[float, ...] = (
    50.0,
    100.0,
    150.0,
    200.0,
    300.0,
    500.0,
    750.0,
    1000.0,
)

DEFAULT_MAGIC_GAINS = "50m,100m,200m,300m,500m,1000m"

ISO_RATE_GUIDES = (800.0, 1000.0, 1200.0, 1500.0, 2000.0, 2500.0)


class _StageProfiler:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self._last = time.perf_counter()

    def lap(self, label: str) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        logging.info("Profile %-18s %.3fs", label, now - self._last)
        self._last = now


def _ensure_matplotlib_style(plt) -> None:
    global _MATPLOTLIB_STYLE_READY
    if not _MATPLOTLIB_STYLE_READY:
        try:
            plt.style.use("ggplot")
        except Exception:
            pass
        _MATPLOTLIB_STYLE_READY = True


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


def _parse_gain_token(token: Union[str, float, int], *, default_unit: str = "m") -> Optional[float]:
    if isinstance(token, (int, float)):
        if math.isnan(float(token)) or float(token) < 0:
            return None
        return float(token)
    s = str(token).strip().lower()
    if not s:
        return None
    unit = default_unit.lower()
    if s.endswith("ft"):
        unit = "ft"
        s = s[:-2]
    elif s.endswith("m"):
        s = s[:-1]
    try:
        value = float(s)
    except ValueError:
        return None
    if not math.isfinite(value) or value < 0:
        return None
    if unit == "ft":
        value *= 0.3048
    return value


def _parse_gain_list(tokens: Sequence[Union[str, float, int]], *, default_unit: str = "m") -> List[float]:
    gains: List[float] = []
    for tok in tokens:
        parsed = _parse_gain_token(tok, default_unit=default_unit)
        if parsed is None:
            continue
        if parsed <= 0:
            continue
        gains.append(parsed)
    unique_sorted = sorted({round(g, 6) for g in gains})
    return [float(g) for g in unique_sorted]


def _expand_gain_tokens(tokens: Sequence[Union[str, float, int]]) -> List[str]:
    expanded: List[str] = []
    for tok in tokens:
        if tok is None:
            continue
        text = str(tok).strip()
        if not text:
            continue
        # Treat commas and whitespace as separators.
        for part in text.replace(",", " ").split():
            if part:
                expanded.append(part)
    return expanded


def _load_gain_tokens_from_file(path_str: str) -> List[str]:
    path = Path(path_str).expanduser()
    data = path.read_text(encoding="utf-8")
    lines = data.splitlines()
    tokens: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens.extend(_expand_gain_tokens([stripped]))
    return tokens


def _fmt_time_hms(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "--:--"
    sec_int = int(round(seconds))
    h, rem = divmod(sec_int, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def _fmt_rate_mph(rate: float) -> str:
    if not math.isfinite(rate) or rate <= 0:
        return "--"
    return f"{rate:.0f} m/h"


def _token_is_likely_gain(token: str, *, default_unit: str = "m") -> bool:
    if not token:
        return False
    if token.startswith("-"):
        return False
    if os.path.sep in token:
        return False
    if os.path.altsep and os.path.altsep in token:
        return False
    lowered = token.lower()
    if lowered.endswith(".fit") or lowered.endswith(".fit.gz"):
        return False
    if "*" in token or "?" in token:
        return False
    parts = token.replace(",", " ").split()
    if not parts:
        parts = [token]
    any_valid = False
    for part in parts:
        parsed = _parse_gain_token(part, default_unit=default_unit)
        if parsed is None:
            return False
        any_valid = True
    return any_valid


def _assert_monotonic_non_decreasing(values: Sequence[float], *, name: str, tol: float = 1e-6) -> None:
    if not logging.getLogger().isEnabledFor(logging.DEBUG):
        return
    if not values:
        return
    prev = float(values[0])
    for idx in range(1, len(values)):
        current = float(values[idx])
        if current + tol < prev:
            raise AssertionError(
                f"{name} violates monotonicity at index {idx}: {current:.6f} < {prev:.6f}"
            )
        if current > prev:
            prev = current


def _is_uniform_1hz(times: Sequence[float], *, tol: float = 1e-6) -> bool:
    if len(times) < 2:
        return False
    dt = float(times[1]) - float(times[0])
    if abs(dt - 1.0) > tol:
        return False
    for idx in range(2, len(times)):
        prev = float(times[idx - 1])
        current = float(times[idx])
        if abs((current - prev) - 1.0) > tol:
            return False
    return True


def _format_gain(value_m: float, unit: str = "m") -> str:
    if not math.isfinite(value_m):
        return "--"
    if unit.lower() == "ft":
        feet = value_m / 0.3048
        return f"{feet:.0f} ft"
    return f"{value_m:.0f} m"


def _convert_gain(value_m: float, unit: str = "m") -> float:
    if unit.lower() == "ft":
        return value_m / 0.3048
    return value_m


def min_time_for_gains(
    times: Sequence[float],
    cumulative_gain: Sequence[float],
    targets: Sequence[float],
) -> List[GainTimePoint]:
    if not times or not cumulative_gain or not targets:
        return []

    t_arr = np.asarray(times, dtype=np.float64)
    g_arr = np.asarray(cumulative_gain, dtype=np.float64)
    if t_arr.shape != g_arr.shape:
        raise ValueError("times and cumulative_gain must have same length")
    if t_arr.size == 0:
        return []

    g_arr = np.maximum.accumulate(g_arr)
    total_gain = float(g_arr[-1])

    merged: List[Optional[GainTimePoint]] = [None] * len(targets)
    positive: List[Tuple[float, int]] = []
    for idx, target in enumerate(targets):
        if not math.isfinite(target) or target < 0.0:
            continue
        if target <= 0.0:
            merged[idx] = GainTimePoint(
                gain_m=0.0,
                min_time_s=0.0,
                avg_rate_m_per_hr=0.0,
                start_offset_s=0.0,
                end_offset_s=0.0,
                note=None,
            )
        else:
            positive.append((float(target), idx))

    if not positive:
        return [pt for pt in merged if pt is not None]

    positive.sort(key=lambda x: x[0])
    eps = 1e-9

    for target_gain, original_idx in positive:
        if target_gain > total_gain + 1e-6:
            merged[original_idx] = GainTimePoint(
                gain_m=target_gain,
                min_time_s=float("inf"),
                avg_rate_m_per_hr=0.0,
                start_offset_s=None,
                end_offset_s=None,
                note="unachievable",
            )
            continue
        left = 0
        best_duration = float("inf")
        best_start = 0
        best_end = 0
        for right in range(t_arr.size):
            while left < right and (g_arr[right] - g_arr[left]) >= target_gain - eps:
                duration = t_arr[right] - t_arr[left]
                if duration > 0.0 and duration + eps < best_duration:
                    best_duration = duration
                    best_start = left
                    best_end = right
                left += 1
        if not math.isfinite(best_duration):
            merged[original_idx] = GainTimePoint(
                gain_m=target_gain,
                min_time_s=float("inf"),
                avg_rate_m_per_hr=0.0,
                start_offset_s=None,
                end_offset_s=None,
                note="unachievable",
            )
            continue
        duration = best_duration
        avg_rate = target_gain / duration * 3600.0 if duration > 0.0 else 0.0
        merged[original_idx] = GainTimePoint(
            gain_m=target_gain,
            min_time_s=duration,
            avg_rate_m_per_hr=avg_rate,
            start_offset_s=float(t_arr[best_start]),
            end_offset_s=float(t_arr[best_end]),
            note=None,
        )

    return [pt for pt in merged if pt is not None]


def min_time_for_gains_numba(
    times: Sequence[float],
    cumulative_gain: Sequence[float],
    targets: Sequence[float],
) -> List[GainTimePoint]:
    if not HAVE_NUMBA:
        raise RuntimeError("Numba is not available")
    if not times or not cumulative_gain or not targets:
        return []

    t_arr = np.asarray(times, dtype=np.float64)
    g_arr = np.asarray(cumulative_gain, dtype=np.float64)
    if t_arr.shape != g_arr.shape or t_arr.size == 0:
        return []

    g_arr = np.maximum.accumulate(g_arr)
    total_gain = float(g_arr[-1])

    merged: List[Optional[GainTimePoint]] = [None] * len(targets)
    positive: List[Tuple[float, int]] = []
    for idx, target in enumerate(targets):
        if not math.isfinite(target) or target < 0.0:
            continue
        if target <= 0.0:
            merged[idx] = GainTimePoint(
                gain_m=0.0,
                min_time_s=0.0,
                avg_rate_m_per_hr=0.0,
                start_offset_s=0.0,
                end_offset_s=0.0,
                note=None,
            )
        elif target > total_gain + 1e-6:
            merged[idx] = GainTimePoint(
                gain_m=float(target),
                min_time_s=float("inf"),
                avg_rate_m_per_hr=0.0,
                start_offset_s=None,
                end_offset_s=None,
                note="unachievable",
            )
        else:
            positive.append((float(target), idx))

    if not positive:
        return [pt for pt in merged if pt is not None]

    positive.sort(key=lambda x: x[0])
    sorted_targets = np.ascontiguousarray(np.asarray([val for val, _ in positive], dtype=np.float64))
    durations, start_idx_arr, end_idx_arr = _numba_min_time_for_gains_kernel(
        np.ascontiguousarray(t_arr),
        np.ascontiguousarray(g_arr),
        sorted_targets,
        NUMBA_EPS,
    )

    for pos_idx, (target_gain, original_idx) in enumerate(positive):
        duration = float(durations[pos_idx])
        start_idx = int(start_idx_arr[pos_idx])
        end_idx = int(end_idx_arr[pos_idx])
        if not math.isfinite(duration) or start_idx < 0 or end_idx < 0:
            merged[original_idx] = GainTimePoint(
                gain_m=target_gain,
                min_time_s=float("inf"),
                avg_rate_m_per_hr=0.0,
                start_offset_s=None,
                end_offset_s=None,
                note="unachievable",
            )
            continue
        avg_rate = target_gain / duration * 3600.0 if duration > 0.0 else 0.0
        merged[original_idx] = GainTimePoint(
            gain_m=target_gain,
            min_time_s=duration,
            avg_rate_m_per_hr=avg_rate,
            start_offset_s=float(t_arr[start_idx]),
            end_offset_s=float(t_arr[end_idx]),
            note=None,
        )

    return [pt for pt in merged if pt is not None]


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


def _export_series_command(
    fit_files: List[str],
    output: str,
    source: str,
    verbose: bool,
    resample_1hz: bool,
    parse_workers: int,
    gain_eps: float,
    session_gap_sec: float,
    qc_enabled: bool,
    qc_spec_path: Optional[str],
    merge_eps_sec: float,
    overlap_policy: str,
    log_file: Optional[str],
    profile: bool,
) -> int:
    _setup_logging(verbose, log_file=log_file)
    profiler = _StageProfiler(profile)
    try:
        series = _load_activity_series(
            fit_files,
            source=source,
            gain_eps=gain_eps,
            session_gap_sec=session_gap_sec,
            qc_enabled=qc_enabled,
            qc_spec_path=qc_spec_path,
            resample_1hz=resample_1hz,
            merge_eps_sec=merge_eps_sec,
            overlap_policy=overlap_policy,
            parse_workers=parse_workers,
            profiler=profiler,
        )
    except Exception as exc:
        logging.error(str(exc))
        return 2

    try:
        with open(output, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["time_s", "cumulative_gain_m", "source"])
            for t, g in zip(series.times, series.values):
                writer.writerow([round(t, 6), round(g, 6), series.selected_raw])
        logging.info("Wrote: %s", output)
        profiler.lap("csv")
    except Exception as exc:
        logging.error(f"Failed to write export: {exc}")
        return 2

    return 0


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
    # Suppress very chatty third-party DEBUG logs (e.g., matplotlib findfont)
    try:
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
        logging.getLogger("PIL").setLevel(logging.INFO)
        logging.getLogger("fontTools").setLevel(logging.INFO)
    except Exception:
        pass


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


def _fit_cache_paths(fit_path: str) -> Tuple[Optional[str], Optional[str], Optional[os.stat_result]]:
    try:
        st = os.stat(fit_path)
    except OSError:
        return None, None, None
    os.makedirs(_PARSED_FIT_CACHE_DIR, exist_ok=True)
    base = os.path.basename(fit_path)
    cache_name = f"{base}.{st.st_mtime_ns}.{st.st_size}.pkl"
    cache_path = os.path.join(_PARSED_FIT_CACHE_DIR, cache_name)
    prefix = os.path.join(_PARSED_FIT_CACHE_DIR, base + ".")
    return cache_path, prefix, st


def _load_fit_cache(fit_path: str) -> Optional[List[Dict[str, Any]]]:
    cache_path, _, st = _fit_cache_paths(fit_path)
    if cache_path is None or st is None:
        return None
    if not os.path.exists(cache_path):
        return None
    try:
        with open(cache_path, "rb") as fh:
            data = pickle.load(fh)
    except Exception:
        return None
    meta = data.get("meta") if isinstance(data, dict) else None
    if not isinstance(meta, dict):
        return None
    if meta.get("mtime_ns") != st.st_mtime_ns or meta.get("size") != st.st_size:
        return None
    records = data.get("records")
    if isinstance(records, list):
        return records
    return None


def _save_fit_cache(fit_path: str, records: List[Dict[str, Any]]) -> None:
    cache_path, prefix, st = _fit_cache_paths(fit_path)
    if cache_path is None or st is None or prefix is None:
        return
    payload = {
        "meta": {"mtime_ns": st.st_mtime_ns, "size": st.st_size},
        "records": records,
    }
    tmp_path = cache_path + ".tmp"
    try:
        with open(tmp_path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_path, cache_path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return
    base = os.path.basename(fit_path)
    try:
        for fname in os.listdir(_PARSED_FIT_CACHE_DIR):
            if not fname.startswith(base + ".") or not fname.endswith(".pkl"):
                continue
            if fname == os.path.basename(cache_path):
                continue
            try:
                os.remove(os.path.join(_PARSED_FIT_CACHE_DIR, fname))
            except Exception:
                pass
    except Exception:
        pass


def _parse_single_fit_records(fit_path: str, file_id: int) -> List[Dict[str, Any]]:
    cached = _load_fit_cache(fit_path)
    if cached is not None:
        return [dict(rec, file_id=file_id) for rec in cached]
    _require_dependency(FitFile, "fitparse", "pip install fitparse")
    fit = FitFile(fit_path)
    fit.parse()
    out: List[Dict[str, Any]] = []
    # Detect a preferred total gain key for this file
    total_gain_key: Optional[str] = None
    for idx, msg in enumerate(fit.get_messages("record")):
        vals = msg.get_values()
        if total_gain_key is None and idx < 200:
            candidate = _pick_total_gain_key(vals)
            if candidate is not None:
                total_gain_key = candidate
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

        speed_val: Optional[float] = None
        for name in ("enhanced_speed", "speed"):
            if vals.get(name) is not None:
                try:
                    sv = float(vals.get(name))
                except Exception:
                    sv = None
                if sv is not None and math.isfinite(sv):
                    speed_val = max(0.0, sv)
                    break

        cad_val: Optional[float] = None
        for name in (
            "cadence",
            "enhanced_running_cadence",
            "enhanced_cadence",
        ):
            if vals.get(name) is not None:
                try:
                    cv = float(vals.get(name))
                except Exception:
                    cv = None
                if cv is not None and math.isfinite(cv):
                    cad_val = max(0.0, cv)
                    break

        out.append({
            "t": t,
            "file_id": file_id,
            "alt": alt_val,
            "tg": tg_val,
            "inc": inc_val,
            "dist": dist_val,
            "dist_prio": dist_prio,
            "speed": speed_val,
            "cad": cad_val,
        })
    try:
        records_for_cache = [
            {
                "t": rec["t"],
                "file_id": rec.get("file_id", 0),
                "alt": rec.get("alt"),
                "tg": rec.get("tg"),
                "inc": rec.get("inc"),
                "dist": rec.get("dist"),
                "dist_prio": rec.get("dist_prio", 0),
                "speed": rec.get("speed"),
                "cad": rec.get("cad"),
            }
            for rec in out
        ]
        _save_fit_cache(fit_path, records_for_cache)
    except Exception:
        pass
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
            if rec.get("speed") is not None and chosen.get("speed") is None:
                chosen["speed"] = rec["speed"]
            if rec.get("cad") is not None and chosen.get("cad") is None:
                chosen["cad"] = rec["cad"]
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
        # Physically-motivated processing on altitude
        alt_raw_t: List[float] = []
        alt_raw_v: List[float] = []
        for r in merged:
            alt = r.get("alt")
            if alt is None:
                continue
            t_rel = r["t"] - t0
            alt_raw_t.append(t_rel)
            alt_raw_v.append(float(alt))
        if len(alt_raw_t) < 2:
            raise RuntimeError("Insufficient altitude data to compute curve after merging.")
        t_alt = np.asarray(alt_raw_t, dtype=np.float64)
        z_alt = np.asarray(alt_raw_v, dtype=np.float64)
        # Estimate median speed and grade from merged
        d_all = [r.get("dist") for r in merged]
        t_all = [r.get("t") - t0 for r in merged]
        speed_med = 0.0
        if any(d is not None for d in d_all):
            d_arr = np.asarray([float(d) if d is not None else np.nan for d in d_all], dtype=np.float64)
            dt_all = np.diff(np.asarray(t_all, dtype=np.float64))
            dd_all = np.diff(d_arr)
            m = (~np.isnan(dd_all)) & (dt_all > 1e-6)
            if np.any(m):
                v = dd_all[m] / dt_all[m]
                if v.size:
                    speed_med = float(np.median(np.clip(v, 0.0, np.inf)))
        inc_all = [r.get("inc") for r in merged]
        grade_med = 0.0
        if any(i is not None for i in inc_all):
            inc_arr = np.asarray([float(x) if x is not None else np.nan for x in inc_all], dtype=np.float64)
            val = inc_arr[~np.isnan(inc_arr)]
            if val.size:
                grade_med = float(np.median(np.clip(val / 100.0, -1.0, 1.0)))
        diag: Dict[str, Any] = {}
        # Additional speed/grade diagnostics
        try:
            # Speed fractions over time
            if any(d is not None for d in d_all):
                d_arr = np.asarray([float(d) if d is not None else np.nan for d in d_all], dtype=np.float64)
                t_arr = np.asarray(t_all, dtype=np.float64)
                dt = np.diff(t_arr)
                dd = np.diff(d_arr)
                mask = (~np.isnan(dd)) & (dt > 1e-6)
                if np.any(mask):
                    v = dd[mask] / dt[mask]
                    dtm = dt[mask]
                    low = dtm[v < 0.5].sum() if np.any(v < 0.5) else 0.0
                    diag["speed_low_time_pct"] = float(low / dtm.sum()) if dtm.sum() > 0 else 0.0
            # Grade fraction by samples
            if any(i is not None for i in inc_all):
                inc_arr = np.asarray([float(x) if x is not None else np.nan for x in inc_all], dtype=np.float64)
                steep = np.count_nonzero(inc_arr > 10.0)
                total = np.count_nonzero(~np.isnan(inc_arr))
                diag["grade_steep_sample_pct"] = float(steep / total) if total > 0 else 0.0
        except Exception:
            pass
        alt_eff = _effective_altitude_path(t_alt, z_alt, speed_med, grade_med, diag)
        indoor_hint = _infer_indoor_mode(merged)
        alt_idle, moving_mask, _ = _apply_idle_detection(
            merged,
            t_alt,
            alt_eff,
            diag,
            t0,
            indoor_hint=indoor_hint,
        )
        # Cumulative ascent (uprun epsilon, baro/GNSS defaults lower)
        eps_gain = 0.02
        cum_series = _cum_ascent_from_alt(
            alt_idle,
            eps_gain,
            mode="uprun",
            diag=diag,
            moving_mask=moving_mask,
        )
        for i in range(t_alt.size):
            times.append(float(t_alt[i]))
            G.append(float(cum_series[i]))
        # Diagnostics
        try:
            logging.debug(
                "Altitude ascent diagnostics: n=%d net=%.1fm peak(start->max)=%.1fm range=%.1fm gross_noeps=%.1fm gross_eps=%.1fm eps_loss=%.1fm(%.0f%%) "
                "up_runs=%d pos_steps=%d eps_mode=%s neg_pre=%.1fm neg_post=%.1fm spikes=%d hampel=%d T=%.2fs speed_med=%.2f m/s grade_med=%.2f%% speed<0.5=%.0f%% grade>10=%.0f%%",
                diag.get("n_alt", 0),
                diag.get("net_gain_m", float('nan')),
                diag.get("peak_gain_from_start_m", float('nan')),
                diag.get("range_max_min_m", float('nan')),
                diag.get("gross_noeps_m", float('nan')),
                diag.get("gross_eps_m", float('nan')),
                diag.get("eps_loss_m", float('nan')),
                100.0 * diag.get("eps_loss_pct", 0.0),
                diag.get("up_runs", 0),
                diag.get("pos_steps", 0),
                str(diag.get("eps_mode", "uprun")),
                diag.get("neg_sum_pre_close_m", float('nan')),
                diag.get("neg_sum_post_m", float('nan')),
                diag.get("spike_count", 0),
                diag.get("hampel_count", 0),
                diag.get("closing_T_s", float('nan')),
                diag.get("speed_med_mps", float('nan')),
                100.0 * diag.get("grade_med_frac", float('nan')),
                100.0 * diag.get("speed_low_time_pct", 0.0),
                100.0 * diag.get("grade_steep_sample_pct", 0.0),
            )
        except Exception:
            pass

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


def _freeze_for_cache(obj: Any) -> Any:
    if isinstance(obj, dict):
        return tuple(sorted((k, _freeze_for_cache(v)) for k, v in obj.items()))
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze_for_cache(v) for v in obj)
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    return repr(obj)


_WR_ENVELOPE_CACHE: Dict[Tuple[Any, ...], Tuple[Any, Dict[str, Any]]] = {}


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


def _hampel_mask_dh(dh: np.ndarray, window: int = 7, n_sigmas: float = 3.0) -> np.ndarray:
    n = dh.size
    if n == 0:
        return np.zeros(0, dtype=bool)
    w = max(1, int(window))
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        lo = max(0, i - w)
        hi = min(n, i + w + 1)
        seg = dh[lo:hi]
        med = np.median(seg)
        mad = np.median(np.abs(seg - med))
        sigma = 1.4826 * mad if mad > 0 else 0.0
        if sigma > 0 and abs(dh[i] - med) > n_sigmas * sigma:
            out[i] = True
    return out


def _local_poly_smooth(t: np.ndarray, y: np.ndarray, min_span_s: float = 0.4, max_points: int = 9, poly: int = 2) -> np.ndarray:
    n = y.size
    if n == 0:
        return y
    max_points = max(3, int(max_points) | 1)  # make odd
    out = np.empty_like(y)
    for i in range(n):
        # Expand window to cover at least min_span_s
        lo = i
        hi = i
        while True:
            span = (t[hi] - t[lo]) if hi > lo else 0.0
            count = hi - lo + 1
            if span >= min_span_s or count >= max_points or (lo == 0 and hi == n - 1):
                break
            if lo > 0:
                lo -= 1
                count += 1
            if count < max_points and hi < n - 1:
                hi += 1
        idx = slice(lo, hi + 1)
        tt = t[idx] - t[i]
        yy = y[idx]
        # Build Vandermonde for degree poly
        X = np.vstack([tt ** k for k in range(poly + 1)]).T
        # Solve normal equations
        try:
            coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
            out[i] = coef[0]
        except Exception:
            out[i] = y[i]
    return out


from collections import defaultdict, deque


def _sliding_extrema_sym(t: np.ndarray, y: np.ndarray, half_window_s: float, mode: str) -> np.ndarray:
    n = y.size
    if n == 0:
        return y
    res = np.empty_like(y)
    dq: deque[int] = deque()
    right = 0
    for i in range(n):
        left_bound = t[i] - half_window_s
        right_bound = t[i] + half_window_s
        while right < n and t[right] <= right_bound:
            # Push right maintaining monotonic deque
            while dq:
                if mode == 'max' and y[right] >= y[dq[-1]]:
                    dq.pop()
                elif mode == 'min' and y[right] <= y[dq[-1]]:
                    dq.pop()
                else:
                    break
            dq.append(right)
            right += 1
        # Pop left indices out of window
        while dq and t[dq[0]] < left_bound:
            dq.popleft()
        if dq:
            res[i] = y[dq[0]]
        else:
            res[i] = y[i]
    return res


def _morphological_closing_time(t: np.ndarray, y: np.ndarray, T: float) -> np.ndarray:
    hw = max(0.0, float(T) * 0.5)
    if hw <= 0:
        return y.copy()
    dil = _sliding_extrema_sym(t, y, hw, mode='max')
    clo = _sliding_extrema_sym(t, dil, hw, mode='min')
    return clo


def _effective_altitude_path(
    t: np.ndarray,
    z: np.ndarray,
    speed_med: float = 1.5,
    grade_med: float = 0.0,
    diag: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    if z.size <= 2:
        return z.copy()
    # Outlier rejection on vertical speed and Hampel on dh
    dt = np.diff(t)
    dz = np.diff(z)
    with np.errstate(divide='ignore', invalid='ignore'):
        v = dz / np.where(dt > 1e-9, dt, 1e-9)
    spike = np.abs(v) > 2.0
    hamp = _hampel_mask_dh(dz, window=5, n_sigmas=3.0)
    mask_bad = spike | hamp
    z1 = z.copy()
    if np.any(mask_bad):
        bad_idx = np.where(mask_bad)[0] + 1  # affect the next sample
        bad = np.zeros(z1.size, dtype=bool)
        bad[np.clip(bad_idx, 0, z1.size - 1)] = True
        # Linear interpolation over bad points
        good = ~bad
        if np.count_nonzero(good) >= 2:
            z1[bad] = np.interp(t[bad], t[good], z1[good])
    # Quantization repair via local quadratic smoothing (tiny bandwidth)
    z2 = _local_poly_smooth(t, z1, min_span_s=0.4, max_points=7, poly=2)
    # Time-scale selection
    sp = max(0.01, float(speed_med))
    base_T = 4.0
    if grade_med > 0.10:
        base_T *= 0.7
    if sp < 1.0:
        base_T *= max(0.6, sp / 1.0)
    T = float(max(3.0, min(6.0, base_T)))
    # Morphological closing to remove micro-descents
    zc = _morphological_closing_time(t, z2, T)
    # Weak post-smoothing to remove quantization
    z3 = _local_poly_smooth(t, zc, min_span_s=0.8, max_points=9, poly=2)
    # Diagnostics (optional)
    if diag is not None:
        try:
            diag["n_alt"] = int(z.size)
            diag["spike_count"] = int(np.count_nonzero(spike))
            diag["hampel_count"] = int(np.count_nonzero(hamp))
            diag["speed_med_mps"] = float(speed_med)
            diag["grade_med_frac"] = float(grade_med)
            diag["closing_T_s"] = float(T)
            # Quantization proxy: MAD of dz before and after smoothing
            dz_raw = np.diff(z)
            dz_s2 = np.diff(z2)
            dz_s3 = np.diff(z3)
            for key, arr in (("dz_raw", dz_raw), ("dz_preclose", dz_s2), ("dz_post", dz_s3)):
                if arr.size:
                    med = float(np.median(arr))
                    mad = float(np.median(np.abs(arr - med)))
                else:
                    med = 0.0; mad = 0.0
                diag[key+"_med"] = med
                diag[key+"_mad"] = mad
            # Negative sums before/after closing
            neg_pre = float(np.sum(np.maximum(0.0, -dz_s2))) if dz_s2.size else 0.0
            neg_post = float(np.sum(np.maximum(0.0, -dz_s3))) if dz_s3.size else 0.0
            gross_post = float(np.sum(np.maximum(0.0, dz_s3))) if dz_s3.size else 0.0
            diag["neg_sum_pre_close_m"] = neg_pre
            diag["neg_sum_post_m"] = neg_post
            diag["gross_noeps_m"] = gross_post
            start_alt = float(z3[0]) if z3.size else 0.0
            max_alt = float(np.max(z3)) if z3.size else start_alt
            min_alt = float(np.min(z3)) if z3.size else start_alt
            end_alt = float(z3[-1]) if z3.size else start_alt
            diag["start_alt_m"] = start_alt
            diag["end_alt_m"] = end_alt
            diag["max_alt_m"] = max_alt
            diag["min_alt_m"] = min_alt
            diag["net_gain_m"] = end_alt - start_alt
            diag["range_max_min_m"] = max_alt - min_alt
            diag["peak_gain_from_start_m"] = max_alt - start_alt
            # Up-run count and pos step stats
            if dz_s3.size:
                pos_mask = dz_s3 > 0
                up_runs = int(np.sum((~pos_mask[:-1]) & pos_mask[1:])) + (1 if pos_mask[0] else 0)
                diag["up_runs"] = up_runs
                diag["pos_steps"] = int(np.count_nonzero(pos_mask))
            else:
                diag["up_runs"] = 0
                diag["pos_steps"] = 0
        except Exception:
            pass
    return z3


def _infer_indoor_mode(records: List[Dict[str, Any]]) -> bool:
    if not records:
        return False
    dev_dist = 0
    inc_samples = 0
    for rec in records:
        if rec.get("dist_prio", 0) and int(rec.get("dist_prio", 0)) >= 3:
            dev_dist += 1
        if rec.get("inc") is not None:
            inc_samples += 1
    total = len(records)
    if total == 0:
        return False
    if dev_dist >= max(5, int(0.4 * total)) and inc_samples > 0:
        return True
    if dev_dist >= max(5, int(0.6 * total)):
        return True
    return False


class _RollingMedian:
    def __init__(self) -> None:
        self._low: List[Tuple[float, int]] = []  # max-heap via negative values
        self._high: List[Tuple[float, int]] = []  # min-heap
        self._invalid_low: Dict[int, int] = defaultdict(int)
        self._invalid_high: Dict[int, int] = defaultdict(int)
        self._entries: Dict[int, float] = {}
        self._size = 0

    def _prune(self, heap: List[Tuple[float, int]], invalid: Dict[int, int]) -> None:
        while heap:
            value, idx = heap[0]
            if invalid.get(idx, 0):
                heapq.heappop(heap)
                invalid[idx] -= 1
                if invalid[idx] <= 0:
                    invalid.pop(idx, None)
            else:
                break

    def _rebalance(self) -> None:
        self._prune(self._low, self._invalid_low)
        self._prune(self._high, self._invalid_high)
        if len(self._low) > len(self._high) + 1:
            value, idx = heapq.heappop(self._low)
            self._prune(self._low, self._invalid_low)
            heapq.heappush(self._high, (-value, idx))
            self._prune(self._high, self._invalid_high)
        elif len(self._high) > len(self._low):
            value, idx = heapq.heappop(self._high)
            self._prune(self._high, self._invalid_high)
            heapq.heappush(self._low, (-value, idx))
            self._prune(self._low, self._invalid_low)

    def add(self, value: float, idx: int) -> None:
        if not math.isfinite(value):
            return
        self._entries[idx] = value
        if not self._low or value <= -self._low[0][0]:
            heapq.heappush(self._low, (-value, idx))
        else:
            heapq.heappush(self._high, (value, idx))
        self._size += 1
        self._rebalance()

    def discard(self, idx: int) -> None:
        if idx not in self._entries:
            return
        value = self._entries.pop(idx)
        if self._low and value <= -self._low[0][0]:
            self._invalid_low[idx] += 1
            if self._low and self._low[0][1] == idx:
                self._prune(self._low, self._invalid_low)
        else:
            self._invalid_high[idx] += 1
            if self._high and self._high[0][1] == idx:
                self._prune(self._high, self._invalid_high)
        self._size -= 1
        if self._size < 0:
            self._size = 0
        self._rebalance()

    def median(self) -> float:
        if self._size <= 0:
            return 0.0
        self._prune(self._low, self._invalid_low)
        self._prune(self._high, self._invalid_high)
        if len(self._low) > len(self._high):
            return -self._low[0][0]
        if len(self._high) > len(self._low):
            return self._high[0][0]
        return (-self._low[0][0] + self._high[0][0]) * 0.5


def _rolling_median_time(t: np.ndarray, values: np.ndarray, window_s: float) -> np.ndarray:
    n = values.size
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    rm = _RollingMedian()
    added = [False] * n
    start = 0
    for i in range(n):
        val = float(values[i])
        if math.isfinite(val):
            rm.add(val, i)
            added[i] = True
        while start <= i and t[i] - t[start] > window_s:
            if added[start]:
                rm.discard(start)
            start += 1
        out[i] = rm.median()
    return out


def _rolling_distance_advance(t: np.ndarray, dist: np.ndarray, window_s: float) -> np.ndarray:
    n = dist.size
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    max_dq: deque[int] = deque()
    min_dq: deque[int] = deque()
    start = 0
    for i in range(n):
        while start <= i and t[i] - t[start] > window_s:
            if max_dq and max_dq[0] == start:
                max_dq.popleft()
            if min_dq and min_dq[0] == start:
                min_dq.popleft()
            start += 1
        while max_dq and dist[max_dq[-1]] <= dist[i]:
            max_dq.pop()
        max_dq.append(i)
        while min_dq and dist[min_dq[-1]] >= dist[i]:
            min_dq.pop()
        min_dq.append(i)
        if max_dq and min_dq:
            out[i] = dist[max_dq[0]] - dist[min_dq[0]]
        else:
            out[i] = 0.0
    return out


def _instantaneous_speed(t: np.ndarray, dist: np.ndarray) -> np.ndarray:
    n = dist.size
    out = np.zeros(n, dtype=np.float64)
    if n <= 1:
        return out
    dt = np.diff(t)
    dd = np.diff(dist)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.divide(dd, dt, out=np.zeros_like(dd), where=dt > 1e-6)
    speed = np.clip(speed, 0.0, 20.0)
    out[1:] = speed
    out[0] = out[1] if n > 1 else 0.0
    return out


def _instantaneous_vertical_speed(t: np.ndarray, alt: np.ndarray) -> np.ndarray:
    n = alt.size
    out = np.zeros(n, dtype=np.float64)
    if n <= 1:
        return out
    dt = np.diff(t)
    dz = np.diff(alt)
    with np.errstate(divide="ignore", invalid="ignore"):
        vs = np.divide(dz, dt, out=np.zeros_like(dz), where=dt > 1e-6)
    vs = np.clip(np.abs(vs), 0.0, 10.0)
    out[1:] = vs
    out[0] = out[1] if n > 1 else 0.0
    return out


def _estimate_speed_noise_floor(speed: np.ndarray) -> Tuple[float, float]:
    if speed.size == 0:
        return 0.15, 0.0
    valid = speed[np.isfinite(speed)]
    if valid.size == 0:
        return 0.15, 0.0
    valid = np.clip(valid, 0.0, None)
    still = valid[valid <= 0.6]
    if still.size > 0:
        baseline = float(np.median(still))
    else:
        try:
            baseline = float(np.percentile(valid, 10))
        except Exception:
            baseline = float(np.median(valid))
    if not math.isfinite(baseline):
        baseline = 0.0
    baseline = max(0.0, baseline)
    noise = max(0.15, baseline * 3.0)
    return float(noise), baseline


def _interp_series(
    target_t: np.ndarray,
    sample_t: np.ndarray,
    sample_v: np.ndarray,
    default: float,
    left: Optional[float] = None,
    right: Optional[float] = None,
) -> np.ndarray:
    if sample_t.size == 0:
        return np.full_like(target_t, default, dtype=np.float64)
    mask = np.isfinite(sample_t) & np.isfinite(sample_v)
    if not np.any(mask):
        return np.full_like(target_t, default, dtype=np.float64)
    st = sample_t[mask]
    sv = sample_v[mask]
    if st.size == 0:
        return np.full_like(target_t, default, dtype=np.float64)
    order = np.argsort(st)
    st = st[order]
    sv = sv[order]
    st_unique, idx = np.unique(st, return_index=True)
    sv = sv[idx]
    st = st_unique
    left_val = sv[0] if left is None else left
    right_val = sv[-1] if right is None else right
    return np.interp(target_t, st, sv, left=left_val, right=right_val)


def _apply_idle_hold(
    t: np.ndarray,
    alt: np.ndarray,
    idle_mask: np.ndarray,
    drift_limit: float,
) -> np.ndarray:
    out = alt.copy()
    if out.size == 0 or not np.any(idle_mask):
        return out
    hold_value = out[0]
    hold_time = t[0]
    if idle_mask[0]:
        hold_value = out[0]
        hold_time = t[0]
    for i in range(1, out.size):
        if idle_mask[i]:
            if not idle_mask[i - 1]:
                hold_value = out[i - 1]
                hold_time = t[i - 1]
            if drift_limit > 0 and t[i] > hold_time:
                max_delta = drift_limit * max(0.0, t[i] - hold_time)
                lower = hold_value - max_delta
                upper = hold_value + max_delta
                val = out[i]
                if val < lower:
                    out[i] = lower
                elif val > upper:
                    out[i] = upper
                else:
                    out[i] = val
            else:
                out[i] = hold_value
        else:
            hold_value = out[i]
            hold_time = t[i]
    return out


def _apply_idle_detection(
    records: List[Dict[str, Any]],
    t: np.ndarray,
    alt_eff: np.ndarray,
    diag: Optional[Dict[str, Any]],
    t0: float,
    indoor_hint: Optional[bool] = None,
    window_s: float = 5.0,
    drift_limit: float = 0.002,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = t.size
    if n == 0:
        return alt_eff.copy(), np.ones(0, dtype=bool), np.zeros(0, dtype=bool)
    indoor = indoor_hint if indoor_hint is not None else _infer_indoor_mode(records)

    dist_samples_t: List[float] = []
    dist_samples_v: List[float] = []
    speed_samples_t: List[float] = []
    speed_samples_v: List[float] = []
    cad_samples_t: List[float] = []
    cad_samples_v: List[float] = []

    for rec in records:
        trel = float(rec.get("t", 0.0) - t0)
        if not math.isfinite(trel):
            continue
        if rec.get("dist") is not None:
            try:
                dist_val = float(rec.get("dist"))
            except Exception:
                dist_val = None
            if dist_val is not None and math.isfinite(dist_val):
                dist_samples_t.append(trel)
                dist_samples_v.append(dist_val)
        if rec.get("speed") is not None:
            try:
                speed_val = float(rec.get("speed"))
            except Exception:
                speed_val = None
            if speed_val is not None and math.isfinite(speed_val):
                speed_samples_t.append(trel)
                speed_samples_v.append(max(0.0, speed_val))
        if rec.get("cad") is not None:
            try:
                cad_val = float(rec.get("cad"))
            except Exception:
                cad_val = None
            if cad_val is not None and math.isfinite(cad_val):
                cad_samples_t.append(trel)
                cad_samples_v.append(max(0.0, cad_val))

    dist_series: Optional[np.ndarray] = None
    if dist_samples_t:
        dist_t = np.asarray(dist_samples_t, dtype=np.float64)
        dist_v = np.asarray(dist_samples_v, dtype=np.float64)
        mask = np.isfinite(dist_t) & np.isfinite(dist_v)
        if np.any(mask):
            dist_t = dist_t[mask]
            dist_v = dist_v[mask]
            if dist_t.size == 1:
                dist_series = np.full(n, float(dist_v[0]), dtype=np.float64)
            elif dist_t.size > 1:
                order = np.argsort(dist_t)
                dist_t = dist_t[order]
                dist_v = dist_v[order]
                dist_v = np.maximum.accumulate(dist_v)
                dist_t, idx = np.unique(dist_t, return_index=True)
                dist_v = dist_v[idx]
                dist_series = np.interp(t, dist_t, dist_v, left=dist_v[0], right=dist_v[-1])

    if dist_series is None:
        speed_series = None
    else:
        speed_series = _instantaneous_speed(t, dist_series)

    if speed_series is None:
        if speed_samples_t:
            speed_series = _interp_series(
                t,
                np.asarray(speed_samples_t, dtype=np.float64),
                np.asarray(speed_samples_v, dtype=np.float64),
                default=0.0,
                left=0.0,
                right=0.0,
            )
        else:
            speed_series = np.zeros(n, dtype=np.float64)

    cad_series = _interp_series(
        t,
        np.asarray(cad_samples_t, dtype=np.float64),
        np.asarray(cad_samples_v, dtype=np.float64),
        default=0.0,
        left=0.0,
        right=0.0,
    ) if cad_samples_t else np.zeros(n, dtype=np.float64)

    cad_series = np.clip(cad_series, 0.0, None)

    if dist_series is None:
        dist_series = np.zeros(n, dtype=np.float64)

    vertical_speed = _instantaneous_vertical_speed(t, alt_eff)

    v_med = _rolling_median_time(t, speed_series, window_s)
    cad_med = _rolling_median_time(t, cad_series, window_s)
    vv_med = _rolling_median_time(t, vertical_speed, window_s)
    ds_adv = _rolling_distance_advance(t, dist_series, window_s)

    v_noise, v_baseline = _estimate_speed_noise_floor(speed_series)
    if indoor:
        V_ON = 0.1
        V_OFF = 0.05
    else:
        V_ON = max(0.25, v_noise * 2.0)
        V_OFF = max(0.15, v_noise * 1.2)

    CAD_ON = 12.0
    CAD_OFF = 6.0
    DS_ON = 3.0
    DS_OFF = 1.5
    VV_ON = 0.05
    VV_OFF = 0.02
    T_enter = 2.0
    T_exit = 1.0

    idle_mask = np.zeros(n, dtype=bool)
    in_idle = False
    below_off = 0.0
    above_on = 0.0

    for i in range(n):
        dt = 0.0 if i == 0 else max(0.0, float(t[i] - t[i - 1]))
        is_moving_on = (
            (v_med[i] >= V_ON)
            or (cad_med[i] >= CAD_ON)
            or (vv_med[i] >= VV_ON)
            or (ds_adv[i] >= DS_ON)
        )
        is_idle_off = (
            (v_med[i] < V_OFF)
            and (cad_med[i] < CAD_OFF)
            and (vv_med[i] < VV_OFF)
            and (ds_adv[i] < DS_OFF)
        )

        if in_idle:
            if is_moving_on:
                above_on += dt
                below_off = 0.0
            else:
                above_on = 0.0
            if above_on >= T_exit:
                in_idle = False
                above_on = 0.0
        else:
            if is_idle_off:
                below_off += dt
                above_on = 0.0
            else:
                below_off = 0.0
            if below_off >= T_enter:
                in_idle = True
                below_off = 0.0
        idle_mask[i] = in_idle

    moving_mask = ~idle_mask
    alt_adj = _apply_idle_hold(t, alt_eff, idle_mask, drift_limit)

    if n <= 0:
        segments_count = 0
    elif n == 1:
        segments_count = 1 if idle_mask[0] else 0
    else:
        transitions = (~idle_mask[:-1]) & idle_mask[1:]
        segments_count = int(np.count_nonzero(transitions))
        if idle_mask[0]:
            segments_count += 1

    if diag is not None:
        try:
            dh_raw = np.diff(alt_eff)
            dh_pos = np.maximum(dh_raw, 0.0)
            moving_step = (~idle_mask[1:]).astype(float) if idle_mask.size > 1 else np.zeros(0)
            gross_before = float(dh_pos.sum())
            gross_after = float((dh_pos * moving_step).sum()) if moving_step.size else gross_before
            gross_removed = max(0.0, gross_before - gross_after)

            durations: List[float] = []
            segments_detail: List[Dict[str, float]] = []
            i = 0
            while i < n:
                if idle_mask[i]:
                    start_idx = i
                    start_t = float(t[i])
                    while i < n and idle_mask[i]:
                        i += 1
                    end_idx = i - 1
                    end_t = float(t[end_idx]) if end_idx >= 0 else start_t
                    dur = max(0.0, end_t - start_t)
                    durations.append(dur)
                    raw_gain_seg = 0.0
                    gated_gain_seg = 0.0
                    if end_idx > start_idx and dh_pos.size >= end_idx:
                        raw_gain_seg = float(dh_pos[start_idx:end_idx].sum())
                        if moving_step.size >= end_idx:
                            gated_gain_seg = float(
                                (dh_pos[start_idx:end_idx] * moving_step[start_idx:end_idx]).sum()
                            )
                    segments_detail.append(
                        {
                            "start_s": start_t,
                            "end_s": end_t,
                            "duration_s": dur,
                            "raw_gain_m": raw_gain_seg,
                            "gated_gain_m": gated_gain_seg,
                        }
                    )
                else:
                    i += 1

            span = float(t[-1] - t[0]) if n > 1 else 0.0
            idle_time = float(sum(durations))
            idle_fraction = (idle_time / span) if span > 1e-9 else 0.0
            median_duration = float(np.median(durations)) if durations else 0.0

            top_segments = sorted(
                segments_detail,
                key=lambda seg: seg.get("raw_gain_m", 0.0),
                reverse=True,
            )[:3]

            diag["idle_time_fraction"] = idle_fraction
            diag["idle_total_time_s"] = idle_time
            diag["idle_segments"] = segments_count
            diag["idle_median_duration_s"] = median_duration
            diag["idle_thresholds"] = {
                "V_ON": float(V_ON),
                "V_OFF": float(V_OFF),
                "CAD_ON": float(CAD_ON),
                "CAD_OFF": float(CAD_OFF),
                "DS_ON": float(DS_ON),
                "DS_OFF": float(DS_OFF),
                "VV_ON": float(VV_ON),
                "VV_OFF": float(VV_OFF),
                "T_enter": float(T_enter),
                "T_exit": float(T_exit),
            }
            diag["idle_v_noise_mps"] = float(v_noise)
            diag["idle_v_baseline_mps"] = float(v_baseline)
            diag["idle_window_s"] = float(window_s)
            diag["idle_drift_limit_mps"] = float(drift_limit)
            diag["idle_indoor_mode"] = bool(indoor)
            diag["idle_has_distance"] = bool(dist_samples_t)
            diag["idle_has_cadence"] = bool(cad_samples_t)
            diag["idle_gross_before_m"] = gross_before
            diag["idle_gross_after_m"] = gross_after
            diag["idle_gross_removed_m"] = gross_removed
            diag["idle_segments_detail"] = top_segments
        except Exception:
            pass

    try:
        idle_pct = 100.0 * float(np.mean(idle_mask)) if n > 0 else 0.0
        logging.debug(
            "Idle detection: indoor=%s idle=%.1f%% segments=%d V_ON=%.2f V_OFF=%.2f cad_on=%.1f vv_on=%.3f",
            indoor,
            idle_pct,
            segments_count,
            V_ON,
            V_OFF,
            CAD_ON,
            VV_ON,
        )
    except Exception:
        pass

    return alt_adj, moving_mask, idle_mask


def _cum_ascent_from_alt(
    alt: np.ndarray,
    eps_gain: float = 0.05,
    mode: str = "uprun",
    diag: Optional[Dict[str, Any]] = None,
    moving_mask: Optional[np.ndarray] = None,
) -> List[float]:
    """Compute cumulative gross ascent from an effective altitude trace.

    - mode='uprun': apply epsilon once per contiguous positive run (recommended).
    - mode='sample': apply epsilon to each positive increment (legacy).
    - mode='none': no epsilon.
    - moving_mask: optional bool mask (length n) where True allows accumulation
      for samples deemed "moving" by idle detection.
    """
    n = alt.size
    if n == 0:
        return []
    eps = max(0.0, float(eps_gain))
    cum = 0.0
    series: List[float] = [0.0]
    up_runs = 0
    pos_steps = 0
    run_gain = 0.0
    moving_flags: Optional[np.ndarray]
    if moving_mask is not None and moving_mask.shape[0] == n:
        moving_flags = moving_mask.astype(bool)
    else:
        moving_flags = None
    for i in range(1, n):
        dv = float(alt[i] - alt[i - 1])
        move_allowed = True
        if moving_flags is not None:
            move_allowed = bool(moving_flags[i])
        if dv > 0 and move_allowed:
            pos_steps += 1
            if mode == "sample":
                cum += max(0.0, dv - eps)
            else:
                run_gain += dv
        else:
            if mode == "uprun" and run_gain > 0.0:
                up_runs += 1
                cum += max(0.0, run_gain - eps)
                run_gain = 0.0
        series.append(cum)
    if mode == "uprun" and run_gain > 0.0:
        up_runs += 1
        cum += max(0.0, run_gain - eps)
        series[-1] = cum

    if diag is not None:
        try:
            dz = np.diff(alt)
            gross_noeps = float(np.sum(np.maximum(0.0, dz))) if dz.size else 0.0
            gross_eps = float(series[-1]) if series else 0.0
            eps_loss = max(0.0, gross_noeps - gross_eps)
            diag["eps_mode"] = mode
            diag["up_runs"] = up_runs
            diag["pos_steps"] = pos_steps
            diag["gross_eps_m"] = gross_eps
            diag["eps_loss_m"] = eps_loss
            diag["eps_loss_pct"] = (eps_loss / gross_noeps) if gross_noeps > 1e-9 else 0.0
        except Exception:
            pass
    return series

def _prepare_envelope_arrays(
    times: List[float],
    cumulative_gain: List[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not times:
        return (
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
            np.asarray([], dtype=np.float64),
        )

    times_np = np.asarray(times, dtype=np.float64)
    gains_np = np.asarray(cumulative_gain, dtype=np.float64)
    if times_np.shape != gains_np.shape:
        raise ValueError("times and cumulative_gain must have the same length")

    slopes = np.zeros_like(gains_np)
    if times_np.size >= 2:
        dt = np.diff(times_np)
        dg = np.diff(gains_np)
        with np.errstate(divide="ignore", invalid="ignore"):
            seg_slopes = np.divide(
                dg,
                dt,
                out=np.zeros_like(dg),
                where=np.abs(dt) > 1e-12,
            )
        slopes[:-1] = seg_slopes
        slopes[-1] = seg_slopes[-1]

    U_at_sample = U_eval_many(times_np, gains_np, slopes, times_np)
    return times_np, gains_np, slopes, U_at_sample


def _compute_curve_numpy(
    times: List[float],
    cumulative_gain: List[float],
    durations: Iterable[int],
    gaps: Optional[List[Gap]] = None,
) -> List[CurvePoint]:
    results: List[CurvePoint] = []

    times_np, gains_np, slopes, U_at_sample = _prepare_envelope_arrays(times, cumulative_gain)
    n = times_np.size
    if n == 0:
        return results

    gaps_list = gaps or []
    start_time_min = float(times_np[0])
    end_time_max = float(times_np[-1])

    eps = 1e-9
    ex = times_np
    ey = gains_np

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


EngineMode = Literal["auto", "numpy", "numba", "stride"]


def _resolve_engine(engine: str) -> EngineMode:
    normalized = engine.strip().lower() if engine else "auto"
    if normalized not in {"auto", "numpy", "numba", "stride"}:
        logging.warning("Unknown engine '%s'; falling back to auto", engine)
        return "auto"
    return normalized  # type: ignore[return-value]


NUMBA_EPS = 1e-9


if HAVE_NUMBA:

    @njit(cache=True, fastmath=True, parallel=True)
    def _numba_curve_kernel(
        times: np.ndarray,
        ex: np.ndarray,
        ey: np.ndarray,
        slopes: np.ndarray,
        U_at_sample: np.ndarray,
        durations: np.ndarray,
        gap_starts: np.ndarray,
        gap_ends: np.ndarray,
        gap_lengths: np.ndarray,
        eps: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        m = durations.shape[0]
        gains = np.zeros(m, dtype=np.float64)
        starts = np.zeros(m, dtype=np.float64)
        n = times.shape[0]
        if n == 0:
            return gains, starts

        t_min = times[0]
        t_max = times[n - 1]
        n_ex = ex.shape[0]
        gap_count = gap_lengths.shape[0]

        for di in prange(m):
            D = durations[di]
            best_gain = -1.0
            best_start = t_min

            if D <= 0.0 or t_min + D > t_max + eps:
                gains[di] = 0.0
                starts[di] = t_min
                continue

            end_idx = 0
            # Start-aligned windows
            for i in range(n):
                t0 = times[i]
                t_plus = t0 + D
                if t_plus > t_max + eps:
                    break
                while end_idx < n_ex and ex[end_idx] < t_plus:
                    end_idx += 1
                if end_idx <= 0:
                    u_end = ey[0]
                elif end_idx >= n_ex:
                    u_end = ey[n_ex - 1]
                else:
                    seg_idx = end_idx - 1
                    u_end = ey[seg_idx] + slopes[seg_idx] * (t_plus - ex[seg_idx])
                gain = u_end - U_at_sample[i]
                if gain > best_gain:
                    best_gain = gain
                    best_start = t0

            # End-aligned windows
            start_ptr = 0
            gap_idx = 0
            for k in range(n):
                t_plus = times[k]
                t_minus = t_plus - D
                if t_minus < t_min - eps:
                    continue

                if gap_count > 0:
                    while gap_idx < gap_count and t_minus > gap_ends[gap_idx]:
                        gap_idx += 1
                    if gap_idx < gap_count:
                        if D <= gap_lengths[gap_idx] + eps:
                            gap_hi = gap_ends[gap_idx] - D
                            gap_lo = gap_starts[gap_idx]
                            if gap_hi >= gap_lo - eps:
                                if t_minus >= gap_lo - eps and t_minus <= gap_hi + eps:
                                    continue

                while start_ptr < n_ex and ex[start_ptr] < t_minus:
                    start_ptr += 1
                if start_ptr <= 0:
                    u_start = ey[0]
                elif start_ptr >= n_ex:
                    u_start = ey[n_ex - 1]
                else:
                    seg_idx2 = start_ptr - 1
                    u_start = ey[seg_idx2] + slopes[seg_idx2] * (t_minus - ex[seg_idx2])

                gain = U_at_sample[k] - u_start
                if gain > best_gain:
                    best_gain = gain
                    best_start = t_minus

            if best_gain < 0.0:
                best_gain = 0.0
                best_start = t_min

            max_start = t_max - D
            if max_start < t_min:
                max_start = t_min
            if best_start < t_min:
                best_start = t_min
            if best_start > max_start:
                best_start = max_start

            gains[di] = best_gain
            starts[di] = best_start

        return gains, starts

    @njit(cache=True, fastmath=True)
    def _numba_min_time_for_gains_kernel(
        times: np.ndarray,
        gains: np.ndarray,
        targets: np.ndarray,
        eps: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        m = targets.shape[0]
        durations = np.empty(m, dtype=np.float64)
        start_idx = np.full(m, -1, dtype=np.int64)
        end_idx = np.full(m, -1, dtype=np.int64)
        n = times.shape[0]
        if n == 0:
            return durations, start_idx, end_idx
        total_gain = gains[n - 1]
        for ti in range(m):
            target = targets[ti]
            if target <= 0.0:
                durations[ti] = 0.0
                start_idx[ti] = 0
                end_idx[ti] = 0
                continue
            if target > total_gain + 1e-6:
                durations[ti] = np.inf
                continue
            left = 0
            best_duration = np.inf
            best_start = -1
            best_end = -1
            for right in range(n):
                while left < right and gains[right] - gains[left] >= target - eps:
                    duration = times[right] - times[left]
                    if duration > 0.0 and duration + eps < best_duration:
                        best_duration = duration
                        best_start = left
                        best_end = right
                    left += 1
                if left > right:
                    left = right
            durations[ti] = best_duration
            start_idx[ti] = best_start
            end_idx[ti] = best_end
        return durations, start_idx, end_idx

else:  # pragma: no cover - fallback when numba missing

    def _numba_curve_kernel(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("Numba engine requested but numba is unavailable")

    def _numba_min_time_for_gains_kernel(*args, **kwargs):  # type: ignore[misc]
        raise RuntimeError("Numba engine requested but numba is unavailable")

def _compute_curve_numba(
    times: List[float],
    cumulative_gain: List[float],
    durations: Iterable[int],
    gaps: Optional[List[Gap]] = None,
) -> List[CurvePoint]:
    if not HAVE_NUMBA:
        return _compute_curve_numpy(times, cumulative_gain, durations, gaps=gaps)

    durations_list = [int(d) for d in durations]
    if not durations_list:
        return []

    times_np, gains_np, slopes, U_at_sample = _prepare_envelope_arrays(times, cumulative_gain)
    if times_np.size == 0:
        return []

    durations_arr = np.ascontiguousarray(np.asarray(durations_list, dtype=np.float64))
    times_arr = np.ascontiguousarray(times_np)
    gains_arr = np.ascontiguousarray(gains_np)
    slopes_arr = np.ascontiguousarray(slopes)
    u_at_arr = np.ascontiguousarray(U_at_sample)

    if gaps:
        gap_starts = np.ascontiguousarray(
            np.asarray([g.start for g in gaps], dtype=np.float64)
        )
        gap_ends = np.ascontiguousarray(
            np.asarray([g.end for g in gaps], dtype=np.float64)
        )
        gap_lengths = np.ascontiguousarray(
            np.asarray([g.length for g in gaps], dtype=np.float64)
        )
    else:
        gap_starts = np.ascontiguousarray(np.zeros(0, dtype=np.float64))
        gap_ends = np.ascontiguousarray(np.zeros(0, dtype=np.float64))
        gap_lengths = np.ascontiguousarray(np.zeros(0, dtype=np.float64))

    gains_out, starts_out = _numba_curve_kernel(
        times_arr,
        times_arr,
        gains_arr,
        slopes_arr,
        u_at_arr,
        durations_arr,
        gap_starts,
        gap_ends,
        gap_lengths,
        NUMBA_EPS,
    )

    start_time_min = float(times_arr[0])
    end_time_max = float(times_arr[-1])

    results: List[CurvePoint] = []
    for idx, D in enumerate(durations_list):
        gain = float(gains_out[idx]) if idx < gains_out.size else 0.0
        start = float(starts_out[idx]) if idx < starts_out.size else start_time_min

        if D <= 0 or start_time_min + D > end_time_max + NUMBA_EPS:
            gain = 0.0
            start = start_time_min

        if gain < 0.0:
            gain = 0.0

        max_start = end_time_max - D
        if max_start < start_time_min:
            max_start = start_time_min
        if start < start_time_min:
            start = start_time_min
        if start > max_start:
            start = max_start

        end_offset = start + D
        if end_offset > end_time_max + 1e-9:
            end_offset = end_time_max
        if end_offset < start:
            end_offset = start

        rate = gain / D * 3600.0 if D > 0 else 0.0
        results.append(
            CurvePoint(
                duration_s=D,
                max_climb_m=gain,
                climb_rate_m_per_hr=rate,
                start_offset_s=start,
                end_offset_s=end_offset,
            )
        )

    return results


def _compute_curve_stride(
    times: List[float],
    cumulative_gain: List[float],
    durations: Iterable[int],
    gaps: Optional[List[Gap]] = None,
) -> List[CurvePoint]:
    durations_list = [int(d) for d in durations]
    if not durations_list:
        return []

    times_np, gains_np, slopes, U_at_sample = _prepare_envelope_arrays(times, cumulative_gain)
    n = times_np.size
    if n == 0:
        return []
    if n < 2:
        start = float(times_np[0])
        return [
            CurvePoint(d, 0.0, 0.0, start, start)
            for d in durations_list
        ]

    dt = np.diff(times_np)
    base_dt = dt[0]
    if np.any(np.abs(dt - base_dt) > 1e-6) or base_dt <= 0:
        return _compute_curve_numpy(times, cumulative_gain, durations, gaps=gaps)

    for D in durations_list:
        if D > 0 and abs(round(D / base_dt) * base_dt - D) > 1e-6:
            return _compute_curve_numpy(times, cumulative_gain, durations, gaps=gaps)

    start_time_min = float(times_np[0])
    end_time_max = float(times_np[-1])
    eps = 1e-9

    results: List[CurvePoint] = []
    for D in durations_list:
        if D <= 0 or start_time_min + D > end_time_max + eps:
            results.append(CurvePoint(D, 0.0, 0.0, start_time_min, start_time_min))
            continue

        stride = int(round(D / base_dt))
        if stride <= 0 or stride >= n:
            results.append(CurvePoint(D, 0.0, 0.0, start_time_min, start_time_min))
            continue

        deltas = U_at_sample[stride:] - U_at_sample[:-stride]
        if deltas.size == 0:
            results.append(CurvePoint(D, 0.0, 0.0, start_time_min, start_time_min))
            continue

        idx = int(np.argmax(deltas))
        gain = float(deltas[idx])
        if gain < 0.0:
            gain = 0.0
        start = float(times_np[idx])
        max_start = end_time_max - D
        if max_start < start_time_min:
            max_start = start_time_min
        if start < start_time_min:
            start = start_time_min
        if start > max_start:
            start = max_start
        end_offset = start + D
        if end_offset > end_time_max + 1e-9:
            end_offset = end_time_max
        rate = gain / D * 3600.0 if D > 0 else 0.0
        results.append(
            CurvePoint(
                duration_s=D,
                max_climb_m=gain,
                climb_rate_m_per_hr=rate,
                start_offset_s=start,
                end_offset_s=end_offset,
            )
        )

    return results


def compute_curve(
    times: List[float],
    cumulative_gain: List[float],
    durations: Iterable[int],
    gaps: Optional[List[Gap]] = None,
    engine: EngineMode = "auto",
) -> List[CurvePoint]:
    resolved = _resolve_engine(engine)
    if resolved == "auto":
        if HAVE_NUMBA:
            return _compute_curve_numba(times, cumulative_gain, durations, gaps=gaps)
        return _compute_curve_numpy(times, cumulative_gain, durations, gaps=gaps)
    if resolved == "numba":
        if HAVE_NUMBA:
            return _compute_curve_numba(times, cumulative_gain, durations, gaps=gaps)
        logging.warning("Numba engine requested but numba is unavailable; using numpy engine")
        return _compute_curve_numpy(times, cumulative_gain, durations, gaps=gaps)
    if resolved == "stride":
        return _compute_curve_stride(times, cumulative_gain, durations, gaps=gaps)
    return _compute_curve_numpy(times, cumulative_gain, durations, gaps=gaps)


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
            step_ratio = pct_step
            if x >= 6 * 3600:
                step_ratio = max(pct_step, 0.015)
            x *= 1.0 + step_ratio
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
    # Alt cumulative (physically-motivated)
    alt_cum: List[Optional[float]] = []
    alt_raw_t: List[float] = []
    alt_raw_v: List[float] = []
    alt_raw_idx: List[int] = []
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

        # Altitude (defer processing)
        alt = r.get("alt")
        if alt is not None:
            alt_raw_t.append(t_rel)
            alt_raw_v.append(float(alt))
            alt_raw_idx.append(len(times) - 1)
            alt_cum.append(0.0)  # placeholder
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

    # Compute effective altitude path and convert to cumulative ascent per spec
    if alt_raw_t:
        t_alt = np.asarray(alt_raw_t, dtype=np.float64)
        z_alt = np.asarray(alt_raw_v, dtype=np.float64)
        # Estimate median speed (m/s) and grade if available from session
        d_all = [r.get("dist") for r in session]
        t_all = [times[i] for i in range(len(session))]
        speed_med = 0.0
        if any(d is not None for d in d_all):
            d_arr = np.asarray([float(d) if d is not None else np.nan for d in d_all], dtype=np.float64)
            dt_all = np.diff(np.asarray(t_all, dtype=np.float64))
            dd_all = np.diff(d_arr)
            m = (~np.isnan(dd_all)) & (dt_all > 1e-6)
            if np.any(m):
                v = dd_all[m] / dt_all[m]
                if v.size:
                    speed_med = float(np.median(np.clip(v, 0.0, np.inf)))
        inc_all = [r.get("inc") for r in session]
        grade_med = 0.0
        if any(i is not None for i in inc_all):
            inc_arr = np.asarray([float(x) if x is not None else np.nan for x in inc_all], dtype=np.float64)
            val = inc_arr[~np.isnan(inc_arr)]
            if val.size:
                grade_med = float(np.median(np.clip(val / 100.0, -1.0, 1.0)))

        diag: Dict[str, Any] = {}
        # Additional diagnostics (speed/grade fractions) similar to _build_timeseries
        try:
            d_all = [r.get("dist") for r in session]
            t_all = [times[i] for i in range(len(session))]
            if any(d is not None for d in d_all):
                d_arr = np.asarray([float(d) if d is not None else np.nan for d in d_all], dtype=np.float64)
                t_arr = np.asarray(t_all, dtype=np.float64)
                dt = np.diff(t_arr)
                dd = np.diff(d_arr)
                mask = (~np.isnan(dd)) & (dt > 1e-6)
                if np.any(mask):
                    v = dd[mask] / dt[mask]
                    dtm = dt[mask]
                    low = dtm[v < 0.5].sum() if np.any(v < 0.5) else 0.0
                    diag["speed_low_time_pct"] = float(low / dtm.sum()) if dtm.sum() > 0 else 0.0
            inc_all = [r.get("inc") for r in session]
            if any(i is not None for i in inc_all):
                inc_arr = np.asarray([float(x) if x is not None else np.nan for x in inc_all], dtype=np.float64)
                steep = np.count_nonzero(inc_arr > 10.0)
                total = np.count_nonzero(~np.isnan(inc_arr))
                diag["grade_steep_sample_pct"] = float(steep / total) if total > 0 else 0.0
        except Exception:
            pass
        alt_eff = _effective_altitude_path(t_alt, z_alt, speed_med, grade_med, diag)
        indoor_hint = _infer_indoor_mode(session)
        alt_idle, moving_mask, _ = _apply_idle_detection(
            session,
            t_alt,
            alt_eff,
            diag,
            t0,
            indoor_hint=indoor_hint,
        )
        eps_gain = 0.02
        cum_series = _cum_ascent_from_alt(
            alt_idle,
            eps_gain,
            mode="uprun",
            diag=diag,
            moving_mask=moving_mask,
        )
        # Log diagnostics at DEBUG level
        try:
            logging.debug(
                "Altitude ascent diagnostics: n=%d net=%.1fm peak(start->max)=%.1fm range=%.1fm gross_noeps=%.1fm gross_eps=%.1fm eps_loss=%.1fm(%.0f%%) "
                "up_runs=%d pos_steps=%d eps_mode=%s neg_pre=%.1fm neg_post=%.1fm spikes=%d hampel=%d T=%.2fs speed_med=%.2f m/s grade_med=%.2f%% speed<0.5=%.0f%% grade>10=%.0f%%",
                diag.get("n_alt", 0),
                diag.get("net_gain_m", float('nan')),
                diag.get("peak_gain_from_start_m", float('nan')),
                diag.get("range_max_min_m", float('nan')),
                diag.get("gross_noeps_m", float('nan')),
                diag.get("gross_eps_m", float('nan')),
                diag.get("eps_loss_m", float('nan')),
                100.0 * diag.get("eps_loss_pct", 0.0),
                diag.get("up_runs", 0),
                diag.get("pos_steps", 0),
                str(diag.get("eps_mode", "uprun")),
                diag.get("neg_sum_pre_close_m", float('nan')),
                diag.get("neg_sum_post_m", float('nan')),
                diag.get("spike_count", 0),
                diag.get("hampel_count", 0),
                diag.get("closing_T_s", float('nan')),
                diag.get("speed_med_mps", float('nan')),
                100.0 * diag.get("grade_med_frac", float('nan')),
                100.0 * diag.get("speed_low_time_pct", 0.0),
                100.0 * diag.get("grade_steep_sample_pct", 0.0),
            )
        except Exception:
            pass
        for j, idx in enumerate(alt_raw_idx):
            alt_cum[idx] = cum_series[j]
    # Ensure monotone in-place
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
) -> Tuple[List[float], int, float, List[Tuple[float, float, float]]]:
    if not qc_spec:
        return list(values), 0, 0.0, []
    vals = list(values)
    n = len(vals)
    segments_removed = 0
    gain_removed = 0.0
    details: List[Tuple[float, float, float]] = []
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
                    details.append((times[i], times[j], delta))
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
    return vals, segments_removed, gain_removed, details


def _load_activity_series(
    fit_files: List[str],
    source: str,
    gain_eps: float,
    session_gap_sec: float,
    qc_enabled: bool,
    qc_spec_path: Optional[str],
    resample_1hz: bool,
    merge_eps_sec: float,
    overlap_policy: str,
    parse_workers: int,
    profiler: Optional[_StageProfiler] = None,
) -> ActivitySeries:
    logging.info("Reading %d FIT file(s)...", len(fit_files))
    records_by_file: List[List[Dict[str, Any]]] = []
    if parse_workers != 1 and len(fit_files) > 1:
        max_workers = parse_workers if parse_workers and parse_workers > 0 else min(len(fit_files), max(1, (os.cpu_count() or 1)))
        records_by_file = [None] * len(fit_files)  # type: ignore[assignment]
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {}
            for idx, path in enumerate(fit_files):
                logging.info("Parsing: %s", path)
                future = executor.submit(_parse_single_fit_records, path, idx)
                future_map[future] = idx
            for future in as_completed(future_map):
                i = future_map[future]
                records_by_file[i] = future.result()  # type: ignore[index]
        records_by_file = [rec if rec is not None else [] for rec in records_by_file]
    else:
        for idx, path in enumerate(fit_files):
            logging.info("Parsing: %s", path)
            records_by_file.append(_parse_single_fit_records(path, file_id=idx))
    if profiler:
        profiler.lap("parse")

    merged = _merge_records(records_by_file, merge_eps_sec=merge_eps_sec, overlap_policy=overlap_policy)
    logging.info("Merged samples: %d", len(merged))
    if profiler:
        profiler.lap("merge")

    if not merged:
        raise RuntimeError("No samples found after merging inputs.")

    inactivity_gaps: List[Tuple[float, float]]
    overall_sources_raw: Set[str]
    if source == "auto":
        times, values, used_sources, inactivity_gaps = _build_canonical_timeseries(
            merged,
            gain_eps=gain_eps,
            dwell_sec=5.0,
            gap_threshold=session_gap_sec,
        )
        overall_sources_raw = {SOURCE_NAME_MAP.get(u, u) for u in used_sources if u}
    else:
        times, values, label = _build_timeseries(merged, source=source, gain_eps=gain_eps)
        inactivity_gaps = []
        overall_sources_raw = {label}

    if not times:
        raise RuntimeError("No data available to compute curve after merging inputs.")

    qc_limits: Optional[Dict[float, float]] = None
    if qc_enabled:
        qc_limits = dict(QC_DEFAULT_SPEC)
        if qc_spec_path:
            try:
                qc_limits.update(_load_qc_spec(qc_spec_path))
            except Exception as exc:
                logging.warning("Failed to load QC spec %s: %s", qc_spec_path, exc)

    if qc_limits:
        values, qc_segments, qc_removed, qc_details = _apply_qc_censor(times, values, qc_limits)
        if qc_segments:
            logging.info(
                "QC censored %d spike segment(s); removed %.1f m of ascent.",
                qc_segments,
                qc_removed,
            )
            if logging.getLogger().isEnabledFor(logging.DEBUG):
                base_time = times[0] if times else 0.0
                preview = qc_details[:5]
                for start, end, delta in preview:
                    logging.debug(
                        "QC window removed %.2f m over %.1f s (start offset %.1f s)",
                        delta,
                        end - start,
                        start - base_time,
                    )
                if len(qc_details) > len(preview):
                    logging.debug(
                        "QC window log truncated (%d additional removal(s) suppressed)",
                        len(qc_details) - len(preview),
                    )
        _assert_monotonic_non_decreasing(values, name="cumulative gain after QC")

    if resample_1hz:
        times, values = _resample_to_1hz(times, values)
        _assert_monotonic_non_decreasing(values, name="cumulative gain after resample")

    session_gap_list = _find_gaps(times, session_gap_sec)
    if session_gap_list:
        preview = ", ".join(_fmt_duration_label(int(round(g.length))) for g in session_gap_list[:5])
        if len(session_gap_list) > 5:
            preview += ", ..."
        logging.info(
            "Detected %d gap(s) longer than %.0fs (skip applied): %s",
            len(session_gap_list),
            session_gap_sec,
            preview,
        )

    full_span_seconds = max(0.0, times[-1] - times[0]) if len(times) > 1 else 0.0

    if not overall_sources_raw:
        selected_raw = "mixed" if source == "auto" else source
    elif len(overall_sources_raw) == 1:
        selected_raw = next(iter(overall_sources_raw))
    else:
        selected_raw = "mixed"
    selected_label = _normalize_source_label(selected_raw)
    _assert_monotonic_non_decreasing(values, name="cumulative gain final")

    return ActivitySeries(
        times=times,
        values=values,
        selected_raw=selected_raw,
        selected_label=selected_label,
        inactivity_gaps=inactivity_gaps,
        session_gaps=session_gap_list,
        full_span_seconds=full_span_seconds,
        used_sources=overall_sources_raw,
    )
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
# Gain-centric helpers
# -----------------

def invert_duration_curve_to_gain_time(
    curve: List[CurvePoint],
    targets: Sequence[float],
    total_span_s: float,
    source_label: str,
) -> GainTimeCurve:
    if not curve:
        return GainTimeCurve(points=[], source_label=source_label, total_span_s=total_span_s)

    durations = np.asarray([float(cp.duration_s) for cp in curve], dtype=np.float64)
    gains = np.asarray([float(cp.max_climb_m) for cp in curve], dtype=np.float64)
    starts = np.asarray([float(cp.start_offset_s) for cp in curve], dtype=np.float64)
    ends = np.asarray([float(cp.end_offset_s) for cp in curve], dtype=np.float64)

    gains = np.maximum.accumulate(gains)

    cleaned_targets = sorted({float(g) for g in targets if math.isfinite(g) and g >= 0.0})
    points: List[GainTimePoint] = []

    if not cleaned_targets or cleaned_targets[0] > 0.0:
        points.append(GainTimePoint(0.0, 0.0, 0.0, 0.0, 0.0, None))

    for gain_target in cleaned_targets:
        if gain_target <= 0.0:
            points.append(GainTimePoint(0.0, 0.0, 0.0, 0.0, 0.0, None))
            continue
        idx = int(np.searchsorted(gains, gain_target, side="left"))
        if idx >= gains.size:
            points.append(
                GainTimePoint(
                    gain_m=gain_target,
                    min_time_s=math.inf,
                    avg_rate_m_per_hr=0.0,
                    start_offset_s=None,
                    end_offset_s=None,
                    note="unachievable",
                )
            )
            continue
        duration = float(durations[idx])
        start = float(starts[idx])
        end = float(ends[idx])
        achieved_gain = float(gains[idx])
        avg_rate = achieved_gain / duration * 3600.0 if duration > 0 else 0.0
        note: Optional[str] = None
        if achieved_gain - gain_target > 1e-6 and idx > 0:
            prev_duration = float(durations[idx - 1])
            if duration - prev_duration > 1.5:
                note = "bounded_by_grid"
        points.append(
            GainTimePoint(
                gain_m=gain_target,
                min_time_s=duration,
                avg_rate_m_per_hr=avg_rate,
                start_offset_s=start,
                end_offset_s=end,
                note=note,
            )
        )

    points_sorted = sorted(points, key=lambda p: p.gain_m)
    deduped: List[GainTimePoint] = []
    seen: Set[float] = set()
    for pt in points_sorted:
        key = round(pt.gain_m, 6)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(pt)
    return GainTimeCurve(points=deduped, source_label=source_label, total_span_s=total_span_s)


def build_gain_time_curve(curve: List[CurvePoint], total_span_s: float, source_label: str) -> GainTimeCurve:
    available_gains = {float(cp.max_climb_m) for cp in curve if cp.max_climb_m > 0}
    targets = sorted(available_gains)
    return invert_duration_curve_to_gain_time(curve, targets, total_span_s=total_span_s, source_label=source_label)


def convert_samples_to_gain_time_curve(
    durations: Sequence[float],
    gains: Sequence[float],
    source_label: str,
) -> GainTimeCurve:
    pts: List[GainTimePoint] = []
    for d, g in zip(durations, gains):
        d_sec = float(d)
        g_m = float(g)
        avg_rate = g_m / d_sec * 3600.0 if d_sec > 0 else 0.0
        pts.append(
            GainTimePoint(
                gain_m=g_m,
                min_time_s=d_sec,
                avg_rate_m_per_hr=avg_rate,
                start_offset_s=None,
                end_offset_s=None,
                note=None,
            )
        )
    total_span = float(durations[-1]) if durations else 0.0
    return GainTimeCurve(points=pts, source_label=source_label, total_span_s=total_span)


def _interpolate_gain_time(points: Sequence[GainTimePoint], gain_m: float) -> Optional[GainTimePoint]:
    if not points:
        return None
    xs = np.asarray([p.gain_m for p in points if math.isfinite(p.min_time_s)], dtype=np.float64)
    ys = np.asarray([p.min_time_s for p in points if math.isfinite(p.min_time_s)], dtype=np.float64)
    if xs.size == 0:
        return None
    if gain_m <= xs[0]:
        base = next(p for p in points if p.gain_m == xs[0])
        return GainTimePoint(gain_m=gain_m, min_time_s=base.min_time_s, avg_rate_m_per_hr=base.avg_rate_m_per_hr, start_offset_s=base.start_offset_s, end_offset_s=base.end_offset_s, note=base.note)
    if gain_m >= xs[-1]:
        base = next(p for p in reversed(points) if math.isfinite(p.min_time_s))
        return GainTimePoint(gain_m=gain_m, min_time_s=base.min_time_s, avg_rate_m_per_hr=base.avg_rate_m_per_hr, start_offset_s=base.start_offset_s, end_offset_s=base.end_offset_s, note=base.note)
    interp_time = float(np.interp(gain_m, xs, ys))
    avg_rate = gain_m / interp_time * 3600.0 if interp_time > 0 else 0.0
    return GainTimePoint(
        gain_m=gain_m,
        min_time_s=interp_time,
        avg_rate_m_per_hr=avg_rate,
        start_offset_s=None,
        end_offset_s=None,
        note=None,
    )

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


def _wr_envelope_cache_key(
    profile_name: str,
    profile_config: Dict[str, Any],
    wr_min_seconds: float,
    wr_anchors_path: Optional[str],
    cap_mode: str,
) -> Tuple[Any, ...]:
    anchors_sig: Optional[Tuple[Any, ...]]
    if wr_anchors_path:
        try:
            stat = os.stat(wr_anchors_path)
            anchors_sig = (wr_anchors_path, stat.st_mtime, stat.st_size)
        except OSError:
            anchors_sig = (wr_anchors_path, None, None)
    else:
        anchors_sig = None
    return (
        profile_name,
        _freeze_for_cache(profile_config),
        float(wr_min_seconds),
        cap_mode,
        anchors_sig,
    )


def _get_wr_envelope(
    profile_name: str,
    profile_config: Dict[str, Any],
    wr_min_seconds: float,
    wr_anchors_path: Optional[str],
    cap_mode: str,
):
    key = _wr_envelope_cache_key(
        profile_name,
        profile_config,
        wr_min_seconds,
        wr_anchors_path,
        cap_mode,
    )
    cached = _WR_ENVELOPE_CACHE.get(key)
    if cached is not None:
        H_WR_cached, env_cached = cached
        return H_WR_cached, copy.deepcopy(env_cached)

    H_WR, wr_env = _build_wr_envelope(
        copy.deepcopy(profile_config),
        wr_min_seconds,
        wr_anchors_path,
        cap_mode,
    )
    _WR_ENVELOPE_CACHE[key] = (H_WR, copy.deepcopy(wr_env))
    return H_WR, wr_env


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
    wr_rates_inst = _rate_curve(sample) * 3600.0
    with np.errstate(divide="ignore", invalid="ignore"):
        wr_rates_avg = np.divide(
            wr_values,
            sample,
            out=np.zeros_like(wr_values),
            where=sample > 0,
        ) * 3600.0

    return H_WR, {
        "durations": sample.tolist(),
        "climbs": wr_values.tolist(),
        "rates": wr_rates_avg.tolist(),
        "rates_inst": wr_rates_inst.tolist(),
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
    wr_curve_sample: Optional[Tuple[np.ndarray, np.ndarray]] = None,
) -> Tuple[List[float], List[float], List[Dict[str, Any]], List[Dict[str, Any]]]:
    import numpy as np
    W_arr = np.asarray(W_s, dtype=np.float64)
    H_user_arr = np.asarray(H_user, dtype=np.float64)
    if wr_curve_sample is not None and wr_curve_sample[0].size and wr_curve_sample[1].size:
        wr_durs, wr_climbs = wr_curve_sample
        H_wr_arr = np.interp(W_arr, wr_durs, wr_climbs, left=wr_climbs[0], right=wr_climbs[-1])
    else:
        H_wr_arr = np.array([H_WR_func(float(w)) for w in W_arr], dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        rat_arr = np.divide(H_user_arr, H_wr_arr, out=np.zeros_like(H_user_arr), where=H_wr_arr > 0)

    mask = (W_arr >= float(min_anchor_s)) & (H_wr_arr > 0)
    r_star = 0.0
    if np.any(mask):
        masked = np.where(mask, rat_arr, -np.inf)
        idx_best = int(np.argmax(masked))
        if masked[idx_best] > 0:
            r_star = float(min(masked[idx_best], 1.0))

    H_pers_arr = r_star * H_wr_arr
    H_wr = H_wr_arr.tolist()
    
    rows: List[Dict[str, Any]] = []
    for w in magic_ws:
        if wr_curve_sample is not None and wr_curve_sample[0].size:
            wr = float(np.interp(float(w), wr_curve_sample[0], wr_curve_sample[1], left=wr_curve_sample[1][0], right=wr_curve_sample[1][-1]))
        else:
            wr = H_WR_func(w)
        u = _interp_monotone(w, W_s, H_user)
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
    return H_wr, H_pers_arr.tolist(), rows, weakest


def _compute_wr_envelope_and_personal(
    curve: List[CurvePoint],
    wr_profile: str,
    wr_anchors_path: Optional[str],
    wr_min_seconds: float,
    wr_short_cap: str,
    magic: Optional[str],
    personal_min_seconds: float,
    goals_topk: int,
    score_output: Optional[str],
) -> WREnvelopeResult:
    wr_curve: Optional[Tuple[List[int], List[float]]] = None
    personal_curve: Optional[Tuple[List[int], List[float]]] = None
    goal_curve: Optional[Tuple[List[int], List[float]]] = None
    magic_rows: Optional[List[Dict[str, Any]]] = None
    H_WR_func: Optional[Any] = None
    wr_rates_env: Optional[List[float]] = None
    wr_sample_arrays: Optional[Tuple[np.ndarray, np.ndarray]] = None
    try:
        profile_config = _wr_profile_config(wr_profile)
        H_WR, wr_env = _get_wr_envelope(
            wr_profile,
            profile_config,
            wr_min_seconds=wr_min_seconds,
            wr_anchors_path=wr_anchors_path,
            cap_mode=wr_short_cap,
        )
        H_WR_func = H_WR
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
        wr_rates_env = wr_env.get("rates") if isinstance(wr_env.get("rates"), list) else None
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

        wr_indices = [idx for idx, w in enumerate(W_s) if w >= wr_min_seconds]
        W_wr = [W_s[idx] for idx in wr_indices]
        parsed_magic = [ _parse_duration_token(t) for t in (magic.split(',') if magic else []) ] if magic else []
        magic_ws = [int(w) for w in parsed_magic if w >= wr_min_seconds]
        if wr_curve and wr_curve[0] and wr_curve[1]:
            wr_sample_arrays = (
                np.asarray(wr_curve[0], dtype=np.float64),
                np.asarray(wr_curve[1], dtype=np.float64),
            )
        elif isinstance(wr_env.get("durations"), list) and isinstance(wr_env.get("climbs"), list):
            wr_sample_arrays = (
                np.asarray(wr_env.get("durations"), dtype=np.float64),
                np.asarray(wr_env.get("climbs"), dtype=np.float64),
            )

        H_wr_arr, H_pers_arr, rows, weakest = _scoring_tables(
            W_s,
            H_user,
            H_WR,
            magic_ws,
            min_anchor_s=personal_min_seconds,
            topk=goals_topk,
            wr_curve_sample=wr_sample_arrays,
        )
        personal_curve = (W_wr, [H_pers_arr[idx] for idx in wr_indices]) if W_wr else None
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
                writer = csv.DictWriter(f, fieldnames=["duration_s","user_gain_m","wr_gain_m","score_pct","personal_gain_m","goal_gain_m"])
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
    except Exception as exc:
        logging.warning(f"WR/scoring computation failed: {exc}")

    return WREnvelopeResult(
        wr_curve=wr_curve,
        wr_rates=wr_rates_env,
        personal_curve=personal_curve,
        goal_curve=goal_curve,
        magic_rows=magic_rows,
        H_WR_func=H_WR_func,
        wr_sample_arrays=wr_sample_arrays,
    )


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
    fast_plot: bool = True,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting. Install with: pip install matplotlib") from e

    _ensure_matplotlib_style(plt)

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

    _ensure_matplotlib_style(plt)
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

    if not dense and not fast_plot:
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
        label_toggle = 1
        for row in weak_rows:
            w = int(row['duration_s'])
            usr = row.get('user_gain_m')
            if not isinstance(usr, (int, float)):
                continue
            wrv = row.get('wr_gain_m') if isinstance(row.get('wr_gain_m'), (int, float)) else None
            goal_val = row.get('goal_gain_m') if isinstance(row.get('goal_gain_m'), (int, float)) else None
            if not fast_plot:
                target = None
                if show_wr and wrv is not None:
                    target = wrv
                elif show_personal and isinstance(row.get('personal_gain_m'), (int, float)):
                    target = row['personal_gain_m']
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
                if goal_val is not None:
                    ax2.plot([w], [goal_val], marker=">", color=GOAL_COLOR)
                    gpct = (goal_val / wrv * 100.0) if isinstance(wrv, (int, float)) and wrv > 0 else None
                    label = f"goal {goal_val:.0f} m" + (f" • {gpct:.0f}%" if gpct is not None else "")
                    goal_offset = 10
                    if not dense and _near_base_duration(w):
                        goal_offset += 6
                    ax2.annotate(
                        label,
                        (w, goal_val),
                        textcoords="offset points",
                        xytext=(-6, goal_offset),
                        ha="right",
                        fontsize=8,
                        color=GOAL_COLOR,
                    )
            else:
                ax2.plot([w], [usr], marker="o", color=USER_COLOR, alpha=0.85)
                pct = row.get('score_pct')
                text = f"{usr:.0f} m"
                if isinstance(pct, (int, float)):
                    text += f" • {pct:.0f}%"
                offset_y = -14 if label_toggle > 0 else -26
                ax2.annotate(
                    text,
                    (w, usr),
                    textcoords="offset points",
                    xytext=(0, offset_y),
                    ha="center",
                    fontsize=8,
                    color="black",
                )
                if goal_val is not None:
                    ax2.plot([w], [goal_val], marker=">", color=GOAL_COLOR, alpha=0.75)
            label_toggle *= -1
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right")

    if fast_plot:
        fig.savefig(out_png, dpi=150)
    else:
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
    fast_plot: bool = True,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting. Install with: pip install matplotlib") from e

    _ensure_matplotlib_style(plt)

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
    fig1, ax1 = plt.subplots(figsize=(12, 7))

    ax1.plot(durs, rates, marker="" if dense else "o", linewidth=1.8, color=USER_COLOR, label="Climb rate (m/h)")

    annotated_magic_rate_durs: Set[int] = set()
    if magic_rows and not fast_plot:
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

    if magic_rows and not fast_plot:
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
    rate_path = out_png_base.replace('.png', '_rate.png') if out_png_base.endswith('.png') else out_png_base + '_rate.png'
    if fast_plot:
        fig1.savefig(rate_path, dpi=150)
    else:
        fig1.tight_layout()
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
            if not isinstance(usr, (int, float)):
                continue
            wrv = row.get('wr_gain_m') if isinstance(row.get('wr_gain_m'), (int, float)) else None
            goal_val = row.get('goal_gain_m') if isinstance(row.get('goal_gain_m'), (int, float)) else None
            if not fast_plot:
                target = None
                if show_wr and wrv is not None:
                    target = wrv
                elif show_personal and isinstance(row.get('personal_gain_m'), (int, float)):
                    target = row['personal_gain_m']
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
                if w in weak_ws and w >= goal_min_seconds and goal_val is not None:
                    axc.plot([w], [goal_val], marker=">", color=GOAL_COLOR)
                    gpct = (goal_val / wrv * 100.0) if isinstance(wrv, (int, float)) and wrv > 0 else None
                    label = f"goal {goal_val:.0f} m" + (f" • {gpct:.0f}%" if gpct is not None else "")
                    goal_offset = 10
                    if not dense and _near_base_duration(w):
                        goal_offset += 6
                    axc.annotate(
                        label,
                        (w, goal_val),
                        textcoords="offset points",
                        xytext=(-6, goal_offset),
                        ha="right",
                        fontsize=8,
                        color=GOAL_COLOR,
                    )
            else:
                axc.plot([w], [usr], marker="o", color=USER_COLOR, alpha=0.85)
                pct = row.get('score_pct')
                text = f"{usr:.0f} m"
                if isinstance(pct, (int, float)):
                    text += f" • {pct:.0f}%"
                offset_y = -14 if label_toggle > 0 else -26
                axc.annotate(
                    text,
                    (w, usr),
                    textcoords="offset points",
                    xytext=(0, offset_y),
                    ha="center",
                    fontsize=8,
                    color="black",
                )
                if goal_val is not None:
                    axc.plot([w], [goal_val], marker=">", color=GOAL_COLOR, alpha=0.75)
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
    climb_path = out_png_base.replace('.png', '_climb.png') if out_png_base.endswith('.png') else out_png_base + '_climb.png'
    if fast_plot:
        fig2.savefig(climb_path, dpi=150)
    else:
        fig2.tight_layout()
        fig2.savefig(climb_path, dpi=200)
    plt.close(fig2)


def _plot_gain_time(
    gain_curve: GainTimeCurve,
    out_png: str,
    source_label: str,
    target_points: Sequence[GainTimePoint],
    wr_curve: Optional[GainTimeCurve] = None,
    personal_curve: Optional[GainTimeCurve] = None,
    magic_gains: Sequence[float] = (),
    show_wr: bool = True,
    show_personal: bool = True,
    unit: str = "m",
    fast_plot: bool = True,
    iso_rates: Sequence[float] = ISO_RATE_GUIDES,
    ylog_time: bool = False,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting. Install with: pip install matplotlib") from exc

    _ensure_matplotlib_style(plt)

    curve_pts = [p for p in gain_curve.points if math.isfinite(p.min_time_s) and p.gain_m >= 0]
    if not curve_pts:
        logging.warning("Gain curve empty; skipping plot generation.")
        return

    gains_m = np.asarray([p.gain_m for p in curve_pts], dtype=np.float64)
    gains_display = np.asarray([_convert_gain(p.gain_m, unit) for p in curve_pts], dtype=np.float64)
    times_minutes = np.asarray([p.min_time_s / 60.0 for p in curve_pts], dtype=np.float64)

    if gains_m.size == 0 or np.max(gains_m) <= 0:
        logging.warning("Gain curve lacks positive ascent; skipping plot generation.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))

    max_gain_m = float(np.max(gains_m))
    max_gain_disp = float(np.max(gains_display)) if gains_display.size else 1.0
    max_time_min = float(np.max(times_minutes)) if times_minutes.size else 1.0

    x_iso = np.linspace(0.0, max_gain_m, 200)
    for rate in iso_rates:
        if rate <= 0:
            continue
        y_iso_min = (x_iso / rate) * 60.0
        x_iso_disp = np.asarray([_convert_gain(val, unit) for val in x_iso], dtype=np.float64)
        ax.plot(x_iso_disp, y_iso_min, linestyle=(0, (2, 4)), linewidth=0.8, color="0.85")
        if x_iso_disp.size:
            ax.text(
                x_iso_disp[-1],
                y_iso_min[-1],
                f"{int(rate)} m/h",
                fontsize=8,
                color="0.6",
                ha="right",
                va="bottom",
            )

    ax.plot(gains_display, times_minutes, color=USER_COLOR, linewidth=1.8, label="Min time")

    if show_wr and wr_curve and wr_curve.points:
        wr_pts = [p for p in wr_curve.points if math.isfinite(p.min_time_s)]
        if wr_pts:
            x_wr = np.asarray([_convert_gain(p.gain_m, unit) for p in wr_pts], dtype=np.float64)
            y_wr = np.asarray([p.min_time_s / 60.0 for p in wr_pts], dtype=np.float64)
            ax.plot(x_wr, y_wr, linestyle=WR_STYLE, color="0.3", alpha=0.9, label="WR time")

    if show_personal and personal_curve and personal_curve.points:
        p_pts = [p for p in personal_curve.points if math.isfinite(p.min_time_s)]
        if p_pts:
            x_p = np.asarray([_convert_gain(p.gain_m, unit) for p in p_pts], dtype=np.float64)
            y_p = np.asarray([p.min_time_s / 60.0 for p in p_pts], dtype=np.float64)
            ax.plot(x_p, y_p, linestyle=PERSONAL_STYLE, color=USER_COLOR, alpha=0.6, linewidth=1.2, label="Personal time")

    for pt in target_points:
        if not math.isfinite(pt.min_time_s) or not math.isfinite(pt.gain_m):
            continue
        if pt.note == "unachievable":
            continue
        x = _convert_gain(pt.gain_m, unit)
        y = pt.min_time_s / 60.0
        ax.scatter([x], [y], color=USER_COLOR, marker="o", zorder=3)
        if not fast_plot:
            label = f"{_format_gain(pt.gain_m, unit)} • {_fmt_time_hms(pt.min_time_s)}"
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(0, -14),
                ha="center",
                fontsize=8,
                color="black",
            )

    for mg in magic_gains:
        interp = _interpolate_gain_time(gain_curve.points, mg)
        if interp is None or not math.isfinite(interp.min_time_s):
            continue
        x = _convert_gain(mg, unit)
        y = interp.min_time_s / 60.0
        ax.scatter([x], [y], color=GOAL_COLOR, marker="^", zorder=3)
        if not fast_plot:
            ax.annotate(
                _format_gain(mg, unit),
                (x, y),
                textcoords="offset points",
                xytext=(0, 12),
                ha="center",
                fontsize=8,
                color=GOAL_COLOR,
            )

    gain_label = "ft" if unit.lower() == "ft" else "m"
    ax.set_xlabel(f"Gain ({gain_label})")
    ax.set_ylabel("Time (min)")
    if ylog_time:
        positive = times_minutes[times_minutes > 0]
        ymin = max(positive.min() * 0.8 if positive.size else 1e-3, 1e-3)
        ax.set_yscale("log")
        ax.set_ylim(bottom=ymin)
    margin_x = max_gain_disp * 0.05 if max_gain_disp > 0 else 1.0
    margin_y = max_time_min * 0.05 if max_time_min > 0 else 1.0
    ax.set_xlim(0, max_gain_disp + margin_x)
    ax.set_ylim(0, max_time_min + margin_y)
    ax.set_title(f"Gain-Time Curve — {source_label}")
    ax.legend(loc="upper left")

    if fast_plot:
        fig.savefig(out_png, dpi=150)
    else:
        fig.tight_layout()
        fig.savefig(out_png, dpi=200)
    plt.close(fig)


def _plot_gain_time_split(
    gain_curve: GainTimeCurve,
    out_png_base: str,
    source_label: str,
    target_points: Sequence[GainTimePoint],
    wr_curve: Optional[GainTimeCurve] = None,
    personal_curve: Optional[GainTimeCurve] = None,
    magic_gains: Sequence[float] = (),
    show_wr: bool = True,
    show_personal: bool = True,
    unit: str = "m",
    fast_plot: bool = True,
    iso_rates: Sequence[float] = ISO_RATE_GUIDES,
    ylog_time: bool = False,
) -> None:
    time_path = out_png_base.replace('.png', '_time.png') if out_png_base.endswith('.png') else out_png_base + '_time.png'
    rate_path = out_png_base.replace('.png', '_rate.png') if out_png_base.endswith('.png') else out_png_base + '_rate.png'

    _plot_gain_time(
        gain_curve,
        time_path,
        source_label,
        target_points,
        wr_curve=wr_curve,
        personal_curve=personal_curve,
        magic_gains=magic_gains,
        show_wr=show_wr,
        show_personal=show_personal,
        unit=unit,
        fast_plot=fast_plot,
        iso_rates=iso_rates,
        ylog_time=ylog_time,
    )

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting. Install with: pip install matplotlib") from exc

    _ensure_matplotlib_style(plt)

    curve_pts = [
        p
        for p in gain_curve.points
        if math.isfinite(p.avg_rate_m_per_hr)
        and math.isfinite(p.min_time_s)
        and p.gain_m > 0
        and p.avg_rate_m_per_hr > 0
    ]
    if not curve_pts:
        logging.warning("Gain curve empty; skipping rate plot.")
        return

    x_vals = np.asarray([_convert_gain(p.gain_m, unit) for p in curve_pts], dtype=np.float64)
    rates = np.asarray([p.avg_rate_m_per_hr for p in curve_pts], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(x_vals, rates, color=USER_COLOR, linewidth=1.8, label="Average rate")

    if show_wr and wr_curve and wr_curve.points:
        wr_pts = [p for p in wr_curve.points if math.isfinite(p.avg_rate_m_per_hr)]
        if wr_pts:
            x_wr = np.asarray([_convert_gain(p.gain_m, unit) for p in wr_pts], dtype=np.float64)
            y_wr = np.asarray([p.avg_rate_m_per_hr for p in wr_pts], dtype=np.float64)
            ax.plot(x_wr, y_wr, linestyle=WR_STYLE, color="0.3", alpha=0.9, label="WR rate")

    if show_personal and personal_curve and personal_curve.points:
        p_pts = [p for p in personal_curve.points if math.isfinite(p.avg_rate_m_per_hr)]
        if p_pts:
            x_p = np.asarray([_convert_gain(p.gain_m, unit) for p in p_pts], dtype=np.float64)
            y_p = np.asarray([p.avg_rate_m_per_hr for p in p_pts], dtype=np.float64)
            ax.plot(x_p, y_p, linestyle=PERSONAL_STYLE, color=USER_COLOR, alpha=0.6, linewidth=1.2, label="Personal rate")

    for pt in target_points:
        if (
            not math.isfinite(pt.avg_rate_m_per_hr)
            or pt.note == "unachievable"
            or not math.isfinite(pt.gain_m)
            or pt.gain_m <= 0
            or pt.avg_rate_m_per_hr <= 0
        ):
            continue
        x = _convert_gain(pt.gain_m, unit)
        y = pt.avg_rate_m_per_hr
        ax.scatter([x], [y], color=USER_COLOR, marker="o", zorder=3)
        if not fast_plot:
            ax.annotate(
                f"{_format_gain(pt.gain_m, unit)} • {y:.0f} m/h",
                (x, y),
                textcoords="offset points",
                xytext=(0, -14),
                ha="center",
                fontsize=8,
                color="black",
            )

    gain_label = "ft" if unit.lower() == "ft" else "m"
    ax.set_xlabel(f"Gain ({gain_label})")
    ax.set_ylabel("Average rate (m/h)")
    ax.set_title(f"Gain-Rate Curve — {source_label}")
    ax.legend(loc="upper right")

    if fast_plot:
        fig.savefig(rate_path, dpi=150)
    else:
        fig.tight_layout()
        fig.savefig(rate_path, dpi=200)
    plt.close(fig)

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
    parse_workers: int = 0,
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
    engine: str = "auto",
    profile: bool = False,
    fast_plot: bool = True,
) -> int:
    _setup_logging(verbose, log_file=log_file)

    selected = "mixed"
    curve: List[CurvePoint] = []
    engine_mode = _resolve_engine(engine)
    profiler = _StageProfiler(profile)
    H_WR_func: Optional[Any] = None
    try:
        series = _load_activity_series(
            fit_files,
            source=source,
            gain_eps=gain_eps,
            session_gap_sec=session_gap_sec,
            qc_enabled=qc_enabled,
            qc_spec_path=qc_spec_path,
            resample_1hz=resample_1hz,
            merge_eps_sec=merge_eps_sec,
            overlap_policy=overlap_policy,
            parse_workers=parse_workers,
            profiler=profiler,
        )
        times = series.times
        values = series.values
        inactivity_gaps = series.inactivity_gaps
        session_gap_list = series.session_gaps
        full_span_seconds = series.full_span_seconds
        selected_raw = series.selected_raw
        selected = series.selected_label
    except Exception as e:
        logging.error(str(e))
        return 2

    if values:
        total_ascent = float(values[-1]) - float(values[0])
    else:
        total_ascent = 0.0
    if total_ascent <= 0.01:
        logging.error(
            "No positive ascent recorded after preprocessing/QC (source=%s); nothing to compute.",
            selected,
        )
        return 3

    duration_grid: List[int] = []

    if all_windows:
        step_for_all = step_s if step_s > 0 else 1
        if not resample_1hz:
            if _is_uniform_1hz(times):
                offset = times[0]
                if abs(offset) > 1e-9:
                    times = [float(t - offset) for t in times]
            else:
                times, values = _resample_to_1hz(times, values)
        curve = all_windows_curve(times, values, step=step_for_all)
        duration_grid = [cp.duration_s for cp in curve]
        profiler.lap("curve")
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

        curve = compute_curve(
            times,
            values,
            durs,
            gaps=session_gap_list,
            engine=engine_mode,
        )
        duration_grid = list(durs)
        profiler.lap("curve")

    if not curve:
        logging.error("Unable to compute curve for requested durations.")
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

    wr_result = _compute_wr_envelope_and_personal(
        curve,
        wr_profile=wr_profile,
        wr_anchors_path=wr_anchors_path,
        wr_min_seconds=wr_min_seconds,
        wr_short_cap=wr_short_cap,
        magic=magic,
        personal_min_seconds=personal_min_seconds,
        goals_topk=goals_topk,
        score_output=score_output,
    )
    wr_curve = wr_result.wr_curve
    personal_curve = wr_result.personal_curve
    goal_curve = wr_result.goal_curve
    magic_rows = wr_result.magic_rows
    H_WR_func = wr_result.H_WR_func
    wr_rates_env = wr_result.wr_rates

    profiler.lap("wr/scoring")

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
                seg_curve = compute_curve(
                    segment["times"],
                    segment["values"],
                    valid_durs,
                    gaps=None,
                    engine=engine_mode,
                )
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
        "wr_climb_m",
        "wr_rate_m_per_hr",
    ]

    wr_gain_arr: Optional[np.ndarray] = None
    if wr_curve and wr_curve[0] and wr_curve[1]:
        wr_durs_arr = np.asarray(wr_curve[0], dtype=np.float64)
        wr_climbs_arr = np.asarray(wr_curve[1], dtype=np.float64)
        cp_durations = np.asarray([cp.duration_s for cp in curve], dtype=np.float64)
        wr_gain_arr = np.interp(cp_durations, wr_durs_arr, wr_climbs_arr, left=wr_climbs_arr[0], right=wr_climbs_arr[-1])
    elif H_WR_func is not None:
        cp_durations = np.asarray([cp.duration_s for cp in curve], dtype=np.float64)
        try:
            wr_gain_arr = np.array([float(H_WR_func(float(d))) for d in cp_durations], dtype=np.float64)
        except Exception:
            wr_gain_arr = None

    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        rows_out: List[List[Any]] = []
        for idx, cp in enumerate(curve):
            wr_gain_val: Optional[float] = None
            if wr_gain_arr is not None and idx < wr_gain_arr.size:
                val = float(wr_gain_arr[idx])
                if math.isfinite(val):
                    wr_gain_val = val
            wr_rate_val: Optional[float] = None
            if wr_gain_val is not None and cp.duration_s > 0:
                wr_rate_val = wr_gain_val / cp.duration_s * 3600.0
            rows_out.append([
                cp.duration_s,
                round(cp.max_climb_m, 3),
                round(cp.climb_rate_m_per_hr, 3),
                round(cp.start_offset_s, 3),
                round(cp.end_offset_s, 3),
                selected_raw,
                round(wr_gain_val, 3) if isinstance(wr_gain_val, float) else None,
                round(wr_rate_val, 3) if isinstance(wr_rate_val, float) else None,
            ])
        writer.writerows(rows_out)

    logging.info("Wrote: %s", output)
    profiler.lap("csv")

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
                    fast_plot=fast_plot,
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
                    fast_plot=fast_plot,
                )
                logging.info("Wrote plot: %s", png_path)
        except Exception as e:
            logging.error(f"Plotting failed: {e}")

    profiler.lap("plot")

    return 0


def _run_gain_time(
    fit_files: List[str],
    output: str,
    gains: List[str],
    source: str,
    verbose: bool,
    png: Optional[str] = None,
    no_plot: bool = False,
    all_windows: bool = True,
    exhaustive: bool = False,
    step_s: int = 1,
    max_duration_s: Optional[int] = None,
    log_file: Optional[str] = None,
    merge_eps_sec: float = 0.5,
    overlap_policy: str = "file:last",
    resample_1hz: bool = False,
    parse_workers: int = 0,
    gain_eps: float = 0.5,
    session_gap_sec: float = 600.0,
    qc_enabled: bool = True,
    qc_spec_path: Optional[str] = None,
    wr_profile: str = "overall",
    wr_anchors_path: Optional[str] = None,
    wr_min_seconds: float = 30.0,
    wr_short_cap: str = "standard",
    plot_wr: bool = False,
    plot_personal: bool = True,
    split_plots: bool = True,
    fast_plot: bool = True,
    magic_gains: Optional[str] = DEFAULT_MAGIC_GAINS,
    goals_topk: int = 3,
    profile: bool = False,
    gain_units: str = "m",
    iso_rates: Sequence[float] = ISO_RATE_GUIDES,
    engine: str = "auto",
    concave_envelope: bool = True,
    ylog_time: bool = False,
) -> int:
    _setup_logging(verbose, log_file=log_file)

    profiler = _StageProfiler(profile)
    engine_mode = _resolve_engine(engine)

    try:
        series = _load_activity_series(
            fit_files,
            source=source,
            gain_eps=gain_eps,
            session_gap_sec=session_gap_sec,
            qc_enabled=qc_enabled,
            qc_spec_path=qc_spec_path,
            resample_1hz=resample_1hz,
            merge_eps_sec=merge_eps_sec,
            overlap_policy=overlap_policy,
            parse_workers=parse_workers,
            profiler=profiler,
        )
    except Exception as exc:
        logging.error(str(exc))
        return 2

    times = series.times
    values = series.values
    inactivity_gaps = series.inactivity_gaps
    session_gap_list = series.session_gaps
    full_span_seconds = series.full_span_seconds
    selected_raw = series.selected_raw
    selected_label = series.selected_label

    curve: List[CurvePoint] = []
    duration_grid: List[int] = []

    if not times:
        logging.error("No data available to compute gain-time report after preprocessing.")
        return 2

    if values:
        total_ascent = float(values[-1]) - float(values[0])
    else:
        total_ascent = 0.0
    if total_ascent <= 0.01:
        logging.error(
            "No positive ascent recorded after preprocessing/QC (source=%s); gain-time report is empty.",
            selected_label,
        )
        return 3

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
            base_durations = [int(d) for d in DEFAULT_DURATIONS]
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

        curve = compute_curve(
            times,
            values,
            durs,
            gaps=session_gap_list,
            engine=engine_mode,
        )
        duration_grid = list(durs)
        profiler.lap("curve")

    if not curve:
        logging.error("Unable to compute duration curve for inversion.")
        return 2

    curve_original = copy.deepcopy(curve)
    apply_concave = concave_envelope and not exhaustive
    _enforce_curve_shape(curve, inactivity_gaps=inactivity_gaps, apply_concave=apply_concave)
    _diagnose_curve_monotonicity(curve, epsilon=1e-6, inactivity_gaps=inactivity_gaps)

    gain_curve = build_gain_time_curve(curve, total_span_s=full_span_seconds, source_label=selected_label)

    wr_result = _compute_wr_envelope_and_personal(
        curve,
        wr_profile=wr_profile,
        wr_anchors_path=wr_anchors_path,
        wr_min_seconds=wr_min_seconds,
        wr_short_cap=wr_short_cap,
        magic=None,
        personal_min_seconds=wr_min_seconds,
        goals_topk=goals_topk,
        score_output=None,
    )

    wr_gain_curve: Optional[GainTimeCurve] = None
    if wr_result.wr_curve:
        wr_gain_curve = convert_samples_to_gain_time_curve(
            wr_result.wr_curve[0],
            wr_result.wr_curve[1],
            source_label="WR",
        )
    personal_gain_curve: Optional[GainTimeCurve] = None
    if wr_result.personal_curve:
        personal_gain_curve = convert_samples_to_gain_time_curve(
            wr_result.personal_curve[0],
            wr_result.personal_curve[1],
            source_label="Personal",
        )

    default_targets = list(DEFAULT_GAIN_TARGETS)
    gain_unit_norm = gain_units.lower()
    target_gains = _parse_gain_list(gains, default_unit=gain_unit_norm) if gains else []
    if gains and not target_gains:
        logging.warning(
            "No valid gain tokens parsed from --gains/--gains-from inputs; "
            "falling back to defaults (%s).",
            ", ".join(f"{val:.0f}m" for val in default_targets),
        )
    if not target_gains:
        target_gains = default_targets
    target_gains = [float(val) for val in sorted({round(v, 6) for v in target_gains}) if val > 0]
    if not target_gains:
        target_gains = list(DEFAULT_GAIN_TARGETS)

    targets_curve = invert_duration_curve_to_gain_time(
        curve,
        target_gains,
        total_span_s=full_span_seconds,
        source_label=selected_label,
    )
    approx_map: Dict[float, GainTimePoint] = {round(pt.gain_m, 6): pt for pt in targets_curve.points}
    prefer_numba = HAVE_NUMBA and engine_mode in ("numba", "auto")
    direct_points: List[GainTimePoint]
    if prefer_numba and target_gains:
        try:
            direct_points = min_time_for_gains_numba(times, values, target_gains)
        except Exception as exc:  # pragma: no cover - numba runtime fallback
            logging.debug("Numba min-time sweep failed (%s); falling back to numpy path.", exc)
            direct_points = min_time_for_gains(times, values, target_gains)
    else:
        direct_points = min_time_for_gains(times, values, target_gains)

    if len(direct_points) != len(target_gains):
        logging.warning("Failed to refine gain targets; falling back to envelope inversion only.")
        merged_points = [approx_map.get(round(val, 6)) for val in target_gains]
        merged_points = [pt for pt in merged_points if pt is not None]
    else:
        merged_points = []
        for gain_value, direct_point in zip(target_gains, direct_points):
            key = round(gain_value, 6)
            approx_pt = approx_map.get(key)
            note = direct_point.note
            if note != "unachievable" and approx_pt is not None:
                if approx_pt.note == "bounded_by_grid":
                    note = "bounded_by_grid"
                elif approx_pt.min_time_s - direct_point.min_time_s > 1.5:
                    note = "bounded_by_grid"
            merged_points.append(
                GainTimePoint(
                    gain_m=direct_point.gain_m,
                    min_time_s=direct_point.min_time_s,
                    avg_rate_m_per_hr=direct_point.avg_rate_m_per_hr,
                    start_offset_s=direct_point.start_offset_s,
                    end_offset_s=direct_point.end_offset_s,
                    note=note,
                )
            )

    if not merged_points:
        logging.warning("No valid gain targets; check input parameters.")

    magic_gain_values: List[float] = []
    magic_tokens = magic_gains.split(",") if magic_gains else []
    for token in magic_tokens:
        val = _parse_gain_token(token, default_unit=gain_unit_norm)
        if val is None or val <= 0:
            continue
        magic_gain_values.append(val)
    magic_gain_values = sorted({round(val, 6) for val in magic_gain_values})

    gain_label = "ft" if gain_unit_norm == "ft" else "m"
    summary_lines = [f"Gain Time Report (source: {selected_raw})"]
    total_gain = values[-1] if values else 0.0
    total_gain_display = _format_gain(total_gain, gain_unit_norm)
    for pt in merged_points:
        gain_str = _format_gain(pt.gain_m, gain_unit_norm)
        if not math.isfinite(pt.min_time_s) or pt.min_time_s <= 0:
            summary_lines.append(f"- {gain_str}: unachievable (max gain {total_gain_display})")
            continue
        rate_str = _fmt_rate_mph(pt.avg_rate_m_per_hr)
        window_start = _fmt_time_hms(pt.start_offset_s or 0.0)
        window_end = _fmt_time_hms(pt.end_offset_s or 0.0)
        note_suffix = f" [{pt.note}]" if pt.note else ""
        summary_lines.append(
            f"- {gain_str}: {_fmt_time_hms(pt.min_time_s)} ({rate_str}) window {window_start}–{window_end}{note_suffix}"
        )

    for line in summary_lines:
        print(line)

    fieldnames = [
        "gain_m",
        "min_time_s",
        "avg_rate_m_per_hr",
        "start_offset_s",
        "end_offset_s",
        "source",
        "note",
    ]
    with open(output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for pt in merged_points:
            writer.writerow([
                round(pt.gain_m, 3),
                round(pt.min_time_s, 3) if math.isfinite(pt.min_time_s) else None,
                round(pt.avg_rate_m_per_hr, 3) if math.isfinite(pt.avg_rate_m_per_hr) else None,
                round(pt.start_offset_s, 3) if isinstance(pt.start_offset_s, (int, float)) and math.isfinite(pt.start_offset_s) else None,
                round(pt.end_offset_s, 3) if isinstance(pt.end_offset_s, (int, float)) and math.isfinite(pt.end_offset_s) else None,
                selected_raw,
                pt.note or "",
            ])

    logging.info("Wrote: %s", output)
    profiler.lap("csv")

    if not no_plot:
        png_path = png
        if png_path is None:
            if output.lower().endswith(".csv"):
                png_path = output[:-4] + ".png"
            else:
                png_path = output + ".png"
        gain_magic = [float(val) for val in magic_gain_values]
        if split_plots:
            _plot_gain_time_split(
                gain_curve,
                png_path,
                selected_label,
                merged_points,
                wr_curve=wr_gain_curve,
                personal_curve=personal_gain_curve,
                magic_gains=gain_magic,
                show_wr=plot_wr,
                show_personal=plot_personal,
                unit=gain_unit_norm,
                fast_plot=fast_plot,
                iso_rates=iso_rates,
                ylog_time=ylog_time,
            )
            logging.info("Wrote plots: %s", png_path)
        else:
            _plot_gain_time(
                gain_curve,
                png_path,
                selected_label,
                merged_points,
                wr_curve=wr_gain_curve,
                personal_curve=personal_gain_curve,
                magic_gains=gain_magic,
                show_wr=plot_wr,
                show_personal=plot_personal,
                unit=gain_unit_norm,
                fast_plot=fast_plot,
                iso_rates=iso_rates,
                ylog_time=ylog_time,
            )
            logging.info("Wrote plot: %s", png_path)

    profiler.lap("plot")
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
        parse_workers: int = typer.Option(0, "--parse-workers", help="Number of worker threads for FIT parsing (0=auto, 1=serial)"),
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
        engine: str = typer.Option("auto", "--engine", help="Curve engine: auto|numpy|numba|stride"),
        profile: bool = typer.Option(False, "--profile/--no-profile", help="Log stage timings for performance profiling"),
        fast_plot: bool = typer.Option(True, "--fast-plot/--no-fast-plot", help="Skip heavy plot annotations for faster rendering"),
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
            parse_workers=parse_workers,
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
            engine=engine,
            profile=profile,
            fast_plot=fast_plot,
        )
        if code != 0:
            raise typer.Exit(code)

    @app.command()
    def time(
        fit_files: List[str] = typer.Argument(..., help="One or more input .fit files"),
        gains: List[str] = typer.Option([], "--gains", "-g", help="Gain targets (accepts suffix m or ft)."),
        gains_from: Optional[str] = typer.Option(
            None,
            "--gains-from",
            help="Path to a file with gain targets (one per line, supports m|ft suffix).",
        ),
        output: str = typer.Option(
            "gain_time.csv",
            "--output",
            "-o",
            help="Output CSV path",
        ),
        source: str = typer.Option(
            "auto",
            "--source",
            help="Data source: auto|runn|altitude",
        ),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
        png: Optional[str] = typer.Option(None, "--png", help="Optional output PNG path (defaults next to CSV)"),
        no_plot: bool = typer.Option(False, "--no-plot", help="Disable PNG generation"),
        all_windows: bool = typer.Option(False, "--all/--no-all", help="Compute per-second duration curve for inversion", show_default=True),
        exhaustive: bool = typer.Option(False, "--exhaustive", help="Evaluate a dense duration grid before inversion"),
        step_s: int = typer.Option(1, "--step", help="Step size in seconds for exhaustive durations"),
        max_duration_s: Optional[int] = typer.Option(None, "--max-duration", help="Maximum duration in seconds for exhaustive evaluation"),
        log_file: Optional[str] = typer.Option(None, "--log-file", help="Optional log file path"),
        merge_eps_sec: float = typer.Option(0.5, "--merge-eps-sec", help="Coalesce tolerance (seconds) for overlapping timestamps"),
        overlap_policy: str = typer.Option("file:last", "--overlap-policy", help="Overlap precedence: file:first|file:last"),
        resample_1hz: bool = typer.Option(False, "--resample-1hz", help="Resample cumulative series to 1 Hz"),
        parse_workers: int = typer.Option(0, "--parse-workers", help="Number of worker threads for FIT parsing (0=auto, 1=serial)"),
        gain_eps: float = typer.Option(0.5, "--gain-eps", help="Altitude hysteresis (m) for ascent from altitude"),
        session_gap_sec: float = typer.Option(600.0, "--session-gap-sec", help="Gap (s) considered inactivity when summarising"),
        qc: bool = typer.Option(True, "--qc/--no-qc", help="Enable QC filtering for ascent spikes"),
        qc_spec: Optional[str] = typer.Option(None, "--qc-spec", help="Path to JSON overriding QC windows"),
        wr_profile: str = typer.Option("overall", "--wr-profile", help="WR envelope profile"),
        wr_anchors: Optional[str] = typer.Option(None, "--wr-anchors", help="Path to JSON defining WR anchors"),
        wr_min_seconds: float = typer.Option(30.0, "--wr-min-seconds", help="Minimum duration (s) for WR overlay"),
        wr_short_cap: str = typer.Option("standard", "--wr-short-cap", help="WR short-duration cap: conservative|standard|aggressive"),
        plot_wr: bool = typer.Option(False, "--plot-wr/--no-plot-wr", help="Show WR envelope on plots"),
        plot_personal: bool = typer.Option(True, "--plot-personal/--no-plot-personal", help="Show personal scaled WR curve"),
        split_plots: bool = typer.Option(True, "--split-plots/--no-split-plots", help="Write separate time/rate PNGs"),
        fast_plot: bool = typer.Option(True, "--fast-plot/--no-fast-plot", help="Skip heavy plot annotations for faster rendering"),
        magic_gain_tokens: Optional[str] = typer.Option(DEFAULT_MAGIC_GAINS, "--magic-gains", help="Comma-separated gains to annotate (accepts m/ft suffix)"),
        goals_topk: int = typer.Option(3, "--goals-topk", help="Reserved for future goal annotations"),
        profile: bool = typer.Option(False, "--profile/--no-profile", help="Log stage timings"),
        gain_units: str = typer.Option("m", "--gain-units", help="Display/parse units: m|ft"),
        engine: str = typer.Option("auto", "--engine", help="Curve engine: auto|numpy|numba|stride"),
        concave_envelope: bool = typer.Option(True, "--concave-envelope/--no-concave-envelope", help="Apply concave envelope smoothing before inversion"),
        ylog_time: bool = typer.Option(False, "--ylog-time/--no-ylog-time", help="Use log scale for time axis"),
    ) -> None:
        _require_dependency(FitFile, "fitparse", "pip install fitparse")
        units_norm = gain_units.lower()
        if units_norm not in ("m", "ft"):
            raise typer.BadParameter("gain-units must be 'm' or 'ft'")
        gains_list: List[str] = _expand_gain_tokens(gains)
        if gains_from:
            try:
                gains_from_tokens = _load_gain_tokens_from_file(gains_from)
            except OSError as exc:
                raise typer.BadParameter(f"Unable to read gains-from file '{gains_from}': {exc}") from exc
            gains_list.extend(gains_from_tokens)

        filtered_files: List[str] = []
        for token in fit_files:
            if _token_is_likely_gain(token, default_unit=units_norm):
                gains_list.append(token)
            else:
                filtered_files.append(token)
        if not filtered_files:
            raise typer.BadParameter("Provide at least one FIT file after gain tokens.")
        code = _run_gain_time(
            filtered_files,
            output,
            gains_list,
            source,
            verbose,
            png=png,
            no_plot=no_plot,
            all_windows=all_windows,
            exhaustive=exhaustive,
            step_s=step_s,
            max_duration_s=max_duration_s,
            log_file=log_file,
            merge_eps_sec=merge_eps_sec,
            overlap_policy=overlap_policy,
            resample_1hz=resample_1hz,
            parse_workers=parse_workers,
            gain_eps=gain_eps,
            session_gap_sec=session_gap_sec,
            qc_enabled=qc,
            qc_spec_path=qc_spec,
            wr_profile=wr_profile,
            wr_anchors_path=wr_anchors,
            wr_min_seconds=wr_min_seconds,
            wr_short_cap=wr_short_cap,
            plot_wr=plot_wr,
            plot_personal=plot_personal,
            split_plots=split_plots,
            fast_plot=fast_plot,
            magic_gains=magic_gain_tokens,
            goals_topk=goals_topk,
            profile=profile,
            gain_units=units_norm,
            engine=engine,
            concave_envelope=concave_envelope,
            ylog_time=ylog_time,
        )
        if code != 0:
            raise typer.Exit(code)

    @app.command(name="export-series")
    def export_series(
        fit_files: List[str] = typer.Argument(..., help="One or more input .fit files"),
        output: str = typer.Option(
            "timeseries.csv",
            "--output",
            "-o",
            help="Output CSV path for the canonical cumulative series",
        ),
        source: str = typer.Option(
            "auto",
            "--source",
            help="Data source: auto|runn|altitude",
        ),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
        resample_1hz: bool = typer.Option(False, "--resample-1hz", help="Resample cumulative series to 1 Hz"),
        parse_workers: int = typer.Option(0, "--parse-workers", help="Number of worker threads for FIT parsing (0=auto, 1=serial)"),
        gain_eps: float = typer.Option(0.5, "--gain-eps", help="Altitude hysteresis (m) for ascent from altitude"),
        session_gap_sec: float = typer.Option(600.0, "--session-gap-sec", help="Gap (s) considered inactivity when summarising"),
        qc: bool = typer.Option(True, "--qc/--no-qc", help="Censor implausible ascent spikes before computing the series"),
        qc_spec: Optional[str] = typer.Option(None, "--qc-spec", help="Path to JSON overriding QC windows {window_s: max_gain_m}"),
        merge_eps_sec: float = typer.Option(0.5, "--merge-eps-sec", help="Coalesce tolerance (seconds) for overlapping timestamps"),
        overlap_policy: str = typer.Option("file:last", "--overlap-policy", help="Overlap precedence for total gain: file:first|file:last"),
        log_file: Optional[str] = typer.Option(None, "--log-file", help="Optional log file path"),
        profile: bool = typer.Option(False, "--profile/--no-profile", help="Log stage timings for performance profiling"),
    ) -> None:
        _require_dependency(FitFile, "fitparse", "pip install fitparse")
        code = _export_series_command(
            fit_files,
            output,
            source,
            verbose,
            resample_1hz,
            parse_workers,
            gain_eps,
            session_gap_sec,
            qc,
            qc_spec,
            merge_eps_sec,
            overlap_policy,
            log_file,
            profile,
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
