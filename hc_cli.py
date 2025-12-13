from __future__ import annotations

# CLI orchestration for hillclimb. Most heavy lifting lives in hc_curve (core)
# and hc_plotting (matplotlib).

from hc_curve import *  # type: ignore
from hc_curve import (
    _StageProfiler,
    _check_fitparse_version,
    _clear_parsed_fit_cache,
    _compute_wr_envelope_and_personal,
    _diagnose_curve_monotonicity,
    _enforce_curve_shape,
    _expand_gain_tokens,
    _export_series_command,
    _fmt_duration_label,
    _fmt_rate_mph,
    _fmt_time_hms,
    _format_gain,
    _is_uniform_1hz,
    _load_activity_series,
    _load_gain_tokens_from_file,
    _parse_gain_list,
    _parse_gain_token,
    _require_dependency,
    _resample_to_1hz,
    _resolve_engine,
    _setup_logging,
    _split_sessions_from_gaps,
    _token_is_likely_gain,
)
import json
import os

from hc_plotting import (
    ISO_RATE_GUIDES,
    _plot_curve,
    _plot_gain_time,
    _plot_gain_time_split,
    _plot_split,
)

try:
    import typer
except Exception:  # pragma: no cover
    typer = None  # type: ignore


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
    smooth_sec: float = 0.0,
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
    json_sidecar: bool = False,
    clear_cache: bool = False,
) -> int:
    _setup_logging(verbose, log_file=log_file)

    if clear_cache:
        _clear_parsed_fit_cache()

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
            smooth_sec=smooth_sec,
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

    if json_sidecar:
        json_path = output[:-4] + ".json" if output.lower().endswith(".csv") else output + ".json"
        try:
            meta = {
                "command": "curve",
                "inputs": list(fit_files),
                "output_csv": output,
                "selected_source": selected,
                "selected_raw": selected_raw,
                "used_sources": sorted(series.used_sources),
                "n_samples": len(times),
                "full_span_s": full_span_seconds,
                "total_gain_m": total_ascent,
                "engine": engine_mode,
                "params": {
                    "durations_s": duration_grid,
                    "exhaustive": exhaustive,
                    "all_windows": all_windows,
                    "step_s": step_s,
                    "max_duration_s": max_duration_s,
                    "resample_1hz": resample_1hz,
                    "parse_workers": parse_workers,
                    "gain_eps": gain_eps,
                    "smooth_sec": smooth_sec,
                    "session_gap_sec": session_gap_sec,
                    "qc_enabled": qc_enabled,
                    "qc_spec_path": qc_spec_path,
                    "merge_eps_sec": merge_eps_sec,
                    "overlap_policy": overlap_policy,
                    "concave_envelope": concave_envelope,
                    "wr_profile": wr_profile,
                    "wr_min_seconds": wr_min_seconds,
                    "wr_short_cap": wr_short_cap,
                    "plot_wr": plot_wr,
                    "plot_personal": plot_personal,
                    "split_plots": split_plots,
                    "ylog_rate": ylog_rate,
                    "ylog_climb": ylog_climb,
                    "goal_min_seconds": goal_min_seconds,
                    "personal_min_seconds": personal_min_seconds,
                    "fast_plot": fast_plot,
                },
            }
            data = {
                "points": [
                    {
                        "duration_s": cp.duration_s,
                        "max_climb_m": cp.max_climb_m,
                        "climb_rate_m_per_hr": cp.climb_rate_m_per_hr,
                        "start_offset_s": cp.start_offset_s,
                        "end_offset_s": cp.end_offset_s,
                    }
                    for cp in curve
                ],
                "wr_curve": wr_curve,
                "wr_rates": wr_rates_env,
                "personal_curve": personal_curve,
                "goal_curve": goal_curve,
                "magic_rows": magic_rows,
                "envelope_curve": envelope_curve,
                "session_curves": session_curves,
            }
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump({"meta": meta, "curves": data}, jf, indent=2)
            logging.info("Wrote JSON: %s", json_path)
        except Exception as exc:
            logging.warning("Failed to write JSON sidecar: %s", exc)

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
    smooth_sec: float = 0.0,
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
    json_sidecar: bool = False,
    clear_cache: bool = False,
) -> int:
    _setup_logging(verbose, log_file=log_file)

    if clear_cache:
        _clear_parsed_fit_cache()

    profiler = _StageProfiler(profile)
    engine_mode = _resolve_engine(engine)

    try:
        series = _load_activity_series(
            fit_files,
            source=source,
            gain_eps=gain_eps,
            smooth_sec=smooth_sec,
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

    if json_sidecar:
        json_path = output[:-4] + ".json" if output.lower().endswith(".csv") else output + ".json"
        try:
            meta = {
                "command": "time",
                "inputs": list(fit_files),
                "output_csv": output,
                "selected_source": selected_label,
                "selected_raw": selected_raw,
                "used_sources": sorted(series.used_sources),
                "n_samples": len(times),
                "full_span_s": full_span_seconds,
                "total_gain_m": total_ascent,
                "engine": engine_mode,
                "params": {
                    "all_windows": all_windows,
                    "exhaustive": exhaustive,
                    "step_s": step_s,
                    "max_duration_s": max_duration_s,
                    "resample_1hz": resample_1hz,
                    "parse_workers": parse_workers,
                    "gain_eps": gain_eps,
                    "smooth_sec": smooth_sec,
                    "session_gap_sec": session_gap_sec,
                    "qc_enabled": qc_enabled,
                    "qc_spec_path": qc_spec_path,
                    "merge_eps_sec": merge_eps_sec,
                    "overlap_policy": overlap_policy,
                    "wr_profile": wr_profile,
                    "wr_min_seconds": wr_min_seconds,
                    "wr_short_cap": wr_short_cap,
                    "plot_wr": plot_wr,
                    "plot_personal": plot_personal,
                    "split_plots": split_plots,
                    "fast_plot": fast_plot,
                    "gain_units": gain_units,
                    "magic_gains": magic_gains,
                    "goals_topk": goals_topk,
                    "iso_rates": list(iso_rates),
                    "concave_envelope": concave_envelope,
                    "ylog_time": ylog_time,
                },
            }
            data = {
                "gain_curve": [pt.__dict__ for pt in gain_curve.points],
                "targets": [pt.__dict__ for pt in target_points],
                "wr_curve": [pt.__dict__ for pt in wr_gain_curve.points] if wr_gain_curve else None,
                "personal_curve": [pt.__dict__ for pt in personal_gain_curve.points] if personal_gain_curve else None,
            }
            with open(json_path, "w", encoding="utf-8") as jf:
                json.dump({"meta": meta, "gain_time": data}, jf, indent=2)
            logging.info("Wrote JSON: %s", json_path)
        except Exception as exc:
            logging.warning("Failed to write JSON sidecar: %s", exc)

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
        smooth_sec: float = typer.Option(0.0, "--smooth", help="Additional altitude smoothing window (seconds), applied after the effective altitude path", show_default=True),
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
        json_sidecar: bool = typer.Option(False, "--json/--no-json", help="Write JSON report next to the CSV"),
        clear_cache: bool = typer.Option(False, "--clear-cache", help="Clear parsed FIT cache before running"),
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
            smooth_sec=smooth_sec,
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
            json_sidecar=json_sidecar,
            clear_cache=clear_cache,
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
        smooth_sec: float = typer.Option(0.0, "--smooth", help="Additional altitude smoothing window (seconds), applied after the effective altitude path", show_default=True),
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
        json_sidecar: bool = typer.Option(False, "--json/--no-json", help="Write JSON report next to the CSV"),
        clear_cache: bool = typer.Option(False, "--clear-cache", help="Clear parsed FIT cache before running"),
    ) -> None:
        _require_dependency(FitFile, "fitparse", "pip install python-fitparse")
        _check_fitparse_version()
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
            smooth_sec=smooth_sec,
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
            json_sidecar=json_sidecar,
            clear_cache=clear_cache,
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
        smooth_sec: float = typer.Option(0.0, "--smooth", help="Additional altitude smoothing window (seconds), applied after the effective altitude path", show_default=True),
        session_gap_sec: float = typer.Option(600.0, "--session-gap-sec", help="Gap (s) considered inactivity when summarising"),
        qc: bool = typer.Option(True, "--qc/--no-qc", help="Censor implausible ascent spikes before computing the series"),
        qc_spec: Optional[str] = typer.Option(None, "--qc-spec", help="Path to JSON overriding QC windows {window_s: max_gain_m}"),
        merge_eps_sec: float = typer.Option(0.5, "--merge-eps-sec", help="Coalesce tolerance (seconds) for overlapping timestamps"),
        overlap_policy: str = typer.Option("file:last", "--overlap-policy", help="Overlap precedence for total gain: file:first|file:last"),
        log_file: Optional[str] = typer.Option(None, "--log-file", help="Optional log file path"),
        profile: bool = typer.Option(False, "--profile/--no-profile", help="Log stage timings for performance profiling"),
        json_sidecar: bool = typer.Option(False, "--json/--no-json", help="Write JSON report next to the CSV"),
        clear_cache: bool = typer.Option(False, "--clear-cache", help="Clear parsed FIT cache before running"),
    ) -> None:
        _require_dependency(FitFile, "fitparse", "pip install python-fitparse")
        _check_fitparse_version()
        if clear_cache:
            _clear_parsed_fit_cache()
        code = _export_series_command(
            fit_files,
            output,
            source,
            verbose,
            resample_1hz,
            parse_workers,
            gain_eps,
            smooth_sec,
            session_gap_sec,
            qc,
            qc_spec,
            merge_eps_sec,
            overlap_policy,
            log_file,
            profile,
            json_sidecar,
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
        _require_dependency(FitFile, "fitparse", "pip install python-fitparse")
        _check_fitparse_version()
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
