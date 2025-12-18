from __future__ import annotations

import logging
import math
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from hc_curve import (
    CurvePoint,
    GainTimeCurve,
    GainTimePoint,
    _convert_gain,
    _format_gain,
    _fmt_duration_label,
    _interpolate_gain_time,
)


USER_COLOR = "C0"
GOAL_COLOR = "tab:green"
PERSONAL_STYLE = (0, (3, 3, 1.5, 3))
WR_STYLE = (0, (6, 4))

ISO_RATE_GUIDES = (800.0, 1000.0, 1200.0, 1500.0, 2000.0, 2500.0)

_MATPLOTLIB_STYLE_READY = False


def _ensure_matplotlib_style(plt) -> None:
    global _MATPLOTLIB_STYLE_READY
    if not _MATPLOTLIB_STYLE_READY:
        try:
            plt.style.use("ggplot")
        except Exception:
            pass
        _MATPLOTLIB_STYLE_READY = True


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
