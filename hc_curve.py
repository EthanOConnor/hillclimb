import sys
import csv
import logging
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict, Any

try:
    import typer
except Exception:  # pragma: no cover
    typer = None  # type: ignore

try:
    from fitparse import FitFile
except Exception:  # pragma: no cover
    FitFile = None  # type: ignore


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


# -----------------
# FIT parsing utils
# -----------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


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


def _parse_single_fit_records(fit_path: str) -> List[Dict[str, Any]]:
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

        out.append({"t": t, "alt": alt_val, "tg": tg_val, "inc": inc_val, "dist": dist_val, "dist_prio": dist_prio})
    return out


def _merge_records(records_by_file: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    # Flatten and sort by time
    allrecs = [r for lst in records_by_file for r in lst]
    if not allrecs:
        return []
    allrecs.sort(key=lambda r: r["t"]) 
    merged: List[Dict[str, Any]] = []
    eps = 0.5  # seconds; coalesce near-duplicate timestamps
    for rec in allrecs:
        if not merged:
            merged.append(rec)
            continue
        last = merged[-1]
        if rec["t"] <= last["t"] + eps:
            # Merge preference: keep tg if present; prefer non-None fields
            chosen = dict(last)
            if rec.get("tg") is not None:
                chosen["tg"] = rec["tg"]
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


def _build_timeseries(
    merged: List[Dict[str, Any]],
    source: str = "auto",
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
        for r in merged:
            t_rel = r["t"] - t0
            tg = r.get("tg")
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
                    # Likely reset; roll base forward
                    base += last_tg
                last_tg = tg
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
                if d > 0:
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
        i = 0
        j = 0
        best_gain = 0.0
        best_window = (0.0, 0.0)
        while i < n and times[i] + D <= times[-1] + 1e-9:
            t_end = times[i] + D
            # advance j so that times[j] >= t_end
            while j < n and times[j] < t_end:
                j += 1
            # Interpolate G at t_end
            G_end = _interp_cum_gain(t_end, times, cumulative_gain, j)
            G_start = cumulative_gain[i]
            gain = G_end - G_start
            if gain > best_gain:
                best_gain = gain
                best_window = (times[i], t_end)
            i += 1

        rate_m_per_hr = best_gain / D * 3600.0 if D > 0 else 0.0
        results.append(
            CurvePoint(
                duration_s=D,
                max_climb_m=best_gain,
                climb_rate_m_per_hr=rate_m_per_hr,
                start_offset_s=best_window[0],
                end_offset_s=best_window[1],
            )
        )
    return results


# -----------------
# CLI
# -----------------

def _plot_curve(curve: List[CurvePoint], out_png: str, source_label: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        raise RuntimeError("matplotlib is required for plotting. Install with: pip install matplotlib") from e

    durs = [cp.duration_s for cp in curve]
    rates = [cp.climb_rate_m_per_hr for cp in curve]
    climbs = [cp.max_climb_m for cp in curve]

    def fmt_dur(s: int) -> str:
        if s < 120:
            return f"{int(s)}s"
        if s < 3600:
            return f"{int(round(s/60))}m"
        return f"{round(s/3600,1)}h"

    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(durs, rates, marker="o", color="C0", label="Climb rate (m/h)")
    for x, y in zip(durs, rates):
        ax.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, color="C0")

    ax.set_xscale("log")
    ax.set_xticks(durs)
    ax.set_xticklabels([fmt_dur(s) for s in durs])
    ax.set_xlabel("Duration")
    ax.set_ylabel("Climb rate (m/h)", color="C0")
    ax.grid(True, which="both", axis="both", linestyle=":", alpha=0.6)

    ax2 = ax.twinx()
    ax2.plot(durs, climbs, marker="s", linestyle="--", color="C1", label="Max climb (m)")
    for x, y in zip(durs, climbs):
        ax2.annotate(f"{y:.0f}", (x, y), textcoords="offset points", xytext=(0, -12), ha="center", fontsize=9, color="C1")
    ax2.set_ylabel("Max climb (m)", color="C1")

    ax.set_title("Critical Hill Climb Rate Curve\nSource: " + source_label)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
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
) -> int:
    _setup_logging(verbose)

    try:
        logging.info("Reading %d FIT file(s)...", len(fit_files))
        records_by_file = []
        for fp in fit_files:
            logging.info("Parsing: %s", fp)
            records_by_file.append(_parse_single_fit_records(fp))
        merged = _merge_records(records_by_file)
        logging.info("Merged samples: %d", len(merged))
        times, G, selected = _build_timeseries(merged, source=source)
        logging.info("Computing curve for %d durations...", len(durations))
        curve = compute_curve(times, G, durations)
    except Exception as e:
        logging.error(str(e))
        return 2

    # Enforce non-increasing rate with increasing duration (monotonic fix)
    best_rate = float("inf")
    for cp in curve:
        if cp.climb_rate_m_per_hr > best_rate:
            cp.climb_rate_m_per_hr = best_rate
        best_rate = cp.climb_rate_m_per_hr

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
                    "source": selected,
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
            _plot_curve(curve, png_path, selected)
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
    ) -> None:
        """Compute the critical hill climb rate curve and save to CSV."""
        code = _run(fit_files, output, durations, source, verbose, png=png, no_plot=no_plot)
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
