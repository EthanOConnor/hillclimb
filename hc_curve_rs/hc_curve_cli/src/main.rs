use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::fs::File;
use std::io::{self, Write};
use std::panic;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use chrono::Utc;
use clap::{ArgAction, Parser, Subcommand, ValueEnum, ValueHint};
use fitparser::de::from_bytes;
use fitparser::profile::MesgNum;
use fitparser::Value as FitValue;
use hc_curve::{
    compute_curves, parse_duration_token, parse_records, Curves, Engine, Params, Source,
};
use ordered_float::OrderedFloat;
use plotters::prelude::*;
use plotters::style::{FontDesc, FontFamily, FontStyle};
use plotters_backend::{
    text_anchor, BackendColor, BackendCoord, BackendStyle, BackendTextStyle, DrawingBackend,
    DrawingErrorKind,
};
use serde_json::Value as JsonValue;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(author, version, about = "Hillclimb curve computation CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Compute the hillclimb curve CSV/plots for one or more FIT/GPX files
    Curve(CurveArgs),
    /// Inspect FIT files for available record keys and ascent candidates
    Diagnose(DiagnoseArgs),
}

#[derive(Parser, Debug)]
struct CurveArgs {
    /// FIT/GPX files to ingest
    #[arg(required = true, value_hint = ValueHint::FilePath)]
    inputs: Vec<PathBuf>,

    /// Output CSV path (`-` for stdout)
    #[arg(short, long, default_value = "curve.csv", value_hint = ValueHint::FilePath)]
    output: PathBuf,

    /// Output PNG figure path (defaults next to CSV)
    #[arg(long, value_hint = ValueHint::FilePath)]
    png: Option<PathBuf>,

    /// Output SVG figure path
    #[arg(long, value_hint = ValueHint::FilePath)]
    svg: Option<PathBuf>,

    /// Disable plot generation
    #[arg(long, action = ArgAction::SetTrue)]
    no_plot: bool,

    /// Explicit durations to evaluate (comma separated seconds)
    #[arg(long)]
    durations: Option<String>,

    /// Evaluate exhaustive multi-resolution grid
    #[arg(long, action = ArgAction::SetTrue)]
    exhaustive: bool,

    /// Compute per-second windows via all-windows sweep
    #[arg(long, action = ArgAction::SetTrue)]
    all: bool,

    /// Step size in seconds for exhaustive/`--all`
    #[arg(long, default_value_t = 1)]
    step: u64,

    /// Maximum duration in seconds
    #[arg(long)]
    max_duration: Option<u64>,

    /// Data source preference
    #[arg(long, value_enum, default_value_t = SourceOpt::Auto)]
    source: SourceOpt,

    /// Gap threshold (seconds) for session segmentation
    #[arg(long, default_value_t = 600.0)]
    session_gap: f64,

    /// Timestamp merge tolerance (seconds)
    #[arg(long, default_value_t = 0.5)]
    merge_eps: f64,

    /// Overlap precedence policy for total gain
    #[arg(long, default_value = "file:last")]
    overlap_policy: String,

    /// Keep native sampling (skip 1 Hz resample)
    #[arg(long, action = ArgAction::SetTrue)]
    raw_sampling: bool,

    /// Disable QC censoring
    #[arg(long, action = ArgAction::SetTrue)]
    no_qc: bool,

    /// Optional QC override JSON path
    #[arg(long, value_hint = ValueHint::FilePath)]
    qc_spec: Option<PathBuf>,

    /// Altitude hysteresis (meters)
    #[arg(long, default_value_t = 0.5)]
    gain_eps: f64,

    /// Altitude smoothing window (seconds)
    #[arg(long, default_value_t = 0.0)]
    smooth: f64,

    /// Optional WR anchors JSON
    #[arg(long, value_hint = ValueHint::FilePath)]
    wr_anchors: Option<PathBuf>,

    /// WR profile identifier
    #[arg(long, default_value = "overall")]
    wr_profile: String,

    /// Minimum duration for WR envelope (seconds)
    #[arg(long, default_value_t = 30.0)]
    wr_min_seconds: f64,

    /// Short-duration WR cap profile (conservative|standard|aggressive)
    #[arg(long, default_value = "standard")]
    wr_short_cap: String,

    /// Magic durations for scoring (comma separated tokens like 60s,0.5h)
    #[arg(long)]
    magic: Option<String>,

    /// Number of weakest scores to highlight
    #[arg(long, default_value_t = 3)]
    goals_topk: usize,

    /// Hide goals below this duration (seconds)
    #[arg(long, default_value_t = 120.0)]
    goal_min_seconds: f64,

    /// Anchor personal scaling above this duration (seconds)
    #[arg(long, default_value_t = 60.0)]
    personal_min_seconds: f64,

    /// Plot WR overlay
    #[arg(long, action = ArgAction::SetTrue)]
    plot_wr: bool,

    /// Disable personal/goal overlays on plots
    #[arg(long, action = ArgAction::SetTrue)]
    no_personal: bool,

    /// Curve engine (auto|numpy|numba|stride)
    #[arg(long, value_enum, default_value_t = EngineOpt::Auto)]
    engine: EngineOpt,

    /// Optional CSV to write scoring table
    #[arg(long, value_hint = ValueHint::FilePath)]
    score_output: Option<PathBuf>,

    /// Verbose logging
    #[arg(long, action = ArgAction::SetTrue)]
    verbose: bool,

    /// Disable concave envelope smoothing
    #[arg(long, action = ArgAction::SetTrue)]
    no_concave_envelope: bool,
}

#[derive(Parser, Debug)]
struct DiagnoseArgs {
    /// FIT files to inspect
    #[arg(required = true, value_hint = ValueHint::FilePath)]
    inputs: Vec<PathBuf>,

    /// Output report path
    #[arg(short, long, default_value = "fit_diagnostics.txt", value_hint = ValueHint::FilePath)]
    output: PathBuf,

    /// Verbose logging
    #[arg(long, action = ArgAction::SetTrue)]
    verbose: bool,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum SourceOpt {
    Auto,
    Runn,
    Altitude,
}

impl From<SourceOpt> for Source {
    fn from(value: SourceOpt) -> Self {
        match value {
            SourceOpt::Auto => Source::Auto,
            SourceOpt::Runn => Source::Runn,
            SourceOpt::Altitude => Source::Altitude,
        }
    }
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum EngineOpt {
    Auto,
    Numpy,
    Numba,
    Stride,
}

impl From<EngineOpt> for Engine {
    fn from(value: EngineOpt) -> Self {
        match value {
            EngineOpt::Auto => Engine::Auto,
            EngineOpt::Numpy => Engine::NumpyStyle,
            EngineOpt::Numba => Engine::NumbaStyle,
            EngineOpt::Stride => Engine::Stride,
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let default_level = match &cli.command {
        Command::Curve(args) => {
            if args.verbose {
                "debug"
            } else {
                "info"
            }
        }
        Command::Diagnose(args) => {
            if args.verbose {
                "debug"
            } else {
                "info"
            }
        }
    };
    let filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_level));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_writer(io::stderr)
        .try_init();

    match cli.command {
        Command::Curve(args) => handle_curve(args),
        Command::Diagnose(args) => handle_diagnose(args),
    }
}

fn handle_curve(args: CurveArgs) -> Result<()> {
    if args.inputs.is_empty() {
        return Err(anyhow!("no input files supplied"));
    }

    let mut params = Params::default();
    params.gain_eps_m = args.gain_eps;
    params.smooth_sec = args.smooth;
    params.session_gap_sec = args.session_gap;
    params.merge_eps_sec = args.merge_eps;
    params.overlap_policy = args.overlap_policy.clone();
    params.resample_1hz = !args.raw_sampling;
    params.qc_enabled = !args.no_qc;
    params.max_duration_s = args.max_duration;
    params.step_s = args.step.max(1);
    params.exhaustive = args.exhaustive;
    params.all_windows = args.all;
    params.source = args.source.into();
    params.wr_anchors_path = args.wr_anchors.clone();
    params.wr_profile = args.wr_profile.clone();
    params.wr_min_seconds = args.wr_min_seconds;
    params.wr_short_cap = args.wr_short_cap.clone();
    params.goals_topk = args.goals_topk;
    params.goal_min_seconds = args.goal_min_seconds;
    params.personal_min_seconds = args.personal_min_seconds;
    params.concave_envelope = !args.no_concave_envelope;
    params.engine = args.engine.into();

    if let Some(spec_path) = args.qc_spec.as_ref() {
        params.qc_spec = Some(load_qc_spec(spec_path)?);
    }

    if let Some(durations_str) = args.durations.as_ref() {
        let durations = parse_duration_list(durations_str)?;
        if durations.is_empty() {
            return Err(anyhow!("--durations list was empty"));
        }
        params.durations = durations;
        params.exhaustive = false;
    }

    if let Some(magic_str) = args.magic.as_ref() {
        let tokens = parse_magic_tokens(magic_str);
        if !tokens.is_empty() {
            params.magic = Some(tokens.clone());
            let mut parsed = Vec::new();
            for token in tokens {
                if let Some(value) = parse_duration_token(&token) {
                    if value > 0.0 {
                        parsed.push(value.round() as u64);
                    }
                }
            }
            if !parsed.is_empty() {
                params.magic_durations = parsed;
            }
        }
    }

    let mut records = Vec::new();
    for (file_id, path) in args.inputs.iter().enumerate() {
        let data = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
        let hint = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("fit");
        let parsed = parse_records(&data, file_id, hint)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        records.push(parsed);
    }

    let curves = compute_curves(records, &params)?;
    info!(
        "Curve computed: {} durations, source {}",
        curves.points.len(),
        curves.selected_source
    );

    if let Some(rows) = curves.magic_rows.as_ref() {
        log_magic_summary(rows, args.goals_topk);
    }

    if let Some(score_path) = args.score_output.as_ref() {
        write_score_output(&curves, score_path)?;
        info!("Wrote scoring table: {}", score_path.display());
    }

    if args.output.as_os_str() == "-" {
        write_curve_stdout(&curves)?;
    } else {
        write_curve_csv(&curves, &args.output)?;
        info!("Wrote curve CSV: {}", args.output.display());
    }

    if !args.no_plot {
        if let Some(path) = args.png.as_ref() {
            if let Err(err) = render_chart_guard(
                &curves,
                path,
                ChartKind::Png,
                args.plot_wr,
                !args.no_personal,
            ) {
                warn!("Skipping PNG render ({}): {}", path.display(), err);
            } else {
                info!("Wrote plot: {}", path.display());
            }
        } else if args.output.as_os_str() != "-" {
            let mut png_path = args.output.clone();
            png_path.set_extension("png");
            if let Err(err) = render_chart_guard(
                &curves,
                &png_path,
                ChartKind::Png,
                args.plot_wr,
                !args.no_personal,
            ) {
                warn!("Skipping PNG render ({}): {}", png_path.display(), err);
            } else {
                info!("Wrote plot: {}", png_path.display());
            }
        }

        if let Some(path) = args.svg.as_ref() {
            if let Err(err) = render_chart_guard(
                &curves,
                path,
                ChartKind::Svg,
                args.plot_wr,
                !args.no_personal,
            ) {
                warn!("Skipping SVG render ({}): {}", path.display(), err);
            } else {
                info!("Wrote plot: {}", path.display());
            }
        }
    }

    Ok(())
}

fn handle_diagnose(args: DiagnoseArgs) -> Result<()> {
    let mut report = String::new();

    for path in &args.inputs {
        let data = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
        let fit =
            from_bytes(&data).with_context(|| format!("failed to parse {}", path.display()))?;

        let mut stats: BTreeMap<String, KeyStats> = BTreeMap::new();
        let mut candidate: HashMap<String, usize> = HashMap::new();
        let mut records = 0usize;
        let mut first_time: Option<f64> = None;
        let mut last_time: Option<f64> = None;

        for message in fit {
            if message.kind() != MesgNum::Record {
                continue;
            }
            records += 1;
            for field in message.fields() {
                let name = field.name().to_string();
                let entry = stats.entry(name.clone()).or_default();
                entry.count += 1;
                if let Some(val) = fit_value_to_f64(field.value()) {
                    entry.numeric += 1;
                    entry.min = Some(entry.min.map_or(val, |m| m.min(val)));
                    entry.max = Some(entry.max.map_or(val, |m| m.max(val)));
                }

                let lower = name.to_ascii_lowercase();
                if (lower.contains("gain") || lower.contains("ascent") || lower.contains("climb"))
                    && (lower.contains("total")
                        || lower.contains("cum")
                        || lower.contains("cumulative"))
                {
                    *candidate.entry(name.clone()).or_insert(0) += 1;
                }

                if name == "timestamp" {
                    if let FitValue::Timestamp(ts) = field.value() {
                        let utc = ts.with_timezone(&Utc);
                        let seconds = utc.timestamp() as f64
                            + (utc.timestamp_subsec_micros() as f64 / 1_000_000.0);
                        first_time.get_or_insert(seconds);
                        last_time = Some(seconds);
                    }
                }
            }
        }

        report.push_str(&format!("FILE: {}\n", path.display()));
        report.push_str(&format!("  records: {}\n", records));
        if let (Some(start), Some(end)) = (first_time, last_time) {
            report.push_str(&format!("  timespan_s: {:.1}\n", end - start));
        }

        if !candidate.is_empty() {
            report.push_str("  candidate_total_gain_keys:\n");
            let mut pairs: Vec<_> = candidate.into_iter().collect();
            pairs.sort_by(|a, b| b.1.cmp(&a.1));
            for (name, count) in pairs.into_iter().take(5) {
                let summary = stats.get(&name).cloned().unwrap_or_default();
                report.push_str(&format!(
                    "    - {} (count={}) first={} last={}\n",
                    name,
                    count,
                    summary.min.map_or("n/a".into(), |v| format!("{:.3}", v)),
                    summary.max.map_or("n/a".into(), |v| format!("{:.3}", v))
                ));
            }
        }

        if !stats.is_empty() {
            report.push_str("  keys:\n");
            let mut entries: Vec<_> = stats.into_iter().collect();
            entries.sort_by(|a, b| b.1.count.cmp(&a.1.count));
            for (name, summary) in entries.into_iter().take(25) {
                report.push_str(&format!(
                    "    - {}: count={}, numeric={}, min={}, max={}\n",
                    name,
                    summary.count,
                    summary.numeric,
                    summary.min.map_or("n/a".into(), |v| format!("{:.3}", v)),
                    summary.max.map_or("n/a".into(), |v| format!("{:.3}", v))
                ));
            }
        }

        report.push('\n');
    }

    fs::write(&args.output, report)
        .with_context(|| format!("failed to write {}", args.output.display()))?;
    info!("Diagnostic report written: {}", args.output.display());
    Ok(())
}

fn parse_duration_list(input: &str) -> Result<Vec<u64>> {
    let mut out = Vec::new();
    for token in input.split(',') {
        let trimmed = token.trim();
        if trimmed.is_empty() {
            continue;
        }
        let value: u64 = trimmed
            .parse()
            .with_context(|| format!("invalid duration '{}': expected integer seconds", trimmed))?;
        if value == 0 {
            return Err(anyhow!("duration tokens must be > 0"));
        }
        out.push(value);
    }
    Ok(out)
}

fn parse_magic_tokens(input: &str) -> Vec<String> {
    input
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn load_qc_spec(path: &Path) -> Result<HashMap<OrderedFloat<f64>, f64>> {
    let text = fs::read_to_string(path)
        .with_context(|| format!("failed to read QC spec {}", path.display()))?;
    let json: JsonValue = serde_json::from_str(&text)
        .with_context(|| format!("{} is not valid JSON", path.display()))?;

    let mut spec = HashMap::new();
    let object = json
        .as_object()
        .ok_or_else(|| anyhow!("QC spec must be a JSON object"))?;
    for (key, value) in object {
        let window: f64 = key
            .parse()
            .with_context(|| format!("invalid QC window '{}': not a float", key))?;
        let limit = value
            .as_f64()
            .ok_or_else(|| anyhow!("invalid QC limit for '{}': expected number", key))?;
        if window > 0.0 && limit > 0.0 {
            spec.insert(OrderedFloat(window), limit);
        }
    }
    if spec.is_empty() {
        warn!("QC spec {} had no usable entries", path.display());
    }
    Ok(spec)
}

fn write_curve_stdout(curves: &Curves) -> Result<()> {
    let stdout = io::stdout();
    let handle = stdout.lock();
    let mut writer = csv::Writer::from_writer(handle);
    write_curve_rows(curves, &mut writer)
}

fn write_curve_csv(curves: &Curves, path: &Path) -> Result<()> {
    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut writer = csv::Writer::from_writer(file);
    write_curve_rows(curves, &mut writer)
}

fn write_curve_rows<W: Write>(curves: &Curves, writer: &mut csv::Writer<W>) -> Result<()> {
    writer.write_record([
        "duration_s",
        "max_climb_m",
        "climb_rate_m_per_hr",
        "start_offset_s",
        "end_offset_s",
        "source",
        "wr_climb_m",
        "wr_rate_m_per_hr",
        "personal_gain_m",
        "goal_gain_m",
    ])?;

    let wr_curve = curves.wr_curve.as_ref();
    let personal_curve = curves.personal_curve.as_ref();
    let goal_curve = curves.goal_curve.as_ref();

    for point in &curves.points {
        let wr_gain = value_from_curve(point.duration_s, wr_curve);
        let personal_gain = value_from_curve(point.duration_s, personal_curve);
        let goal_gain = value_from_curve(point.duration_s, goal_curve);
        let wr_rate = wr_gain.map(|g| {
            if point.duration_s > 0 {
                g * 3600.0 / point.duration_s as f64
            } else {
                0.0
            }
        });

        writer.write_record([
            point.duration_s.to_string(),
            format!("{:.3}", point.max_climb_m),
            format!("{:.3}", point.climb_rate_m_per_hr),
            format!("{:.3}", point.start_offset_s),
            format!("{:.3}", point.end_offset_s),
            curves.selected_source.clone(),
            wr_gain
                .map(|v| format!("{:.3}", v))
                .unwrap_or_else(|| "".into()),
            wr_rate
                .map(|v| format!("{:.3}", v))
                .unwrap_or_else(|| "".into()),
            personal_gain
                .map(|v| format!("{:.3}", v))
                .unwrap_or_else(|| "".into()),
            goal_gain
                .map(|v| format!("{:.3}", v))
                .unwrap_or_else(|| "".into()),
        ])?;
    }

    writer.flush()?;
    Ok(())
}

fn value_from_curve(duration: u64, curve: Option<&(Vec<u64>, Vec<f64>)>) -> Option<f64> {
    let (durations, gains) = curve?;
    if durations.is_empty() || gains.is_empty() {
        return None;
    }
    let target = duration as f64;
    let xs: Vec<f64> = durations.iter().map(|&d| d as f64).collect();
    Some(interpolate_curve_value(target, &xs, gains))
}

fn write_score_output(curves: &Curves, path: &Path) -> Result<()> {
    let rows = match curves.magic_rows.as_ref() {
        Some(rows) if !rows.is_empty() => rows,
        _ => {
            warn!("No magic rows available; skipping score output");
            return Ok(());
        }
    };

    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut writer = csv::Writer::from_writer(file);
    writer.write_record([
        "duration_s",
        "user_gain_m",
        "wr_gain_m",
        "score_pct",
        "personal_gain_m",
        "goal_gain_m",
    ])?;

    for row in rows {
        let duration = row.get("duration_s").copied().unwrap_or(0.0);
        let user = row.get("user_gain_m").copied().unwrap_or(0.0);
        let wr = row.get("wr_gain_m").copied().unwrap_or(0.0);
        let score = row.get("score_pct").copied().unwrap_or(0.0);
        let personal = row.get("personal_gain_m").copied().unwrap_or(0.0);
        let goal = row.get("goal_gain_m").copied().unwrap_or(0.0);
        writer.write_record([
            format!("{:.0}", duration),
            format!("{:.3}", user),
            format!("{:.3}", wr),
            format!("{:.1}", score),
            format!("{:.3}", personal),
            format!("{:.3}", goal),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

#[derive(Clone, Debug)]
struct ChartSeries<'a> {
    label: &'a str,
    durations: Vec<f64>,
    values: Vec<f64>,
    color: RGBColor,
}

enum ChartKind {
    Png,
    Svg,
}

fn render_chart_guard(
    curves: &Curves,
    path: &Path,
    kind: ChartKind,
    show_wr: bool,
    show_personal: bool,
) -> Result<(), String> {
    panic::catch_unwind(panic::AssertUnwindSafe(|| {
        render_chart(curves, path, kind, show_wr, show_personal)
    }))
    .map_err(|_| "plotting backend panicked".to_string())?
    .map_err(|err| format!("plotting error: {}", err))
}

fn render_chart(
    curves: &Curves,
    path: &Path,
    kind: ChartKind,
    show_wr: bool,
    show_personal: bool,
) -> Result<()> {
    if curves.points.is_empty() {
        return Ok(());
    }

    let durations: Vec<f64> = curves
        .points
        .iter()
        .map(|p| p.duration_s as f64 / 60.0)
        .collect();
    let gains: Vec<f64> = curves.points.iter().map(|p| p.max_climb_m).collect();
    let rates: Vec<f64> = curves
        .points
        .iter()
        .map(|p| p.climb_rate_m_per_hr)
        .collect();

    let mut overlays: Vec<ChartSeries> = Vec::new();
    if show_wr {
        if let Some((wr_durs, wr_gains)) = curves.wr_curve.as_ref() {
            let wr_minutes: Vec<f64> = wr_durs.iter().map(|d| *d as f64 / 60.0).collect();
            overlays.push(ChartSeries {
                label: "WR climb",
                durations: wr_minutes,
                values: wr_gains.clone(),
                color: RGBColor(90, 90, 90),
            });
        }
    }

    if show_personal {
        if let Some((durs, gains)) = curves.personal_curve.as_ref() {
            overlays.push(ChartSeries {
                label: "Personal",
                durations: durs.iter().map(|d| *d as f64 / 60.0).collect(),
                values: gains.clone(),
                color: RGBColor(30, 144, 255),
            });
        }
        if let Some((durs, gains)) = curves.goal_curve.as_ref() {
            overlays.push(ChartSeries {
                label: "Goal",
                durations: durs.iter().map(|d| *d as f64 / 60.0).collect(),
                values: gains.clone(),
                color: RGBColor(34, 139, 34),
            });
        }
    }

    let sessions = curves.session_curves.clone();

    let x_max = durations.iter().copied().fold(f64::MIN, f64::max).max(1.0);
    let y_max = gains.iter().copied().fold(f64::MIN, f64::max).max(1.0);

    match kind {
        ChartKind::Png => {
            let backend = BitMapBackend::new(path, (1280, 760));
            let root = FontSafeBackend::new(backend).into_drawing_area();
            draw_chart(
                root, &durations, &gains, &rates, x_max, y_max, &overlays, sessions,
            )?;
        }
        ChartKind::Svg => {
            let backend = SVGBackend::new(path, (1280, 760));
            let root = FontSafeBackend::new(backend).into_drawing_area();
            draw_chart(
                root, &durations, &gains, &rates, x_max, y_max, &overlays, sessions,
            )?;
        }
    }

    Ok(())
}

fn draw_chart<DB>(
    root: DrawingArea<DB, plotters::coord::Shift>,
    durations: &[f64],
    gains: &[f64],
    rates: &[f64],
    x_max: f64,
    y_max: f64,
    overlays: &[ChartSeries],
    sessions: Vec<hc_curve::SessionCurve>,
) -> Result<()>
where
    DB: DrawingBackend,
    DB::ErrorType: 'static,
{
    let area = root;
    area.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&area)
        .margin(25)
        .set_label_area_size(LabelAreaPosition::Left, 50)
        .set_label_area_size(LabelAreaPosition::Right, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d(0.0..x_max, 0.0..y_max)?;

    let axis_font = FontDesc::new(FontFamily::SansSerif, 20.0, FontStyle::Normal);

    chart
        .configure_mesh()
        .light_line_style(&TRANSPARENT)
        .bold_line_style(&TRANSPARENT)
        .x_label_formatter(&|v| format!("{:.0}", v))
        .y_label_formatter(&|v| format!("{:.0}", v))
        .label_style(axis_font.clone().color(&BLACK.mix(0.85)))
        .draw()?;

    let rate_style = RGBColor(200, 0, 100);
    let rate_color = rate_style;

    chart
        .draw_series(LineSeries::new(
            durations
                .iter()
                .copied()
                .zip(rates.iter().map(|v| v / 100.0)),
            &rate_style,
        ))?
        .label("Climb rate (m/h)")
        .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 30, y)], rate_color));

    let gain_style = ShapeStyle {
        color: RGBColor(50, 50, 50).to_rgba(),
        filled: false,
        stroke_width: 2,
    };
    chart
        .draw_series(LineSeries::new(
            durations.iter().copied().zip(gains.iter().copied()),
            gain_style,
        ))?
        .label("Climb (m)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 30, y)], &RGBColor(50, 50, 50)));

    for overlay in overlays {
        let color = overlay.color;
        let legend_color = color;
        let style = ShapeStyle {
            color: color.to_rgba(),
            filled: false,
            stroke_width: 2,
        };
        chart
            .draw_series(LineSeries::new(
                overlay
                    .durations
                    .iter()
                    .copied()
                    .zip(overlay.values.iter().copied()),
                style,
            ))?
            .label(overlay.label)
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 30, y)], legend_color));
    }

    if !sessions.is_empty() {
        let colors = [
            RGBColor(135, 206, 250),
            RGBColor(250, 128, 114),
            RGBColor(152, 251, 152),
            RGBColor(216, 191, 216),
        ];
        for (idx, session) in sessions.into_iter().enumerate() {
            let color = colors[idx % colors.len()].mix(0.5);
            let durations: Vec<f64> = session.durations.iter().map(|d| *d as f64 / 60.0).collect();
            let gains = session.climbs;
            chart.draw_series(LineSeries::new(
                durations.into_iter().zip(gains.into_iter()),
                &color,
            ))?;
        }
    }

    let legend_font = FontDesc::new(FontFamily::SansSerif, 18.0, FontStyle::Normal);

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.7))
        .border_style(&BLACK.mix(0.3))
        .label_font(legend_font.color(&BLACK))
        .position(SeriesLabelPosition::UpperLeft)
        .draw()?;

    area.present()?;
    Ok(())
}

struct FontSafeBackend<DB> {
    inner: DB,
}

impl<DB> FontSafeBackend<DB> {
    fn new(inner: DB) -> Self {
        Self { inner }
    }
}

impl<DB: DrawingBackend> DrawingBackend for FontSafeBackend<DB> {
    type ErrorType = DB::ErrorType;

    fn get_size(&self) -> (u32, u32) {
        self.inner.get_size()
    }

    fn ensure_prepared(&mut self) -> Result<(), DrawingErrorKind<Self::ErrorType>> {
        self.inner.ensure_prepared()
    }

    fn present(&mut self) -> Result<(), DrawingErrorKind<Self::ErrorType>> {
        self.inner.present()
    }

    fn draw_pixel(
        &mut self,
        point: BackendCoord,
        color: BackendColor,
    ) -> Result<(), DrawingErrorKind<DB::ErrorType>> {
        self.inner.draw_pixel(point, color)
    }

    fn draw_line<S: BackendStyle>(
        &mut self,
        from: BackendCoord,
        to: BackendCoord,
        style: &S,
    ) -> Result<(), DrawingErrorKind<DB::ErrorType>> {
        self.inner.draw_line(from, to, style)
    }

    fn draw_rect<S: BackendStyle>(
        &mut self,
        upper_left: BackendCoord,
        bottom_right: BackendCoord,
        style: &S,
        fill: bool,
    ) -> Result<(), DrawingErrorKind<DB::ErrorType>> {
        self.inner.draw_rect(upper_left, bottom_right, style, fill)
    }

    fn draw_path<S: BackendStyle, I: IntoIterator<Item = BackendCoord>>(
        &mut self,
        path: I,
        style: &S,
    ) -> Result<(), DrawingErrorKind<DB::ErrorType>> {
        self.inner.draw_path(path, style)
    }

    fn draw_circle<S: BackendStyle>(
        &mut self,
        center: BackendCoord,
        radius: u32,
        style: &S,
        fill: bool,
    ) -> Result<(), DrawingErrorKind<Self::ErrorType>> {
        self.inner.draw_circle(center, radius, style, fill)
    }

    fn blit_bitmap(
        &mut self,
        pos: BackendCoord,
        (iw, ih): (u32, u32),
        src: &[u8],
    ) -> Result<(), DrawingErrorKind<Self::ErrorType>> {
        self.inner.blit_bitmap(pos, (iw, ih), src)
    }

    fn draw_text<TStyle: BackendTextStyle>(
        &mut self,
        text: &str,
        style: &TStyle,
        pos: BackendCoord,
    ) -> Result<(), DrawingErrorKind<Self::ErrorType>> {
        match panic::catch_unwind(panic::AssertUnwindSafe(|| {
            self.inner.draw_text(text, style, pos)
        })) {
            Ok(result) => result,
            Err(_) => self.draw_text_fallback(text, style, pos),
        }
    }

    fn estimate_text_size<TStyle: BackendTextStyle>(
        &self,
        text: &str,
        style: &TStyle,
    ) -> Result<(u32, u32), DrawingErrorKind<Self::ErrorType>> {
        self.inner.estimate_text_size(text, style)
    }
}

impl<DB: DrawingBackend> FontSafeBackend<DB> {
    fn draw_text_fallback<TStyle: BackendTextStyle>(
        &mut self,
        text: &str,
        style: &TStyle,
        pos: BackendCoord,
    ) -> Result<(), DrawingErrorKind<DB::ErrorType>> {
        let color = style.color();
        if color.alpha == 0.0 || text.trim().is_empty() {
            return Ok(());
        }

        let layout = style
            .layout_box(text)
            .map_err(|e| DrawingErrorKind::FontError(Box::new(e)))?;
        let ((min_x, min_y), (max_x, max_y)) = layout;
        let raw_height = (max_y - min_y).max(1);
        let scale = (raw_height as f64 / FALLBACK_FONT_HEIGHT as f64)
            .max(1.0)
            .round() as i32;

        let width = max_x - min_x;
        let dx = match style.anchor().h_pos {
            text_anchor::HPos::Left => 0,
            text_anchor::HPos::Right => -width,
            text_anchor::HPos::Center => -width / 2,
        };
        let dy = match style.anchor().v_pos {
            text_anchor::VPos::Top => 0,
            text_anchor::VPos::Center => -(raw_height / 2),
            text_anchor::VPos::Bottom => -raw_height,
        };

        let mut cursor_x = pos.0 + dx - min_x;
        let baseline_y = pos.1 + dy - min_y;
        for ch in text.chars() {
            if ch == ' ' {
                cursor_x += scale * (FALLBACK_SPACE_WIDTH as i32);
                continue;
            }
            if let Some(glyph) = fallback_glyph(ch) {
                for (row, pattern) in glyph.rows.iter().enumerate() {
                    for col in 0..glyph.width {
                        if pattern & (1 << (glyph.width - 1 - col)) != 0 {
                            self.draw_scaled_pixel_block(
                                cursor_x + col as i32 * scale,
                                baseline_y + row as i32 * scale,
                                scale,
                                color,
                            )?;
                        }
                    }
                }
                cursor_x += scale * (glyph.width as i32 + 1);
            } else {
                cursor_x += scale * (FALLBACK_SPACE_WIDTH as i32);
            }
        }
        Ok(())
    }

    fn draw_scaled_pixel_block(
        &mut self,
        x: i32,
        y: i32,
        scale: i32,
        color: BackendColor,
    ) -> Result<(), DrawingErrorKind<DB::ErrorType>> {
        for dx in 0..scale {
            for dy in 0..scale {
                self.inner.draw_pixel((x + dx, y + dy), color.clone())?;
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct Glyph {
    width: u8,
    rows: [u8; FALLBACK_FONT_HEIGHT],
}

const FALLBACK_FONT_HEIGHT: usize = 7;
const FALLBACK_SPACE_WIDTH: usize = 3;

fn fallback_glyph(ch: char) -> Option<Glyph> {
    let upper = ch.to_ascii_uppercase();
    Some(match upper {
        'A' => Glyph {
            width: 5,
            rows: [
                0b01110, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
            ],
        },
        'B' => Glyph {
            width: 5,
            rows: [
                0b11110, 0b10001, 0b10001, 0b11110, 0b10001, 0b10001, 0b11110,
            ],
        },
        'C' => Glyph {
            width: 5,
            rows: [
                0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110,
            ],
        },
        'D' => Glyph {
            width: 5,
            rows: [
                0b11100, 0b10010, 0b10001, 0b10001, 0b10001, 0b10010, 0b11100,
            ],
        },
        'E' => Glyph {
            width: 5,
            rows: [
                0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111,
            ],
        },
        'F' => Glyph {
            width: 5,
            rows: [
                0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b10000,
            ],
        },
        'G' => Glyph {
            width: 5,
            rows: [
                0b01110, 0b10001, 0b10000, 0b10111, 0b10001, 0b10001, 0b01111,
            ],
        },
        'H' => Glyph {
            width: 5,
            rows: [
                0b10001, 0b10001, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001,
            ],
        },
        'I' => Glyph {
            width: 3,
            rows: [0b111, 0b010, 0b010, 0b010, 0b010, 0b010, 0b111],
        },
        'K' => Glyph {
            width: 5,
            rows: [
                0b10001, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010, 0b10001,
            ],
        },
        'L' => Glyph {
            width: 5,
            rows: [
                0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111,
            ],
        },
        'M' => Glyph {
            width: 5,
            rows: [
                0b10001, 0b11011, 0b10101, 0b10001, 0b10001, 0b10001, 0b10001,
            ],
        },
        'N' => Glyph {
            width: 5,
            rows: [
                0b10001, 0b11001, 0b10101, 0b10011, 0b10001, 0b10001, 0b10001,
            ],
        },
        'O' => Glyph {
            width: 5,
            rows: [
                0b01110, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
            ],
        },
        'P' => Glyph {
            width: 5,
            rows: [
                0b11110, 0b10001, 0b10001, 0b11110, 0b10000, 0b10000, 0b10000,
            ],
        },
        'R' => Glyph {
            width: 5,
            rows: [
                0b11110, 0b10001, 0b10001, 0b11110, 0b10100, 0b10010, 0b10001,
            ],
        },
        'S' => Glyph {
            width: 5,
            rows: [
                0b01111, 0b10000, 0b10000, 0b01110, 0b00001, 0b00001, 0b11110,
            ],
        },
        'T' => Glyph {
            width: 5,
            rows: [
                0b11111, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100, 0b00100,
            ],
        },
        'U' => Glyph {
            width: 5,
            rows: [
                0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b10001, 0b01110,
            ],
        },
        'V' => Glyph {
            width: 5,
            rows: [
                0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b01010, 0b00100,
            ],
        },
        'W' => Glyph {
            width: 5,
            rows: [
                0b10001, 0b10001, 0b10001, 0b10001, 0b10101, 0b11011, 0b10001,
            ],
        },
        'Y' => Glyph {
            width: 5,
            rows: [
                0b10001, 0b10001, 0b01010, 0b00100, 0b00100, 0b00100, 0b00100,
            ],
        },
        'Z' => Glyph {
            width: 5,
            rows: [
                0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b11111,
            ],
        },
        '0' => Glyph {
            width: 5,
            rows: [
                0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110,
            ],
        },
        '1' => Glyph {
            width: 3,
            rows: [0b010, 0b110, 0b010, 0b010, 0b010, 0b010, 0b111],
        },
        '2' => Glyph {
            width: 5,
            rows: [
                0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111,
            ],
        },
        '3' => Glyph {
            width: 5,
            rows: [
                0b11110, 0b00001, 0b00001, 0b00110, 0b00001, 0b00001, 0b11110,
            ],
        },
        '4' => Glyph {
            width: 5,
            rows: [
                0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010,
            ],
        },
        '5' => Glyph {
            width: 5,
            rows: [
                0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110,
            ],
        },
        '6' => Glyph {
            width: 5,
            rows: [
                0b01110, 0b10001, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110,
            ],
        },
        '7' => Glyph {
            width: 5,
            rows: [
                0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000,
            ],
        },
        '8' => Glyph {
            width: 5,
            rows: [
                0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110,
            ],
        },
        '9' => Glyph {
            width: 5,
            rows: [
                0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b10001, 0b01110,
            ],
        },
        '-' => Glyph {
            width: 3,
            rows: [0b000, 0b000, 0b000, 0b111, 0b000, 0b000, 0b000],
        },
        '/' => Glyph {
            width: 3,
            rows: [0b001, 0b001, 0b010, 0b010, 0b100, 0b100, 0b100],
        },
        '(' => Glyph {
            width: 3,
            rows: [0b001, 0b010, 0b100, 0b100, 0b100, 0b010, 0b001],
        },
        ')' => Glyph {
            width: 3,
            rows: [0b100, 0b010, 0b001, 0b001, 0b001, 0b010, 0b100],
        },
        ':' => Glyph {
            width: 1,
            rows: [0b0, 0b1, 0b0, 0b0, 0b0, 0b1, 0b0],
        },
        _ => return None,
    })
}

fn interpolate_curve_value(target: f64, durations: &[f64], gains: &[f64]) -> f64 {
    if durations.is_empty() || gains.is_empty() {
        return 0.0;
    }
    let len = durations.len().min(gains.len());
    if target <= durations[0] {
        return gains[0];
    }
    for i in 1..len {
        if target <= durations[i] {
            let x0 = durations[i - 1];
            let x1 = durations[i];
            let y0 = gains[i - 1];
            let y1 = gains[i];
            if (x1 - x0).abs() < f64::EPSILON {
                return y1.max(y0);
            }
            let frac = ((target - x0) / (x1 - x0)).clamp(0.0, 1.0);
            return y0 + (y1 - y0) * frac;
        }
    }
    gains[len - 1]
}

fn log_magic_summary(rows: &[HashMap<String, f64>], topk: usize) {
    let mut scored: Vec<_> = rows
        .iter()
        .filter_map(|row| {
            let duration = row.get("duration_s").copied()?;
            let wr = row.get("wr_gain_m").copied()?;
            let user = row.get("user_gain_m").copied()?;
            let score = row.get("score_pct").copied()?;
            Some((duration, wr, user, score))
        })
        .collect();
    if scored.is_empty() {
        return;
    }
    scored.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));
    let summary: Vec<String> = scored
        .into_iter()
        .take(topk.max(1))
        .map(|(duration, wr, user, score)| {
            format!(
                "{}s: user {:.0} m ({:.0}%), WR {:.0} m",
                duration, user, score, wr
            )
        })
        .collect();
    if !summary.is_empty() {
        info!("WR checkpoints: {}", summary.join("; "));
    }
}

#[derive(Default, Clone)]
struct KeyStats {
    count: usize,
    numeric: usize,
    min: Option<f64>,
    max: Option<f64>,
}

fn fit_value_to_f64(value: &FitValue) -> Option<f64> {
    match value {
        FitValue::Float32(v) => Some(*v as f64),
        FitValue::Float64(v) => Some(*v),
        FitValue::SInt8(v) => Some(*v as f64),
        FitValue::SInt16(v) => Some(*v as f64),
        FitValue::SInt32(v) => Some(*v as f64),
        FitValue::SInt64(v) => Some(*v as f64),
        FitValue::UInt8(v) => Some(*v as f64),
        FitValue::UInt16(v) => Some(*v as f64),
        FitValue::UInt32(v) => Some(*v as f64),
        FitValue::UInt64(v) => Some(*v as f64),
        FitValue::String(s) => s.parse().ok(),
        FitValue::Array(values) => values.iter().find_map(fit_value_to_f64),
        _ => None,
    }
}
