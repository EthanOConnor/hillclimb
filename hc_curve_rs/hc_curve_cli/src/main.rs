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
    compute_curves, compute_gain_time, compute_timeseries_export, parse_duration_token,
    parse_records, Curves, Engine, GainTimePoint, GainTimeResult, Params, Source, TimeseriesExport,
};
use ordered_float::OrderedFloat;
use plotters::prelude::IntoLogRange;
use plotters::prelude::*;
use plotters::style::{FontDesc, FontFamily, FontStyle};
use plotters_backend::{
    text_anchor, BackendColor, BackendCoord, BackendStyle, BackendTextStyle, DrawingBackend,
    DrawingErrorKind,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::json;
use serde_json::Value as JsonValue;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::time::Instant;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

const GAIN_TIME_ISO_RATES: &[f64] = &[800.0, 1_000.0, 1_200.0, 1_500.0, 2_000.0, 2_500.0];
const DEFAULT_MAGIC_GAIN_TOKENS: &[&str] = &[
    "50m", "100m", "150m", "200m", "300m", "500m", "750m", "1000m",
];
const FIT_CACHE_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CachePayload {
    schema_version: u32,
    records: Vec<hc_curve::FitRecord>,
}

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
    /// Compute gain-centric report: minimum time for target gains
    GainTime(GainTimeArgs),
    /// Export canonical cumulative gain series for external tooling
    ExportSeries(ExportSeriesArgs),
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

    /// Write JSON report next to CSV
    #[arg(long, action = ArgAction::SetTrue)]
    json: bool,

    /// Clear parsed FIT cache before running
    #[arg(long, action = ArgAction::SetTrue)]
    clear_cache: bool,

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

    /// Generate separate rate and climb plots
    #[arg(long = "split-plots", action = ArgAction::SetTrue)]
    split_plots: bool,

    /// Disable split plot output
    #[arg(long = "no-split-plots", action = ArgAction::SetTrue)]
    no_split_plots: bool,

    /// Enable lightweight plotting (default)
    #[arg(long = "fast-plot", action = ArgAction::SetTrue)]
    fast_plot: bool,

    /// Disable lightweight plotting
    #[arg(long = "no-fast-plot", action = ArgAction::SetTrue)]
    no_fast_plot: bool,

    /// Use logarithmic scale on rate axis (split plots only)
    #[arg(long = "ylog-rate", action = ArgAction::SetTrue)]
    ylog_rate: bool,

    /// Use logarithmic scale on climb axis (split plots only)
    #[arg(long = "ylog-climb", action = ArgAction::SetTrue)]
    ylog_climb: bool,

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

    /// Profile major stages with timings
    #[arg(long, action = ArgAction::SetTrue)]
    profile: bool,
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

#[derive(Parser, Debug)]
struct GainTimeArgs {
    /// FIT/GPX files to ingest
    #[arg(required = true, value_hint = ValueHint::FilePath)]
    inputs: Vec<PathBuf>,

    /// Output CSV path (`-` for stdout)
    #[arg(short, long, default_value = "gain_time.csv", value_hint = ValueHint::FilePath)]
    output: PathBuf,

    /// Output PNG figure path (defaults next to CSV)
    #[arg(long, value_hint = ValueHint::FilePath)]
    png: Option<PathBuf>,

    /// Write JSON report next to CSV
    #[arg(long, action = ArgAction::SetTrue)]
    json: bool,

    /// Clear parsed FIT cache before running
    #[arg(long, action = ArgAction::SetTrue)]
    clear_cache: bool,

    /// Disable plot generation
    #[arg(long, action = ArgAction::SetTrue)]
    no_plot: bool,

    /// Gain targets (meters by default, accepts suffix m|ft, comma separated)
    #[arg(short = 'g', long = "gains", value_delimiter = ',', num_args = 1..)]
    gains: Vec<String>,

    /// Read gain targets from file (one per line, supports m|ft suffix)
    #[arg(long = "gains-from", value_hint = ValueHint::FilePath)]
    gains_from: Option<PathBuf>,

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

    /// Plot WR overlay
    #[arg(long, action = ArgAction::SetTrue)]
    plot_wr: bool,

    /// Disable personal overlay on plots
    #[arg(long, action = ArgAction::SetTrue)]
    no_personal: bool,

    /// Generate separate time/rate plots
    #[arg(long = "split-plots", action = ArgAction::SetTrue)]
    split_plots: bool,

    /// Disable split plot output
    #[arg(long = "no-split-plots", action = ArgAction::SetTrue)]
    no_split_plots: bool,

    /// Enable lightweight plotting (default)
    #[arg(long = "fast-plot", action = ArgAction::SetTrue)]
    fast_plot: bool,

    /// Disable lightweight plotting
    #[arg(long = "no-fast-plot", action = ArgAction::SetTrue)]
    no_fast_plot: bool,

    /// Use logarithmic scale on time axis
    #[arg(long = "ylog-time", action = ArgAction::SetTrue)]
    ylog_time: bool,

    /// Gain units for parsing/display
    #[arg(long = "gain-units", value_enum, default_value_t = GainUnitOpt::Meters)]
    gain_units: GainUnitOpt,

    /// Magic gains for annotations (comma separated, accepts m|ft suffix)
    #[arg(long = "magic-gains")]
    magic_gains: Option<String>,

    /// Disable concave envelope smoothing
    #[arg(long, action = ArgAction::SetTrue)]
    no_concave_envelope: bool,

    /// Curve engine (auto|numpy|numba|stride)
    #[arg(long, value_enum, default_value_t = EngineOpt::Auto)]
    engine: EngineOpt,

    /// Altitude smoothing window (seconds)
    #[arg(long, default_value_t = 0.0)]
    smooth: f64,

    /// Profile major stages with timings
    #[arg(long, action = ArgAction::SetTrue)]
    profile: bool,

    /// Verbose logging
    #[arg(long, action = ArgAction::SetTrue)]
    verbose: bool,
}

#[derive(Parser, Debug)]
struct ExportSeriesArgs {
    /// FIT/GPX files to ingest
    #[arg(required = true, value_hint = ValueHint::FilePath)]
    inputs: Vec<PathBuf>,

    /// Output CSV path (`-` for stdout)
    #[arg(short, long, default_value = "timeseries.csv", value_hint = ValueHint::FilePath)]
    output: PathBuf,

    /// Write JSON report next to CSV
    #[arg(long, action = ArgAction::SetTrue)]
    json: bool,

    /// Clear parsed FIT cache before running
    #[arg(long, action = ArgAction::SetTrue)]
    clear_cache: bool,

    /// Data source preference
    #[arg(long, value_enum, default_value_t = SourceOpt::Auto)]
    source: SourceOpt,

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

    /// Gap threshold (seconds) for session segmentation
    #[arg(long, default_value_t = 600.0)]
    session_gap: f64,

    /// Timestamp merge tolerance (seconds)
    #[arg(long, default_value_t = 0.5)]
    merge_eps: f64,

    /// Overlap precedence policy for total gain
    #[arg(long, default_value = "file:last")]
    overlap_policy: String,

    /// Verbose logging
    #[arg(long, action = ArgAction::SetTrue)]
    verbose: bool,

    /// Profile major stages with timings
    #[arg(long, action = ArgAction::SetTrue)]
    profile: bool,

    /// Optional log file
    #[arg(long, value_hint = ValueHint::FilePath)]
    log_file: Option<PathBuf>,
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

#[derive(Copy, Clone, Debug, ValueEnum)]
enum GainUnitOpt {
    #[clap(alias = "m")]
    Meters,
    #[clap(alias = "ft")]
    Feet,
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
        Command::GainTime(args) => {
            if args.verbose {
                "debug"
            } else {
                "info"
            }
        }
        Command::ExportSeries(args) => {
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
        Command::GainTime(args) => handle_gain_time(args),
        Command::ExportSeries(args) => handle_export_series(args),
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

    // Prepare cache dir
    let cache_dir = PathBuf::from(".cache").join("parsed_fit");
    if args.clear_cache {
        let _ = fs::remove_dir_all(&cache_dir);
    }
    let _ = fs::create_dir_all(&cache_dir);

    // Parse inputs (possibly in parallel) with caching
    let t_parse = Instant::now();
    let inputs: Vec<(usize, PathBuf)> = args.inputs.iter().cloned().enumerate().collect();

    let mut records: Vec<(usize, Vec<hc_curve::FitRecord>)> = inputs
        .par_iter()
        .map(
            |(file_id, path)| -> Result<(usize, Vec<hc_curve::FitRecord>)> {
                let key = cache_key(path)?;
                if let Some(mut cached) = read_cache(&cache_dir, &key) {
                    for record in &mut cached {
                        record.file_id = *file_id;
                    }
                    return Ok((*file_id, cached));
                }
                let data =
                    fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
                let hint = path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .unwrap_or("fit");
                let parsed = parse_records(&data, *file_id, hint)
                    .with_context(|| format!("failed to parse {}", path.display()))?;
                // Best-effort cache write
                let _ = write_cache(&cache_dir, &key, &parsed);
                Ok((*file_id, parsed))
            },
        )
        .collect::<Result<Vec<_>>>()?;

    // Restore original ordering by file_id
    records.sort_by_key(|(id, _)| *id);
    let records: Vec<Vec<hc_curve::FitRecord>> = records.into_iter().map(|(_, r)| r).collect();

    if args.profile || args.verbose {
        info!(
            "Parse stage: {:.1} ms",
            t_parse.elapsed().as_secs_f64() * 1000.0
        );
    }

    let records_for_json = if args.json {
        Some(records.clone())
    } else {
        None
    };

    let t_compute = Instant::now();
    let curves = compute_curves(records, &params)?;
    if args.profile || args.verbose {
        info!(
            "Compute stage: {:.1} ms ({} points)",
            t_compute.elapsed().as_secs_f64() * 1000.0,
            curves.points.len()
        );
    }
    info!(
        "Curve computed: {} durations, source {}",
        curves.points.len(),
        curves.selected_source
    );

    if let Some(rows) = curves.magic_rows.as_ref() {
        log_magic_summary(rows, args.goals_topk);
    }

    let split_plots = if args.no_split_plots {
        false
    } else if args.split_plots {
        true
    } else {
        false
    };

    let fast_plot = if args.no_fast_plot {
        false
    } else if args.fast_plot {
        true
    } else {
        true
    };

    let plot_opts = PlotOptions {
        split_plots,
        fast_plot,
        show_wr: args.plot_wr,
        show_personal: !args.no_personal,
        ylog_rate: args.ylog_rate,
        ylog_climb: args.ylog_climb,
    };

    if let Some(score_path) = args.score_output.as_ref() {
        write_score_output(&curves, score_path)?;
        info!("Wrote scoring table: {}", score_path.display());
    }

    if args.output.as_os_str() == "-" {
        write_curve_stdout(&curves)?;
    } else {
        let t_csv = Instant::now();
        write_curve_csv(&curves, &args.output)?;
        if args.profile || args.verbose {
            info!(
                "CSV stage: {:.1} ms ({} rows)",
                t_csv.elapsed().as_secs_f64() * 1000.0,
                curves.points.len()
            );
        }
        info!("Wrote curve CSV: {}", args.output.display());

        if args.json {
            let mut json_path = args.output.clone();
            json_path.set_extension("json");

            let (n_samples, total_span_s, total_gain_m) = if let Some(rec_sets) = records_for_json.clone() {
                match compute_timeseries_export(rec_sets, &params) {
                    Ok(ts) => {
                        let n = ts.times.len();
                        let span = ts.times.last().copied().unwrap_or(0.0);
                        let gain = ts.gain.last().copied().unwrap_or(0.0) - ts.gain.first().copied().unwrap_or(0.0);
                        (n, span, gain)
                    }
                    Err(err) => {
                        warn!("Unable to build timeseries for JSON report: {}", err);
                        (0, 0.0, 0.0)
                    }
                }
            } else {
                (0, 0.0, 0.0)
            };

            let meta = json!({
                "command": "curve",
                "inputs": args.inputs.iter().map(|p| p.to_string_lossy()).collect::<Vec<_>>(),
                "output_csv": args.output.to_string_lossy(),
                "selected_source": curves.selected_source,
                "n_samples": n_samples,
                "total_span_s": total_span_s,
                "total_gain_m": total_gain_m,
                "engine": format!("{:?}", params.engine).to_lowercase(),
                "qc_enabled": params.qc_enabled,
                "qc_spec": args.qc_spec.as_ref().map(|p| p.to_string_lossy()),
                "resample_1hz": params.resample_1hz,
                "gain_eps": params.gain_eps_m,
                "smooth_sec": params.smooth_sec,
                "durations_s": curves.points.iter().map(|p| p.duration_s).collect::<Vec<_>>(),
                "exhaustive": params.exhaustive,
                "all_windows": params.all_windows,
                "step_s": params.step_s,
                "max_duration_s": params.max_duration_s,
            });

            let payload = json!({
                "meta": meta,
                "curves": curves,
            });

            let text = serde_json::to_string_pretty(&payload)?;
            fs::write(&json_path, text)
                .with_context(|| format!("failed to write {}", json_path.display()))?;
            info!("Wrote JSON: {}", json_path.display());
        }
    }

    if !args.no_plot {
        if let Some(path) = args.png.as_ref() {
            if let Err(err) = render_chart_guard(&curves, path, ChartKind::Png, &plot_opts) {
                warn!("Skipping PNG render ({}): {}", path.display(), err);
            } else if plot_opts.split_plots {
                info!("Wrote plots: {} (_rate/_climb)", path.display());
            } else {
                info!("Wrote plot: {}", path.display());
            }
        } else if args.output.as_os_str() != "-" {
            let mut png_path = args.output.clone();
            png_path.set_extension("png");
            let t_plot = Instant::now();
            if let Err(err) = render_chart_guard(&curves, &png_path, ChartKind::Png, &plot_opts) {
                warn!("Skipping PNG render ({}): {}", png_path.display(), err);
            } else if plot_opts.split_plots {
                info!("Wrote plots: {} (_rate/_climb)", png_path.display());
            } else {
                info!("Wrote plot: {}", png_path.display());
            }
            if args.profile || args.verbose {
                info!(
                    "Plot stage: {:.1} ms",
                    t_plot.elapsed().as_secs_f64() * 1000.0
                );
            }
        }

        if let Some(path) = args.svg.as_ref() {
            if let Err(err) = render_chart_guard(&curves, path, ChartKind::Svg, &plot_opts) {
                warn!("Skipping SVG render ({}): {}", path.display(), err);
            } else if plot_opts.split_plots {
                info!("Wrote plots: {} (_rate/_climb)", path.display());
            } else {
                info!("Wrote plot: {}", path.display());
            }
        }
    }

    Ok(())
}

fn handle_gain_time(args: GainTimeArgs) -> Result<()> {
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
    params.exhaustive = args.exhaustive;
    params.all_windows = args.all;
    params.step_s = args.step.max(1);
    params.max_duration_s = args.max_duration;
    params.source = args.source.into();
    params.wr_profile = args.wr_profile.clone();
    params.wr_anchors_path = args.wr_anchors.clone();
    params.wr_min_seconds = args.wr_min_seconds;
    params.wr_short_cap = args.wr_short_cap.clone();
    params.concave_envelope = !args.no_concave_envelope;
    params.engine = args.engine.into();
    params.magic = None;

    if let Some(spec_path) = args.qc_spec.as_ref() {
        params.qc_spec = Some(load_qc_spec(spec_path)?);
    }

    let gain_unit = args.gain_units;
    let mut gain_tokens = args.gains.clone();
    if let Some(path) = args.gains_from.as_ref() {
        let from_file = read_gain_tokens_from_file(path)?;
        gain_tokens.extend(from_file);
    }
    let gain_targets = parse_gain_value_list(&gain_tokens, gain_unit);
    if !gain_tokens.is_empty() && gain_targets.is_empty() {
        warn!("No valid gain targets parsed from --gains/--gains-from; falling back to defaults.");
    }
    let mut magic_gain_targets = parse_magic_gain_tokens(args.magic_gains.as_deref(), gain_unit);
    if magic_gain_targets.is_empty() {
        magic_gain_targets = DEFAULT_MAGIC_GAIN_TOKENS
            .iter()
            .filter_map(|token| parse_gain_token(token, GainUnitOpt::Meters))
            .collect();
    }

    let cache_dir = PathBuf::from(".cache").join("parsed_fit");
    if args.clear_cache {
        let _ = fs::remove_dir_all(&cache_dir);
    }
    let _ = fs::create_dir_all(&cache_dir);

    let t_parse = Instant::now();
    let inputs: Vec<(usize, PathBuf)> = args.inputs.iter().cloned().enumerate().collect();

    let mut records: Vec<(usize, Vec<hc_curve::FitRecord>)> = inputs
        .par_iter()
        .map(
            |(file_id, path)| -> Result<(usize, Vec<hc_curve::FitRecord>)> {
                let key = cache_key(path)?;
                if let Some(mut cached) = read_cache(&cache_dir, &key) {
                    for record in &mut cached {
                        record.file_id = *file_id;
                    }
                    return Ok((*file_id, cached));
                }
                let data =
                    fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
                let hint = path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .unwrap_or("fit");
                let parsed = parse_records(&data, *file_id, hint)
                    .with_context(|| format!("failed to parse {}", path.display()))?;
                let _ = write_cache(&cache_dir, &key, &parsed);
                Ok((*file_id, parsed))
            },
        )
        .collect::<Result<Vec<_>>>()?;

    records.sort_by_key(|(id, _)| *id);
    let records: Vec<Vec<hc_curve::FitRecord>> = records.into_iter().map(|(_, r)| r).collect();

    if args.profile || args.verbose {
        info!(
            "Parse stage: {:.1} ms",
            t_parse.elapsed().as_secs_f64() * 1000.0
        );
    }

    let records_for_json = if args.json {
        Some(records.clone())
    } else {
        None
    };

    let t_compute = Instant::now();
    let result = compute_gain_time(records, &params, &gain_targets)?;
    if args.profile || args.verbose {
        info!(
            "Compute stage: {:.1} ms ({} curve points)",
            t_compute.elapsed().as_secs_f64() * 1000.0,
            result.curve.len()
        );
    }
    info!(
        "Gain-time computed: {} targets, source {}",
        result.targets.len(),
        result.selected_source
    );

    print_gain_time_summary(&result, gain_unit);

    if args.output.as_os_str() == "-" {
        write_gain_time_stdout(&result)?;
    } else {
        let t_csv = Instant::now();
        write_gain_time_csv(&result, &args.output)?;
        if args.profile || args.verbose {
            info!(
                "CSV stage: {:.1} ms ({} rows)",
                t_csv.elapsed().as_secs_f64() * 1000.0,
                result.targets.len()
            );
        }
        info!("Wrote gain-time CSV: {}", args.output.display());

        if args.json {
            let mut json_path = args.output.clone();
            json_path.set_extension("json");

            let n_samples = if let Some(rec_sets) = records_for_json {
                match compute_timeseries_export(rec_sets, &params) {
                    Ok(ts) => ts.times.len(),
                    Err(err) => {
                        warn!("Unable to build timeseries for JSON report: {}", err);
                        0
                    }
                }
            } else {
                0
            };

            let meta = json!({
                "command": "gain-time",
                "inputs": args.inputs.iter().map(|p| p.to_string_lossy()).collect::<Vec<_>>(),
                "output_csv": args.output.to_string_lossy(),
                "selected_source": result.selected_source,
                "n_samples": n_samples,
                "total_span_s": result.total_span_s,
                "total_gain_m": result.total_gain_m,
                "engine": format!("{:?}", params.engine).to_lowercase(),
                "qc_enabled": params.qc_enabled,
                "qc_spec": args.qc_spec.as_ref().map(|p| p.to_string_lossy()),
                "resample_1hz": params.resample_1hz,
                "gain_eps": params.gain_eps_m,
                "smooth_sec": params.smooth_sec,
                "exhaustive": params.exhaustive,
                "all_windows": params.all_windows,
                "step_s": params.step_s,
                "max_duration_s": params.max_duration_s,
                "gain_units": format!("{:?}", gain_unit).to_lowercase(),
                "magic_gains": magic_gain_targets.clone(),
            });

            let payload = json!({
                "meta": meta,
                "gain_time": result,
            });

            let text = serde_json::to_string_pretty(&payload)?;
            fs::write(&json_path, text)
                .with_context(|| format!("failed to write {}", json_path.display()))?;
            info!("Wrote JSON: {}", json_path.display());
        }
    }

    if !args.no_plot {
        let split_plots = if args.no_split_plots {
            false
        } else if args.split_plots {
            true
        } else {
            false
        };

        let fast_plot = if args.no_fast_plot {
            false
        } else if args.fast_plot {
            true
        } else {
            true
        };

        let plot_opts = GainTimePlotOptions {
            split_plots,
            fast_plot,
            show_wr: args.plot_wr,
            show_personal: !args.no_personal,
            ylog_time: args.ylog_time,
            unit: gain_unit,
            magic_gains: magic_gain_targets.clone(),
        };

        if let Some(path) = args.png.as_ref() {
            if let Err(err) = render_gain_time_guard(&result, path, ChartKind::Png, &plot_opts) {
                warn!("Skipping PNG render ({}): {}", path.display(), err);
            } else if plot_opts.split_plots {
                info!("Wrote plots: {} (_time/_rate)", path.display());
            } else {
                info!("Wrote plot: {}", path.display());
            }
        } else if args.output.as_os_str() != "-" {
            let mut png_path = args.output.clone();
            png_path.set_extension("png");
            let t_plot = Instant::now();
            if let Err(err) = render_gain_time_guard(&result, &png_path, ChartKind::Png, &plot_opts)
            {
                warn!("Skipping PNG render ({}): {}", png_path.display(), err);
            } else if plot_opts.split_plots {
                info!("Wrote plots: {} (_time/_rate)", png_path.display());
            } else {
                info!("Wrote plot: {}", png_path.display());
            }
            if args.profile || args.verbose {
                info!(
                    "Plot stage: {:.1} ms",
                    t_plot.elapsed().as_secs_f64() * 1000.0
                );
            }
        }
    }

    Ok(())
}

fn handle_export_series(args: ExportSeriesArgs) -> Result<()> {
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
    params.source = args.source.into();
    params.all_windows = false;
    params.exhaustive = false;
    params.step_s = 1;
    params.max_duration_s = None;

    if let Some(spec_path) = args.qc_spec.as_ref() {
        params.qc_spec = Some(load_qc_spec(spec_path)?);
    }

    let cache_dir = PathBuf::from(".cache").join("parsed_fit");
    if args.clear_cache {
        let _ = fs::remove_dir_all(&cache_dir);
    }
    let _ = fs::create_dir_all(&cache_dir);

    let inputs: Vec<(usize, PathBuf)> = args.inputs.iter().cloned().enumerate().collect();

    let t_parse = Instant::now();
    let mut records: Vec<(usize, Vec<hc_curve::FitRecord>)> = inputs
        .par_iter()
        .map(
            |(file_id, path)| -> Result<(usize, Vec<hc_curve::FitRecord>)> {
                let key = cache_key(path)?;
                if let Some(mut cached) = read_cache(&cache_dir, &key) {
                    for record in &mut cached {
                        record.file_id = *file_id;
                    }
                    return Ok((*file_id, cached));
                }
                let data =
                    fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
                let hint = path
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .unwrap_or("fit");
                let parsed = parse_records(&data, *file_id, hint)
                    .with_context(|| format!("failed to parse {}", path.display()))?;
                let _ = write_cache(&cache_dir, &key, &parsed);
                Ok((*file_id, parsed))
            },
        )
        .collect::<Result<Vec<_>>>()?;

    records.sort_by_key(|(id, _)| *id);
    let records: Vec<Vec<hc_curve::FitRecord>> = records.into_iter().map(|(_, r)| r).collect();

    if args.profile || args.verbose {
        info!(
            "Parse stage: {:.1} ms",
            t_parse.elapsed().as_secs_f64() * 1000.0
        );
    }

    let t_compute = Instant::now();
    let series = compute_timeseries_export(records, &params)?;
    if args.profile || args.verbose {
        info!(
            "Compute stage: {:.1} ms ({} samples)",
            t_compute.elapsed().as_secs_f64() * 1000.0,
            series.times.len()
        );
    }
    info!("Timeseries computed: {} samples", series.times.len());

    if args.output.as_os_str() == "-" {
        write_timeseries_stdout(&series)?;
        info!("Wrote timeseries CSV to stdout");
    } else {
        write_timeseries_csv(&series, &args.output)?;
        info!("Wrote timeseries CSV: {}", args.output.display());

        if args.json {
            let mut json_path = args.output.clone();
            json_path.set_extension("json");

            let meta = json!({
                "command": "export-series",
                "inputs": args.inputs.iter().map(|p| p.to_string_lossy()).collect::<Vec<_>>(),
                "output_csv": args.output.to_string_lossy(),
                "selected_source": series.selected_source,
                "n_samples": series.times.len(),
                "total_span_s": series.times.last().copied().unwrap_or(0.0),
                "total_gain_m": series.gain.last().copied().unwrap_or(0.0) - series.gain.first().copied().unwrap_or(0.0),
                "qc_enabled": params.qc_enabled,
                "qc_spec": args.qc_spec.as_ref().map(|p| p.to_string_lossy()),
                "resample_1hz": params.resample_1hz,
                "gain_eps": params.gain_eps_m,
                "smooth_sec": params.smooth_sec,
                "session_gap_sec": params.session_gap_sec,
                "merge_eps_sec": params.merge_eps_sec,
                "overlap_policy": params.overlap_policy,
            });

            let payload = json!({
                "meta": meta,
                "series": series,
            });

            let text = serde_json::to_string_pretty(&payload)?;
            fs::write(&json_path, text)
                .with_context(|| format!("failed to write {}", json_path.display()))?;
            info!("Wrote JSON: {}", json_path.display());
        }
    }

    Ok(())
}

fn cache_key(path: &Path) -> Result<String> {
    use std::time::SystemTime;
    let meta = fs::metadata(path).with_context(|| format!("failed to stat {}", path.display()))?;
    let size = meta.len();
    let modified = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
    let dur = modified
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let mtime = (dur.as_secs(), dur.subsec_nanos());

    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    path.to_string_lossy().hash(&mut hasher);
    size.hash(&mut hasher);
    mtime.hash(&mut hasher);
    Ok(format!("{:016x}", hasher.finish()))
}

fn read_cache(dir: &Path, key: &str) -> Option<Vec<hc_curve::FitRecord>> {
    let path = dir.join(format!("{}.json", key));
    let text = fs::read_to_string(&path).ok()?;
    let payload: CachePayload = serde_json::from_str(&text).ok()?;
    if payload.schema_version != FIT_CACHE_SCHEMA_VERSION {
        return None;
    }
    Some(payload.records)
}

fn write_cache(dir: &Path, key: &str, records: &[hc_curve::FitRecord]) -> Result<()> {
    let path = dir.join(format!("{}.json", key));
    let payload = CachePayload {
        schema_version: FIT_CACHE_SCHEMA_VERSION,
        records: records.to_vec(),
    };
    let text = serde_json::to_string(&payload)?;
    fs::write(&path, text).with_context(|| format!("failed to write {}", path.display()))?;
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

fn parse_gain_value_list(values: &[String], unit: GainUnitOpt) -> Vec<f64> {
    let mut out = Vec::new();
    for token in values {
        for part in token.split(|c: char| c == ',' || c.is_whitespace()) {
            if part.is_empty() {
                continue;
            }
            if let Some(value) = parse_gain_token(part, unit) {
                out.push(value);
            }
        }
    }
    out
}

fn read_gain_tokens_from_file(path: &Path) -> Result<Vec<String>> {
    let text =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let mut tokens = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        tokens.push(trimmed.to_string());
    }
    Ok(tokens)
}

fn parse_magic_gain_tokens(input: Option<&str>, unit: GainUnitOpt) -> Vec<f64> {
    let mut out = Vec::new();
    if let Some(text) = input {
        for part in text.split(|c: char| c == ',' || c.is_whitespace()) {
            if part.is_empty() {
                continue;
            }
            if let Some(value) = parse_gain_token(part, unit) {
                out.push(value);
            }
        }
    }
    out
}

fn parse_gain_token(token: &str, default_unit: GainUnitOpt) -> Option<f64> {
    let trimmed = token.trim().to_ascii_lowercase();
    if trimmed.is_empty() {
        return None;
    }
    let (value_str, unit) = if let Some(stripped) = trimmed.strip_suffix("ft") {
        (stripped.trim(), GainUnitOpt::Feet)
    } else if let Some(stripped) = trimmed.strip_suffix('m') {
        (stripped.trim(), GainUnitOpt::Meters)
    } else {
        (trimmed.as_str(), default_unit)
    };
    let value: f64 = value_str.parse().ok()?;
    if !value.is_finite() || value < 0.0 {
        return None;
    }
    match unit {
        GainUnitOpt::Meters => Some(value),
        GainUnitOpt::Feet => Some(value * 0.3048),
    }
}

fn gain_to_display(value_m: f64, unit: GainUnitOpt) -> f64 {
    match unit {
        GainUnitOpt::Meters => value_m,
        GainUnitOpt::Feet => value_m / 0.3048,
    }
}

fn display_gain_label(unit: GainUnitOpt) -> &'static str {
    match unit {
        GainUnitOpt::Meters => "m",
        GainUnitOpt::Feet => "ft",
    }
}

fn format_gain_value(value_m: f64, unit: GainUnitOpt) -> String {
    match unit {
        GainUnitOpt::Meters => format!("{:.0} m", value_m),
        GainUnitOpt::Feet => format!("{:.0} ft", value_m / 0.3048),
    }
}

fn format_duration_hms(seconds: f64) -> String {
    if !seconds.is_finite() || seconds < 0.0 {
        return "--:--".to_string();
    }
    let total_seconds = seconds.round() as i64;
    let hours = total_seconds / 3_600;
    let minutes = (total_seconds % 3_600) / 60;
    let secs = total_seconds % 60;
    if hours > 0 {
        format!("{}:{:02}:{:02}", hours, minutes, secs)
    } else {
        format!("{}:{:02}", minutes, secs)
    }
}

fn approximate_time_from_curve(curve: &[GainTimePoint], gain_m: f64) -> Option<f64> {
    if curve.is_empty() {
        return None;
    }
    if gain_m <= curve[0].gain_m {
        return Some(curve[0].min_time_s);
    }
    for window in curve.windows(2) {
        let a = &window[0];
        let b = &window[1];
        if gain_m <= b.gain_m {
            let g0 = a.gain_m;
            let g1 = b.gain_m;
            let t0 = a.min_time_s;
            let t1 = b.min_time_s;
            if (g1 - g0).abs() < f64::EPSILON {
                return Some(t1.max(t0));
            }
            let frac = ((gain_m - g0) / (g1 - g0)).clamp(0.0, 1.0);
            return Some(t0 + (t1 - t0) * frac);
        }
    }
    let last = curve.last().unwrap();
    if gain_m > last.gain_m + 1e-6 {
        None
    } else {
        Some(last.min_time_s)
    }
}

fn print_gain_time_summary(result: &GainTimeResult, unit: GainUnitOpt) {
    println!("Gain Time Report (source: {})", result.selected_source);
    let max_gain_display = format_gain_value(result.total_gain_m, unit);
    for point in &result.targets {
        let gain_label = format_gain_value(point.gain_m, unit);
        if !point.min_time_s.is_finite() {
            println!(
                "- {}: unachievable (max gain {})",
                gain_label, max_gain_display
            );
            continue;
        }
        let time_label = format_duration_hms(point.min_time_s);
        let rate_label = if point.avg_rate_m_per_hr.is_finite() && point.avg_rate_m_per_hr > 0.0 {
            format!("{:.0} m/h", point.avg_rate_m_per_hr)
        } else {
            "--".to_string()
        };
        let start_label = point
            .start_offset_s
            .map(format_duration_hms)
            .unwrap_or_else(|| "--:--".to_string());
        let end_label = point
            .end_offset_s
            .map(format_duration_hms)
            .unwrap_or_else(|| "--:--".to_string());
        if let Some(note) = point.note.as_deref() {
            println!(
                "- {}: {} ({}) window {}–{} [{}]",
                gain_label, time_label, rate_label, start_label, end_label, note
            );
        } else {
            println!(
                "- {}: {} ({}) window {}–{}",
                gain_label, time_label, rate_label, start_label, end_label
            );
        }
    }
}

fn write_gain_time_stdout(result: &GainTimeResult) -> Result<()> {
    let mut writer = csv::Writer::from_writer(io::stdout());
    write_gain_time_records(&mut writer, result)?;
    writer.flush()?;
    Ok(())
}

fn write_gain_time_csv(result: &GainTimeResult, path: &Path) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    write_gain_time_records(&mut writer, result)?;
    writer.flush()?;
    Ok(())
}

fn write_gain_time_records<W: io::Write>(
    writer: &mut csv::Writer<W>,
    result: &GainTimeResult,
) -> Result<()> {
    writer.write_record([
        "gain_m",
        "min_time_s",
        "avg_rate_m_per_hr",
        "start_offset_s",
        "end_offset_s",
        "source",
        "note",
    ])?;

    for point in &result.targets {
        writer.write_record([
            format!("{:.3}", point.gain_m),
            if point.min_time_s.is_finite() {
                format!("{:.3}", point.min_time_s)
            } else {
                String::new()
            },
            if point.avg_rate_m_per_hr.is_finite() {
                format!("{:.3}", point.avg_rate_m_per_hr)
            } else {
                String::new()
            },
            point
                .start_offset_s
                .map(|v| format!("{:.3}", v))
                .unwrap_or_default(),
            point
                .end_offset_s
                .map(|v| format!("{:.3}", v))
                .unwrap_or_default(),
            result.selected_source.clone(),
            point.note.clone().unwrap_or_default(),
        ])?;
    }
    Ok(())
}

fn write_timeseries_stdout(series: &TimeseriesExport) -> Result<()> {
    let mut writer = csv::Writer::from_writer(io::stdout());
    write_timeseries_records(&mut writer, series)?;
    writer.flush()?;
    Ok(())
}

fn write_timeseries_csv(series: &TimeseriesExport, path: &Path) -> Result<()> {
    let mut writer = csv::Writer::from_path(path)?;
    write_timeseries_records(&mut writer, series)?;
    writer.flush()?;
    Ok(())
}

fn write_timeseries_records<W: io::Write>(
    writer: &mut csv::Writer<W>,
    series: &TimeseriesExport,
) -> Result<()> {
    writer.write_record(["time_s", "cumulative_gain_m", "source"])?;
    for (t, g) in series.times.iter().zip(series.gain.iter()) {
        writer.write_record([
            format!("{:.6}", t),
            format!("{:.6}", g),
            series.selected_source.clone(),
        ])?;
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct ChartSeries<'a> {
    label: &'a str,
    durations: Vec<f64>,
    values: Vec<f64>,
    color: RGBColor,
}

#[derive(Clone, Debug)]
struct PlotOptions {
    split_plots: bool,
    fast_plot: bool,
    show_wr: bool,
    show_personal: bool,
    ylog_rate: bool,
    ylog_climb: bool,
}

#[derive(Clone, Debug)]
struct GainTimePlotOptions {
    split_plots: bool,
    fast_plot: bool,
    show_wr: bool,
    show_personal: bool,
    ylog_time: bool,
    unit: GainUnitOpt,
    magic_gains: Vec<f64>,
}

fn render_gain_time_guard(
    result: &GainTimeResult,
    path: &Path,
    kind: ChartKind,
    opts: &GainTimePlotOptions,
) -> Result<(), String> {
    let render = || -> Result<(), String> {
        if opts.split_plots {
            render_gain_time_split(result, path, kind, opts)
        } else {
            render_gain_time_combined(result, path, kind, opts)
                .map_err(|e| format!("plotting error: {}", e))
        }
    };

    panic::catch_unwind(panic::AssertUnwindSafe(render))
        .map_err(|_| "plotting backend panicked".to_string())?
        .map_err(|err| err)
}

fn render_gain_time_combined(
    result: &GainTimeResult,
    path: &Path,
    kind: ChartKind,
    opts: &GainTimePlotOptions,
) -> Result<()> {
    if result.curve.is_empty() {
        return Ok(());
    }

    match kind {
        ChartKind::Png => {
            let backend = BitMapBackend::new(path, (1280, 720));
            let root = FontSafeBackend::new(backend).into_drawing_area();
            draw_gain_time_chart(root, result, opts)?;
        }
        ChartKind::Svg => {
            let backend = SVGBackend::new(path, (1280, 720));
            let root = FontSafeBackend::new(backend).into_drawing_area();
            draw_gain_time_chart(root, result, opts)?;
        }
    }

    Ok(())
}

fn render_gain_time_split(
    result: &GainTimeResult,
    base_path: &Path,
    kind: ChartKind,
    opts: &GainTimePlotOptions,
) -> Result<(), String> {
    if result.curve.is_empty() {
        return Ok(());
    }

    let (time_path, rate_path) = derive_gain_time_paths(base_path);

    match kind {
        ChartKind::Png => {
            let backend_time = BitMapBackend::new(&time_path, (1280, 720));
            let root_time = FontSafeBackend::new(backend_time).into_drawing_area();
            draw_gain_time_chart(root_time, result, opts)
                .map_err(|e| format!("plotting error: {}", e))?;

            let backend_rate = BitMapBackend::new(&rate_path, (1280, 720));
            let root_rate = FontSafeBackend::new(backend_rate).into_drawing_area();
            draw_gain_rate_chart(root_rate, result, opts)
                .map_err(|e| format!("plotting error: {}", e))?;
        }
        ChartKind::Svg => {
            let backend_time = SVGBackend::new(&time_path, (1280, 720));
            let root_time = FontSafeBackend::new(backend_time).into_drawing_area();
            draw_gain_time_chart(root_time, result, opts)
                .map_err(|e| format!("plotting error: {}", e))?;

            let backend_rate = SVGBackend::new(&rate_path, (1280, 720));
            let root_rate = FontSafeBackend::new(backend_rate).into_drawing_area();
            draw_gain_rate_chart(root_rate, result, opts)
                .map_err(|e| format!("plotting error: {}", e))?;
        }
    }

    Ok(())
}

fn derive_gain_time_paths(base: &Path) -> (PathBuf, PathBuf) {
    let stem = base
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("gain_time");
    let ext = base.extension().and_then(|s| s.to_str()).unwrap_or("png");
    let time_name = format!("{}_time.{}", stem, ext);
    let rate_name = format!("{}_rate.{}", stem, ext);
    let time_path = base.with_file_name(time_name);
    let rate_path = base.with_file_name(rate_name);
    (time_path, rate_path)
}

fn draw_gain_time_chart<DB>(
    root: DrawingArea<DB, plotters::coord::Shift>,
    result: &GainTimeResult,
    opts: &GainTimePlotOptions,
) -> Result<()>
where
    DB: DrawingBackend,
    DB::ErrorType: 'static,
{
    let area = root;
    area.fill(&WHITE)?;

    let gains_display: Vec<f64> = result
        .curve
        .iter()
        .map(|p| gain_to_display(p.gain_m, opts.unit))
        .collect();
    let times_min: Vec<f64> = result.curve.iter().map(|p| p.min_time_s / 60.0).collect();
    let max_gain_display = gains_display
        .iter()
        .copied()
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let max_gain_m = result
        .curve
        .iter()
        .map(|p| p.gain_m)
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let mut max_time = times_min.iter().copied().fold(0.0_f64, f64::max).max(1.0);
    if !max_time.is_finite() || max_time <= 0.0 {
        max_time = 1.0;
    }

    macro_rules! draw_body {
        ($chart:ident) => {{
            let iso_color = RGBColor(180, 180, 180);
            for &rate in GAIN_TIME_ISO_RATES {
                if rate <= 0.0 {
                    continue;
                }
                let mut series: Vec<(f64, f64)> = Vec::new();
                let steps = 64;
                for idx in 0..=steps {
                    let gain_m = max_gain_m * idx as f64 / steps as f64;
                    let gain_disp = gain_to_display(gain_m, opts.unit);
                    let time_min = (gain_m / rate) * 60.0;
                    series.push((gain_disp, time_min));
                }
                $chart.draw_series(LineSeries::new(series.clone(), &iso_color.mix(0.4)))?;
                if let Some(last) = series.last() {
                    let text_style = FontDesc::new(FontFamily::SansSerif, 14.0, FontStyle::Italic)
                        .color(&iso_color);
                    $chart.draw_series(std::iter::once(Text::new(
                        format!("{:.0} m/h", rate),
                        *last,
                        text_style,
                    )))?;
                }
            }

            $chart
                .draw_series(LineSeries::new(
                    gains_display
                        .iter()
                        .zip(times_min.iter())
                        .map(|(&g, &t)| (g, t)),
                    &RGBColor(0, 114, 178),
                ))?
                .label("Min time")
                .legend(|(x, y)| {
                    PathElement::new(vec![(x, y), (x + 20, y)], RGBColor(0, 114, 178))
                });

            if opts.show_wr {
                if let Some(curve) = result.wr_curve.as_ref() {
                    $chart
                        .draw_series(LineSeries::new(
                            curve.iter().map(|pt| {
                                (gain_to_display(pt.gain_m, opts.unit), pt.min_time_s / 60.0)
                            }),
                            &RGBColor(90, 90, 90),
                        ))?
                        .label("WR")
                        .legend(|(x, y)| {
                            PathElement::new(vec![(x, y), (x + 20, y)], RGBColor(90, 90, 90))
                        });
                }
            }

            if opts.show_personal {
                if let Some(curve) = result.personal_curve.as_ref() {
                    $chart
                        .draw_series(LineSeries::new(
                            curve.iter().map(|pt| {
                                (gain_to_display(pt.gain_m, opts.unit), pt.min_time_s / 60.0)
                            }),
                            &RGBColor(30, 144, 255),
                        ))?
                        .label("Personal")
                        .legend(|(x, y)| {
                            PathElement::new(vec![(x, y), (x + 20, y)], RGBColor(30, 144, 255))
                        });
                }
            }

            $chart.draw_series(result.targets.iter().filter_map(|pt| {
                if !pt.min_time_s.is_finite() {
                    return None;
                }
                let gain_disp = gain_to_display(pt.gain_m, opts.unit);
                let time_min = pt.min_time_s / 60.0;
                Some(Circle::new(
                    (gain_disp, time_min),
                    4,
                    RGBColor(0, 114, 178).filled(),
                ))
            }))?;

            if !opts.fast_plot {
                for point in &result.targets {
                    if !point.min_time_s.is_finite() {
                        continue;
                    }
                    let text_style =
                        FontDesc::new(FontFamily::SansSerif, 14.0, FontStyle::Normal).color(&BLACK);
                    $chart.draw_series(std::iter::once(Text::new(
                        format_duration_hms(point.min_time_s),
                        (
                            gain_to_display(point.gain_m, opts.unit),
                            point.min_time_s / 60.0,
                        ),
                        text_style,
                    )))?;
                }
            }

            if !opts.magic_gains.is_empty() {
                for gain_m in &opts.magic_gains {
                    if let Some(time_s) = approximate_time_from_curve(&result.curve, *gain_m) {
                        let gain_disp = gain_to_display(*gain_m, opts.unit);
                        let time_min = time_s / 60.0;
                        $chart.draw_series(std::iter::once(TriangleMarker::new(
                            (gain_disp, time_min),
                            6,
                            RGBColor(34, 139, 34).filled(),
                        )))?;
                    }
                }
            }

            $chart
                .configure_series_labels()
                .background_style(&WHITE.mix(0.8))
                .border_style(&BLACK)
                .draw()?;
        }};
    }

    if opts.ylog_time {
        let min_positive = times_min
            .iter()
            .copied()
            .filter(|v| *v > 0.0)
            .fold(f64::INFINITY, f64::min)
            .max(1e-2);
        let mut chart = ChartBuilder::on(&area)
            .margin(25)
            .set_label_area_size(LabelAreaPosition::Left, 60)
            .set_label_area_size(LabelAreaPosition::Bottom, 50)
            .build_cartesian_2d(
                0.0..(max_gain_display * 1.05),
                (min_positive..(max_time * 1.1)).log_scale(),
            )?;
        chart
            .configure_mesh()
            .x_desc(format!("Gain ({})", display_gain_label(opts.unit)))
            .y_desc("Time (min)")
            .x_label_formatter(&|v| format!("{:.0}", v))
            .y_label_formatter(&|v| format!("{:.1}", v))
            .label_style(FontDesc::new(
                FontFamily::SansSerif,
                16.0,
                FontStyle::Normal,
            ))
            .draw()?;
        draw_body!(chart);
    } else {
        let mut chart = ChartBuilder::on(&area)
            .margin(25)
            .set_label_area_size(LabelAreaPosition::Left, 60)
            .set_label_area_size(LabelAreaPosition::Bottom, 50)
            .build_cartesian_2d(0.0..(max_gain_display * 1.05), 0.0..(max_time * 1.1))?;
        chart
            .configure_mesh()
            .x_desc(format!("Gain ({})", display_gain_label(opts.unit)))
            .y_desc("Time (min)")
            .x_label_formatter(&|v| format!("{:.0}", v))
            .y_label_formatter(&|v| format!("{:.0}", v))
            .label_style(FontDesc::new(
                FontFamily::SansSerif,
                16.0,
                FontStyle::Normal,
            ))
            .draw()?;
        draw_body!(chart);
    }

    Ok(())
}

fn draw_gain_rate_chart<DB>(
    root: DrawingArea<DB, plotters::coord::Shift>,
    result: &GainTimeResult,
    opts: &GainTimePlotOptions,
) -> Result<()>
where
    DB: DrawingBackend,
    DB::ErrorType: 'static,
{
    let area = root;
    area.fill(&WHITE)?;

    let gains_display: Vec<f64> = result
        .curve
        .iter()
        .map(|pt| gain_to_display(pt.gain_m, opts.unit))
        .collect();
    let rates: Vec<f64> = result.curve.iter().map(|pt| pt.avg_rate_m_per_hr).collect();

    let x_max = gains_display.iter().copied().fold(0.0, f64::max).max(1.0);
    let mut y_max = rates.iter().copied().fold(0.0, f64::max).max(1.0);
    if !y_max.is_finite() || y_max <= 0.0 {
        y_max = 1.0;
    }

    let mut chart = ChartBuilder::on(&area)
        .margin(25)
        .set_label_area_size(LabelAreaPosition::Left, 60)
        .set_label_area_size(LabelAreaPosition::Bottom, 50)
        .build_cartesian_2d(0.0..(x_max * 1.05), 0.0..(y_max * 1.1))?;

    chart
        .configure_mesh()
        .x_desc(format!("Gain ({})", display_gain_label(opts.unit)))
        .y_desc("Average rate (m/h)")
        .x_label_formatter(&|v| format!("{:.0}", v))
        .y_label_formatter(&|v| format!("{:.0}", v))
        .label_style(FontDesc::new(
            FontFamily::SansSerif,
            16.0,
            FontStyle::Normal,
        ))
        .draw()?;

    chart
        .draw_series(LineSeries::new(
            gains_display
                .iter()
                .zip(rates.iter())
                .map(|(&g, &r)| (g, r)),
            &RGBColor(0, 114, 178),
        ))?
        .label("Average rate")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RGBColor(0, 114, 178)));

    if opts.show_wr {
        if let Some(curve) = result.wr_curve.as_ref() {
            chart
                .draw_series(LineSeries::new(
                    curve
                        .iter()
                        .map(|pt| (gain_to_display(pt.gain_m, opts.unit), pt.avg_rate_m_per_hr)),
                    &RGBColor(90, 90, 90),
                ))?
                .label("WR rate")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RGBColor(90, 90, 90)));
        }
    }

    if opts.show_personal {
        if let Some(curve) = result.personal_curve.as_ref() {
            chart
                .draw_series(LineSeries::new(
                    curve
                        .iter()
                        .map(|pt| (gain_to_display(pt.gain_m, opts.unit), pt.avg_rate_m_per_hr)),
                    &RGBColor(30, 144, 255),
                ))?
                .label("Personal rate")
                .legend(|(x, y)| {
                    PathElement::new(vec![(x, y), (x + 20, y)], RGBColor(30, 144, 255))
                });
        }
    }

    chart.draw_series(result.targets.iter().filter_map(|pt| {
        if !pt.avg_rate_m_per_hr.is_finite() {
            return None;
        }
        let gain_disp = gain_to_display(pt.gain_m, opts.unit);
        Some(Circle::new(
            (gain_disp, pt.avg_rate_m_per_hr),
            4,
            RGBColor(0, 114, 178).filled(),
        ))
    }))?;

    chart
        .configure_series_labels()
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

#[derive(Clone, Debug)]
struct MagicPoint {
    duration_s: f64,
    user_gain: f64,
}

enum ChartKind {
    Png,
    Svg,
}

fn render_chart_guard(
    curves: &Curves,
    path: &Path,
    kind: ChartKind,
    opts: &PlotOptions,
) -> Result<(), String> {
    let render = || -> Result<(), String> {
        if opts.split_plots {
            render_split_charts(curves, path, kind, opts)
        } else {
            render_combined_chart(curves, path, kind, opts)
                .map_err(|e| format!("plotting error: {}", e))
        }
    };

    panic::catch_unwind(panic::AssertUnwindSafe(render))
        .map_err(|_| "plotting backend panicked".to_string())?
        .map_err(|err| err)
}

fn render_combined_chart(
    curves: &Curves,
    path: &Path,
    kind: ChartKind,
    opts: &PlotOptions,
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
    if opts.show_wr {
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

    if opts.show_personal {
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
                root,
                &durations,
                &gains,
                &rates,
                x_max,
                y_max,
                &overlays,
                sessions,
                opts.fast_plot,
            )?;
        }
        ChartKind::Svg => {
            let backend = SVGBackend::new(path, (1280, 760));
            let root = FontSafeBackend::new(backend).into_drawing_area();
            draw_chart(
                root,
                &durations,
                &gains,
                &rates,
                x_max,
                y_max,
                &overlays,
                sessions,
                opts.fast_plot,
            )?;
        }
    }

    Ok(())
}

fn collect_magic_points(curves: &Curves) -> Vec<MagicPoint> {
    let mut points = Vec::new();
    if let Some(rows) = curves.magic_rows.as_ref() {
        for row in rows {
            let duration = row.get("duration_s").copied().unwrap_or(0.0);
            let user_gain = row.get("user_gain_m").copied().unwrap_or(0.0);
            if duration <= 0.0 || user_gain <= 0.0 {
                continue;
            }
            points.push(MagicPoint {
                duration_s: duration,
                user_gain,
            });
        }
    }
    points.sort_by(|a, b| {
        a.duration_s
            .partial_cmp(&b.duration_s)
            .unwrap_or(Ordering::Equal)
    });
    points
}

fn render_split_charts(
    curves: &Curves,
    base_path: &Path,
    kind: ChartKind,
    opts: &PlotOptions,
) -> Result<(), String> {
    if curves.points.is_empty() {
        return Ok(());
    }

    let (rate_path, climb_path) = derive_split_paths(base_path);

    match kind {
        ChartKind::Png => {
            let backend_rate = BitMapBackend::new(&rate_path, (1280, 720));
            let root_rate = FontSafeBackend::new(backend_rate).into_drawing_area();
            draw_rate_chart(root_rate, curves, opts)
                .map_err(|e| format!("plotting error: {}", e))?;

            let backend_climb = BitMapBackend::new(&climb_path, (1280, 720));
            let root_climb = FontSafeBackend::new(backend_climb).into_drawing_area();
            draw_climb_chart(root_climb, curves, opts)
                .map_err(|e| format!("plotting error: {}", e))?;
        }
        ChartKind::Svg => {
            let backend_rate = SVGBackend::new(&rate_path, (1280, 720));
            let root_rate = FontSafeBackend::new(backend_rate).into_drawing_area();
            draw_rate_chart(root_rate, curves, opts)
                .map_err(|e| format!("plotting error: {}", e))?;

            let backend_climb = SVGBackend::new(&climb_path, (1280, 720));
            let root_climb = FontSafeBackend::new(backend_climb).into_drawing_area();
            draw_climb_chart(root_climb, curves, opts)
                .map_err(|e| format!("plotting error: {}", e))?;
        }
    }

    Ok(())
}

fn derive_split_paths(base: &Path) -> (PathBuf, PathBuf) {
    let stem = base.file_stem().and_then(|s| s.to_str()).unwrap_or("curve");
    let ext = base.extension().and_then(|s| s.to_str()).unwrap_or("png");
    let rate_name = format!("{}_rate.{}", stem, ext);
    let climb_name = format!("{}_climb.{}", stem, ext);
    let rate_path = base.with_file_name(rate_name);
    let climb_path = base.with_file_name(climb_name);
    (rate_path, climb_path)
}

fn draw_rate_chart<DB>(
    root: DrawingArea<DB, plotters::coord::Shift>,
    curves: &Curves,
    opts: &PlotOptions,
) -> Result<()>
where
    DB: DrawingBackend,
    DB::ErrorType: 'static,
{
    let durations: Vec<f64> = curves
        .points
        .iter()
        .map(|p| p.duration_s as f64 / 60.0)
        .collect();
    let rates: Vec<f64> = curves
        .points
        .iter()
        .map(|p| p.climb_rate_m_per_hr)
        .collect();

    let x_max = durations.iter().copied().fold(1.0, f64::max);
    let mut y_max = rates.iter().copied().fold(1.0, f64::max);
    if !y_max.is_finite() || y_max <= 0.0 {
        y_max = 1.0;
    }

    let magic = if opts.fast_plot {
        Vec::new()
    } else {
        collect_magic_points(curves)
    };

    let area = root;
    area.fill(&WHITE)?;

    if opts.ylog_rate {
        let mut min_pos = rates
            .iter()
            .copied()
            .filter(|v| *v > 0.0)
            .fold(f64::INFINITY, f64::min);
        if !min_pos.is_finite() {
            min_pos = 1e-2;
        } else {
            min_pos = min_pos.max(1e-2);
        }
        let mut chart = ChartBuilder::on(&area)
            .margin(25)
            .set_label_area_size(LabelAreaPosition::Left, 60)
            .set_label_area_size(LabelAreaPosition::Bottom, 40)
            .build_cartesian_2d(0.0..x_max, (min_pos..(y_max * 1.1)).log_scale())?;

        chart
            .configure_mesh()
            .x_label_formatter(&|v| format!("{:.0}", v))
            .y_label_formatter(&|v| format!("{:.0}", v))
            .label_style(FontDesc::new(
                FontFamily::SansSerif,
                18.0,
                FontStyle::Normal,
            ))
            .draw()?;

        // Draw main rate series (log Y)
        chart
            .draw_series(LineSeries::new(
                durations.iter().copied().zip(rates.iter().copied()),
                &RGBColor(200, 0, 100),
            ))?
            .label("Climb rate (m/h)")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 30, y)], &RGBColor(200, 0, 100)));

        // WR overlay
        if opts.show_wr {
            if let Some((wr_durs, wr_climbs)) = curves.wr_curve.as_ref() {
                let wr_minutes: Vec<f64> = wr_durs.iter().map(|d| *d as f64 / 60.0).collect();
                let wr_rates: Vec<f64> = if let Some(rates) = curves.wr_rates.as_ref() {
                    rates.clone()
                } else {
                    wr_durs
                        .iter()
                        .zip(wr_climbs.iter())
                        .map(|(&d, &g)| if d == 0 { 0.0 } else { g * 3600.0 / d as f64 })
                        .collect()
                };
                chart
                    .draw_series(LineSeries::new(
                        wr_minutes.into_iter().zip(wr_rates.into_iter()),
                        RGBColor(90, 90, 90),
                    ))?
                    .label("WR rate")
                    .legend(|(x, y)| {
                        PathElement::new(vec![(x, y), (x + 30, y)], &RGBColor(90, 90, 90))
                    });
            }
        }

        // Personal/Goal overlays
        if opts.show_personal {
            if let Some((durs, gains)) = curves.personal_curve.as_ref() {
                let series: Vec<(f64, f64)> = durs
                    .iter()
                    .zip(gains.iter())
                    .filter_map(|(&d, &g)| {
                        if d == 0 {
                            None
                        } else {
                            Some((d as f64 / 60.0, g * 3600.0 / d as f64))
                        }
                    })
                    .collect();
                if !series.is_empty() {
                    chart
                        .draw_series(LineSeries::new(series.into_iter(), RGBColor(30, 144, 255)))?
                        .label("Personal rate")
                        .legend(|(x, y)| {
                            PathElement::new(vec![(x, y), (x + 30, y)], &RGBColor(30, 144, 255))
                        });
                }
            }
            if let Some((durs, gains)) = curves.goal_curve.as_ref() {
                let series: Vec<(f64, f64)> = durs
                    .iter()
                    .zip(gains.iter())
                    .filter_map(|(&d, &g)| {
                        if d == 0 {
                            None
                        } else {
                            Some((d as f64 / 60.0, g * 3600.0 / d as f64))
                        }
                    })
                    .collect();
                if !series.is_empty() {
                    chart
                        .draw_series(LineSeries::new(series.into_iter(), RGBColor(34, 139, 34)))?
                        .label("Goal rate")
                        .legend(|(x, y)| {
                            PathElement::new(vec![(x, y), (x + 30, y)], &RGBColor(34, 139, 34))
                        });
                }
            }
        }

        // Magic annotations
        if !magic.is_empty() {
            let y_hint = rates.iter().cloned().fold(1.0_f64, f64::max);
            let dy = (y_hint * 0.02).max(1.0);
            for point in &magic {
                let rate = point.user_gain * 3600.0 / point.duration_s.max(1.0);
                let x = point.duration_s / 60.0;
                let y = rate;
                chart.draw_series(std::iter::once(Circle::new(
                    (x, y),
                    4,
                    RGBColor(0, 0, 0).filled(),
                )))?;
                chart.draw_series(std::iter::once(Text::new(
                    format!("{:.0} m/h", rate),
                    (x, y + dy),
                    FontDesc::new(FontFamily::SansSerif, 14.0, FontStyle::Normal).color(&BLACK),
                )))?;
            }
        }

        if !opts.fast_plot {
            chart
                .configure_series_labels()
                .background_style(&WHITE.mix(0.7))
                .border_style(&BLACK.mix(0.3))
                .label_font(FontDesc::new(
                    FontFamily::SansSerif,
                    16.0,
                    FontStyle::Normal,
                ))
                .position(SeriesLabelPosition::UpperLeft)
                .draw()?;
        }
    } else {
        let mut chart = ChartBuilder::on(&area)
            .margin(25)
            .set_label_area_size(LabelAreaPosition::Left, 60)
            .set_label_area_size(LabelAreaPosition::Bottom, 40)
            .build_cartesian_2d(0.0..x_max, 0.0..(y_max * 1.1))?;

        chart
            .configure_mesh()
            .x_label_formatter(&|v| format!("{:.0}", v))
            .y_label_formatter(&|v| format!("{:.0}", v))
            .label_style(FontDesc::new(
                FontFamily::SansSerif,
                18.0,
                FontStyle::Normal,
            ))
            .draw()?;

        // Draw main rate series (linear Y)
        chart
            .draw_series(LineSeries::new(
                durations.iter().copied().zip(rates.iter().copied()),
                &RGBColor(200, 0, 100),
            ))?
            .label("Climb rate (m/h)")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 30, y)], &RGBColor(200, 0, 100)));

        // WR overlay
        if opts.show_wr {
            if let Some((wr_durs, wr_climbs)) = curves.wr_curve.as_ref() {
                let wr_minutes: Vec<f64> = wr_durs.iter().map(|d| *d as f64 / 60.0).collect();
                let wr_rates: Vec<f64> = if let Some(rates) = curves.wr_rates.as_ref() {
                    rates.clone()
                } else {
                    wr_durs
                        .iter()
                        .zip(wr_climbs.iter())
                        .map(|(&d, &g)| if d == 0 { 0.0 } else { g * 3600.0 / d as f64 })
                        .collect()
                };
                chart
                    .draw_series(LineSeries::new(
                        wr_minutes.into_iter().zip(wr_rates.into_iter()),
                        RGBColor(90, 90, 90),
                    ))?
                    .label("WR rate")
                    .legend(|(x, y)| {
                        PathElement::new(vec![(x, y), (x + 30, y)], &RGBColor(90, 90, 90))
                    });
            }
        }

        // Personal/Goal overlays
        if opts.show_personal {
            if let Some((durs, gains)) = curves.personal_curve.as_ref() {
                let series: Vec<(f64, f64)> = durs
                    .iter()
                    .zip(gains.iter())
                    .filter_map(|(&d, &g)| {
                        if d == 0 {
                            None
                        } else {
                            Some((d as f64 / 60.0, g * 3600.0 / d as f64))
                        }
                    })
                    .collect();
                if !series.is_empty() {
                    chart
                        .draw_series(LineSeries::new(series.into_iter(), RGBColor(30, 144, 255)))?
                        .label("Personal rate")
                        .legend(|(x, y)| {
                            PathElement::new(vec![(x, y), (x + 30, y)], &RGBColor(30, 144, 255))
                        });
                }
            }
            if let Some((durs, gains)) = curves.goal_curve.as_ref() {
                let series: Vec<(f64, f64)> = durs
                    .iter()
                    .zip(gains.iter())
                    .filter_map(|(&d, &g)| {
                        if d == 0 {
                            None
                        } else {
                            Some((d as f64 / 60.0, g * 3600.0 / d as f64))
                        }
                    })
                    .collect();
                if !series.is_empty() {
                    chart
                        .draw_series(LineSeries::new(series.into_iter(), RGBColor(34, 139, 34)))?
                        .label("Goal rate")
                        .legend(|(x, y)| {
                            PathElement::new(vec![(x, y), (x + 30, y)], &RGBColor(34, 139, 34))
                        });
                }
            }
        }

        // Magic annotations
        if !magic.is_empty() {
            let y_hint = rates.iter().cloned().fold(1.0_f64, f64::max);
            let dy = (y_hint * 0.02).max(1.0);
            for point in &magic {
                let rate = point.user_gain * 3600.0 / point.duration_s.max(1.0);
                let x = point.duration_s / 60.0;
                let y = rate;
                chart.draw_series(std::iter::once(Circle::new(
                    (x, y),
                    4,
                    RGBColor(0, 0, 0).filled(),
                )))?;
                chart.draw_series(std::iter::once(Text::new(
                    format!("{:.0} m/h", rate),
                    (x, y + dy),
                    FontDesc::new(FontFamily::SansSerif, 14.0, FontStyle::Normal).color(&BLACK),
                )))?;
            }
        }

        if !opts.fast_plot {
            chart
                .configure_series_labels()
                .background_style(&WHITE.mix(0.7))
                .border_style(&BLACK.mix(0.3))
                .label_font(FontDesc::new(
                    FontFamily::SansSerif,
                    16.0,
                    FontStyle::Normal,
                ))
                .position(SeriesLabelPosition::UpperLeft)
                .draw()?;
        }
    }

    area.present()?;
    Ok(())
}

fn draw_climb_chart<DB>(
    root: DrawingArea<DB, plotters::coord::Shift>,
    curves: &Curves,
    opts: &PlotOptions,
) -> Result<()>
where
    DB: DrawingBackend,
    DB::ErrorType: 'static,
{
    let durations: Vec<f64> = curves
        .points
        .iter()
        .map(|p| p.duration_s as f64 / 60.0)
        .collect();
    let gains: Vec<f64> = curves.points.iter().map(|p| p.max_climb_m).collect();

    let x_max = durations.iter().copied().fold(1.0, f64::max);
    let mut y_max = gains.iter().copied().fold(1.0, f64::max);
    if !y_max.is_finite() || y_max <= 0.0 {
        y_max = 1.0;
    }

    let magic = if opts.fast_plot {
        Vec::new()
    } else {
        collect_magic_points(curves)
    };

    let area = root;
    area.fill(&WHITE)?;

    if opts.ylog_climb {
        let mut min_pos = gains
            .iter()
            .copied()
            .filter(|v| *v > 0.0)
            .fold(f64::INFINITY, f64::min);
        if !min_pos.is_finite() {
            min_pos = 1e-2;
        } else {
            min_pos = min_pos.max(1e-2);
        }
        let mut chart = ChartBuilder::on(&area)
            .margin(25)
            .set_label_area_size(LabelAreaPosition::Left, 60)
            .set_label_area_size(LabelAreaPosition::Bottom, 40)
            .build_cartesian_2d(0.0..x_max, (min_pos..(y_max * 1.15)).log_scale())?;

        chart
            .configure_mesh()
            .x_label_formatter(&|v| format!("{:.0}", v))
            .y_label_formatter(&|v| format!("{:.0}", v))
            .label_style(FontDesc::new(
                FontFamily::SansSerif,
                18.0,
                FontStyle::Normal,
            ))
            .draw()?;

        // Draw main climb series (log Y)
        chart
            .draw_series(LineSeries::new(
                durations.iter().copied().zip(gains.iter().copied()),
                &RGBColor(50, 50, 50),
            ))?
            .label("Climb (m)")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 30, y)], &RGBColor(50, 50, 50)));

        if opts.show_wr {
            if let Some((wr_durs, wr_gains)) = curves.wr_curve.as_ref() {
                let wr_minutes: Vec<f64> = wr_durs.iter().map(|d| *d as f64 / 60.0).collect();
                chart
                    .draw_series(LineSeries::new(
                        wr_minutes.into_iter().zip(wr_gains.iter().copied()),
                        RGBColor(90, 90, 90),
                    ))?
                    .label("WR climb")
                    .legend(|(x, y)| {
                        PathElement::new(vec![(x, y), (x + 30, y)], &RGBColor(90, 90, 90))
                    });
            }
        }

        if opts.show_personal {
            if let Some((durs, gains_p)) = curves.personal_curve.as_ref() {
                chart
                    .draw_series(LineSeries::new(
                        durs.iter()
                            .map(|d| *d as f64 / 60.0)
                            .zip(gains_p.iter().copied()),
                        RGBColor(30, 144, 255),
                    ))?
                    .label("Personal climb")
                    .legend(|(x, y)| {
                        PathElement::new(vec![(x, y), (x + 30, y)], &RGBColor(30, 144, 255))
                    });
            }
            if let Some((durs, gains_g)) = curves.goal_curve.as_ref() {
                chart
                    .draw_series(LineSeries::new(
                        durs.iter()
                            .map(|d| *d as f64 / 60.0)
                            .zip(gains_g.iter().copied()),
                        RGBColor(34, 139, 34),
                    ))?
                    .label("Goal climb")
                    .legend(|(x, y)| {
                        PathElement::new(vec![(x, y), (x + 30, y)], &RGBColor(34, 139, 34))
                    });
            }
        }

        if !magic.is_empty() {
            let y_hint = gains.iter().cloned().fold(1.0_f64, f64::max);
            let dy = (y_hint * 0.02).max(1.0);
            for point in &magic {
                let x = point.duration_s / 60.0;
                let y = point.user_gain;
                chart.draw_series(std::iter::once(Circle::new(
                    (x, y),
                    4,
                    RGBColor(0, 0, 0).filled(),
                )))?;
                chart.draw_series(std::iter::once(Text::new(
                    format!("{:.0} m", point.user_gain),
                    (x, y + dy),
                    FontDesc::new(FontFamily::SansSerif, 14.0, FontStyle::Normal).color(&BLACK),
                )))?;
            }
        }

        if !opts.fast_plot {
            chart
                .configure_series_labels()
                .background_style(&WHITE.mix(0.7))
                .border_style(&BLACK.mix(0.3))
                .label_font(FontDesc::new(
                    FontFamily::SansSerif,
                    16.0,
                    FontStyle::Normal,
                ))
                .position(SeriesLabelPosition::UpperLeft)
                .draw()?;
        }
    } else {
        let mut chart = ChartBuilder::on(&area)
            .margin(25)
            .set_label_area_size(LabelAreaPosition::Left, 60)
            .set_label_area_size(LabelAreaPosition::Bottom, 40)
            .build_cartesian_2d(0.0..x_max, 0.0..(y_max * 1.15))?;

        chart
            .configure_mesh()
            .x_label_formatter(&|v| format!("{:.0}", v))
            .y_label_formatter(&|v| format!("{:.0}", v))
            .label_style(FontDesc::new(
                FontFamily::SansSerif,
                18.0,
                FontStyle::Normal,
            ))
            .draw()?;

        // Draw main climb series (linear Y)
        chart
            .draw_series(LineSeries::new(
                durations.iter().copied().zip(gains.iter().copied()),
                &RGBColor(50, 50, 50),
            ))?
            .label("Climb (m)")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 30, y)], &RGBColor(50, 50, 50)));

        if opts.show_wr {
            if let Some((wr_durs, wr_gains)) = curves.wr_curve.as_ref() {
                let wr_minutes: Vec<f64> = wr_durs.iter().map(|d| *d as f64 / 60.0).collect();
                chart
                    .draw_series(LineSeries::new(
                        wr_minutes.into_iter().zip(wr_gains.iter().copied()),
                        RGBColor(90, 90, 90),
                    ))?
                    .label("WR climb")
                    .legend(|(x, y)| {
                        PathElement::new(vec![(x, y), (x + 30, y)], &RGBColor(90, 90, 90))
                    });
            }
        }

        if opts.show_personal {
            if let Some((durs, gains_p)) = curves.personal_curve.as_ref() {
                chart
                    .draw_series(LineSeries::new(
                        durs.iter()
                            .map(|d| *d as f64 / 60.0)
                            .zip(gains_p.iter().copied()),
                        RGBColor(30, 144, 255),
                    ))?
                    .label("Personal climb")
                    .legend(|(x, y)| {
                        PathElement::new(vec![(x, y), (x + 30, y)], &RGBColor(30, 144, 255))
                    });
            }
            if let Some((durs, gains_g)) = curves.goal_curve.as_ref() {
                chart
                    .draw_series(LineSeries::new(
                        durs.iter()
                            .map(|d| *d as f64 / 60.0)
                            .zip(gains_g.iter().copied()),
                        RGBColor(34, 139, 34),
                    ))?
                    .label("Goal climb")
                    .legend(|(x, y)| {
                        PathElement::new(vec![(x, y), (x + 30, y)], &RGBColor(34, 139, 34))
                    });
            }
        }

        if !magic.is_empty() {
            let y_hint = gains.iter().cloned().fold(1.0_f64, f64::max);
            let dy = (y_hint * 0.02).max(1.0);
            for point in &magic {
                let x = point.duration_s / 60.0;
                let y = point.user_gain;
                chart.draw_series(std::iter::once(Circle::new(
                    (x, y),
                    4,
                    RGBColor(0, 0, 0).filled(),
                )))?;
                chart.draw_series(std::iter::once(Text::new(
                    format!("{:.0} m", point.user_gain),
                    (x, y + dy),
                    FontDesc::new(FontFamily::SansSerif, 14.0, FontStyle::Normal).color(&BLACK),
                )))?;
            }
        }

        if !opts.fast_plot {
            chart
                .configure_series_labels()
                .background_style(&WHITE.mix(0.7))
                .border_style(&BLACK.mix(0.3))
                .label_font(FontDesc::new(
                    FontFamily::SansSerif,
                    16.0,
                    FontStyle::Normal,
                ))
                .position(SeriesLabelPosition::UpperLeft)
                .draw()?;
        }
    }

    area.present()?;
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
    fast_plot: bool,
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

    if !sessions.is_empty() && !fast_plot {
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

    if !fast_plot {
        chart
            .configure_series_labels()
            .background_style(&WHITE.mix(0.7))
            .border_style(&BLACK.mix(0.3))
            .label_font(legend_font.color(&BLACK))
            .position(SeriesLabelPosition::UpperLeft)
            .draw()?;
    }

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
