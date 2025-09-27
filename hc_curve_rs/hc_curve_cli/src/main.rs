use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{ArgGroup, Parser};
use hc_curve::{compute_curves, parse_records, CurvePoint, Params, Source};
use plotters::prelude::*;
use tracing_subscriber::EnvFilter;

#[derive(Parser, Debug)]
#[command(author, version, about = "Hillclimb curve computation CLI", long_about = None)]
#[command(group(ArgGroup::new("plots").args(["png", "svg"])))]
struct Args {
    /// FIT/GPX files to ingest
    #[arg(required = true)]
    inputs: Vec<PathBuf>,

    /// Output CSV path (stdout if omitted)
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Output PNG figure path
    #[arg(long)]
    png: Option<PathBuf>,

    /// Output SVG figure path
    #[arg(long)]
    svg: Option<PathBuf>,

    /// Disable QC censoring
    #[arg(long)]
    no_qc: bool,

    /// Disable 1Hz resampling
    #[arg(long)]
    raw_sampling: bool,

    /// Gain epsilon in meters
    #[arg(long, default_value_t = 0.5)]
    gain_eps: f64,

    /// Smoothing window in seconds
    #[arg(long, default_value_t = 0.0)]
    smooth: f64,

    /// Explicit durations (comma separated seconds)
    #[arg(long)]
    durations: Option<String>,

    /// Maximum duration for exhaustive search
    #[arg(long)]
    max_duration: Option<u64>,

    /// Step size for exhaustive search
    #[arg(long, default_value_t = 1)]
    step: u64,

    /// Session gap in seconds
    #[arg(long, default_value_t = 600.0)]
    session_gap: f64,

    /// Source override (auto|runn|altitude)
    #[arg(long, default_value = "auto")]
    source: String,
}

fn main() -> Result<()> {
    let _guard = tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(io::stderr)
        .try_init();

    let args = Args::parse();

    let mut params = Params::default();
    params.gain_eps_m = args.gain_eps;
    params.smooth_sec = args.smooth;
    params.max_duration_s = args.max_duration;
    params.step_s = args.step.max(1);
    params.session_gap_sec = args.session_gap;
    params.qc_enabled = !args.no_qc;
    params.resample_1hz = !args.raw_sampling;
    params.source = match args.source.as_str() {
        "auto" => Source::Auto,
        "runn" => Source::Runn,
        "altitude" => Source::Altitude,
        other => {
            eprintln!("Unknown source '{}', defaulting to auto", other);
            Source::Auto
        }
    };
    if let Some(durations) = args.durations.as_ref() {
        params.durations = durations
            .split(',')
            .filter_map(|s| s.trim().parse::<u64>().ok())
            .collect();
        params.exhaustive = false;
    }

    let mut records = Vec::new();
    for (file_id, path) in args.inputs.iter().enumerate() {
        let data =
            std::fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
        let hint = path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("fit");
        let parsed = parse_records(&data, file_id, hint)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        records.push(parsed);
    }

    let curves = compute_curves(records, &params)?;

    write_csv(&curves.points, args.output.as_deref())?;

    if let Some(path) = args.png.as_ref() {
        render_chart(&curves.points, path, ChartKind::Png)?;
    }
    if let Some(path) = args.svg.as_ref() {
        render_chart(&curves.points, path, ChartKind::Svg)?;
    }

    Ok(())
}

fn write_csv(points: &[CurvePoint], dest: Option<&Path>) -> Result<()> {
    #[derive(serde::Serialize)]
    struct Row<'a> {
        duration_s: u64,
        max_climb_m: f64,
        climb_rate_m_per_hr: f64,
        start_offset_s: f64,
        end_offset_s: f64,
        #[serde(skip_serializing)]
        _marker: std::marker::PhantomData<&'a ()>,
    }

    if let Some(path) = dest {
        let file =
            File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
        let mut writer = csv::Writer::from_writer(file);
        for point in points {
            writer.serialize(Row {
                duration_s: point.duration_s,
                max_climb_m: point.max_climb_m,
                climb_rate_m_per_hr: point.climb_rate_m_per_hr,
                start_offset_s: point.start_offset_s,
                end_offset_s: point.end_offset_s,
                _marker: std::marker::PhantomData,
            })?;
        }
        writer.flush()?;
    } else {
        let mut stdout = csv::Writer::from_writer(io::stdout());
        for point in points {
            stdout.serialize((
                &point.duration_s,
                &point.max_climb_m,
                &point.climb_rate_m_per_hr,
            ))?;
        }
        stdout.flush()?;
    }
    Ok(())
}

enum ChartKind {
    Png,
    Svg,
}

fn render_chart(points: &[CurvePoint], path: &Path, kind: ChartKind) -> Result<()> {
    if points.is_empty() {
        return Ok(());
    }

    let durations: Vec<f64> = points.iter().map(|p| p.duration_s as f64 / 60.0).collect();
    let gains: Vec<f64> = points.iter().map(|p| p.max_climb_m).collect();
    let rate: Vec<f64> = points.iter().map(|p| p.climb_rate_m_per_hr).collect();

    let x_max = durations.iter().copied().fold(f64::MIN, f64::max).max(1.0);
    let y_max = gains.iter().copied().fold(f64::MIN, f64::max).max(1.0);

    match kind {
        ChartKind::Png => {
            let path_buf = path.to_path_buf();
            let root = BitMapBackend::new(&path_buf, (1200, 720)).into_drawing_area();
            draw_chart(root, &durations, &gains, &rate, x_max, y_max)?;
        }
        ChartKind::Svg => {
            let path_buf = path.to_path_buf();
            let root = SVGBackend::new(&path_buf, (1200, 720)).into_drawing_area();
            draw_chart(root, &durations, &gains, &rate, x_max, y_max)?;
        }
    }
    Ok(())
}

fn draw_chart<DB>(
    root: DrawingArea<DB, plotters::coord::Shift>,
    durations: &[f64],
    gains: &[f64],
    rate: &[f64],
    x_max: f64,
    y_max: f64,
) -> Result<()>
where
    DB: DrawingBackend,
    DB::ErrorType: 'static,
{
    let area = root;
    area.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&area)
        .margin(20)
        .set_label_area_size(LabelAreaPosition::Left, 40)
        .set_label_area_size(LabelAreaPosition::Bottom, 40)
        .build_cartesian_2d(0.0..x_max, 0.0..y_max)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .bold_line_style(&TRANSPARENT)
        .light_line_style(&TRANSPARENT)
        .axis_style(&BLACK.mix(0.8))
        .label_style(TextStyle::from(("Helvetica", 20).into_font()).color(&BLACK.mix(0.8)))
        .x_label_formatter(&|v| format!("{:.0}", v))
        .y_label_formatter(&|v| format!("{:.0}", v))
        .draw()?;

    let style_gain = ShapeStyle {
        color: RGBColor(34, 34, 34).to_rgba(),
        filled: false,
        stroke_width: 2,
    };

    chart
        .draw_series(plotters::prelude::LineSeries::new(
            durations.iter().copied().zip(gains.iter().copied()),
            style_gain,
        ))?
        .label("Climb (m)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RGBColor(34, 34, 34)));

    chart.draw_series(plotters::prelude::LineSeries::new(
        durations
            .iter()
            .copied()
            .zip(rate.iter().map(|v| v / 100.0)),
        &RGBColor(200, 0, 100),
    ))?;

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .margin(10)
        .background_style(&WHITE.mix(0.1))
        .label_font(TextStyle::from(("Helvetica", 18).into_font()).color(&BLACK))
        .draw()?;

    area.present()?;
    Ok(())
}
