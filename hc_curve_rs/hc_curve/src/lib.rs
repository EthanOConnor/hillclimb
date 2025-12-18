//! Core hillclimb curve computation library implemented in Rust.

use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet, VecDeque};
use std::path::PathBuf;

use chrono::{DateTime, Utc};
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use thiserror::Error;

mod wr;
use wr::build_wr_envelope;

mod ascent;
pub use ascent::{
    compute_ascent_algorithm, compute_ascent_compare, list_ascent_algorithms, AscentAlgorithmConfig,
    AscentAlgorithmInfo, AscentAlgorithmResult, AscentCompareAlgorithmEntry, AscentCompareReport,
    AscentDiagnostics, AscentRequirement, AscentSeries,
};

const DEFAULT_DURATIONS: &[u64] = &[60, 120, 300, 600, 1_200, 1_800, 3_600];

const DEFAULT_GAIN_TARGETS: &[f64] = &[50.0, 100.0, 150.0, 200.0, 300.0, 500.0, 750.0, 1_000.0];

const DEFAULT_MAGIC_TOKENS: &[&str] = &[
    "60s", "300s", "600s", "1800s", "3600s", "0.481h", "7200s", "21600s", "43200s",
];

const QC_DEFAULT_SPEC: &[(f64, f64)] = &[
    (5.0, 8.0),
    (10.0, 12.0),
    (30.0, 25.0),
    (60.0, 40.0),
    (300.0, 150.0),
    (600.0, 250.0),
    (1_800.0, 500.0),
    (3_600.0, 900.0),
];

pub fn parse_duration_token(token: &str) -> Option<f64> {
    let trimmed = token.trim().to_ascii_lowercase();
    if trimmed.is_empty() {
        return None;
    }
    if let Some(stripped) = trimmed.strip_suffix('h') {
        stripped.parse::<f64>().ok().map(|v| v * 3_600.0)
    } else if let Some(stripped) = trimmed.strip_suffix('m') {
        stripped.parse::<f64>().ok().map(|v| v * 60.0)
    } else if let Some(stripped) = trimmed.strip_suffix('s') {
        stripped.parse::<f64>().ok()
    } else {
        trimmed.parse::<f64>().ok()
    }
}

fn default_magic_durations() -> Vec<u64> {
    DEFAULT_MAGIC_TOKENS
        .iter()
        .filter_map(|token| parse_duration_token(token).map(|v| v.max(0.0)))
        .map(|seconds| seconds.round() as u64)
        .collect()
}

#[derive(Error, Debug)]
pub enum HcError {
    #[error("unsupported file format: {0}")]
    UnsupportedFormat(String),
    #[error("failed to parse FIT file: {0}")]
    FitParse(String),
    #[error("failed to parse GPX file: {0}")]
    GpxParse(String),
    #[error("insufficient data for curve computation")]
    InsufficientData,
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("world-record anchors unavailable")]
    MissingWorldRecord,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Source {
    Auto,
    Runn,
    Altitude,
}

impl Default for Source {
    fn default() -> Self {
        Source::Auto
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum Engine {
    Auto,
    NumpyStyle,
    NumbaStyle,
    Stride,
}

impl Default for Engine {
    fn default() -> Self {
        Engine::Auto
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Params {
    pub source: Source,
    pub gain_eps_m: f64,
    pub idle_speed_mps: f64,
    pub smooth_sec: f64,
    pub durations: Vec<u64>,
    pub exhaustive: bool,
    pub all_windows: bool,
    pub step_s: u64,
    pub max_duration_s: Option<u64>,
    pub session_gap_sec: f64,
    pub merge_eps_sec: f64,
    pub overlap_policy: String,
    pub resample_1hz: bool,
    pub resample_max_gap_sec: f64,
    pub resample_max_points: usize,
    pub qc_enabled: bool,
    pub qc_spec: Option<HashMap<OrderedFloat<f64>, f64>>,
    pub wr_profile: String,
    pub wr_anchors_path: Option<PathBuf>,
    pub wr_min_seconds: f64,
    pub wr_short_cap: String,
    pub magic: Option<Vec<String>>,
    pub magic_durations: Vec<u64>,
    pub goals_topk: usize,
    pub goal_min_seconds: f64,
    pub personal_min_seconds: f64,
    pub concave_envelope: bool,
    pub engine: Engine,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            source: Source::Auto,
            gain_eps_m: 0.5,
            idle_speed_mps: 0.15,
            smooth_sec: 0.0,
            durations: Vec::new(),
            exhaustive: true,
            all_windows: false,
            step_s: 1,
            max_duration_s: None,
            session_gap_sec: 600.0,
            merge_eps_sec: 0.5,
            overlap_policy: "file:last".to_string(),
            resample_1hz: true,
            resample_max_gap_sec: 2.0 * 3600.0,
            resample_max_points: 500_000,
            qc_enabled: true,
            qc_spec: None,
            wr_profile: "overall".to_string(),
            wr_anchors_path: None,
            wr_min_seconds: 30.0,
            wr_short_cap: "standard".to_string(),
            magic: None,
            magic_durations: default_magic_durations(),
            goals_topk: 3,
            goal_min_seconds: 120.0,
            personal_min_seconds: 60.0,
            concave_envelope: true,
            engine: Engine::Auto,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CurvePoint {
    pub duration_s: u64,
    pub max_climb_m: f64,
    pub climb_rate_m_per_hr: f64,
    pub start_offset_s: f64,
    pub end_offset_s: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionCurve {
    pub durations: Vec<u64>,
    pub climbs: Vec<f64>,
    pub span: f64,
    pub order: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GainTimePoint {
    pub gain_m: f64,
    pub min_time_s: f64,
    pub avg_rate_m_per_hr: f64,
    pub start_offset_s: Option<f64>,
    pub end_offset_s: Option<f64>,
    pub note: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GainTimeResult {
    pub curve: Vec<GainTimePoint>,
    pub targets: Vec<GainTimePoint>,
    pub wr_curve: Option<Vec<GainTimePoint>>,
    pub personal_curve: Option<Vec<GainTimePoint>>,
    pub selected_source: String,
    pub total_span_s: f64,
    pub total_gain_m: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Curves {
    pub points: Vec<CurvePoint>,
    pub wr_curve: Option<(Vec<u64>, Vec<f64>)>,
    pub wr_rates: Option<Vec<f64>>,
    pub personal_curve: Option<(Vec<u64>, Vec<f64>)>,
    pub goal_curve: Option<(Vec<u64>, Vec<f64>)>,
    pub envelope_curve: Option<(Vec<u64>, Vec<f64>)>,
    pub session_curves: Vec<SessionCurve>,
    pub selected_source: String,
    pub magic_rows: Option<Vec<HashMap<String, f64>>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TimeseriesExport {
    pub times: Vec<f64>,
    pub gain: Vec<f64>,
    pub selected_source: String,
}

impl Default for Curves {
    fn default() -> Self {
        Self {
            points: Vec::new(),
            wr_curve: None,
            wr_rates: None,
            personal_curve: None,
            goal_curve: None,
            envelope_curve: None,
            session_curves: Vec::new(),
            selected_source: String::new(),
            magic_rows: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitRecord {
    pub timestamp_s: f64,
    pub file_id: usize,
    pub alt: Option<f64>,
    pub tg: Option<f64>,
    pub inc: Option<f64>,
    pub dist: Option<f64>,
    pub dist_prio: u8,
    pub speed: Option<f64>,
    pub cad: Option<f64>,
}

impl FitRecord {
    fn new(timestamp_s: f64, file_id: usize) -> Self {
        Self {
            timestamp_s,
            file_id,
            alt: None,
            tg: None,
            inc: None,
            dist: None,
            dist_prio: 0,
            speed: None,
            cad: None,
        }
    }
}

#[derive(Clone, Debug)]
struct MergedRecord {
    time_s: f64,
    file_id: usize,
    alt: Option<f64>,
    tg: Option<f64>,
    inc: Option<f64>,
    dist: Option<f64>,
    dist_prio: u8,
    speed: Option<f64>,
    cad: Option<f64>,
}

impl From<&FitRecord> for MergedRecord {
    fn from(rec: &FitRecord) -> Self {
        Self {
            time_s: rec.timestamp_s,
            file_id: rec.file_id,
            alt: rec.alt,
            tg: rec.tg,
            inc: rec.inc,
            dist: rec.dist,
            dist_prio: rec.dist_prio,
            speed: rec.speed,
            cad: rec.cad,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Gap {
    pub start: f64,
    pub end: f64,
    pub length: f64,
}

/// Parse FIT or GPX records from bytes using the provided format hint (extension).
pub fn parse_records(
    input: &[u8],
    file_id: usize,
    format: &str,
) -> Result<Vec<FitRecord>, HcError> {
    let format_lc = format.to_ascii_lowercase();
    if format_lc.ends_with(".fit") || format_lc == "fit" {
        parse_fit_records(input, file_id)
    } else if format_lc.ends_with(".gpx") || format_lc == "gpx" {
        parse_gpx_records(input, file_id)
    } else {
        Err(HcError::UnsupportedFormat(format.to_string()))
    }
}

fn parse_fit_records(input: &[u8], file_id: usize) -> Result<Vec<FitRecord>, HcError> {
    use fitparser::de::from_bytes;
    use fitparser::profile::MesgNum;
    let records = from_bytes(input).map_err(|e| HcError::FitParse(e.to_string()))?;
    let mut out = Vec::new();

    for record in records.into_iter() {
        if record.kind() != MesgNum::Record {
            continue;
        }
        let mut timestamp: Option<DateTime<Utc>> = None;
        let mut alt: Option<f64> = None;
        let mut tg: Option<f64> = None;
        let mut inc: Option<f64> = None;
        let mut dist: Option<f64> = None;
        let mut dist_prio: u8 = 0;
        let mut speed: Option<f64> = None;
        let mut cad: Option<f64> = None;
        for field in record.fields() {
            match field.name() {
                "timestamp" => {
                    if let fitparser::Value::Timestamp(ts) = field.value() {
                        let utc = ts.with_timezone(&Utc);
                        timestamp = Some(utc);
                    }
                }
                "altitude" => {
                    if alt.is_none() {
                        if let Some(val) = fit_value_to_f64(field.value()) {
                            alt = Some(val);
                        }
                    }
                }
                "enhanced_altitude" => {
                    if let Some(val) = fit_value_to_f64(field.value()) {
                        alt = Some(val);
                    }
                }
                "total_ascent" | "accumulated_climb" | "runn_total_gain" => {
                    if let Some(val) = fit_value_to_f64(field.value()) {
                        tg = Some(val);
                    }
                }
                "grade" | "vertical_oscillation" | "vert_distance" => {
                    if let Some(val) = fit_value_to_f64(field.value()) {
                        inc = Some(val);
                    }
                }
                "unknown_field_135" => {
                    if let Some(val) = fit_value_to_f64(field.value()) {
                        inc = Some(val / 10.0);
                    }
                }
                "unknown_field_141" => {
                    if let Some(val) = fit_value_to_f64(field.value()) {
                        // Observed on NPE Runn developer data (inclineRunn) encoded as tenths of a percent.
                        inc = Some(val / 10.0);
                    }
                }
                "distance" | "enhanced_distance" => {
                    if let Some(val) = fit_value_to_f64(field.value()) {
                        dist = Some(val);
                        if field.name() == "enhanced_distance" {
                            dist_prio = 2;
                        } else {
                            dist_prio = dist_prio.max(1);
                        }
                    }
                }
                "speed" | "enhanced_speed" => {
                    if let Some(val) = fit_value_to_f64(field.value()) {
                        speed = Some(val.max(0.0));
                    }
                }
                "cadence" => {
                    if let Some(val) = fit_value_to_f64(field.value()) {
                        cad = Some(val.max(0.0));
                    }
                }
                other => {
                    let name = other.to_ascii_lowercase();
                    if name.contains("incline") || name.contains("grade") {
                        if let Some(val) = fit_value_to_f64(field.value()) {
                            inc = Some(val);
                        }
                    } else if name.contains("total_distance")
                        || name.contains("distance_total")
                        || name.contains("developer_distance")
                    {
                        if let Some(val) = fit_value_to_f64(field.value()) {
                            dist = Some(val);
                            dist_prio = dist_prio.max(3);
                        }
                    }
                }
            }
        }
        if let Some(ts) = timestamp {
            let seconds =
                ts.timestamp() as f64 + f64::from(ts.timestamp_subsec_micros()) / 1_000_000.0;
            let mut row = FitRecord::new(seconds, file_id);
            row.alt = alt;
            row.tg = tg;
            row.inc = inc;
            row.dist = dist;
            row.dist_prio = dist_prio;
            row.speed = speed;
            row.cad = cad;
            out.push(row);
        }
    }

    Ok(out)
}

fn fit_value_to_f64(value: &fitparser::Value) -> Option<f64> {
    match value {
        fitparser::Value::Float32(v) => Some(*v as f64),
        fitparser::Value::Float64(v) => Some(*v),
        fitparser::Value::SInt16(v) => Some(*v as f64),
        fitparser::Value::UInt16(v) => Some(*v as f64),
        fitparser::Value::SInt32(v) => Some(*v as f64),
        fitparser::Value::UInt32(v) => Some(*v as f64),
        fitparser::Value::SInt64(v) => Some(*v as f64),
        fitparser::Value::UInt64(v) => Some(*v as f64),
        fitparser::Value::UInt16z(v) => Some(*v as f64),
        fitparser::Value::UInt32z(v) => Some(*v as f64),
        fitparser::Value::UInt64z(v) => Some(*v as f64),
        fitparser::Value::Byte(v) => Some(*v as f64),
        fitparser::Value::UInt8(v) => Some(*v as f64),
        fitparser::Value::UInt8z(v) => Some(*v as f64),
        fitparser::Value::SInt8(v) => Some(*v as f64),
        fitparser::Value::Array(values) => values.iter().find_map(fit_value_to_f64),
        _ => None,
    }
}

fn parse_gpx_records(input: &[u8], file_id: usize) -> Result<Vec<FitRecord>, HcError> {
    use gpx::read;
    use std::io::Cursor;

    let mut cursor = Cursor::new(input);
    let gpx = read(&mut cursor).map_err(|e| HcError::GpxParse(e.to_string()))?;
    let mut out = Vec::new();
    let mut cumulative_dist = 0.0;
    let mut last_lat_lon: Option<(f64, f64)> = None;

    for track in gpx.tracks {
        for segment in track.segments {
            for point in segment.points {
                if let Some(time) = point.time {
                    let iso = time
                        .format()
                        .map_err(|e| HcError::GpxParse(e.to_string()))?;
                    let utc = DateTime::parse_from_rfc3339(&iso)
                        .map_err(|e| HcError::GpxParse(e.to_string()))?
                        .with_timezone(&Utc);
                    let seconds = utc.timestamp() as f64
                        + f64::from(utc.timestamp_subsec_micros()) / 1_000_000.0;
                    let mut record = FitRecord::new(seconds, file_id);
                    if let Some(ele) = point.elevation {
                        record.alt = Some(ele);
                    }
                    let point_geo = point.point();
                    let lat = point_geo.y();
                    let lon = point_geo.x();
                    if let Some((last_lat, last_lon)) = last_lat_lon {
                        cumulative_dist += haversine_distance(last_lat, last_lon, lat, lon);
                        record.dist = Some(cumulative_dist);
                        record.dist_prio = 1;
                    }
                    last_lat_lon = Some((lat, lon));
                    out.push(record);
                }
            }
        }
    }
    Ok(out)
}

fn haversine_distance(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> f64 {
    let r = 6_371_000.0_f64;
    let to_rad = |deg: f64| deg.to_radians();
    let dlat = to_rad(lat2 - lat1);
    let dlon = to_rad(lon2 - lon1);
    let a = (dlat / 2.0).sin().powi(2)
        + to_rad(lat1).cos() * to_rad(lat2).cos() * (dlon / 2.0).sin().powi(2);
    let c = 2.0 * a.sqrt().atan2((1.0 - a).sqrt());
    r * c
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SourceKind {
    RunnTotalGain,
    RunnIncline,
    Altitude,
    Mixed,
}

#[derive(Clone, Debug)]
struct Timeseries {
    times: Vec<f64>,
    gain: Vec<f64>,
    source: SourceKind,
    gaps: Vec<Gap>,
    inactivity_gaps: Vec<(f64, f64)>,
    span: f64,
    used_sources: BTreeSet<String>,
}

impl Timeseries {
    fn len(&self) -> usize {
        self.times.len()
    }
}

pub fn compute_timeseries_export(
    records_by_file: Vec<Vec<FitRecord>>,
    params: &Params,
) -> Result<TimeseriesExport, HcError> {
    if records_by_file.is_empty() {
        return Err(HcError::InsufficientData);
    }

    let merged = merge_records(
        &records_by_file,
        params.merge_eps_sec,
        &params.overlap_policy,
    );
    if merged.is_empty() {
        return Err(HcError::InsufficientData);
    }

    let mut timeseries = build_timeseries(&merged, params)?;
    if timeseries.len() < 2 {
        return Err(HcError::InsufficientData);
    }

    if params.qc_enabled {
        apply_qc(&mut timeseries, params);
    }

    let (times_out, gain_out) = if params.resample_1hz {
        let samples: Vec<(f64, f64)> = timeseries
            .times
            .iter()
            .zip(timeseries.gain.iter())
            .map(|(&t, &g)| (t, g))
            .collect();
        if let Some(reason) = resample_guard_reason(
            &samples,
            params.resample_max_gap_sec,
            params.resample_max_points,
        ) {
            tracing::warn!("Skipping 1 Hz resample: {}", reason);
            (timeseries.times.clone(), timeseries.gain.clone())
        } else {
            let (mut t_resampled, mut g_resampled) = resample_1hz(&samples);
            normalize_timeseries(
                &mut t_resampled,
                &mut g_resampled,
                Option::<&mut Vec<bool>>::None,
            );
            (t_resampled, g_resampled)
        }
    } else {
        (timeseries.times.clone(), timeseries.gain.clone())
    };

    let selected_source = if timeseries.used_sources.is_empty() {
        match params.source {
            Source::Auto => "mixed".to_string(),
            Source::Runn => "runn".to_string(),
            Source::Altitude => "altitude".to_string(),
        }
    } else if timeseries.used_sources.len() == 1 {
        timeseries
            .used_sources
            .iter()
            .next()
            .cloned()
            .unwrap_or_else(|| "mixed".to_string())
    } else {
        "mixed".to_string()
    };

    Ok(TimeseriesExport {
        times: times_out,
        gain: gain_out,
        selected_source,
    })
}

/// Compute hillclimb curves from per-file FIT/GPX records.
pub fn compute_curves(
    records_by_file: Vec<Vec<FitRecord>>,
    params: &Params,
) -> Result<Curves, HcError> {
    if records_by_file.is_empty() {
        return Err(HcError::InsufficientData);
    }

    let merged = merge_records(
        &records_by_file,
        params.merge_eps_sec,
        &params.overlap_policy,
    );
    if merged.is_empty() {
        return Err(HcError::InsufficientData);
    }

    let mut timeseries = build_timeseries(&merged, params)?;
    if timeseries.len() < 3 {
        return Err(HcError::InsufficientData);
    }

    if params.qc_enabled {
        apply_qc(&mut timeseries, params);
    }

    let total_gain =
        if let (Some(first), Some(last)) = (timeseries.gain.first(), timeseries.gain.last()) {
            last - first
        } else {
            0.0
        };
    if total_gain <= 0.01 {
        return Err(HcError::InvalidParameter(
            "no positive ascent recorded after preprocessing".into(),
        ));
    }

    let mut points: Vec<CurvePoint>;
    let base_durations: Vec<u64>;

    if params.all_windows {
        let step = params.step_s.max(1);
        let (series_times, series_gain) = if params.resample_1hz {
            let pairs: Vec<(f64, f64)> = timeseries
                .times
                .iter()
                .zip(timeseries.gain.iter())
                .map(|(&t, &g)| (t, g))
                .collect();
            if let Some(reason) = resample_guard_reason(
                &pairs,
                params.resample_max_gap_sec,
                params.resample_max_points,
            ) {
                return Err(HcError::InvalidParameter(format!(
                    "1 Hz resample blocked ({}); use --raw-sampling or increase --resample-max-gap-sec/--resample-max-points",
                    reason
                )));
            }
            let (mut t_resampled, mut g_resampled) = resample_1hz(&pairs);
            normalize_timeseries(
                &mut t_resampled,
                &mut g_resampled,
                Option::<&mut Vec<bool>>::None,
            );
            (t_resampled, g_resampled)
        } else {
            (timeseries.times.clone(), timeseries.gain.clone())
        };

        ensure_uniform_sampling(&series_times)?;

        points = compute_curve_all_windows(&series_times, &series_gain, step);
        if points.is_empty() {
            return Err(HcError::InsufficientData);
        }
        base_durations = points.iter().map(|p| p.duration_s).collect();
    } else {
        let durations = build_duration_list(timeseries.span, params);
        if durations.is_empty() {
            return Err(HcError::InvalidParameter("no durations available".into()));
        }
        points = compute_curve_points(&timeseries, &durations, params);
        base_durations = durations;
    }

    enforce_curve_shape(
        &mut points,
        &timeseries.inactivity_gaps,
        params.concave_envelope && !params.exhaustive,
    );

    let session_curves = compute_session_curves(&timeseries, params, &base_durations);

    let envelope = if params.concave_envelope {
        Some(build_envelope(&points))
    } else {
        None
    };

    let mut wr_curve: Option<(Vec<u64>, Vec<f64>)> = None;
    let mut wr_rates: Option<Vec<f64>> = None;
    let mut wr_gains_interp: Option<Vec<f64>> = None;

    match build_wr_envelope(
        &params.wr_profile.to_lowercase(),
        params.wr_min_seconds,
        params.wr_anchors_path.as_ref().map(|p| p.as_path()),
        &params.wr_short_cap,
    ) {
        Ok(env) => {
            if !env.cap_info.is_empty() {
                if let Some(v_cap) = env.cap_info.get("v_cap") {
                    tracing::info!("WR cap {:.3} m/s", v_cap);
                }
            }
            let (dur_u64, climbs_dedup, rates_dedup) =
                quantize_wr_samples(&env.durations, &env.climbs, &env.rates_avg);
            wr_curve = Some((dur_u64, climbs_dedup));
            wr_rates = Some(rates_dedup);
            wr_gains_interp = Some(
                base_durations
                    .iter()
                    .map(|&d| env.model.evaluate(d as f64))
                    .collect(),
            );
        }
        Err(err) => {
            tracing::warn!("WR envelope unavailable: {}", err);
        }
    }

    let envelope_curve = envelope
        .as_ref()
        .map(|(dur, gains)| (dur.clone(), gains.clone()));

    let (personal_curve, goal_curve, magic_rows) =
        build_scoring_overlays(&points, &base_durations, params, wr_gains_interp.as_deref());

    let selected_source = if timeseries.used_sources.is_empty() {
        match params.source {
            Source::Auto => "mixed".to_string(),
            Source::Runn => "runn".to_string(),
            Source::Altitude => "altitude".to_string(),
        }
    } else if timeseries.used_sources.len() == 1 {
        timeseries
            .used_sources
            .iter()
            .next()
            .cloned()
            .unwrap_or_else(|| "mixed".to_string())
    } else {
        "mixed".to_string()
    };

    Ok(Curves {
        points,
        wr_curve,
        wr_rates,
        personal_curve,
        goal_curve,
        envelope_curve,
        session_curves,
        selected_source,
        magic_rows,
    })
}

pub fn compute_gain_time(
    records_by_file: Vec<Vec<FitRecord>>,
    params: &Params,
    targets: &[f64],
) -> Result<GainTimeResult, HcError> {
    if records_by_file.is_empty() {
        return Err(HcError::InsufficientData);
    }

    let merged = merge_records(
        &records_by_file,
        params.merge_eps_sec,
        &params.overlap_policy,
    );
    if merged.is_empty() {
        return Err(HcError::InsufficientData);
    }

    let mut timeseries = build_timeseries(&merged, params)?;
    if timeseries.len() < 3 {
        return Err(HcError::InsufficientData);
    }

    if params.qc_enabled {
        apply_qc(&mut timeseries, params);
    }

    let mut points: Vec<CurvePoint>;
    let base_durations: Vec<u64>;

    if params.all_windows {
        let step = params.step_s.max(1);
        let (series_times, series_gain) = if params.resample_1hz {
            let samples: Vec<(f64, f64)> = timeseries
                .times
                .iter()
                .zip(timeseries.gain.iter())
                .map(|(&t, &g)| (t, g))
                .collect();
            if let Some(reason) = resample_guard_reason(
                &samples,
                params.resample_max_gap_sec,
                params.resample_max_points,
            ) {
                return Err(HcError::InvalidParameter(format!(
                    "1 Hz resample blocked ({}); use --raw-sampling or increase --resample-max-gap-sec/--resample-max-points",
                    reason
                )));
            }
            let (mut t_resampled, mut g_resampled) = resample_1hz(&samples);
            normalize_timeseries(
                &mut t_resampled,
                &mut g_resampled,
                Option::<&mut Vec<bool>>::None,
            );
            (t_resampled, g_resampled)
        } else {
            (timeseries.times.clone(), timeseries.gain.clone())
        };

        ensure_uniform_sampling(&series_times)?;

        points = compute_curve_all_windows(&series_times, &series_gain, step);
        if points.is_empty() {
            return Err(HcError::InsufficientData);
        }
        base_durations = points.iter().map(|p| p.duration_s).collect();
    } else {
        let durations = build_duration_list(timeseries.span, params);
        if durations.is_empty() {
            return Err(HcError::InvalidParameter("no durations available".into()));
        }
        points = compute_curve_points(&timeseries, &durations, params);
        base_durations = durations;
    }

    enforce_curve_shape(
        &mut points,
        &timeseries.inactivity_gaps,
        params.concave_envelope && !params.exhaustive,
    );

    let mut wr_curve: Option<(Vec<u64>, Vec<f64>)> = None;
    let mut wr_gains_interp: Option<Vec<f64>> = None;

    match build_wr_envelope(
        &params.wr_profile.to_lowercase(),
        params.wr_min_seconds,
        params.wr_anchors_path.as_ref().map(|p| p.as_path()),
        &params.wr_short_cap,
    ) {
        Ok(env) => {
            if !env.cap_info.is_empty() {
                if let Some(v_cap) = env.cap_info.get("v_cap") {
                    tracing::info!("WR cap {:.3} m/s", v_cap);
                }
            }
            let (dur_u64, climbs_dedup, _) =
                quantize_wr_samples(&env.durations, &env.climbs, &env.rates_avg);
            wr_curve = Some((dur_u64, climbs_dedup));
            wr_gains_interp = Some(
                base_durations
                    .iter()
                    .map(|&d| env.model.evaluate(d as f64))
                    .collect(),
            );
        }
        Err(err) => {
            tracing::warn!("WR envelope unavailable: {}", err);
        }
    }

    let (personal_curve, _, _) =
        build_scoring_overlays(&points, &base_durations, params, wr_gains_interp.as_deref());

    let selected_source = if timeseries.used_sources.is_empty() {
        match params.source {
            Source::Auto => "mixed".to_string(),
            Source::Runn => "runn".to_string(),
            Source::Altitude => "altitude".to_string(),
        }
    } else if timeseries.used_sources.len() == 1 {
        timeseries
            .used_sources
            .iter()
            .next()
            .cloned()
            .unwrap_or_else(|| "mixed".to_string())
    } else {
        "mixed".to_string()
    };

    let total_gain_m = timeseries.gain.last().copied().unwrap_or(0.0);
    let total_span_s = timeseries.span;

    let gain_curve = gain_time_curve_from_points(&points);
    let wr_gain_curve = wr_curve
        .as_ref()
        .map(|(durs, gains)| convert_duration_series_to_gain_time(durs, gains));
    let personal_gain_curve = personal_curve
        .as_ref()
        .map(|(durs, gains)| convert_duration_series_to_gain_time(durs, gains));

    let mut target_values: Vec<f64> = targets
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v >= 0.0)
        .collect();
    if target_values.is_empty() {
        target_values.extend_from_slice(DEFAULT_GAIN_TARGETS);
    }
    target_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    target_values.dedup_by(|a, b| (*a - *b).abs() < 1e-6);

    let approx_points = invert_curve_targets(&points, &target_values);
    let direct_points = min_time_for_targets(&timeseries.times, &timeseries.gain, &target_values);

    let targets_points = if direct_points.len() != target_values.len() {
        tracing::warn!("gain-time refinement failed; falling back to envelope inversion");
        approx_points
    } else {
        let mut merged = Vec::with_capacity(target_values.len());
        let mut approx_map: HashMap<i64, GainTimePoint> = HashMap::new();
        for pt in approx_points.into_iter() {
            let key = (pt.gain_m * 1_000_000.0).round() as i64;
            approx_map.insert(key, pt);
        }
        for (gain_value, direct_pt) in target_values.iter().copied().zip(direct_points.into_iter())
        {
            let key = (gain_value * 1_000_000.0).round() as i64;
            let approx_pt = approx_map.get(&key);
            let mut note = direct_pt.note.clone();
            if note.as_deref() != Some("unachievable") {
                if let Some(approx) = approx_pt {
                    if approx.note.as_deref() == Some("bounded_by_grid")
                        || (approx.min_time_s - direct_pt.min_time_s) > 1.5
                    {
                        note = Some("bounded_by_grid".to_string());
                    }
                }
            }
            merged.push(GainTimePoint {
                gain_m: direct_pt.gain_m,
                min_time_s: direct_pt.min_time_s,
                avg_rate_m_per_hr: direct_pt.avg_rate_m_per_hr,
                start_offset_s: direct_pt.start_offset_s,
                end_offset_s: direct_pt.end_offset_s,
                note,
            });
        }
        merged
    };

    Ok(GainTimeResult {
        curve: gain_curve,
        targets: targets_points,
        wr_curve: wr_gain_curve,
        personal_curve: personal_gain_curve,
        selected_source,
        total_span_s,
        total_gain_m,
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

fn quantize_wr_samples(
    durations: &[f64],
    climbs: &[f64],
    rates_avg: &[f64],
) -> (Vec<u64>, Vec<f64>, Vec<f64>) {
    let len = durations.len().min(climbs.len()).min(rates_avg.len());
    let mut out_durations: Vec<u64> = Vec::with_capacity(len);
    let mut out_climbs: Vec<f64> = Vec::with_capacity(len);
    let mut out_rates: Vec<f64> = Vec::with_capacity(len);
    let mut last_duration: Option<u64> = None;
    for idx in 0..len {
        let mut sec = durations[idx].round();
        if sec < 1.0 {
            sec = 1.0;
        }
        let duration_u = sec as u64;
        if Some(duration_u) == last_duration {
            if let Some(last_climb) = out_climbs.last_mut() {
                *last_climb = (*last_climb).max(climbs[idx]);
            }
            if let Some(last_rate) = out_rates.last_mut() {
                *last_rate = (*last_rate).max(rates_avg[idx]);
            }
        } else {
            out_durations.push(duration_u);
            out_climbs.push(climbs[idx]);
            out_rates.push(rates_avg[idx]);
            last_duration = Some(duration_u);
        }
    }
    (out_durations, out_climbs, out_rates)
}

fn interpolate_monotone(target: f64, durations: &[u64], gains: &[f64]) -> f64 {
    if durations.is_empty() || gains.is_empty() {
        return 0.0;
    }
    let len = durations.len().min(gains.len());
    let first = durations[0] as f64;
    if target <= first {
        return gains[0];
    }
    for i in 1..len {
        let current = durations[i] as f64;
        if target <= current {
            let prev = durations[i - 1] as f64;
            let y0 = gains[i - 1];
            let y1 = gains[i];
            if (current - prev).abs() < f64::EPSILON {
                return y1.max(y0);
            }
            let frac = ((target - prev) / (current - prev)).clamp(0.0, 1.0);
            return y0 + (y1 - y0) * frac;
        }
    }
    gains[len - 1]
}

fn can_use_stride(series: &Timeseries, durations: &[u64]) -> bool {
    if series.times.len() < 2 {
        return false;
    }
    let dt = series.times[1] - series.times[0];
    if dt <= 0.0 {
        return false;
    }
    if !series
        .times
        .windows(2)
        .all(|w| (w[1] - w[0] - dt).abs() <= 1e-6)
    {
        return false;
    }
    for &d in durations {
        if d == 0 {
            continue;
        }
        let ratio = (d as f64) / dt;
        if (ratio.round() - ratio).abs() > 1e-6 {
            return false;
        }
    }
    true
}

fn build_scoring_overlays(
    points: &[CurvePoint],
    durations: &[u64],
    params: &Params,
    wr_gains: Option<&[f64]>,
) -> (
    Option<(Vec<u64>, Vec<f64>)>,
    Option<(Vec<u64>, Vec<f64>)>,
    Option<Vec<HashMap<String, f64>>>,
) {
    let wr_gains = match wr_gains {
        Some(values) if values.len() == durations.len() => values,
        _ => return (None, None, None),
    };

    let mut user_map: BTreeMap<u64, f64> = BTreeMap::new();
    for point in points {
        user_map
            .entry(point.duration_s)
            .and_modify(|existing| {
                if point.max_climb_m > *existing {
                    *existing = point.max_climb_m;
                }
            })
            .or_insert(point.max_climb_m);
    }

    if user_map.is_empty() {
        return (None, None, None);
    }

    let (map_durations, map_gains): (Vec<u64>, Vec<f64>) =
        user_map.iter().map(|(&d, &g)| (d, g)).unzip();
    let mut user_gains: Vec<f64> = Vec::with_capacity(durations.len());
    for &duration in durations {
        if let Some(gain) = user_map.get(&duration) {
            user_gains.push(*gain);
        } else {
            user_gains.push(interpolate_monotone(
                duration as f64,
                &map_durations,
                &map_gains,
            ));
        }
    }

    let mut r_star = 0.0;
    for (&duration, (&user_gain, &wr_gain)) in
        durations.iter().zip(user_gains.iter().zip(wr_gains.iter()))
    {
        if (duration as f64) >= params.personal_min_seconds && wr_gain > 0.0 {
            let ratio = (user_gain / wr_gain).clamp(0.0, 1.0);
            if ratio > r_star {
                r_star = ratio;
            }
        }
    }

    if r_star <= 0.0 {
        r_star = 1.0;
    }

    let personal_gains: Vec<f64> = wr_gains.iter().map(|&g| g.max(0.0) * r_star).collect();
    let goal_gains: Vec<f64> = durations
        .iter()
        .zip(user_gains.iter().zip(personal_gains.iter()))
        .map(|(&duration, (&user_gain, &personal_gain))| {
            if (duration as f64) >= params.goal_min_seconds {
                user_gain + (personal_gain - user_gain) * (2.0 / 3.0)
            } else {
                user_gain
            }
        })
        .collect();

    let personal_curve: Vec<(u64, f64)> = durations
        .iter()
        .zip(personal_gains.iter())
        .filter(|(&duration, _)| (duration as f64) >= params.wr_min_seconds)
        .map(|(&d, &g)| (d, g))
        .collect();

    let magic_tokens = if let Some(tokens) = params.magic.as_ref() {
        tokens
            .iter()
            .filter_map(|token| parse_duration_token(token).map(|v| v.max(0.0)))
            .map(|v| v.round() as u64)
            .collect::<Vec<u64>>()
    } else {
        params.magic_durations.clone()
    };

    let mut magic_set: BTreeSet<u64> = BTreeSet::new();
    for value in magic_tokens {
        if value > 0 {
            magic_set.insert(value);
        }
    }

    let durations_f64: Vec<f64> = durations.iter().map(|&d| d as f64).collect();
    let mut magic_rows: Vec<HashMap<String, f64>> = Vec::new();

    for duration in magic_set {
        let duration_f = duration as f64;
        let wr_gain = interpolate_curve_value(duration_f, &durations_f64, wr_gains);
        if wr_gain <= 0.0 {
            continue;
        }
        let user_gain = interpolate_monotone(duration_f, &map_durations, &map_gains);
        let personal_gain = r_star * wr_gain;
        let goal_gain = user_gain + (personal_gain - user_gain) * (2.0 / 3.0);
        let score_pct = if wr_gain > 0.0 {
            (user_gain / wr_gain * 100.0).clamp(0.0, 1_000.0)
        } else {
            0.0
        };

        let mut row = HashMap::new();
        row.insert("duration_s".to_string(), duration_f);
        row.insert("user_gain_m".to_string(), user_gain);
        row.insert("wr_gain_m".to_string(), wr_gain);
        row.insert("personal_gain_m".to_string(), personal_gain);
        row.insert("goal_gain_m".to_string(), goal_gain);
        row.insert("score_pct".to_string(), score_pct);
        row.insert(
            "wr_rate_m_per_hr".to_string(),
            if duration_f > 0.0 {
                wr_gain * 3600.0 / duration_f
            } else {
                0.0
            },
        );
        row.insert(
            "user_rate_m_per_hr".to_string(),
            if duration_f > 0.0 {
                user_gain * 3600.0 / duration_f
            } else {
                0.0
            },
        );
        magic_rows.push(row);
    }

    let personal_curve_out = if personal_curve.is_empty() {
        None
    } else {
        Some((
            personal_curve.iter().map(|(d, _)| *d).collect(),
            personal_curve.iter().map(|(_, g)| *g).collect(),
        ))
    };

    let goal_curve_out = Some((durations.to_vec(), goal_gains));
    let magic_rows_out = if magic_rows.is_empty() {
        None
    } else {
        Some(magic_rows)
    };

    (personal_curve_out, goal_curve_out, magic_rows_out)
}

fn merge_records(
    records_by_file: &[Vec<FitRecord>],
    merge_eps_sec: f64,
    overlap_policy: &str,
) -> Vec<MergedRecord> {
    let mut all: Vec<MergedRecord> = records_by_file
        .iter()
        .flat_map(|records| records.iter().map(MergedRecord::from))
        .collect();

    all.sort_by(|a, b| {
        match a
            .time_s
            .partial_cmp(&b.time_s)
            .unwrap_or(std::cmp::Ordering::Equal)
        {
            std::cmp::Ordering::Equal => a.file_id.cmp(&b.file_id),
            other => other,
        }
    });

    let tg_spans = compute_tg_overlap_spans(records_by_file, overlap_policy);
    let mut merged: Vec<MergedRecord> = Vec::with_capacity(all.len());

    for mut record in all.into_iter() {
        if record.tg.is_some() {
            if let Some(winner) = lookup_overlap_winner(&tg_spans, record.time_s) {
                if record.file_id != winner {
                    record.tg = None;
                }
            }
        }

        if let Some(last) = merged.last_mut() {
            if record.time_s <= last.time_s + merge_eps_sec {
                merge_record_in_place(last, &record);
                continue;
            }
        }

        merged.push(record);
    }

    merged
}

fn merge_record_in_place(dest: &mut MergedRecord, src: &MergedRecord) {
    if src.tg.is_some() {
        dest.tg = src.tg;
        dest.file_id = src.file_id;
    }
    if dest.alt.is_none() && src.alt.is_some() {
        dest.alt = src.alt;
    }
    if dest.inc.is_none() && src.inc.is_some() {
        dest.inc = src.inc;
    }
    if dest.speed.is_none() && src.speed.is_some() {
        dest.speed = src.speed;
    }
    if dest.cad.is_none() && src.cad.is_some() {
        dest.cad = src.cad;
    }

    let dest_prio = dest.dist_prio;
    let src_prio = src.dist_prio;
    if src.dist.is_some() && (dest.dist.is_none() || src_prio > dest_prio) {
        dest.dist = src.dist;
        dest.dist_prio = src_prio;
    }
}

fn compute_tg_overlap_spans(
    records_by_file: &[Vec<FitRecord>],
    policy: &str,
) -> Vec<(f64, f64, usize)> {
    let mut intervals: Vec<(f64, f64, usize)> = Vec::new();
    for (file_id, recs) in records_by_file.iter().enumerate() {
        let mut timestamps: Vec<f64> = recs
            .iter()
            .filter(|r| r.tg.is_some())
            .map(|r| r.timestamp_s)
            .collect();
        if timestamps.is_empty() {
            continue;
        }
        timestamps.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if let (Some(min), Some(max)) = (timestamps.first(), timestamps.last()) {
            intervals.push((*min, *max, file_id));
        }
    }
    if intervals.is_empty() {
        return Vec::new();
    }

    #[derive(Clone, Copy)]
    struct Event {
        time: f64,
        delta: i32,
        file_id: usize,
    }

    let mut events: Vec<Event> = Vec::with_capacity(intervals.len() * 2);
    for (start, end, file_id) in intervals.iter().copied() {
        events.push(Event {
            time: start,
            delta: 1,
            file_id,
        });
        events.push(Event {
            time: end,
            delta: -1,
            file_id,
        });
    }
    events.sort_by(|a, b| {
        match a
            .time
            .partial_cmp(&b.time)
            .unwrap_or(std::cmp::Ordering::Equal)
        {
            std::cmp::Ordering::Equal => a.delta.cmp(&b.delta),
            other => other,
        }
    });

    use std::collections::HashMap;
    let mut active: HashMap<usize, i32> = HashMap::new();
    let mut spans: Vec<(f64, f64, usize)> = Vec::new();
    let mut last_time: Option<f64> = None;

    for event in events {
        if let Some(prev) = last_time {
            if event.time > prev {
                let mut files: Vec<usize> = active
                    .iter()
                    .filter_map(|(&fid, &count)| if count > 0 { Some(fid) } else { None })
                    .collect();
                if files.len() >= 2 {
                    files.sort_unstable();
                    let winner = match policy {
                        "file:first" => files[0],
                        "file:last" => files[files.len() - 1],
                        _ => files[files.len() - 1],
                    };
                    spans.push((prev, event.time, winner));
                }
            }
        }

        let counter = active.entry(event.file_id).or_insert(0);
        *counter += event.delta;
        if *counter <= 0 {
            active.remove(&event.file_id);
        }
        last_time = Some(event.time);
    }

    spans
}

fn lookup_overlap_winner(spans: &[(f64, f64, usize)], target: f64) -> Option<usize> {
    if spans.is_empty() {
        return None;
    }
    let mut lo = 0usize;
    let mut hi = spans.len();
    while lo < hi {
        let mid = (lo + hi) / 2;
        if spans[mid].0 <= target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    if lo == 0 {
        return None;
    }
    let (start, end, winner) = spans[lo - 1];
    if target >= start && target <= end {
        Some(winner)
    } else {
        None
    }
}

fn build_timeseries(records: &[MergedRecord], params: &Params) -> Result<Timeseries, HcError> {
    if records.is_empty() {
        return Err(HcError::InsufficientData);
    }

    let t0 = records.first().map(|r| r.time_s).unwrap_or(0.0);

    let mut used_sources: BTreeSet<String> = BTreeSet::new();
    let (mut times, mut gains, mut inactivity_gaps): (Vec<f64>, Vec<f64>, Vec<(f64, f64)>);
    let source_kind: SourceKind;

    match params.source {
        Source::Auto => {
            let canonical = build_canonical_timeseries(records, params, t0)?;
            times = canonical.times;
            gains = canonical.gain;
            inactivity_gaps = canonical.inactivity_gaps;
            used_sources = canonical.used_sources;
            source_kind = determine_source_kind(&used_sources);
        }
        Source::Runn => {
            let any_tg = records.iter().any(|r| r.tg.is_some());
            let any_incline = records.iter().any(|r| r.inc.is_some() && r.dist.is_some());
            if any_tg {
                let (t, g) = build_timeseries_from_tg(records, t0);
                times = t;
                gains = g;
                inactivity_gaps = Vec::new();
                used_sources.insert("runn_total_gain".to_string());
                source_kind = SourceKind::RunnTotalGain;
            } else if any_incline {
                let (t, g) = build_timeseries_from_incline(records, t0);
                times = t;
                gains = g;
                inactivity_gaps = Vec::new();
                used_sources.insert("runn_incline".to_string());
                source_kind = SourceKind::RunnIncline;
            } else {
                return Err(HcError::InsufficientData);
            }
        }
        Source::Altitude => {
            let any_alt = records.iter().any(|r| r.alt.is_some());
            if !any_alt {
                return Err(HcError::InsufficientData);
            }
            let (t, g) = build_timeseries_from_altitude(records, t0, params)?;
            times = t;
            gains = g;
            inactivity_gaps = Vec::new();
            used_sources.insert("altitude".to_string());
            source_kind = SourceKind::Altitude;
        }
    }

    if times.len() < 2 {
        return Err(HcError::InsufficientData);
    }

    let offset = times.first().copied().unwrap_or(0.0);
    normalize_timeseries(&mut times, &mut gains, None);

    if offset != 0.0 && !inactivity_gaps.is_empty() {
        for gap in inactivity_gaps.iter_mut() {
            gap.0 = (gap.0 - offset).max(0.0);
            gap.1 = (gap.1 - offset).max(gap.0);
        }
    }

    let span = times.last().copied().unwrap_or(0.0);
    let mut gaps = detect_gaps(&times, params.session_gap_sec);

    // Merge inactivity gaps inferred from canonical stitching with session gaps when needed
    if !inactivity_gaps.is_empty() {
        inactivity_gaps.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        inactivity_gaps.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-6 && (a.1 - b.1).abs() < 1e-6);
    }

    if gaps.is_empty() && params.source == Source::Auto {
        // Recompute gaps from inactivity segments when canonically inferred gaps exist
        if !inactivity_gaps.is_empty() {
            gaps = inactivity_gaps
                .iter()
                .map(|(start, end)| Gap {
                    start: *start,
                    end: *end,
                    length: (end - start).max(0.0),
                })
                .collect();
        }
    }

    Ok(Timeseries {
        times,
        gain: gains,
        source: source_kind,
        gaps,
        inactivity_gaps,
        span,
        used_sources,
    })
}

#[derive(Default)]
struct CanonicalTimeseries {
    times: Vec<f64>,
    gain: Vec<f64>,
    inactivity_gaps: Vec<(f64, f64)>,
    used_sources: BTreeSet<String>,
}

struct PerSourceCumulative {
    times: Vec<f64>,
    tg: Vec<Option<f64>>,
    incline: Vec<Option<f64>>,
    altitude: Vec<Option<f64>>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum CanonicalSource {
    Tg,
    Incline,
    Altitude,
}

fn canonical_source_rank(src: CanonicalSource) -> i32 {
    match src {
        CanonicalSource::Tg => 3,
        CanonicalSource::Incline => 2,
        CanonicalSource::Altitude => 1,
    }
}

fn canonical_source_label(src: CanonicalSource) -> &'static str {
    match src {
        CanonicalSource::Tg => "runn_total_gain",
        CanonicalSource::Incline => "runn_incline",
        CanonicalSource::Altitude => "altitude",
    }
}

fn determine_source_kind(used: &BTreeSet<String>) -> SourceKind {
    if used.len() == 1 {
        match used.iter().next().map(String::as_str) {
            Some("runn_total_gain") => SourceKind::RunnTotalGain,
            Some("runn_incline") => SourceKind::RunnIncline,
            Some("altitude") => SourceKind::Altitude,
            _ => SourceKind::Mixed,
        }
    } else if used.is_empty() {
        SourceKind::Mixed
    } else {
        SourceKind::Mixed
    }
}

fn enforce_optional_monotone(values: &mut [Option<f64>]) {
    let mut last = None;
    for entry in values.iter_mut() {
        if let Some(val) = entry {
            if let Some(prev) = last {
                if *val < prev {
                    *val = prev;
                }
            }
            last = Some(*val);
        }
    }
}

fn build_canonical_timeseries(
    records: &[MergedRecord],
    params: &Params,
    t0: f64,
) -> Result<CanonicalTimeseries, HcError> {
    let per_source = build_per_source_cumulative(records, params, t0);
    if per_source.times.len() < 2 {
        return Err(HcError::InsufficientData);
    }

    let dwell_sec = 5.0;
    let gap_threshold = params.session_gap_sec.max(0.0);

    let mut canonical = Vec::with_capacity(per_source.times.len());
    let mut used_sources: BTreeSet<String> = BTreeSet::new();
    let mut current_source: Option<CanonicalSource> = None;
    let mut base = 0.0;
    let mut last_value: Option<f64> = None;
    let mut last_switch: Option<f64> = None;
    let mut inferred_gaps: Vec<(f64, f64)> = Vec::new();

    let mut last_time = per_source.times[0];
    for (idx, &time) in per_source.times.iter().enumerate() {
        if idx > 0 {
            let delta = time - last_time;
            if delta > gap_threshold {
                inferred_gaps.push((last_time, time));
            }
            last_time = time;
        }

        let mut avail: Vec<(CanonicalSource, f64)> = Vec::with_capacity(3);
        if let Some(v) = per_source.tg[idx] {
            avail.push((CanonicalSource::Tg, v));
        }
        if let Some(v) = per_source.incline[idx] {
            avail.push((CanonicalSource::Incline, v));
        }
        if let Some(v) = per_source.altitude[idx] {
            avail.push((CanonicalSource::Altitude, v));
        }

        if avail.is_empty() {
            if let Some(last) = canonical.last().copied() {
                canonical.push(last);
            } else {
                canonical.push(0.0);
            }
            continue;
        }

        let preferred = avail
            .iter()
            .max_by_key(|(src, _)| canonical_source_rank(*src))
            .map(|(src, _)| *src)
            .unwrap_or(CanonicalSource::Altitude);

        let active_available = current_source.filter(|src| avail.iter().any(|(s, _)| s == src));

        if active_available.is_none() {
            current_source = Some(preferred);
            let value = avail
                .iter()
                .find(|(src, _)| *src == preferred)
                .map(|(_, v)| *v)
                .unwrap_or(0.0);
            let last_val = canonical.last().copied().unwrap_or(0.0);
            base = last_val - value;
            last_switch = Some(time);
        } else if let Some(active) = active_available {
            if preferred != active {
                let rank_improved =
                    canonical_source_rank(preferred) > canonical_source_rank(active);
                let dwell_elapsed = last_switch.map(|sw| time - sw >= dwell_sec).unwrap_or(true);
                if rank_improved || dwell_elapsed {
                    current_source = Some(preferred);
                    let value = avail
                        .iter()
                        .find(|(src, _)| *src == preferred)
                        .map(|(_, v)| *v)
                        .unwrap_or(0.0);
                    let last_val = canonical.last().copied().unwrap_or(0.0);
                    base = last_val - value;
                    last_switch = Some(time);
                }
            }
        }

        let active_source = current_source.unwrap_or(preferred);
        let value = avail
            .iter()
            .find(|(src, _)| *src == active_source)
            .map(|(_, v)| *v)
            .unwrap_or(0.0);
        let mut cumulative = base + value;
        if let Some(last) = last_value {
            if cumulative < last {
                cumulative = last;
            }
        }
        canonical.push(cumulative);
        last_value = Some(cumulative);
        used_sources.insert(canonical_source_label(active_source).to_string());
    }

    if !inferred_gaps.is_empty() {
        inferred_gaps.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        inferred_gaps.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-6 && (a.1 - b.1).abs() < 1e-6);
    }

    Ok(CanonicalTimeseries {
        times: per_source.times,
        gain: canonical,
        inactivity_gaps: inferred_gaps,
        used_sources,
    })
}

fn build_per_source_cumulative(
    records: &[MergedRecord],
    params: &Params,
    t0: f64,
) -> PerSourceCumulative {
    let mut times = Vec::new();
    let mut tg_cum: Vec<Option<f64>> = Vec::new();
    let mut inc_cum: Vec<Option<f64>> = Vec::new();
    let mut alt_cum: Vec<Option<f64>> = Vec::new();

    let mut base_tg = 0.0;
    let mut last_tg: Option<f64> = None;
    let mut last_file: Option<usize> = None;
    let mut stitched: HashSet<(usize, usize)> = HashSet::new();

    let mut cum_inc = 0.0;
    let mut last_dist: Option<f64> = None;
    let mut last_inc: Option<f64> = None;

    let mut alt_raw_t: Vec<f64> = Vec::new();
    let mut alt_raw_v: Vec<f64> = Vec::new();
    let mut alt_raw_idx: Vec<usize> = Vec::new();

    for (idx, record) in records.iter().enumerate() {
        let time = record.time_s - t0;
        times.push(time);

        if let Some(tg_val) = record.tg {
            if let Some(prev) = last_tg {
                if tg_val + 1.0 < prev {
                    let prev_file = last_file.unwrap_or(record.file_id);
                    let key = (prev_file, record.file_id);
                    if prev_file == record.file_id {
                        base_tg += prev;
                    } else if !stitched.contains(&key) {
                        base_tg += prev;
                        stitched.insert(key);
                    }
                }
            }
            last_tg = Some(tg_val);
            last_file = Some(record.file_id);
            let mut cumulative = base_tg + tg_val;
            if let Some(prev) = tg_cum.iter().rev().find_map(|v| *v) {
                if cumulative < prev {
                    cumulative = prev;
                }
            }
            tg_cum.push(Some(cumulative.max(0.0)));
        } else {
            tg_cum.push(None);
        }

        if let Some(dist) = record.dist {
            if let Some(prev_dist) = last_dist {
                let mut delta = dist - prev_dist;
                if delta < 0.0 {
                    delta = 0.0;
                }
                if let Some(inc) = record.inc.or(last_inc) {
                    if inc > 0.0 {
                        cum_inc += delta * (inc / 100.0);
                    }
                }
            }
            last_dist = Some(dist);
        }
        if record.inc.is_some() {
            last_inc = record.inc;
        }
        let incline_available = last_dist.is_some() && last_inc.is_some();
        if incline_available {
            inc_cum.push(Some(cum_inc.max(0.0)));
        } else {
            inc_cum.push(None);
        }

        if let Some(alt) = record.alt {
            alt_raw_t.push(time);
            alt_raw_v.push(alt);
            alt_raw_idx.push(idx);
        }
        alt_cum.push(None);
    }

    enforce_optional_monotone(&mut tg_cum);
    enforce_optional_monotone(&mut inc_cum);

    if alt_raw_t.len() >= 2 {
        let speed_med = estimate_session_speed(records, t0);
        let grade_med = estimate_session_grade(records);
        let mut altitude_eff = effective_altitude_path(&alt_raw_t, &alt_raw_v, speed_med, grade_med);
        if params.smooth_sec > 0.0 {
            altitude_eff = rolling_median_time(&alt_raw_t, &altitude_eff, params.smooth_sec);
        }
        let (altitude_adj, moving_mask, _idle_mask) =
            apply_idle_detection(records, t0, &alt_raw_t, &altitude_eff);
        let gain_series =
            cumulative_ascent_from_altitude(&altitude_adj, params.gain_eps_m, &moving_mask);
        for (series_idx, value) in alt_raw_idx.iter().zip(gain_series.iter()) {
            if let Some(entry) = alt_cum.get_mut(*series_idx) {
                *entry = Some(value.max(0.0));
            }
        }
        enforce_optional_monotone(&mut alt_cum);
    }

    PerSourceCumulative {
        times,
        tg: tg_cum,
        incline: inc_cum,
        altitude: alt_cum,
    }
}

fn build_timeseries_from_tg(records: &[MergedRecord], t0: f64) -> (Vec<f64>, Vec<f64>) {
    let mut times = Vec::new();
    let mut gains = Vec::new();
    let mut base = 0.0;
    let mut last_tg: Option<f64> = None;
    let mut last_file: Option<usize> = None;
    let mut stitched: HashSet<(usize, usize)> = HashSet::new();

    for record in records {
        let time = record.time_s - t0;
        if let Some(tg_val) = record.tg {
            if let Some(prev) = last_tg {
                if tg_val + 1.0 < prev {
                    let prev_file = last_file.unwrap_or(record.file_id);
                    let key = (prev_file, record.file_id);
                    if prev_file == record.file_id {
                        base += prev;
                    } else if !stitched.contains(&key) {
                        base += prev;
                        stitched.insert(key);
                    }
                }
            }
            last_tg = Some(tg_val);
            last_file = Some(record.file_id);
            let mut cumulative = base + tg_val;
            if let Some(last_gain) = gains.last() {
                if cumulative < *last_gain {
                    cumulative = *last_gain;
                }
            }
            times.push(time);
            gains.push(cumulative.max(0.0));
        } else if let Some(&last_gain) = gains.last() {
            times.push(time);
            gains.push(last_gain);
        }
    }

    (times, gains)
}

fn build_timeseries_from_incline(records: &[MergedRecord], t0: f64) -> (Vec<f64>, Vec<f64>) {
    let mut times = Vec::new();
    let mut gains = Vec::new();
    let mut cum = 0.0;
    let mut last_dist: Option<f64> = None;
    let mut last_inc: Option<f64> = None;
    let mut started = false;

    for record in records {
        let dist = record.dist;
        let raw_inc = record.inc;
        let inc = raw_inc.or(last_inc);

        if let Some(d) = dist {
            if let Some(prev) = last_dist {
                let mut delta = d - prev;
                if delta < 0.0 {
                    delta = 0.0;
                }
                if let Some(grade) = inc {
                    if grade > 0.0 {
                        cum += delta * (grade / 100.0);
                    }
                }
            }
            last_dist = Some(d);
            started = true;
        }

        if raw_inc.is_some() {
            last_inc = raw_inc;
        }

        if started {
            let time = record.time_s - t0;
            times.push(time);
            gains.push(cum.max(0.0));
        }
    }

    (times, gains)
}

fn build_timeseries_from_altitude(
    records: &[MergedRecord],
    t0: f64,
    params: &Params,
) -> Result<(Vec<f64>, Vec<f64>), HcError> {
    let (times_raw, gain, _diag) = build_altitude_ascent(
        records,
        t0,
        params.resample_1hz,
        params.resample_max_gap_sec,
        params.resample_max_points,
        params.smooth_sec,
        params.gain_eps_m,
    )?;
    Ok((times_raw, gain))
}

#[derive(Clone, Debug)]
struct AltitudeAscentDiagnostics {
    idle_time_pct: f64,
    resample_applied: bool,
    resample_skipped_reason: Option<String>,
}

#[derive(Clone, Debug)]
struct AltitudePreprocess {
    times: Vec<f64>,
    altitude_eff: Vec<f64>,
    altitude_idle_hold: Vec<f64>,
    moving_mask: Vec<bool>,
    idle_time_pct: f64,
    resample_applied: bool,
    resample_skipped_reason: Option<String>,
}

fn preprocess_altitude(
    records: &[MergedRecord],
    t0: f64,
    resample_1hz_requested: bool,
    resample_max_gap_sec: f64,
    resample_max_points: usize,
    smooth_sec: f64,
) -> Result<AltitudePreprocess, HcError> {
    let mut points: Vec<(f64, f64)> = records
        .iter()
        .filter_map(|r| r.alt.map(|alt| (r.time_s - t0, alt)))
        .collect();

    if points.len() < 2 {
        return Err(HcError::InsufficientData);
    }

    points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    points.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-6);

    let mut resample_applied = false;
    let mut resample_skipped_reason: Option<String> = None;

    let (times_raw, altitude_raw) = if resample_1hz_requested {
        if let Some(reason) =
            resample_guard_reason(&points, resample_max_gap_sec, resample_max_points)
        {
            tracing::warn!("Skipping 1 Hz altitude resample: {}", reason);
            resample_skipped_reason = Some(reason);
            points.into_iter().unzip()
        } else {
            resample_applied = true;
            resample_1hz(&points)
        }
    } else {
        points.into_iter().unzip()
    };

    if times_raw.len() < 2 {
        return Err(HcError::InsufficientData);
    }

    let speed_med = estimate_session_speed(records, t0);
    let grade_med = estimate_session_grade(records);

    let mut altitude_eff = effective_altitude_path(&times_raw, &altitude_raw, speed_med, grade_med);
    if smooth_sec > 0.0 {
        altitude_eff = rolling_median_time(&times_raw, &altitude_eff, smooth_sec);
    }

    let (altitude_idle_hold, moving_mask, _idle_mask) =
        apply_idle_detection(records, t0, &times_raw, &altitude_eff);

    let idle_time_pct = compute_idle_time_pct(&times_raw, &moving_mask);

    Ok(AltitudePreprocess {
        times: times_raw,
        altitude_eff,
        altitude_idle_hold,
        moving_mask,
        idle_time_pct,
        resample_applied,
        resample_skipped_reason,
    })
}

fn build_altitude_ascent(
    records: &[MergedRecord],
    t0: f64,
    resample_1hz_requested: bool,
    resample_max_gap_sec: f64,
    resample_max_points: usize,
    smooth_sec: f64,
    gain_eps_m: f64,
) -> Result<(Vec<f64>, Vec<f64>, AltitudeAscentDiagnostics), HcError> {
    let pre = preprocess_altitude(
        records,
        t0,
        resample_1hz_requested,
        resample_max_gap_sec,
        resample_max_points,
        smooth_sec,
    )?;

    let gain = cumulative_ascent_from_altitude(&pre.altitude_idle_hold, gain_eps_m, &pre.moving_mask);

    Ok((
        pre.times,
        gain,
        AltitudeAscentDiagnostics {
            idle_time_pct: pre.idle_time_pct,
            resample_applied: pre.resample_applied,
            resample_skipped_reason: pre.resample_skipped_reason,
        },
    ))
}

fn compute_idle_time_pct(times: &[f64], moving_mask: &[bool]) -> f64 {
    if times.len() < 2 || moving_mask.len() != times.len() {
        return 0.0;
    }
    let mut total = 0.0;
    let mut moving = 0.0;
    for i in 0..(times.len() - 1) {
        let dt = (times[i + 1] - times[i]).max(0.0);
        total += dt;
        if moving_mask[i] {
            moving += dt;
        }
    }
    if total <= 0.0 {
        0.0
    } else {
        ((total - moving).max(0.0) / total).clamp(0.0, 1.0)
    }
}

fn estimate_session_speed(records: &[MergedRecord], t0: f64) -> f64 {
    let mut samples: Vec<(f64, f64)> = records
        .iter()
        .filter_map(|r| r.dist.map(|d| (r.time_s - t0, d)))
        .collect();
    if samples.len() < 2 {
        return 0.0;
    }
    samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    samples.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-9);
    if samples.len() < 2 {
        return 0.0;
    }
    let mut speeds = Vec::new();
    for w in samples.windows(2) {
        let dt = w[1].0 - w[0].0;
        if dt <= 1e-6 {
            continue;
        }
        let mut dd = w[1].1 - w[0].1;
        if dd < 0.0 {
            dd = 0.0;
        }
        let v = (dd / dt).clamp(0.0, 20.0);
        if v.is_finite() {
            speeds.push(v);
        }
    }
    median(&mut speeds).unwrap_or(0.0)
}

fn estimate_session_grade(records: &[MergedRecord]) -> f64 {
    let mut grades: Vec<f64> = records
        .iter()
        .filter_map(|r| r.inc)
        .map(|inc| (inc / 100.0).clamp(-1.0, 1.0))
        .collect();
    median(&mut grades).unwrap_or(0.0)
}

fn effective_altitude_path(
    times: &[f64],
    altitude: &[f64],
    speed_med: f64,
    grade_med: f64,
) -> Vec<f64> {
    let n = altitude.len();
    if n <= 2 {
        return altitude.to_vec();
    }

    let dz: Vec<f64> = altitude.windows(2).map(|w| w[1] - w[0]).collect();
    let dt: Vec<f64> = times.windows(2).map(|w| (w[1] - w[0]).max(1e-9)).collect();

    let mut spike = vec![false; n - 1];
    for i in 0..dz.len() {
        let v = dz[i] / dt[i];
        if v.abs() > 2.0 {
            spike[i] = true;
        }
    }

    let hampel = hampel_mask_dh(&dz, 5, 3.0);
    let mut bad = vec![false; n];
    for i in 0..dz.len() {
        if spike[i] || hampel[i] {
            let idx = (i + 1).min(n - 1);
            bad[idx] = true;
        }
    }

    let mut z1 = altitude.to_vec();
    if bad.iter().any(|&b| b) {
        let good_times: Vec<f64> = times
            .iter()
            .enumerate()
            .filter_map(|(idx, &t)| if !bad[idx] { Some(t) } else { None })
            .collect();
        let good_values: Vec<f64> = z1
            .iter()
            .enumerate()
            .filter_map(|(idx, &v)| if !bad[idx] { Some(v) } else { None })
            .collect();
        if good_times.len() >= 2 {
            for (idx, flag) in bad.iter().enumerate() {
                if *flag {
                    z1[idx] = linear_interpolate(times[idx], &good_times, &good_values);
                }
            }
        }
    }

    let z2 = local_poly_smooth(times, &z1, 0.4, 7, 2);

    let mut base_t = 4.0;
    if grade_med > 0.10 {
        base_t *= 0.7;
    }
    if speed_med < 1.0 {
        base_t *= (speed_med / 1.0).max(0.6);
    }
    let closing_t = base_t.clamp(3.0, 6.0);

    let zc = morphological_closing_time(times, &z2, closing_t);
    local_poly_smooth(times, &zc, 0.8, 9, 2)
}

fn apply_idle_detection(
    records: &[MergedRecord],
    t0: f64,
    times: &[f64],
    altitude: &[f64],
) -> (Vec<f64>, Vec<bool>, Vec<bool>) {
    let n = times.len();
    if n == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let (dist_series, speed_series, cad_series) = build_motion_series(records, t0, times);
    let vertical_speed = instantaneous_vertical_speed(times, altitude);

    let window_s = 5.0;
    let drift_limit = 0.002;

    let v_med = rolling_median_time(times, &speed_series, window_s);
    let cad_med = rolling_median_time(times, &cad_series, window_s);
    let vv_med = rolling_median_time(times, &vertical_speed, window_s);
    let ds_adv = rolling_distance_advance(times, &dist_series, window_s);

    let (v_noise, _v_baseline) = estimate_speed_noise_floor(&speed_series);
    let indoor = infer_indoor_mode(records);
    let (v_on, v_off) = if indoor {
        (0.1, 0.05)
    } else {
        (v_noise.max(0.15) * 2.0, (v_noise * 1.2).max(0.15))
    };

    let cad_on = 12.0;
    let cad_off = 6.0;
    let ds_on = 3.0;
    let ds_off = 1.5;
    let vv_on = 0.05;
    let vv_off = 0.02;
    let t_enter = 2.0;
    let t_exit = 1.0;

    let mut idle_mask = vec![false; n];
    let mut in_idle = false;
    let mut below_off = 0.0;
    let mut above_on = 0.0;

    for i in 0..n {
        let dt = if i == 0 {
            0.0
        } else {
            (times[i] - times[i - 1]).max(0.0)
        };

        let speed_ok = v_med[i] >= v_on;
        let speed_low = v_med[i] <= v_off;
        let cad_ok = cad_med[i] >= cad_on;
        let cad_low = cad_med[i] <= cad_off;
        let ds_ok = ds_adv[i] >= ds_on;
        let ds_low = ds_adv[i] <= ds_off;
        let vv_ok = vv_med[i] >= vv_on;
        let vv_low = vv_med[i] <= vv_off;

        let moving_criteria = speed_ok || cad_ok || ds_ok || vv_ok;
        let idle_criteria = speed_low && cad_low && ds_low && vv_low;

        if moving_criteria {
            above_on += dt;
        } else {
            above_on = 0.0;
        }

        if idle_criteria {
            below_off += dt;
        } else {
            below_off = 0.0;
        }

        if in_idle {
            if moving_criteria && above_on >= t_exit {
                in_idle = false;
                above_on = 0.0;
            }
        } else if idle_criteria && below_off >= t_enter {
            in_idle = true;
            below_off = 0.0;
        }

        idle_mask[i] = in_idle;
    }

    let moving_mask: Vec<bool> = idle_mask.iter().map(|&b| !b).collect();
    let alt_adj = apply_idle_hold(times, altitude, &idle_mask, drift_limit);

    (alt_adj, moving_mask, idle_mask)
}

fn cumulative_ascent_from_altitude(
    altitude: &[f64],
    eps_gain: f64,
    moving_mask: &[bool],
) -> Vec<f64> {
    if altitude.is_empty() {
        return Vec::new();
    }
    let n = altitude.len();
    let eps = eps_gain.max(0.0);
    let mut out = Vec::with_capacity(n);
    let mut cum = 0.0;
    let mut run_gain = 0.0;
    out.push(0.0);
    for i in 1..n {
        let dv = altitude[i] - altitude[i - 1];
        let moving = moving_mask.get(i).copied().unwrap_or(true);
        if dv > 0.0 && moving {
            run_gain += dv;
        } else if run_gain > 0.0 {
            cum += (run_gain - eps).max(0.0);
            run_gain = 0.0;
        }
        out.push(cum);
    }
    if run_gain > 0.0 {
        let last = out.last_mut().unwrap();
        *last = cum + (run_gain - eps).max(0.0);
    }
    out
}

fn hampel_mask_dh(dh: &[f64], window: usize, n_sigmas: f64) -> Vec<bool> {
    let n = dh.len();
    if n == 0 {
        return Vec::new();
    }
    let w = window.max(1);
    let mut out = vec![false; n];
    for i in 0..n {
        let lo = i.saturating_sub(w);
        let hi = (i + w + 1).min(n);
        let segment = &dh[lo..hi];
        let mut seg_vec = segment.to_vec();
        if let Some(med) = median(&mut seg_vec) {
            let mut abs_dev: Vec<f64> = segment.iter().map(|v| (v - med).abs()).collect();
            if let Some(mad) = median(&mut abs_dev) {
                let sigma = if mad > 0.0 { 1.4826 * mad } else { 0.0 };
                if sigma > 0.0 && (dh[i] - med).abs() > n_sigmas * sigma {
                    out[i] = true;
                }
            }
        }
    }
    out
}

fn local_poly_smooth(
    times: &[f64],
    values: &[f64],
    min_span_s: f64,
    max_points: usize,
    poly: usize,
) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n);
    let max_points = max_points.max(3) | 1;
    let degree = poly.min(2);

    for i in 0..n {
        let mut lo = i;
        let mut hi = i;
        loop {
            let span = if hi > lo { times[hi] - times[lo] } else { 0.0 };
            let count = hi - lo + 1;
            if span >= min_span_s || count >= max_points || (lo == 0 && hi == n - 1) {
                break;
            }
            if lo > 0 {
                lo -= 1;
            }
            if (hi - lo + 1) < max_points && hi < n - 1 {
                hi += 1;
            }
            if lo == 0 && hi == n - 1 {
                break;
            }
        }

        let mut tt = Vec::with_capacity(hi - lo + 1);
        let mut yy = Vec::with_capacity(hi - lo + 1);
        for idx in lo..=hi {
            tt.push(times[idx] - times[i]);
            yy.push(values[idx]);
        }

        let estimate = match degree {
            0 => {
                let mut temp = yy.clone();
                median(&mut temp).unwrap_or(values[i])
            }
            1 => solve_linear_fit(&tt, &yy, 1).unwrap_or(values[i]),
            _ => solve_linear_fit(&tt, &yy, 2).unwrap_or(values[i]),
        };
        out.push(estimate);
    }
    out
}

fn solve_linear_fit(tt: &[f64], yy: &[f64], degree: usize) -> Option<f64> {
    if tt.is_empty() {
        return None;
    }
    match degree {
        1 => {
            let mut s0 = 0.0;
            let mut s1 = 0.0;
            let mut s2 = 0.0;
            let mut t0 = 0.0;
            let mut t1 = 0.0;
            for (&x, &y) in tt.iter().zip(yy.iter()) {
                s0 += 1.0;
                s1 += x;
                s2 += x * x;
                t0 += y;
                t1 += x * y;
            }
            let det = s0 * s2 - s1 * s1;
            if det.abs() < 1e-9 {
                return Some(yy[yy.len() / 2]);
            }
            let a0 = (t0 * s2 - s1 * t1) / det;
            Some(a0)
        }
        _ => {
            let mut s0 = 0.0;
            let mut s1 = 0.0;
            let mut s2 = 0.0;
            let mut s3 = 0.0;
            let mut s4 = 0.0;
            let mut t0 = 0.0;
            let mut t1 = 0.0;
            let mut t2 = 0.0;
            for (&x, &y) in tt.iter().zip(yy.iter()) {
                let x2 = x * x;
                s0 += 1.0;
                s1 += x;
                s2 += x2;
                s3 += x2 * x;
                s4 += x2 * x2;
                t0 += y;
                t1 += x * y;
                t2 += x2 * y;
            }
            let det =
                s0 * (s2 * s4 - s3 * s3) - s1 * (s1 * s4 - s2 * s3) + s2 * (s1 * s3 - s2 * s2);
            if det.abs() < 1e-9 {
                return Some(yy[yy.len() / 2]);
            }
            let det0 =
                t0 * (s2 * s4 - s3 * s3) - s1 * (t1 * s4 - s3 * t2) + s2 * (t1 * s3 - s2 * t2);
            Some(det0 / det)
        }
    }
}

fn morphological_closing_time(times: &[f64], values: &[f64], t: f64) -> Vec<f64> {
    let half_window = (t * 0.5).max(0.0);
    if half_window == 0.0 {
        return values.to_vec();
    }
    let dil = sliding_extrema_sym(times, values, half_window, true);
    sliding_extrema_sym(times, &dil, half_window, false)
}

fn sliding_extrema_sym(times: &[f64], values: &[f64], half_window: f64, is_max: bool) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }
    let mut res = Vec::with_capacity(n);
    let mut dq: VecDeque<usize> = VecDeque::new();
    let mut right = 0usize;
    for i in 0..n {
        let left_bound = times[i] - half_window;
        let right_bound = times[i] + half_window;
        while right < n && times[right] <= right_bound {
            while let Some(&back) = dq.back() {
                let cond = if is_max {
                    values[right] >= values[back]
                } else {
                    values[right] <= values[back]
                };
                if cond {
                    dq.pop_back();
                } else {
                    break;
                }
            }
            dq.push_back(right);
            right += 1;
        }
        while let Some(&front) = dq.front() {
            if times[front] < left_bound {
                dq.pop_front();
            } else {
                break;
            }
        }
        if let Some(&front) = dq.front() {
            res.push(values[front]);
        } else {
            res.push(values[i]);
        }
    }
    res
}

fn build_motion_series(
    records: &[MergedRecord],
    t0: f64,
    times: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut dist_samples: Vec<(f64, f64, u8)> = Vec::new();
    let mut speed_samples: Vec<(f64, f64)> = Vec::new();
    let mut cad_samples: Vec<(f64, f64)> = Vec::new();

    for rec in records {
        let trel = rec.time_s - t0;
        if !trel.is_finite() {
            continue;
        }
        if let Some(dist) = rec.dist {
            dist_samples.push((trel, dist.max(0.0), rec.dist_prio));
        }
        if let Some(speed) = rec.speed {
            speed_samples.push((trel, speed.max(0.0)));
        }
        if let Some(cad) = rec.cad {
            cad_samples.push((trel, cad.max(0.0)));
        }
    }

    dist_samples.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    if let Some(first_prio) = dist_samples.iter().map(|(_, _, p)| *p).max() {
        dist_samples.retain(|(_, _, p)| *p == first_prio);
    }
    dist_samples.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-9);

    let dist_series = if dist_samples.len() >= 2 {
        let st: Vec<f64> = dist_samples.iter().map(|(t, _, _)| *t).collect();
        let mut sv: Vec<f64> = dist_samples.iter().map(|(_, v, _)| *v).collect();
        // enforce monotonicity
        let mut max_val = sv[0];
        for v in sv.iter_mut() {
            if *v < max_val {
                *v = max_val;
            } else {
                max_val = *v;
            }
        }
        interp_series(times, &st, &sv, sv[0], sv[0], sv[sv.len() - 1])
    } else {
        vec![0.0; times.len()]
    };

    let speed_series = if !speed_samples.is_empty() {
        let st: Vec<f64> = speed_samples.iter().map(|(t, _)| *t).collect();
        let sv: Vec<f64> = speed_samples.iter().map(|(_, v)| *v).collect();
        interp_series(times, &st, &sv, 0.0, 0.0, 0.0)
    } else {
        instantaneous_speed(times, &dist_series)
    };

    let cad_series = if !cad_samples.is_empty() {
        let st: Vec<f64> = cad_samples.iter().map(|(t, _)| *t).collect();
        let sv: Vec<f64> = cad_samples.iter().map(|(_, v)| *v).collect();
        interp_series(times, &st, &sv, 0.0, 0.0, 0.0)
    } else {
        vec![0.0; times.len()]
    };

    (dist_series, speed_series, cad_series)
}

fn instantaneous_speed(times: &[f64], dist: &[f64]) -> Vec<f64> {
    let n = dist.len();
    if n == 0 {
        return Vec::new();
    }
    let mut out = vec![0.0; n];
    for i in 1..n {
        let dt = (times[i] - times[i - 1]).max(1e-6);
        let mut dd = dist[i] - dist[i - 1];
        if dd < 0.0 {
            dd = 0.0;
        }
        out[i] = (dd / dt).clamp(0.0, 20.0);
    }
    if n > 1 {
        out[0] = out[1];
    }
    out
}

fn instantaneous_vertical_speed(times: &[f64], alt: &[f64]) -> Vec<f64> {
    let n = alt.len();
    if n == 0 {
        return Vec::new();
    }
    let mut out = vec![0.0; n];
    for i in 1..n {
        let dt = (times[i] - times[i - 1]).max(1e-6);
        let dv = alt[i] - alt[i - 1];
        out[i] = dv.abs().clamp(0.0, 10.0) / dt;
    }
    if n > 1 {
        out[0] = out[1];
    }
    out
}

fn rolling_median_time(times: &[f64], values: &[f64], window: f64) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }
    let mut rm = RollingMedian::new();
    let mut out = Vec::with_capacity(n);
    let mut added = vec![false; n];
    let mut start = 0usize;
    for i in 0..n {
        let val = values[i];
        if val.is_finite() {
            rm.add(val, i);
            added[i] = true;
        }
        while start <= i && times[i] - times[start] > window {
            if added[start] {
                rm.discard(start);
            }
            start += 1;
        }
        out.push(rm.median());
    }
    out
}

fn rolling_distance_advance(times: &[f64], dist: &[f64], window: f64) -> Vec<f64> {
    let n = dist.len();
    if n == 0 {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(n);
    let mut maxdq: VecDeque<usize> = VecDeque::new();
    let mut mindq: VecDeque<usize> = VecDeque::new();
    let mut start = 0usize;
    for i in 0..n {
        while start <= i && times[i] - times[start] > window {
            if maxdq.front() == Some(&start) {
                maxdq.pop_front();
            }
            if mindq.front() == Some(&start) {
                mindq.pop_front();
            }
            start += 1;
        }
        while let Some(&idx) = maxdq.back() {
            if dist[idx] <= dist[i] {
                maxdq.pop_back();
            } else {
                break;
            }
        }
        maxdq.push_back(i);
        while let Some(&idx) = mindq.back() {
            if dist[idx] >= dist[i] {
                mindq.pop_back();
            } else {
                break;
            }
        }
        mindq.push_back(i);
        if let (Some(&max_i), Some(&min_i)) = (maxdq.front(), mindq.front()) {
            out.push((dist[max_i] - dist[min_i]).max(0.0));
        } else {
            out.push(0.0);
        }
    }
    out
}

fn estimate_speed_noise_floor(speed: &[f64]) -> (f64, f64) {
    let mut valid: Vec<f64> = speed.iter().copied().filter(|v| v.is_finite()).collect();
    if valid.is_empty() {
        return (0.15, 0.0);
    }
    valid
        .iter_mut()
        .for_each(|v| *v = v.clamp(0.0, f64::INFINITY));
    let still: Vec<f64> = valid.iter().copied().filter(|&v| v <= 0.6).collect();
    let baseline = if !still.is_empty() {
        let mut tmp = still;
        median(&mut tmp).unwrap_or(0.0)
    } else {
        let mut tmp = valid.clone();
        median(&mut tmp).unwrap_or(0.0)
    };
    let baseline = baseline.max(0.0);
    let noise = (baseline * 3.0).max(0.15);
    (noise, baseline)
}

fn interp_series(
    target_t: &[f64],
    sample_t: &[f64],
    sample_v: &[f64],
    default: f64,
    left: f64,
    right: f64,
) -> Vec<f64> {
    if sample_t.is_empty() {
        return vec![default; target_t.len()];
    }
    let mut paired: Vec<(f64, f64)> = sample_t
        .iter()
        .copied()
        .zip(sample_v.iter().copied())
        .filter(|(t, v)| t.is_finite() && v.is_finite())
        .collect();
    if paired.is_empty() {
        return vec![default; target_t.len()];
    }
    paired.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    paired.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-9);
    let st: Vec<f64> = paired.iter().map(|(t, _)| *t).collect();
    let sv: Vec<f64> = paired.iter().map(|(_, v)| *v).collect();
    target_t
        .iter()
        .map(|&t| {
            if t <= st[0] {
                left
            } else if t >= st[st.len() - 1] {
                right
            } else {
                linear_interpolate(t, &st, &sv)
            }
        })
        .collect()
}

fn apply_idle_hold(
    times: &[f64],
    altitude: &[f64],
    idle_mask: &[bool],
    drift_limit: f64,
) -> Vec<f64> {
    let mut out = altitude.to_vec();
    if out.is_empty() || idle_mask.iter().all(|&b| !b) {
        return out;
    }
    let mut hold_value = out[0];
    let mut hold_time = times[0];
    for i in 0..out.len() {
        if idle_mask[i] {
            if i > 0 && !idle_mask[i - 1] {
                hold_value = out[i - 1];
                hold_time = times[i - 1];
            }
            if drift_limit > 0.0 {
                let dt = (times[i] - hold_time).max(0.0);
                let max_delta = drift_limit * dt;
                let lower = hold_value - max_delta;
                let upper = hold_value + max_delta;
                out[i] = out[i].clamp(lower, upper);
            } else {
                out[i] = hold_value;
            }
        } else {
            hold_value = out[i];
            hold_time = times[i];
        }
    }
    out
}

fn infer_indoor_mode(records: &[MergedRecord]) -> bool {
    if records.is_empty() {
        return false;
    }
    let mut dev_dist = 0usize;
    let mut inc_samples = 0usize;
    for rec in records {
        if rec.dist_prio >= 3 {
            dev_dist += 1;
        }
        if rec.inc.is_some() {
            inc_samples += 1;
        }
    }
    let total = records.len();
    if total == 0 {
        return false;
    }
    if dev_dist >= total.max(5) * 4 / 10 && inc_samples > 0 {
        return true;
    }
    dev_dist >= total.max(5) * 6 / 10
}

struct RollingMedian {
    low: BinaryHeap<(OrderedFloat<f64>, usize)>,
    high: BinaryHeap<(Reverse<OrderedFloat<f64>>, usize)>,
    invalid_low: HashMap<usize, usize>,
    invalid_high: HashMap<usize, usize>,
}

impl RollingMedian {
    fn new() -> Self {
        Self {
            low: BinaryHeap::new(),
            high: BinaryHeap::new(),
            invalid_low: HashMap::new(),
            invalid_high: HashMap::new(),
        }
    }

    fn add(&mut self, value: f64, idx: usize) {
        let ord = OrderedFloat(value);
        if self.low.is_empty() || ord <= self.low.peek().map(|(v, _)| *v).unwrap_or(ord) {
            self.low.push((ord, idx));
        } else {
            self.high.push((Reverse(ord), idx));
        }
        self.rebalance();
    }

    fn discard(&mut self, idx: usize) {
        if self.low.iter().any(|&(_, stored_idx)| stored_idx == idx) {
            *self.invalid_low.entry(idx).or_insert(0) += 1;
        } else {
            *self.invalid_high.entry(idx).or_insert(0) += 1;
        }
        self.prune();
        self.rebalance();
    }

    fn median(&mut self) -> f64 {
        self.prune();
        if self.low.len() > self.high.len() {
            self.low.peek().map(|(v, _)| v.0).unwrap_or(0.0)
        } else if self.high.len() > self.low.len() {
            self.high.peek().map(|(Reverse(v), _)| v.0).unwrap_or(0.0)
        } else {
            let low = self.low.peek().map(|(v, _)| v.0).unwrap_or(0.0);
            let high = self.high.peek().map(|(Reverse(v), _)| v.0).unwrap_or(low);
            0.5 * (low + high)
        }
    }

    fn rebalance(&mut self) {
        self.prune();
        if self.low.len() > self.high.len() + 1 {
            if let Some((val, idx)) = self.low.pop() {
                self.high.push((Reverse(val), idx));
            }
        } else if self.high.len() > self.low.len() {
            if let Some((Reverse(val), idx)) = self.high.pop() {
                self.low.push((val, idx));
            }
        }
        self.prune();
    }

    fn prune(&mut self) {
        while let Some(&(_val, idx)) = self.low.peek() {
            if let Some(count) = self.invalid_low.get_mut(&idx) {
                if *count > 0 {
                    self.low.pop();
                    *count -= 1;
                    if *count == 0 {
                        self.invalid_low.remove(&idx);
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }
        while let Some(&(Reverse(_val), idx)) = self.high.peek() {
            if let Some(count) = self.invalid_high.get_mut(&idx) {
                if *count > 0 {
                    self.high.pop();
                    *count -= 1;
                    if *count == 0 {
                        self.invalid_high.remove(&idx);
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }
}

fn enforce_curve_shape(
    points: &mut [CurvePoint],
    inactivity_gaps: &[(f64, f64)],
    apply_concave: bool,
) {
    if points.is_empty() {
        return;
    }

    let durations: Vec<u64> = points.iter().map(|p| p.duration_s).collect();
    let mut climbs: Vec<f64> = points.iter().map(|p| p.max_climb_m).collect();

    let mut best = 0.0;
    for val in climbs.iter_mut() {
        if *val < best {
            *val = best;
        }
        if *val > best {
            best = *val;
        }
    }

    let mut span_gaps = vec![false; points.len()];
    if !inactivity_gaps.is_empty() {
        for (idx, cp) in points.iter().enumerate() {
            let mut spans = false;
            for &(start, end) in inactivity_gaps {
                if cp.start_offset_s <= end + 1e-6 && cp.end_offset_s >= start - 1e-6 {
                    spans = true;
                    break;
                }
            }
            span_gaps[idx] = spans;
        }
    }

    let mut concave = climbs.clone();
    if apply_concave {
        if span_gaps.iter().any(|&g| g) {
            let mut idx = 0usize;
            while idx < concave.len() {
                let flag = span_gaps[idx];
                let mut j = idx;
                while j < concave.len() && span_gaps[j] == flag {
                    j += 1;
                }
                if !flag {
                    let segment_d = durations[idx..j].to_vec();
                    let segment_c = concave[idx..j].to_vec();
                    let adjusted = upper_concave_envelope(&segment_d, &segment_c);
                    for (offset, value) in adjusted.into_iter().enumerate() {
                        concave[idx + offset] = value;
                    }
                }
                idx = j;
            }
        } else {
            concave = upper_concave_envelope(&durations, &concave);
        }
    }

    let mut best_climb = 0.0;
    let mut best_rate = f64::INFINITY;
    let mut prev_flag = span_gaps.get(0).copied().unwrap_or(false);
    for (idx, cp) in points.iter_mut().enumerate() {
        let target = if apply_concave {
            concave[idx]
        } else {
            climbs[idx]
        };
        let current_flag = span_gaps[idx];
        if idx == 0 {
            prev_flag = current_flag;
        } else if current_flag != prev_flag {
            if current_flag {
                best_rate = f64::INFINITY;
            }
            prev_flag = current_flag;
        }
        let mut gain = target.max(best_climb);
        let mut rate = if cp.duration_s > 0 {
            gain * 3600.0 / cp.duration_s as f64
        } else {
            0.0
        };
        if !current_flag && rate > best_rate {
            rate = best_rate;
            gain = rate / 3600.0 * cp.duration_s as f64;
        } else {
            best_rate = rate;
        }
        best_climb = best_climb.max(gain);
        cp.max_climb_m = gain;
        cp.climb_rate_m_per_hr = rate;
    }
}

fn upper_concave_envelope(durations: &[u64], climbs: &[f64]) -> Vec<f64> {
    if climbs.is_empty() {
        return Vec::new();
    }
    if climbs.len() == 1 {
        return climbs.to_vec();
    }
    let mut hull: Vec<usize> = Vec::new();
    for (idx, (&d, &c)) in durations.iter().zip(climbs.iter()).enumerate() {
        hull.push(idx);
        while hull.len() >= 3 {
            let len = hull.len();
            let i1 = hull[len - 3];
            let i2 = hull[len - 2];
            let i3 = hull[len - 1];
            let x1 = durations[i1] as f64;
            let x2 = durations[i2] as f64;
            let x3 = durations[i3] as f64;
            if x2 <= x1 || x3 <= x2 {
                hull.remove(len - 2);
                continue;
            }
            let slope12 = (climbs[i2] - climbs[i1]) / (x2 - x1);
            let slope23 = (climbs[i3] - climbs[i2]) / (x3 - x2);
            if slope23 > slope12 + 1e-9 {
                hull.remove(len - 2);
            } else {
                break;
            }
        }
        // reuse idx and c to avoid unused warnings
        let _ = (d, c);
    }

    let mut adjusted = climbs.to_vec();
    for pair in hull.windows(2) {
        let a = pair[0];
        let b = pair[1];
        let x1 = durations[a] as f64;
        let x2 = durations[b] as f64;
        if (x2 - x1).abs() < 1e-12 {
            continue;
        }
        let y1 = climbs[a];
        let y2 = climbs[b];
        let slope = (y2 - y1) / (x2 - x1);
        for j in (a + 1)..b {
            let x = durations[j] as f64;
            adjusted[j] = y1 + slope * (x - x1);
        }
    }
    adjusted
}

fn median(values: &mut Vec<f64>) -> Option<f64> {
    values.retain(|v| v.is_finite());
    if values.is_empty() {
        return None;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        Some(values[mid])
    } else {
        Some((values[mid - 1] + values[mid]) * 0.5)
    }
}

fn linear_interpolate(target: f64, xs: &[f64], ys: &[f64]) -> f64 {
    if xs.is_empty() || ys.is_empty() {
        return 0.0;
    }
    if target <= xs[0] {
        return ys[0];
    }
    if target >= xs[xs.len() - 1] {
        return ys[ys.len() - 1];
    }
    for i in 1..xs.len() {
        if target <= xs[i] {
            let x0 = xs[i - 1];
            let x1 = xs[i];
            let y0 = ys[i - 1];
            let y1 = ys[i];
            if (x1 - x0).abs() < 1e-12 {
                return y1.max(y0);
            }
            let frac = ((target - x0) / (x1 - x0)).clamp(0.0, 1.0);
            return y0 + (y1 - y0) * frac;
        }
    }
    ys[ys.len() - 1]
}

fn normalize_timeseries(
    times: &mut Vec<f64>,
    values: &mut Vec<f64>,
    moving: Option<&mut Vec<bool>>,
) {
    if times.is_empty() {
        return;
    }

    let mut paired: Vec<(f64, f64, bool)> = Vec::with_capacity(times.len());
    if let Some(ref mv) = moving {
        for ((&t, &v), &m) in times.iter().zip(values.iter()).zip(mv.iter()) {
            paired.push((t, v, m));
        }
    } else {
        for (&t, &v) in times.iter().zip(values.iter()) {
            paired.push((t, v, true));
        }
    }

    paired.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));

    let mut deduped: Vec<(f64, f64, bool)> = Vec::with_capacity(paired.len());
    for (t, v, m) in paired {
        if let Some((last_t, last_v, last_m)) = deduped.last_mut() {
            if (t - *last_t).abs() < 1e-6 {
                *last_v = v;
                *last_m = m;
                continue;
            }
        }
        deduped.push((t, v, m));
    }

    let offset = deduped.first().map(|(t, _, _)| *t).unwrap_or(0.0);

    times.clear();
    values.clear();
    match moving {
        Some(mv) => {
            mv.clear();
            for (t, v, m) in deduped {
                let shifted_time = (t - offset).max(0.0);
                times.push(shifted_time);
                values.push(v.max(0.0));
                mv.push(m);
            }
        }
        None => {
            for (t, v, _) in deduped {
                let shifted_time = (t - offset).max(0.0);
                times.push(shifted_time);
                values.push(v.max(0.0));
            }
        }
    }
}

fn detect_gaps(times: &[f64], gap_sec: f64) -> Vec<Gap> {
    if gap_sec <= 0.0 {
        return Vec::new();
    }
    let mut gaps = Vec::new();
    for window in times.windows(2) {
        let dt = window[1] - window[0];
        if dt > gap_sec {
            gaps.push(Gap {
                start: window[0],
                end: window[1],
                length: dt,
            });
        }
    }
    gaps
}

fn resample_guard_reason(points: &[(f64, f64)], max_gap_sec: f64, max_points: usize) -> Option<String> {
    if points.len() < 2 {
        return None;
    }

    if max_gap_sec > 0.0 {
        let mut max_gap = 0.0;
        for window in points.windows(2) {
            let dt = window[1].0 - window[0].0;
            if dt > max_gap {
                max_gap = dt;
            }
        }
        if max_gap > max_gap_sec {
            return Some(format!(
                "max timestamp gap {:.0}s exceeds resample_max_gap_sec {:.0}s",
                max_gap, max_gap_sec
            ));
        }
    }

    if max_points > 0 {
        let start = points[0].0.floor();
        let end = points.last().unwrap().0.ceil();
        let len = (end - start).max(0.0) as usize + 1;
        if len > max_points {
            return Some(format!(
                "would allocate {} points (> resample_max_points {})",
                len, max_points
            ));
        }
    }

    None
}

fn resample_1hz(points: &[(f64, f64)]) -> (Vec<f64>, Vec<f64>) {
    if points.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let start = points[0].0.floor();
    let end = points.last().unwrap().0.ceil();
    let len = (end - start).max(0.0) as usize + 1;
    let mut times = Vec::with_capacity(len);
    let mut altitude = Vec::with_capacity(len);
    let mut idx = 0;
    for step in 0..len {
        let target = start + step as f64;
        while idx + 1 < points.len() && points[idx + 1].0 < target {
            idx += 1;
        }
        let (t0, a0) = points[idx];
        if idx + 1 < points.len() {
            let (t1, a1) = points[idx + 1];
            let frac = if (t1 - t0).abs() > f64::EPSILON {
                ((target - t0) / (t1 - t0)).clamp(0.0, 1.0)
            } else {
                0.0
            };
            altitude.push(a0 + (a1 - a0) * frac);
        } else {
            altitude.push(a0);
        }
        times.push(target);
    }
    (times, altitude)
}

fn ensure_uniform_sampling(times: &[f64]) -> Result<(), HcError> {
    if times.len() < 2 {
        return Err(HcError::InsufficientData);
    }
    let dt = times[1] - times[0];
    if dt <= 0.0 {
        return Err(HcError::InvalidParameter(
            "non-positive sampling interval".to_string(),
        ));
    }
    let tol = 1e-4;
    for (idx, &time) in times.iter().enumerate().skip(2) {
        let expected = times[0] + idx as f64 * dt;
        if (time - expected).abs() > tol {
            return Err(HcError::InvalidParameter(
                "all_windows requires uniformly sampled data".to_string(),
            ));
        }
    }
    Ok(())
}

fn build_duration_list(span: f64, params: &Params) -> Vec<u64> {
    if !params.durations.is_empty() {
        return params
            .durations
            .iter()
            .copied()
            .filter(|&d| d > 0)
            .collect();
    }

    let total_span = params
        .max_duration_s
        .map(|d| d as f64)
        .unwrap_or(span)
        .max(0.0);
    let total_span_u = total_span.ceil() as u64;
    if total_span_u == 0 {
        return DEFAULT_DURATIONS.to_vec();
    }

    if params.exhaustive {
        let fine_until = total_span_u.min(2 * 3_600);
        let step = params.step_s.max(1);
        let mut durations = nice_durations_for_span(total_span_u, fine_until, step, 0.01);
        if durations.is_empty() {
            durations.push(total_span_u);
        }
        return durations;
    }

    let base_list: Vec<u64> = DEFAULT_DURATIONS.to_vec();

    let mut set: HashSet<u64> = base_list.iter().copied().filter(|&d| d > 0).collect();
    let mut durations: Vec<u64>;

    let max_base = base_list.iter().copied().max().unwrap_or(0);
    let target_span = total_span_u.max(1);

    if target_span > max_base {
        let fine_until = target_span.min(std::cmp::max(max_base, 2_u64 * 3_600_u64));
        let step_eval = params.step_s.max(60);
        for extra in nice_durations_for_span(target_span, fine_until, step_eval, 0.02) {
            if extra >= 60 {
                set.insert(extra);
            }
        }
        set.insert(target_span);
    } else {
        set.insert(target_span);
    }

    if set.is_empty() {
        durations = DEFAULT_DURATIONS.to_vec();
    } else {
        durations = set.into_iter().collect();
        durations.sort_unstable();
    }

    durations
}

fn nice_durations_for_span(
    total_span_s: u64,
    fine_until_s: u64,
    fine_step_s: u64,
    pct_step: f64,
) -> Vec<u64> {
    if total_span_s == 0 {
        return Vec::new();
    }

    let fine_step = fine_step_s.max(1);
    let fine_limit = fine_until_s.max(fine_step).min(total_span_s);

    let mut out: Vec<u64> = Vec::new();
    let mut current = fine_step;
    while current <= fine_limit {
        out.push(current);
        current += fine_step;
    }
    if total_span_s >= 1 && !out.iter().any(|&d| d == 1) {
        out.push(1);
    }

    let mut curated: Vec<u64> = Vec::new();
    let curated_hours = [
        3_u64, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 22, 24, 28, 32, 36, 40, 44, 48, 54,
        60, 66, 72, 84, 96, 120, 132, 144, 168,
    ];
    for hours in curated_hours {
        let secs = hours * 3_600;
        if secs <= total_span_s {
            curated.push(secs);
        }
    }

    for days in 1..15 {
        let secs = days * 86_400;
        if secs <= total_span_s {
            curated.push(secs);
        }
    }

    for k in 1..16 {
        let secs = 2_u64 * 86_400 * k;
        if secs <= total_span_s {
            curated.push(secs);
        }
    }

    for k in 5..53 {
        let secs = 7_u64 * 86_400 * k;
        if secs <= total_span_s {
            curated.push(secs);
        }
    }

    let pct = if pct_step > 0.0 { pct_step } else { 0.01 };
    if (total_span_s as f64) > (fine_limit as f64) {
        let mut x = fine_limit as f64;
        while x < total_span_s as f64 {
            let mut step_ratio = pct;
            if x >= 6.0 * 3_600.0 {
                step_ratio = step_ratio.max(0.015);
            }
            x *= 1.0 + step_ratio;
            let rounded = round_nice_seconds(x);
            if rounded > total_span_s {
                break;
            }
            curated.push(rounded);
        }
    }

    let mut grid: HashSet<u64> = out.into_iter().filter(|&d| d <= total_span_s).collect();
    for value in curated {
        if value > 0 && value <= total_span_s {
            grid.insert(value);
        }
    }
    grid.insert(total_span_s);

    let mut durations: Vec<u64> = grid.into_iter().collect();
    durations.sort_unstable();
    durations
}

fn round_nice_seconds(value: f64) -> u64 {
    if value <= 0.0 {
        return 1;
    }
    let rounded = if value < 3.0 * 3_600.0 {
        (value / 30.0).round() * 30.0
    } else if value < 24.0 * 3_600.0 {
        (value / 60.0).round() * 60.0
    } else if value < 7.0 * 86_400.0 {
        (value / 600.0).round() * 600.0
    } else {
        (value / 3_600.0).round() * 3_600.0
    };
    rounded.max(1.0).round() as u64
}

fn compute_curve_points(
    series: &Timeseries,
    durations: &[u64],
    params: &Params,
) -> Vec<CurvePoint> {
    let engine = match params.engine {
        Engine::Auto => {
            if can_use_stride(series, durations) {
                Engine::Stride
            } else {
                Engine::NumpyStyle
            }
        }
        other => other,
    };

    match engine {
        Engine::Stride => {
            compute_curve_stride(&series.times, &series.gain, durations, &series.gaps)
        }
        Engine::NumbaStyle | Engine::NumpyStyle => {
            compute_curve_numpy(&series.times, &series.gain, durations, &series.gaps)
        }
        Engine::Auto => compute_curve_numpy(&series.times, &series.gain, durations, &series.gaps),
    }
}

fn compute_curve_all_windows(times: &[f64], gains: &[f64], step: u64) -> Vec<CurvePoint> {
    if times.len() < 2 || gains.len() < 2 {
        return Vec::new();
    }
    let n = times.len().min(gains.len());
    if n < 2 {
        return Vec::new();
    }

    let base_dt = times[1] - times[0];
    if base_dt <= 0.0 {
        return Vec::new();
    }

    let total_span = times[n - 1] - times[0];
    if total_span <= 0.0 {
        return Vec::new();
    }

    let step_seconds = step.max(1) as f64;
    let mut duration = step_seconds;
    let mut candidates: BTreeMap<u64, CurvePoint> = BTreeMap::new();
    let eps = 1e-6;

    while duration <= total_span + eps {
        let stride = (duration / base_dt).round() as usize;
        if stride == 0 || stride >= n {
            break;
        }
        let mut best_gain = f64::NEG_INFINITY;
        let mut best_idx = 0usize;
        for idx in 0..(n - stride) {
            let gain = gains[idx + stride] - gains[idx];
            if gain > best_gain {
                best_gain = gain;
                best_idx = idx;
            }
        }

        if !best_gain.is_finite() {
            best_gain = 0.0;
        }
        if best_gain < 0.0 {
            best_gain = 0.0;
        }

        let start = times[best_idx];
        let end = times[best_idx + stride];
        let actual_duration = (end - start).max(base_dt);
        let duration_s = actual_duration.round() as u64;
        let gain_clamped = best_gain.max(0.0);
        let rate = if actual_duration > 0.0 {
            gain_clamped * 3600.0 / actual_duration
        } else {
            0.0
        };

        candidates
            .entry(duration_s.max(1))
            .and_modify(|existing| {
                if gain_clamped > existing.max_climb_m {
                    existing.max_climb_m = gain_clamped;
                    existing.climb_rate_m_per_hr = rate.max(0.0);
                    existing.start_offset_s = start;
                    existing.end_offset_s = end;
                }
            })
            .or_insert(CurvePoint {
                duration_s: duration_s.max(1),
                max_climb_m: gain_clamped,
                climb_rate_m_per_hr: rate.max(0.0),
                start_offset_s: start,
                end_offset_s: end,
            });

        duration += step_seconds;
    }

    candidates.into_iter().map(|(_, point)| point).collect()
}

fn compute_curve_numpy(
    times: &[f64],
    gains: &[f64],
    durations: &[u64],
    gaps: &[Gap],
) -> Vec<CurvePoint> {
    if times.is_empty() {
        return Vec::new();
    }

    let (ex, ey, slopes, u_at_sample) = prepare_envelope_arrays(times, gains);
    let start_time_min = times[0];
    let end_time_max = *times.last().unwrap();
    let eps = 1e-9;

    let mut results = Vec::with_capacity(durations.len());

    for &duration in durations {
        let d = duration as f64;
        if d <= 0.0 || start_time_min + d > end_time_max + eps {
            results.push(CurvePoint {
                duration_s: duration,
                max_climb_m: 0.0,
                climb_rate_m_per_hr: 0.0,
                start_offset_s: start_time_min,
                end_offset_s: start_time_min,
            });
            continue;
        }

        let mut best_gain = f64::NEG_INFINITY;
        let mut best_start = start_time_min;

        for (idx, &t0) in times.iter().enumerate() {
            let t_plus = t0 + d;
            if t_plus > end_time_max + eps {
                break;
            }
            if spans_gap(t0, t_plus, d, gaps) {
                continue;
            }
            let u_end = u_eval(&ex, &ey, &slopes, t_plus);
            let gain = u_end - u_at_sample[idx];
            if gain > best_gain {
                best_gain = gain;
                best_start = t0;
            }
        }

        for (idx, &t_end) in times.iter().enumerate().rev() {
            let t_start = t_end - d;
            if t_start < start_time_min - eps {
                continue;
            }
            if spans_gap(t_start, t_end, d, gaps) {
                continue;
            }
            let u_start = u_eval(&ex, &ey, &slopes, t_start);
            let gain = u_at_sample[idx] - u_start;
            if gain > best_gain {
                best_gain = gain;
                best_start = t_start;
            }
        }

        if !best_gain.is_finite() || best_gain < 0.0 {
            best_gain = 0.0;
            best_start = start_time_min;
        }

        let max_start = (end_time_max - d).max(start_time_min);
        let mut start = best_start.clamp(start_time_min, max_start);
        if start.is_nan() {
            start = start_time_min;
        }
        let mut end_offset = start + d;
        if end_offset > end_time_max {
            end_offset = end_time_max;
        }
        let rate = if d > 0.0 { best_gain * 3600.0 / d } else { 0.0 };

        results.push(CurvePoint {
            duration_s: duration,
            max_climb_m: best_gain.max(0.0),
            climb_rate_m_per_hr: rate.max(0.0),
            start_offset_s: start,
            end_offset_s: end_offset,
        });
    }

    results
}

fn compute_curve_stride(
    times: &[f64],
    gains: &[f64],
    durations: &[u64],
    gaps: &[Gap],
) -> Vec<CurvePoint> {
    let n = times.len().min(gains.len());
    if n < 2 {
        let start = times.first().copied().unwrap_or(0.0);
        return durations
            .iter()
            .map(|&d| CurvePoint {
                duration_s: d,
                max_climb_m: 0.0,
                climb_rate_m_per_hr: 0.0,
                start_offset_s: start,
                end_offset_s: start,
            })
            .collect();
    }

    let base_dt = times[1] - times[0];
    if base_dt <= 0.0 {
        return Vec::new();
    }
    if !times
        .windows(2)
        .all(|w| (w[1] - w[0] - base_dt).abs() <= 1e-6)
    {
        return compute_curve_numpy(times, gains, durations, gaps);
    }

    let start_time_min = times[0];
    let end_time_max = times[n - 1];
    let mut results = Vec::with_capacity(durations.len());

    for &duration in durations {
        let d = duration as f64;
        if d <= 0.0 || start_time_min + d > end_time_max + 1e-9 {
            results.push(CurvePoint {
                duration_s: duration,
                max_climb_m: 0.0,
                climb_rate_m_per_hr: 0.0,
                start_offset_s: start_time_min,
                end_offset_s: start_time_min,
            });
            continue;
        }

        let stride = ((d / base_dt).round() as usize).clamp(1, n - 1);
        if stride == 0 || stride >= n {
            results.push(CurvePoint {
                duration_s: duration,
                max_climb_m: 0.0,
                climb_rate_m_per_hr: 0.0,
                start_offset_s: start_time_min,
                end_offset_s: start_time_min,
            });
            continue;
        }

        let mut best_gain = f64::NEG_INFINITY;
        let mut best_idx = 0usize;
        for i in 0..(n - stride) {
            let start = times[i];
            let end = times[i + stride];
            if spans_gap(start, end, d, gaps) {
                continue;
            }
            let gain = gains[i + stride] - gains[i];
            if gain > best_gain {
                best_gain = gain;
                best_idx = i;
            }
        }

        if !best_gain.is_finite() || best_gain < 0.0 {
            best_gain = 0.0;
        }

        let start = times[best_idx];
        let end = (start + d).min(end_time_max);
        let rate = if d > 0.0 { best_gain * 3600.0 / d } else { 0.0 };
        results.push(CurvePoint {
            duration_s: duration,
            max_climb_m: best_gain.max(0.0),
            climb_rate_m_per_hr: rate.max(0.0),
            start_offset_s: start,
            end_offset_s: end,
        });
    }

    results
}

fn prepare_envelope_arrays(
    times: &[f64],
    gains: &[f64],
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let ex = times.to_vec();
    let ey = gains.to_vec();
    let n = ex.len();
    let mut slopes = vec![0.0; n];
    if n >= 2 {
        for i in 0..(n - 1) {
            let dt = ex[i + 1] - ex[i];
            if dt.abs() > 1e-12 {
                slopes[i] = (ey[i + 1] - ey[i]) / dt;
            } else {
                slopes[i] = 0.0;
            }
        }
        slopes[n - 1] = slopes[n - 2];
    }
    let u_at_sample = ey.clone();
    (ex, ey, slopes, u_at_sample)
}

fn u_eval(ex: &[f64], ey: &[f64], slopes: &[f64], t: f64) -> f64 {
    if ex.is_empty() {
        return 0.0;
    }
    if t <= ex[0] {
        return ey[0];
    }
    if t >= ex[ex.len() - 1] {
        return ey[ey.len() - 1];
    }
    match ex.binary_search_by(|probe| probe.partial_cmp(&t).unwrap_or(Ordering::Equal)) {
        Ok(idx) => ey[idx],
        Err(idx) => {
            let k = idx.saturating_sub(1);
            ey[k] + slopes[k] * (t - ex[k])
        }
    }
}

fn spans_gap(start: f64, _end: f64, duration: f64, gaps: &[Gap]) -> bool {
    if gaps.is_empty() {
        return false;
    }
    let eps = 1e-9;
    for gap in gaps {
        if duration > gap.length + eps {
            continue;
        }
        let gap_lo = gap.start;
        let gap_hi = gap.end - duration;
        if gap_hi < gap_lo {
            continue;
        }
        if start >= gap_lo - eps && start <= gap_hi + eps {
            return true;
        }
    }
    false
}

fn compute_session_curves(
    series: &Timeseries,
    params: &Params,
    base_durations: &[u64],
) -> Vec<SessionCurve> {
    let bounds = session_bounds(&series.times, params.session_gap_sec);
    let mut out = Vec::with_capacity(bounds.len());

    for (order, (start, end)) in bounds.into_iter().enumerate() {
        if end <= start + 1 {
            continue;
        }
        let times_slice = &series.times[start..end];
        let gains_slice = &series.gain[start..end];
        let start_time = times_slice.first().copied().unwrap_or(0.0);
        let base_gain = gains_slice.first().copied().unwrap_or(0.0);
        let span = times_slice.last().copied().unwrap_or(start_time) - start_time;
        if span <= 0.0 {
            continue;
        }

        let mut seg_series = Timeseries {
            times: times_slice.iter().map(|t| t - start_time).collect(),
            gain: gains_slice.iter().map(|g| g - base_gain).collect(),
            source: series.source,
            gaps: Vec::new(),
            inactivity_gaps: Vec::new(),
            span,
            used_sources: series.used_sources.clone(),
        };
        seg_series.gaps = detect_gaps(&seg_series.times, params.session_gap_sec);

        let mut durations: Vec<u64> = base_durations
            .iter()
            .copied()
            .filter(|&d| d > 0 && (d as f64) <= span + 1e-6)
            .collect();
        durations.sort_unstable();
        if durations.is_empty() {
            continue;
        }

        let points = compute_curve_points(&seg_series, &durations, params);
        if points.is_empty() {
            continue;
        }

        let mut climb_map: HashMap<u64, f64> = HashMap::with_capacity(points.len());
        for point in points {
            climb_map
                .entry(point.duration_s)
                .and_modify(|existing| {
                    if point.max_climb_m > *existing {
                        *existing = point.max_climb_m;
                    }
                })
                .or_insert(point.max_climb_m);
        }

        let climbs = durations
            .iter()
            .map(|d| climb_map.get(d).copied().unwrap_or(0.0))
            .collect();

        out.push(SessionCurve {
            durations,
            climbs,
            span,
            order,
        });
    }

    out
}

fn session_bounds(times: &[f64], gap_threshold: f64) -> Vec<(usize, usize)> {
    if times.is_empty() {
        return Vec::new();
    }
    let mut bounds = Vec::new();
    let mut start = 0usize;
    for i in 1..times.len() {
        if times[i] - times[i - 1] > gap_threshold {
            bounds.push((start, i));
            start = i;
        }
    }
    bounds.push((start, times.len()));
    bounds
}

fn apply_qc(series: &mut Timeseries, params: &Params) {
    let spec = resolve_qc_spec(params);
    if spec.is_empty() {
        return;
    }
    if series.times.len() != series.gain.len() || series.times.len() < 2 {
        return;
    }
    let (_segments, _removed) = apply_qc_censor(&series.times, &mut series.gain, &spec);
}

fn resolve_qc_spec(params: &Params) -> Vec<(f64, f64)> {
    let mut combined: BTreeMap<OrderedFloat<f64>, f64> = BTreeMap::new();
    for &(window, limit) in QC_DEFAULT_SPEC {
        combined.insert(OrderedFloat(window), limit);
    }
    if let Some(custom) = params.qc_spec.as_ref() {
        for (window, limit) in custom {
            combined.insert(*window, *limit);
        }
    }
    combined
        .into_iter()
        .filter_map(|(window, limit)| {
            let w = window.into_inner();
            if w > 0.0 && limit > 0.0 {
                Some((w, limit))
            } else {
                None
            }
        })
        .collect()
}

fn apply_qc_censor(times: &[f64], gains: &mut [f64], spec: &[(f64, f64)]) -> (usize, f64) {
    if spec.is_empty() || gains.is_empty() || times.len() != gains.len() {
        return (0, 0.0);
    }

    let n = gains.len();
    let mut segments_removed = 0usize;
    let mut gain_removed = 0.0;
    let eps = 1e-9;

    for &(window_sec, limit_gain) in spec {
        if window_sec <= 0.0 || limit_gain <= 0.0 {
            continue;
        }
        let mut i = 0usize;
        let mut j = 0usize;
        while i < n {
            if j <= i {
                j = i + 1;
            }
            while j < n && times[j] - times[i] <= window_sec + eps {
                let gain = gains[j] - gains[i];
                if gain > limit_gain + eps {
                    let base = gains[i];
                    let delta = gain;
                    if delta <= 0.0 {
                        j += 1;
                        continue;
                    }
                    for k in (i + 1)..=j {
                        gains[k] = base;
                    }
                    for k in (j + 1)..n {
                        gains[k] = (gains[k] - delta).max(0.0);
                    }
                    segments_removed += 1;
                    gain_removed += delta;
                    continue;
                }
                j += 1;
            }
            i += 1;
        }
    }

    let mut last = gains[0].max(0.0);
    gains[0] = last;
    for value in gains.iter_mut().skip(1) {
        if *value < last {
            *value = last;
        } else {
            if !value.is_finite() {
                *value = last;
            } else {
                *value = value.max(last);
                last = *value;
            }
        }
    }

    (segments_removed, gain_removed)
}

fn build_envelope(points: &[CurvePoint]) -> (Vec<u64>, Vec<f64>) {
    let mut unique: BTreeMap<u64, f64> = BTreeMap::new();
    for point in points {
        unique
            .entry(point.duration_s)
            .and_modify(|g| {
                if point.max_climb_m > *g {
                    *g = point.max_climb_m;
                }
            })
            .or_insert(point.max_climb_m);
    }

    let sorted: Vec<(f64, f64)> = unique.iter().map(|(&d, &g)| (d as f64, g)).collect();

    if sorted.len() <= 1 {
        let (durations, gains): (Vec<u64>, Vec<f64>) = unique.into_iter().unzip();
        return (durations, gains);
    }

    let mut hull: Vec<(f64, f64)> = Vec::new();
    for point in sorted.iter().copied() {
        while hull.len() >= 2 {
            let len = hull.len();
            let (x1, y1) = hull[len - 2];
            let (x2, y2) = hull[len - 1];
            let (x3, y3) = point;
            let slope1 = if (x2 - x1).abs() > f64::EPSILON {
                (y2 - y1) / (x2 - x1)
            } else {
                f64::INFINITY
            };
            let slope2 = if (x3 - x2).abs() > f64::EPSILON {
                (y3 - y2) / (x3 - x2)
            } else {
                f64::INFINITY
            };
            if slope2 > slope1 + 1e-9 {
                hull.pop();
            } else {
                break;
            }
        }
        hull.push(point);
    }

    let durations_sorted: Vec<u64> = unique.keys().copied().collect();
    let mut envelope = Vec::with_capacity(durations_sorted.len());

    let mut hull_idx = 0usize;
    for &duration in &durations_sorted {
        let duration_f = duration as f64;
        while hull_idx + 1 < hull.len() && hull[hull_idx + 1].0 <= duration_f {
            hull_idx += 1;
        }
        if hull_idx + 1 >= hull.len() {
            envelope.push(hull[hull_idx].1);
            continue;
        }
        let (x0, y0) = hull[hull_idx];
        let (x1, y1) = hull[hull_idx + 1];
        let value = if (x1 - x0).abs() > f64::EPSILON {
            let frac = ((duration_f - x0) / (x1 - x0)).clamp(0.0, 1.0);
            y0 + (y1 - y0) * frac
        } else {
            y0.max(y1)
        };
        envelope.push(value);
    }

    (durations_sorted, envelope)
}

fn gain_time_curve_from_points(points: &[CurvePoint]) -> Vec<GainTimePoint> {
    let mut curve: Vec<GainTimePoint> = points
        .iter()
        .map(|cp| {
            let duration = cp.duration_s as f64;
            let gain = cp.max_climb_m.max(0.0);
            let rate = if duration > 0.0 {
                gain / duration * 3_600.0
            } else {
                0.0
            };
            GainTimePoint {
                gain_m: gain,
                min_time_s: duration,
                avg_rate_m_per_hr: rate,
                start_offset_s: Some(cp.start_offset_s),
                end_offset_s: Some(cp.end_offset_s),
                note: None,
            }
        })
        .collect();

    curve.sort_by(|a, b| a.gain_m.partial_cmp(&b.gain_m).unwrap_or(Ordering::Equal));

    let mut deduped: Vec<GainTimePoint> = Vec::new();
    for point in curve.into_iter() {
        if let Some(last) = deduped.last_mut() {
            if (point.gain_m - last.gain_m).abs() < 1e-9 {
                if point.min_time_s < last.min_time_s {
                    *last = point;
                }
            } else {
                deduped.push(point);
            }
        } else {
            deduped.push(point);
        }
    }

    deduped
}

fn convert_duration_series_to_gain_time(durations: &[u64], gains: &[f64]) -> Vec<GainTimePoint> {
    let len = durations.len().min(gains.len());
    if len == 0 {
        return Vec::new();
    }

    let mut curve: Vec<GainTimePoint> = durations
        .iter()
        .zip(gains.iter())
        .take(len)
        .map(|(&duration, &gain)| {
            let duration_f = duration as f64;
            let gain_f = gain.max(0.0);
            let rate = if duration_f > 0.0 {
                gain_f / duration_f * 3_600.0
            } else {
                0.0
            };
            GainTimePoint {
                gain_m: gain_f,
                min_time_s: duration_f,
                avg_rate_m_per_hr: rate,
                start_offset_s: None,
                end_offset_s: None,
                note: None,
            }
        })
        .collect();

    curve.sort_by(|a, b| a.gain_m.partial_cmp(&b.gain_m).unwrap_or(Ordering::Equal));

    let mut deduped: Vec<GainTimePoint> = Vec::new();
    for point in curve.into_iter() {
        if let Some(last) = deduped.last_mut() {
            if (point.gain_m - last.gain_m).abs() < 1e-9 {
                if point.min_time_s < last.min_time_s {
                    *last = point;
                }
            } else {
                deduped.push(point);
            }
        } else {
            deduped.push(point);
        }
    }

    deduped
}

fn invert_curve_targets(points: &[CurvePoint], targets: &[f64]) -> Vec<GainTimePoint> {
    let mut out = Vec::with_capacity(targets.len());
    if points.is_empty() {
        return out;
    }

    let mut sorted: Vec<&CurvePoint> = points.iter().collect();
    sorted.sort_by(|a, b| a.duration_s.cmp(&b.duration_s));

    let mut gains: Vec<f64> = sorted.iter().map(|cp| cp.max_climb_m.max(0.0)).collect();
    for idx in 1..gains.len() {
        if gains[idx] < gains[idx - 1] {
            gains[idx] = gains[idx - 1];
        }
    }
    let max_gain = gains.last().copied().unwrap_or(0.0);

    for &target in targets {
        if target <= 0.0 {
            out.push(GainTimePoint {
                gain_m: target.max(0.0),
                min_time_s: 0.0,
                avg_rate_m_per_hr: 0.0,
                start_offset_s: Some(0.0),
                end_offset_s: Some(0.0),
                note: None,
            });
            continue;
        }
        if target > max_gain + 1e-6 {
            out.push(GainTimePoint {
                gain_m: target,
                min_time_s: f64::INFINITY,
                avg_rate_m_per_hr: 0.0,
                start_offset_s: None,
                end_offset_s: None,
                note: Some("unachievable".to_string()),
            });
            continue;
        }

        let mut idx = 0usize;
        while idx < gains.len() && gains[idx] < target - 1e-6 {
            idx += 1;
        }
        if idx >= gains.len() {
            out.push(GainTimePoint {
                gain_m: target,
                min_time_s: f64::INFINITY,
                avg_rate_m_per_hr: 0.0,
                start_offset_s: None,
                end_offset_s: None,
                note: Some("unachievable".to_string()),
            });
            continue;
        }

        let point = sorted[idx];
        let duration = point.duration_s as f64;
        let achieved_gain = gains[idx];
        let rate = if duration > 0.0 {
            achieved_gain / duration * 3_600.0
        } else {
            0.0
        };
        let mut note = None;
        if achieved_gain - target > 1e-6 && idx > 0 {
            let prev = sorted[idx - 1];
            let delta = point.duration_s as f64 - prev.duration_s as f64;
            if delta.abs() > 1.5 {
                note = Some("bounded_by_grid".to_string());
            }
        }

        out.push(GainTimePoint {
            gain_m: target,
            min_time_s: duration,
            avg_rate_m_per_hr: rate,
            start_offset_s: Some(point.start_offset_s),
            end_offset_s: Some(point.end_offset_s),
            note,
        });
    }

    out
}

fn min_time_for_targets(times: &[f64], gains: &[f64], targets: &[f64]) -> Vec<GainTimePoint> {
    if times.is_empty() || gains.is_empty() || targets.is_empty() {
        return Vec::new();
    }

    let mut gains_monotone = gains.to_vec();
    for idx in 1..gains_monotone.len() {
        if gains_monotone[idx] < gains_monotone[idx - 1] {
            gains_monotone[idx] = gains_monotone[idx - 1];
        }
    }
    let total_gain = gains_monotone.last().copied().unwrap_or(0.0);

    let mut merged: Vec<Option<GainTimePoint>> = vec![None; targets.len()];
    let mut zero_indices = Vec::new();
    let mut positives = Vec::new();
    for (idx, &target) in targets.iter().enumerate() {
        if !target.is_finite() || target < 0.0 {
            continue;
        }
        if target <= 0.0 {
            zero_indices.push(idx);
        } else {
            positives.push((target, idx));
        }
    }

    for idx in zero_indices {
        merged[idx] = Some(GainTimePoint {
            gain_m: 0.0,
            min_time_s: 0.0,
            avg_rate_m_per_hr: 0.0,
            start_offset_s: Some(0.0),
            end_offset_s: Some(0.0),
            note: None,
        });
    }

    if positives.is_empty() {
        return merged.into_iter().filter_map(|x| x).collect();
    }

    positives.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    let eps = 1e-9;

    for (target_gain, original_idx) in positives.iter().copied() {
        if target_gain > total_gain + 1e-6 {
            merged[original_idx] = Some(GainTimePoint {
                gain_m: target_gain,
                min_time_s: f64::INFINITY,
                avg_rate_m_per_hr: 0.0,
                start_offset_s: None,
                end_offset_s: None,
                note: Some("unachievable".to_string()),
            });
            continue;
        }
        let mut left = 0usize;
        let mut best_duration = f64::INFINITY;
        let mut best_range = None;
        for right in 0..times.len() {
            while left < right
                && (gains_monotone[right] - gains_monotone[left]) >= target_gain - eps
            {
                let duration = times[right] - times[left];
                if duration > 0.0 && duration + eps < best_duration {
                    best_duration = duration;
                    best_range = Some((left, right));
                }
                left += 1;
            }
            if left > right {
                left = right;
            }
        }
        if let Some((start_idx, end_idx)) = best_range {
            let avg_rate = if best_duration > 0.0 {
                target_gain / best_duration * 3_600.0
            } else {
                0.0
            };
            merged[original_idx] = Some(GainTimePoint {
                gain_m: target_gain,
                min_time_s: best_duration,
                avg_rate_m_per_hr: avg_rate,
                start_offset_s: Some(times[start_idx]),
                end_offset_s: Some(times[end_idx]),
                note: None,
            });
        } else {
            merged[original_idx] = Some(GainTimePoint {
                gain_m: target_gain,
                min_time_s: f64::INFINITY,
                avg_rate_m_per_hr: 0.0,
                start_offset_s: None,
                end_offset_s: None,
                note: Some("unachievable".to_string()),
            });
        }
    }

    merged.into_iter().filter_map(|x| x).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haversine_distance() {
        let dist = haversine_distance(0.0, 0.0, 0.0, 1.0);
        assert!((dist - 111_195.0).abs() < 200.0);
    }

    #[test]
    fn test_spans_gap_only_flags_gap_only_windows() {
        let gaps = vec![Gap {
            start: 10.0,
            end: 110.0,
            length: 100.0,
        }];

        // Window fully inside the gap should be skipped (duration <= gap length, start within [start, end-duration]).
        assert!(spans_gap(50.0, 60.0, 10.0, &gaps));
        assert!(spans_gap(100.0, 110.0, 10.0, &gaps));

        // Windows that cross the gap boundary are allowed by design (gap is treated as "no-gain time", not a hard boundary).
        assert!(!spans_gap(105.0, 115.0, 10.0, &gaps));
        assert!(!spans_gap(0.0, 10.0, 10.0, &gaps));

        // Durations longer than the gap are not treated as "gap-only" windows.
        assert!(!spans_gap(0.0, 120.0, 120.0, &gaps));
    }

    #[test]
    fn test_compute_curve_points_simple() {
        let mut used_sources = BTreeSet::new();
        used_sources.insert("altitude".to_string());
        let series = Timeseries {
            times: vec![0.0, 1.0, 2.0, 3.0],
            gain: vec![0.0, 20.0, 30.0, 35.0],
            source: SourceKind::Altitude,
            gaps: Vec::new(),
            inactivity_gaps: Vec::new(),
            span: 3.0,
            used_sources,
        };
        let params = Params::default();
        let durations = vec![1, 2];
        let points = compute_curve_points(&series, &durations, &params);
        assert_eq!(points.len(), 2);
        let mut map = std::collections::HashMap::new();
        for point in points {
            map.insert(point.duration_s, point.max_climb_m);
        }
        assert!((map.get(&1).copied().unwrap_or_default() - 20.0).abs() < 1e-6);
        assert!((map.get(&2).copied().unwrap_or_default() - 30.0).abs() < 1e-6);
    }

    #[test]
    fn test_resample_guard_reason_flags_large_gaps_and_allocations() {
        let points = vec![(0.0, 0.0), (1.0, 0.0), (10_000.0, 10.0)];

        let reason_gap = resample_guard_reason(&points, 2.0 * 3600.0, 500_000);
        assert!(reason_gap.is_some());

        let reason_points = resample_guard_reason(&points, 0.0, 5_000);
        assert!(reason_points.is_some());

        let ok = resample_guard_reason(&points, 0.0, 20_000);
        assert!(ok.is_none());
    }
}
