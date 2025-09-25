//! Core hillclimb curve computation library implemented in Rust.

use std::collections::HashMap;
use std::path::PathBuf;

use chrono::{DateTime, Utc};
use ndarray::Array1;
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};
use thiserror::Error;

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
    pub qc_enabled: bool,
    pub qc_spec: Option<HashMap<OrderedFloat<f64>, f64>>,
    pub wr_profile: String,
    pub wr_anchors_path: Option<PathBuf>,
    pub wr_min_seconds: f64,
    pub wr_short_cap: String,
    pub magic: Option<Vec<String>>,
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
            qc_enabled: true,
            qc_spec: None,
            wr_profile: "overall".to_string(),
            wr_anchors_path: None,
            wr_min_seconds: 30.0,
            wr_short_cap: "standard".to_string(),
            magic: None,
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
pub struct Curves {
    pub points: Vec<CurvePoint>,
    pub wr_curve: Option<(Vec<u64>, Vec<f64>)>,
    pub wr_rates: Option<Vec<f64>>,
    pub personal_curve: Option<(Vec<u64>, Vec<f64>)>,
    pub goal_curve: Option<(Vec<u64>, Vec<f64>)>,
    pub envelope_curve: Option<(Vec<u64>, Vec<f64>)>,
    pub session_curves: Vec<SessionCurve>,
    pub magic_rows: Option<Vec<HashMap<String, f64>>>,
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
            magic_rows: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FitRecord {
    pub t: f64,
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
    fn new(t: f64, file_id: usize) -> Self {
        Self {
            t,
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
    let mut t0: Option<DateTime<Utc>> = None;

    for record in records.into_iter() {
        if record.kind() != MesgNum::Record {
            continue;
        }
        let mut row = FitRecord::new(0.0, file_id);
        let mut timestamp: Option<DateTime<Utc>> = None;
        for field in record.fields() {
            match field.name() {
                "timestamp" => {
                    if let fitparser::Value::Timestamp(ts) = field.value() {
                        let utc = ts.with_timezone(&Utc);
                        timestamp = Some(utc);
                        if t0.is_none() {
                            t0 = Some(utc);
                        }
                        if let Some(base) = t0 {
                            row.t = (utc - base).num_milliseconds() as f64 / 1000.0;
                        }
                    }
                }
                "altitude" | "enhanced_altitude" => {
                    if let Some(val) = fit_value_to_f64(field.value()) {
                        row.alt = Some(val);
                    }
                }
                "total_ascent" | "accumulated_climb" | "runn_total_gain" => {
                    if let Some(val) = fit_value_to_f64(field.value()) {
                        row.tg = Some(val);
                    }
                }
                "grade" | "vertical_oscillation" | "vert_distance" => {
                    if let Some(val) = fit_value_to_f64(field.value()) {
                        row.inc = Some(val);
                    }
                }
                "distance" | "enhanced_distance" => {
                    if let Some(val) = fit_value_to_f64(field.value()) {
                        row.dist = Some(val);
                        if field.name() == "enhanced_distance" {
                            row.dist_prio = 2;
                        } else {
                            row.dist_prio = row.dist_prio.max(1);
                        }
                    }
                }
                "speed" | "enhanced_speed" => {
                    if let Some(val) = fit_value_to_f64(field.value()) {
                        row.speed = Some(val);
                    }
                }
                "cadence" => {
                    if let Some(val) = fit_value_to_f64(field.value()) {
                        row.cad = Some(val);
                    }
                }
                _ => {}
            }
        }
        if timestamp.is_some() {
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
    let mut base: Option<DateTime<Utc>> = None;
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
                    if base.is_none() {
                        base = Some(utc);
                    }
                    let mut record = FitRecord::new(
                        base.map(|b| (utc - b).num_milliseconds() as f64 / 1000.0)
                            .unwrap_or(0.0),
                        file_id,
                    );
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

#[derive(Clone, Debug)]
struct Timeseries {
    times: Array1<f64>,
    _altitude: Array1<f64>,
    gain: Array1<f64>,
}

impl Timeseries {
    fn len(&self) -> usize {
        self.times.len()
    }
}

/// Compute hillclimb curves from per-file FIT/GPX records.
pub fn compute_curves(
    records_by_file: Vec<Vec<FitRecord>>,
    params: &Params,
) -> Result<Curves, HcError> {
    if records_by_file.is_empty() {
        return Err(HcError::InsufficientData);
    }

    let merged = merge_records(records_by_file, params.session_gap_sec);
    if merged.is_empty() {
        return Err(HcError::InsufficientData);
    }

    let timeseries = build_timeseries(&merged, params)?;
    if timeseries.len() < 3 {
        return Err(HcError::InsufficientData);
    }

    let durations = build_duration_list(&timeseries, params);
    if durations.is_empty() {
        return Err(HcError::InvalidParameter("no durations available".into()));
    }

    let mut points = Vec::with_capacity(durations.len());
    for &duration in &durations {
        if let Some(point) = compute_window(&timeseries, duration as usize) {
            points.push(point);
        }
    }

    if params.qc_enabled {
        apply_qc(&mut points, params);
    }

    let envelope = if params.concave_envelope {
        Some(build_envelope(&points))
    } else {
        None
    };

    let wr_curve = params
        .wr_anchors_path
        .as_ref()
        .and_then(|path| load_wr_curve(path).ok());
    let wr_rates = wr_curve.as_ref().map(|(_, gains)| {
        durations
            .iter()
            .zip(gains.iter())
            .map(|(&d, &g)| if d == 0 { 0.0 } else { g * 3600.0 / d as f64 })
            .collect::<Vec<_>>()
    });

    let envelope_curve = envelope
        .as_ref()
        .map(|(dur, gains)| (dur.clone(), gains.clone()));

    Ok(Curves {
        points,
        wr_curve,
        wr_rates,
        personal_curve: None,
        goal_curve: None,
        envelope_curve,
        session_curves: Vec::new(),
        magic_rows: None,
    })
}

fn merge_records(records_by_file: Vec<Vec<FitRecord>>, session_gap: f64) -> Vec<FitRecord> {
    let mut all = Vec::new();
    for records in records_by_file {
        all.extend(records);
    }
    all.sort_by(|a, b| a.t.partial_cmp(&b.t).unwrap_or(std::cmp::Ordering::Equal));

    let mut merged = Vec::with_capacity(all.len());
    let mut prev_time: Option<f64> = None;
    for record in all.into_iter() {
        if let Some(prev) = prev_time {
            if record.t - prev > session_gap {
                // Insert a small gap marker by pushing a duplicate with NaN altitude to retain boundaries.
                let mut gap = record.clone();
                gap.alt = None;
                merged.push(gap);
            }
        }
        prev_time = Some(record.t);
        merged.push(record);
    }
    merged
}

fn build_timeseries(records: &[FitRecord], params: &Params) -> Result<Timeseries, HcError> {
    let mut points: Vec<(f64, f64)> = records.iter().filter_map(|r| Some((r.t, r.alt?))).collect();
    if points.len() < 2 {
        return Err(HcError::InsufficientData);
    }
    points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    points.dedup_by(|a, b| (a.0 - b.0).abs() < 1e-6);

    let (times, altitude) = if params.resample_1hz {
        resample_1hz(&points)
    } else {
        let mut t = Vec::with_capacity(points.len());
        let mut a = Vec::with_capacity(points.len());
        for (time, alt) in points.into_iter() {
            t.push(time);
            a.push(alt);
        }
        (t, a)
    };

    let altitude = if params.smooth_sec > 1e-3 {
        let window = (params.smooth_sec.max(1.0)).round() as usize;
        smooth_median(&altitude, window)
    } else {
        Array1::from_vec(altitude)
    };

    let gain = cumulative_gain(&altitude, params.gain_eps_m);
    Ok(Timeseries {
        times: Array1::from_vec(times),
        _altitude: altitude,
        gain,
    })
}

fn resample_1hz(points: &[(f64, f64)]) -> (Vec<f64>, Vec<f64>) {
    if points.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let start = points[0].0.floor();
    let end = points.last().unwrap().0.ceil();
    let len = (end - start) as usize + 1;
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

fn smooth_median(data: &[f64], window: usize) -> Array1<f64> {
    if window <= 1 {
        return Array1::from_vec(data.to_vec());
    }
    let radius = window / 2;
    let mut out = Vec::with_capacity(data.len());
    for i in 0..data.len() {
        let start = i.saturating_sub(radius);
        let end = (i + radius + 1).min(data.len());
        let mut slice = data[start..end].to_vec();
        slice.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if slice.len() % 2 == 0 {
            let mid = slice.len() / 2;
            (slice[mid - 1] + slice[mid]) / 2.0
        } else {
            slice[slice.len() / 2]
        };
        out.push(median);
    }
    Array1::from_vec(out)
}

fn cumulative_gain(altitude: &Array1<f64>, eps: f64) -> Array1<f64> {
    let mut gain = Vec::with_capacity(altitude.len());
    let mut total = 0.0;
    gain.push(0.0);
    for w in altitude.windows(2) {
        let delta = w[1] - w[0];
        if delta > eps {
            total += delta;
        }
        gain.push(total);
    }
    Array1::from_vec(gain)
}

fn build_duration_list(series: &Timeseries, params: &Params) -> Vec<u64> {
    if !params.durations.is_empty() {
        return params.durations.clone();
    }
    let span = if let (Some(start), Some(end)) = (series.times.first(), series.times.last()) {
        *end - *start
    } else {
        series.times.len() as f64
    };
    let max_duration = params.max_duration_s.map(|d| d as f64).unwrap_or(span);
    let mut durations = Vec::new();
    if params.exhaustive {
        let step = params.step_s.max(1);
        let mut current = step as f64;
        while current <= max_duration {
            durations.push(current as u64);
            current += step as f64;
        }
    } else {
        durations.push(max_duration.round() as u64);
    }
    durations
}

fn compute_window(series: &Timeseries, duration_s: usize) -> Option<CurvePoint> {
    if duration_s < 1 || duration_s >= series.gain.len() {
        return None;
    }
    let mut best_gain = 0.0;
    let mut best_start_idx = 0usize;
    let mut best_end_idx = duration_s;
    let gain = &series.gain;
    let times = &series.times;
    let mut end = duration_s;
    while end < gain.len() {
        let start = end - duration_s;
        let climb = gain[end] - gain[start];
        if climb > best_gain {
            let span = times[end] - times[start];
            if span >= duration_s as f64 - 1.0 {
                best_gain = climb;
                best_start_idx = start;
                best_end_idx = end;
            }
        }
        end += 1;
    }
    if best_gain <= 0.0 {
        return None;
    }
    let duration = times[best_end_idx] - times[best_start_idx];
    Some(CurvePoint {
        duration_s: duration_s as u64,
        max_climb_m: best_gain,
        climb_rate_m_per_hr: if duration > 0.0 {
            best_gain * 3600.0 / duration
        } else {
            0.0
        },
        start_offset_s: times[best_start_idx],
        end_offset_s: times[best_end_idx],
    })
}

fn apply_qc(points: &mut Vec<CurvePoint>, params: &Params) {
    if let Some(spec) = params.qc_spec.as_ref() {
        points.retain(|p| {
            if let Some(&limit) = spec.get(&OrderedFloat(p.duration_s as f64)) {
                p.max_climb_m <= limit
            } else {
                true
            }
        });
    }
}

fn build_envelope(points: &[CurvePoint]) -> (Vec<u64>, Vec<f64>) {
    let mut pairs: Vec<(u64, f64)> = points
        .iter()
        .map(|p| (p.duration_s, p.max_climb_m))
        .collect();
    pairs.sort_by_key(|(d, _)| *d);
    let mut env = Vec::with_capacity(pairs.len());
    let mut best = 0.0;
    for (d, g) in pairs.into_iter() {
        if g > best {
            best = g;
        }
        env.push((d, best));
    }
    let (durations, gains): (Vec<_>, Vec<_>) = env.into_iter().unzip();
    (durations, gains)
}

fn load_wr_curve(path: &PathBuf) -> Result<(Vec<u64>, Vec<f64>), HcError> {
    #[cfg(feature = "wasm")]
    {
        let _ = path;
        Err(HcError::MissingWorldRecord)
    }
    #[cfg(not(feature = "wasm"))]
    {
        use std::fs;
        let data = fs::read_to_string(path).map_err(|_| HcError::MissingWorldRecord)?;
        let anchors: HashMap<String, f64> =
            serde_json::from_str(&data).map_err(|_| HcError::MissingWorldRecord)?;
        let mut pairs: Vec<(u64, f64)> = anchors
            .into_iter()
            .filter_map(|(k, v)| k.parse::<u64>().ok().map(|d| (d, v)))
            .collect();
        pairs.sort_by_key(|(d, _)| *d);
        let (durations, gains): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        Ok((durations, gains))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_haversine_distance() {
        let dist = haversine_distance(0.0, 0.0, 0.0, 1.0);
        assert!((dist - 111_195.0).abs() < 200.0);
    }
}
