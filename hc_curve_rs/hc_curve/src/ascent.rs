use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use sha2::{Digest, Sha256};

use super::{FitRecord, Gap, HcError, Params};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub enum AscentRequirement {
    Altitude,
    Distance,
    Incline,
    TotalGain,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AscentAlgorithmInfo {
    pub id: String,
    pub name: String,
    pub description: String,
    pub requirements: Vec<AscentRequirement>,
    pub default_params: JsonValue,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct AscentDiagnostics {
    pub qc_segments_removed: usize,
    pub qc_gain_removed_m: f64,
    pub idle_time_pct: Option<f64>,
    pub gain_eps_m: Option<f64>,
    pub smooth_sec: Option<f64>,
    pub resample_1hz_requested: bool,
    pub resample_applied: Option<bool>,
    pub resample_skipped_reason: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AscentSeries {
    pub times_s: Vec<f64>,
    pub gain_m: Vec<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AscentAlgorithmResult {
    pub algorithm_id: String,
    pub params: JsonValue,
    pub params_hash: String,
    pub diagnostics: AscentDiagnostics,
    pub series: AscentSeries,
    pub total_span_s: f64,
    pub total_gain_m: f64,
    pub gaps: Vec<Gap>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "id", content = "params")]
pub enum AscentAlgorithmConfig {
    #[serde(rename = "hc.source.runn_total_gain.v1")]
    RunnTotalGainV1,
    #[serde(rename = "hc.source.runn_incline.v1")]
    RunnInclineV1,
    #[serde(rename = "hc.altitude.canonical.v1")]
    AltitudeCanonicalV1 { gain_eps_m: f64, smooth_sec: f64 },
}

impl AscentAlgorithmConfig {
    pub fn id(&self) -> &'static str {
        match self {
            AscentAlgorithmConfig::RunnTotalGainV1 => "hc.source.runn_total_gain.v1",
            AscentAlgorithmConfig::RunnInclineV1 => "hc.source.runn_incline.v1",
            AscentAlgorithmConfig::AltitudeCanonicalV1 { .. } => "hc.altitude.canonical.v1",
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            AscentAlgorithmConfig::RunnTotalGainV1 => "Runn total gain",
            AscentAlgorithmConfig::RunnInclineV1 => "Runn incline integration",
            AscentAlgorithmConfig::AltitudeCanonicalV1 { .. } => "Canonical altitude (Hillclimb)",
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            AscentAlgorithmConfig::RunnTotalGainV1 => "Device-provided cumulative total gain (developer field).",
            AscentAlgorithmConfig::RunnInclineV1 => "Integrate positive vertical from incline percent and distance.",
            AscentAlgorithmConfig::AltitudeCanonicalV1 { .. } => {
                "Effective altitude path + smoothing + idle gating + hysteresis."
            }
        }
    }

    pub fn requirements(&self) -> &'static [AscentRequirement] {
        match self {
            AscentAlgorithmConfig::RunnTotalGainV1 => &[AscentRequirement::TotalGain],
            AscentAlgorithmConfig::RunnInclineV1 => &[AscentRequirement::Incline, AscentRequirement::Distance],
            AscentAlgorithmConfig::AltitudeCanonicalV1 { .. } => &[AscentRequirement::Altitude],
        }
    }

    pub fn default_for_id(id: &str) -> Option<Self> {
        let normalized = id.trim();
        match normalized {
            "hc.source.runn_total_gain.v1" => Some(AscentAlgorithmConfig::RunnTotalGainV1),
            "hc.source.runn_incline.v1" => Some(AscentAlgorithmConfig::RunnInclineV1),
            "hc.altitude.canonical.v1" => Some(AscentAlgorithmConfig::AltitudeCanonicalV1 {
                gain_eps_m: 0.5,
                smooth_sec: 0.0,
            }),
            _ => None,
        }
    }

    pub fn params_json(&self) -> JsonValue {
        serde_json::to_value(self).unwrap_or(JsonValue::Null)
    }

    pub fn params_hash_sha256(&self) -> Result<String, HcError> {
        let bytes = serde_json::to_vec(self).map_err(|e| HcError::InvalidParameter(e.to_string()))?;
        Ok(sha256_hex(&bytes))
    }
}

pub fn list_ascent_algorithms() -> Vec<AscentAlgorithmInfo> {
    let defaults = vec![
        AscentAlgorithmConfig::RunnTotalGainV1,
        AscentAlgorithmConfig::RunnInclineV1,
        AscentAlgorithmConfig::AltitudeCanonicalV1 {
            gain_eps_m: 0.5,
            smooth_sec: 0.0,
        },
    ];

    defaults
        .into_iter()
        .map(|cfg| AscentAlgorithmInfo {
            id: cfg.id().to_string(),
            name: cfg.name().to_string(),
            description: cfg.description().to_string(),
            requirements: cfg.requirements().to_vec(),
            default_params: cfg.params_json(),
        })
        .collect()
}

pub fn compute_ascent_algorithm(
    records_by_file: Vec<Vec<FitRecord>>,
    params: &Params,
    algo: &AscentAlgorithmConfig,
) -> Result<AscentAlgorithmResult, HcError> {
    if records_by_file.is_empty() {
        return Err(HcError::InsufficientData);
    }

    let merged = super::merge_records(
        &records_by_file,
        params.merge_eps_sec,
        &params.overlap_policy,
    );
    if merged.is_empty() {
        return Err(HcError::InsufficientData);
    }

    let t0 = merged.first().map(|r| r.time_s).unwrap_or(0.0);

    let mut diagnostics = AscentDiagnostics {
        resample_1hz_requested: params.resample_1hz,
        ..AscentDiagnostics::default()
    };

    let (mut times, mut gain) = match algo {
        AscentAlgorithmConfig::RunnTotalGainV1 => {
            if !merged.iter().any(|r| r.tg.is_some()) {
                return Err(HcError::InsufficientData);
            }
            super::build_timeseries_from_tg(&merged, t0)
        }
        AscentAlgorithmConfig::RunnInclineV1 => {
            let any = merged.iter().any(|r| r.inc.is_some() && r.dist.is_some());
            if !any {
                return Err(HcError::InsufficientData);
            }
            super::build_timeseries_from_incline(&merged, t0)
        }
        AscentAlgorithmConfig::AltitudeCanonicalV1 { gain_eps_m, smooth_sec } => {
            let any_alt = merged.iter().any(|r| r.alt.is_some());
            if !any_alt {
                return Err(HcError::InsufficientData);
            }
            let (t, g, diag) = super::build_altitude_ascent(
                &merged,
                t0,
                params.resample_1hz,
                params.resample_max_gap_sec,
                params.resample_max_points,
                *smooth_sec,
                *gain_eps_m,
            )?;
            diagnostics.idle_time_pct = Some(diag.idle_time_pct);
            diagnostics.gain_eps_m = Some(*gain_eps_m);
            diagnostics.smooth_sec = Some(*smooth_sec);
            diagnostics.resample_applied = Some(diag.resample_applied);
            diagnostics.resample_skipped_reason = diag.resample_skipped_reason;
            (t, g)
        }
    };

    if times.len() < 2 || times.len() != gain.len() {
        return Err(HcError::InsufficientData);
    }

    if params.qc_enabled {
        let spec = super::resolve_qc_spec(params);
        let (segments_removed, gain_removed) = super::apply_qc_censor(&times, &mut gain, &spec);
        diagnostics.qc_segments_removed = segments_removed;
        diagnostics.qc_gain_removed_m = gain_removed;
    }

    super::normalize_timeseries(&mut times, &mut gain, Option::<&mut Vec<bool>>::None);
    let gaps = super::detect_gaps(&times, params.session_gap_sec);

    let total_span_s = times.last().copied().unwrap_or(0.0);
    let total_gain_m = gain.last().copied().unwrap_or(0.0) - gain.first().copied().unwrap_or(0.0);

    Ok(AscentAlgorithmResult {
        algorithm_id: algo.id().to_string(),
        params: algo.params_json(),
        params_hash: algo.params_hash_sha256()?,
        diagnostics,
        series: AscentSeries {
            times_s: times,
            gain_m: gain,
        },
        total_span_s,
        total_gain_m,
        gaps,
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(digest.len() * 2);
    for b in digest {
        use std::fmt::Write;
        let _ = write!(&mut out, "{:02x}", b);
    }
    out
}

