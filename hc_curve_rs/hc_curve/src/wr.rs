// World-record envelope modeling logic ported from Python implementation in hc_curve.py.
// Provides facilities to construct the WR curve for different profiles, fit parameters,
// and evaluate/sample the resulting model.

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::Path;

use serde::Deserialize;

use ordered_float::OrderedFloat;

use crate::HcError;

const WR_VCAP_HARD_MAX: f64 = 1.05;
const WR_SAMPLE_SECONDS_MIN: f64 = 1e-3;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WrModelKind {
    Sbpl1,
    Sbpl2,
}

#[derive(Clone, Debug)]
pub enum WrModel {
    Sbpl1 {
        v_cap: f64,
        s_inf: f64,
        t_star: f64,
        k: f64,
    },
    Sbpl2 {
        v_cap: f64,
        s1: f64,
        s2: f64,
        t1: f64,
        t2: f64,
        k1: f64,
        k2: f64,
    },
}

impl WrModel {
    pub fn kind(&self) -> WrModelKind {
        match self {
            WrModel::Sbpl1 { .. } => WrModelKind::Sbpl1,
            WrModel::Sbpl2 { .. } => WrModelKind::Sbpl2,
        }
    }

    pub fn evaluate(&self, duration_s: f64) -> f64 {
        match self {
            WrModel::Sbpl1 {
                v_cap,
                s_inf,
                t_star,
                k,
            } => h_sbpl_cap_scalar(duration_s, *v_cap, *s_inf, *t_star, *k),
            WrModel::Sbpl2 {
                v_cap,
                s1,
                s2,
                t1,
                t2,
                k1,
                k2,
            } => h_sbpl_two_break_scalar(duration_s, *v_cap, *s1, *s2, *t1, *t2, *k1, *k2),
        }
    }

    pub fn instantaneous_rate(&self, duration_s: f64) -> f64 {
        match self {
            WrModel::Sbpl1 {
                v_cap,
                s_inf,
                t_star,
                k,
            } => d_h_sbpl_cap_scalar(duration_s, *v_cap, *s_inf, *t_star, *k),
            WrModel::Sbpl2 { .. } => {
                // No closed form derivative; approximate using symmetric difference.
                let eps = (1e-3_f64).max(duration_s * 0.005);
                let lo = (duration_s - eps).max(WR_SAMPLE_SECONDS_MIN);
                let hi = duration_s + eps;
                let gain_lo = self.evaluate(lo);
                let gain_hi = self.evaluate(hi);
                ((gain_hi - gain_lo) / (hi - lo)).max(0.0)
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct WrEnvelope {
    pub model: WrModel,
    pub durations: Vec<f64>,
    pub climbs: Vec<f64>,
    pub rates_avg: Vec<f64>,
    pub rates_inst: Vec<f64>,
    pub anchors: Vec<WrAnchor>,
    pub cap_info: BTreeMap<String, f64>,
}

#[derive(Clone, Debug)]
pub struct WrAnchor {
    pub duration_s: f64,
    pub gain_m: f64,
    pub weight: f64,
}

#[derive(Clone, Debug)]
struct TreadmillCap {
    grade: f64,
    f_max: f64,
    l_max: f64,
}

#[derive(Clone, Debug)]
struct StairsCap {
    riser: f64,
    f_max: f64,
}

#[derive(Clone, Debug)]
struct EnergyCap {
    v200_mps: f64,
    cost_per_m: f64,
    efficiency: f64,
}

#[derive(Clone, Debug)]
struct Sbpl1Bounds {
    s_inf: (f64, f64),
    t_star: (f64, f64),
    k: (f64, f64),
}

#[derive(Clone, Debug)]
struct Sbpl2Bounds {
    s1: (f64, f64),
    s2: (f64, f64),
    t1: (f64, f64),
    t2: (f64, f64),
}

#[derive(Clone, Debug)]
struct WrProfileConfig {
    model: WrModelKind,
    anchors: Vec<(f64, f64)>,
    anchor_scale: f64,
    treadmill: Option<TreadmillCap>,
    stairs: Option<StairsCap>,
    energy: Option<EnergyCap>,
    bounds_sbpl1: Sbpl1Bounds,
    bounds_sbpl2: Option<Sbpl2Bounds>,
    k1: f64,
    k2: f64,
    coarse_grid_sbpl1: (usize, usize, usize),
    refine_grid_sbpl1: (usize, usize, usize),
    coarse_grid_sbpl2: Option<(usize, usize, usize, usize)>,
    refine_grid_sbpl2: Option<(usize, usize, usize, usize)>,
    anchor_weights: HashMap<OrderedFloat<f64>, f64>,
    short_anchor_weight: f64,
    default_anchor_weight: f64,
}

impl WrProfileConfig {
    fn base_overall() -> Self {
        Self {
            model: WrModelKind::Sbpl2,
            anchors: vec![
                (0.481 * 3600.0, 1000.0),
                (1.0 * 3600.0, 1616.0),
                (12.0 * 3600.0, 13145.65),
                (24.0 * 3600.0, 21720.0),
            ],
            anchor_scale: 1.0,
            treadmill: Some(TreadmillCap {
                grade: 0.40,
                f_max: 3.4,
                l_max: 0.85,
            }),
            stairs: Some(StairsCap {
                riser: 0.17,
                f_max: 3.4,
            }),
            energy: Some(EnergyCap {
                v200_mps: 10.4,
                cost_per_m: 4.3,
                efficiency: 0.25,
            }),
            bounds_sbpl1: Sbpl1Bounds {
                s_inf: (0.55, 0.90),
                t_star: (35.0, 7200.0),
                k: (1.0, 6.0),
            },
            bounds_sbpl2: Some(Sbpl2Bounds {
                s1: (0.70, 0.96),
                s2: (0.45, 0.78),
                t1: (35.0, 1800.0),
                t2: (3600.0, 90000.0),
            }),
            k1: 4.0,
            k2: 4.0,
            coarse_grid_sbpl1: (15, 15, 12),
            refine_grid_sbpl1: (9, 9, 9),
            coarse_grid_sbpl2: Some((12, 12, 10, 10)),
            refine_grid_sbpl2: Some((9, 9, 7, 7)),
            anchor_weights: HashMap::from([
                (OrderedFloat(0.481 * 3600.0), 45.0),
                (OrderedFloat(1.0 * 3600.0), 60.0),
                (OrderedFloat(12.0 * 3600.0), 20.0),
                (OrderedFloat(24.0 * 3600.0), 10.0),
            ]),
            short_anchor_weight: 15.0,
            default_anchor_weight: 1.0,
        }
    }

    fn with_stairs() -> Self {
        let mut cfg = Self::base_overall();
        cfg.anchors = vec![
            (0.481 * 3600.0, 1000.0),
            (1.0 * 3600.0, 1616.0),
            (12.0 * 3600.0, 13145.65),
            (24.0 * 3600.0, 18713.0),
        ];
        cfg
    }

    fn for_profile(name: &str) -> Self {
        match name {
            "overall" => Self::base_overall(),
            "stairs" => Self::with_stairs(),
            "female_overall" => {
                let mut cfg = Self::base_overall();
                cfg.anchor_scale = 0.87;
                cfg.treadmill = Some(TreadmillCap {
                    grade: 0.40,
                    f_max: 3.2,
                    l_max: 0.80,
                });
                cfg.stairs = Some(StairsCap {
                    riser: 0.17,
                    f_max: 3.2,
                });
                cfg.energy = Some(EnergyCap {
                    v200_mps: 9.2,
                    cost_per_m: 4.2,
                    efficiency: 0.25,
                });
                cfg
            }
            "female_stairs" => {
                let mut cfg = Self::with_stairs();
                cfg.anchor_scale = 0.87;
                cfg.treadmill = Some(TreadmillCap {
                    grade: 0.40,
                    f_max: 3.2,
                    l_max: 0.80,
                });
                cfg.stairs = Some(StairsCap {
                    riser: 0.17,
                    f_max: 3.2,
                });
                cfg.energy = Some(EnergyCap {
                    v200_mps: 9.2,
                    cost_per_m: 4.2,
                    efficiency: 0.25,
                });
                cfg
            }
            other => {
                // Unknown profile -> fall back to overall but keep anchor scale unchanged.
                let mut cfg = Self::base_overall();
                eprintln!("Unknown WR profile '{}'; falling back to overall", other);
                cfg
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct AnchorOverride {
    w_s: f64,
    #[serde(default)]
    gain_m: Option<f64>,
    #[serde(default)]
    gain: Option<f64>,
    #[serde(default)]
    weight: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct AnchorOverrideFile {
    #[serde(default)]
    anchors: Option<Vec<AnchorOverride>>,
    #[serde(default)]
    points: Option<Vec<AnchorOverride>>,
    #[serde(default)]
    v_cap: Option<f64>,
    #[serde(default)]
    vcap: Option<f64>,
}

pub fn build_wr_envelope(
    profile_name: &str,
    wr_min_seconds: f64,
    anchors_path: Option<&Path>,
    cap_mode: &str,
) -> Result<WrEnvelope, HcError> {
    let mut profile = WrProfileConfig::for_profile(profile_name);

    let cap_multiplier = match cap_mode {
        "conservative" => 0.95,
        "aggressive" => 1.05,
        _ => 1.0,
    };
    apply_cap_mode(&mut profile, cap_multiplier);

    let (anchor_overrides, v_cap_override) = match anchors_path {
        Some(path) => load_anchor_override(path)?,
        None => (Vec::new(), None),
    };

    let cap_info = resolve_vertical_cap(&profile, v_cap_override);
    let v_cap = cap_info
        .get("v_cap")
        .copied()
        .ok_or_else(|| HcError::MissingWorldRecord)?;

    let anchors = build_wr_anchors(&profile, wr_min_seconds, v_cap, anchor_overrides);
    if anchors.is_empty() {
        return Err(HcError::MissingWorldRecord);
    }

    let filtered: Vec<WrAnchor> = anchors
        .into_iter()
        .filter(|a| a.duration_s >= wr_min_seconds)
        .collect();
    if filtered.is_empty() {
        return Err(HcError::MissingWorldRecord);
    }

    let t_vals: Vec<f64> = filtered.iter().map(|a| a.duration_s).collect();
    let h_vals: Vec<f64> = filtered.iter().map(|a| a.gain_m).collect();
    let mut weights: Vec<f64> = filtered.iter().map(|a| a.weight.max(0.0)).collect();
    normalize_weights(&mut weights);

    let model = fit_wr_model(&profile, &t_vals, &h_vals, &weights, v_cap, wr_min_seconds)?;

    let (durations, climbs, rates_avg, rates_inst) =
        sample_wr_model(&model, &t_vals, wr_min_seconds);

    Ok(WrEnvelope {
        model,
        durations,
        climbs,
        rates_avg,
        rates_inst,
        anchors: filtered,
        cap_info,
    })
}

fn apply_cap_mode(profile: &mut WrProfileConfig, multiplier: f64) {
    if (multiplier - 1.0).abs() < 1e-6 {
        return;
    }
    if let Some(tread) = profile.treadmill.as_mut() {
        tread.f_max *= multiplier;
        tread.l_max *= multiplier;
    }
    if let Some(stairs) = profile.stairs.as_mut() {
        stairs.f_max *= multiplier;
    }
    if let Some(energy) = profile.energy.as_mut() {
        energy.v200_mps *= multiplier;
    }
}

fn load_anchor_override(path: &Path) -> Result<(Vec<WrAnchor>, Option<f64>), HcError> {
    let data = fs::read(path).map_err(|_| HcError::MissingWorldRecord)?;
    let json: serde_json::Value =
        serde_json::from_slice(&data).map_err(|_| HcError::MissingWorldRecord)?;

    let mut anchors: Vec<WrAnchor> = Vec::new();
    let mut v_cap_override: Option<f64> = None;

    match json {
        serde_json::Value::Object(map) => {
            let anchors_val = map.get("anchors").or_else(|| map.get("points"));
            if let Some(val) = anchors_val {
                anchors = parse_anchor_array(val)?;
            }
            if let Some(val) = map.get("v_cap").and_then(|v| v.as_f64()) {
                v_cap_override = Some(val);
            } else if let Some(val) = map.get("vcap").and_then(|v| v.as_f64()) {
                v_cap_override = Some(val);
            }
        }
        serde_json::Value::Array(arr) => {
            anchors = parse_anchor_array(&serde_json::Value::Array(arr))?;
        }
        _ => {}
    }

    Ok((anchors, v_cap_override))
}

fn parse_anchor_array(value: &serde_json::Value) -> Result<Vec<WrAnchor>, HcError> {
    let mut anchors: Vec<WrAnchor> = Vec::new();
    match value {
        serde_json::Value::Array(items) => {
            for item in items {
                match item {
                    serde_json::Value::Array(pair) => {
                        if pair.len() >= 2 {
                            let duration = pair[0].as_f64().unwrap_or(0.0);
                            let gain = pair[1].as_f64().unwrap_or(0.0);
                            if duration > 0.0 && gain > 0.0 {
                                anchors.push(WrAnchor {
                                    duration_s: duration,
                                    gain_m: gain,
                                    weight: 1.0,
                                });
                            }
                        }
                    }
                    serde_json::Value::Object(obj) => {
                        let duration = obj.get("w_s").or_else(|| obj.get("seconds"));
                        let gain = obj.get("gain_m").or_else(|| obj.get("gain"));
                        if let Some(d) = duration.and_then(|v| v.as_f64()) {
                            let g = gain.and_then(|v| v.as_f64()).unwrap_or(0.0);
                            let weight = obj.get("weight").and_then(|v| v.as_f64()).unwrap_or(1.0);
                            anchors.push(WrAnchor {
                                duration_s: d,
                                gain_m: g,
                                weight,
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
        _ => {}
    }
    Ok(anchors)
}

fn normalize_weights(weights: &mut [f64]) {
    let sum: f64 = weights.iter().copied().sum();
    if sum > 0.0 {
        for w in weights.iter_mut() {
            *w /= sum;
        }
    } else {
        let n = weights.len().max(1) as f64;
        for w in weights.iter_mut() {
            *w = 1.0 / n;
        }
    }
}

fn resolve_vertical_cap(
    profile: &WrProfileConfig,
    override_v_cap: Option<f64>,
) -> BTreeMap<String, f64> {
    let mut caps = BTreeMap::new();
    if let Some(t) = profile.treadmill.as_ref() {
        caps.insert(
            "treadmill".to_string(),
            treadmill_cap(t.grade, t.f_max, t.l_max),
        );
    }
    if let Some(s) = profile.stairs.as_ref() {
        caps.insert("stairs".to_string(), stairs_cap_double(s.riser, s.f_max));
    }
    if let Some(e) = profile.energy.as_ref() {
        caps.insert(
            "energy".to_string(),
            energy_cap_from_200m(e.v200_mps, e.cost_per_m, e.efficiency),
        );
    }
    if caps.is_empty() {
        caps.insert("default".to_string(), 1.0);
    }
    let mut v_cap = caps.values().copied().fold(f64::INFINITY, f64::min);
    if let Some(override_val) = override_v_cap {
        v_cap = override_val;
    }
    v_cap = v_cap.min(WR_VCAP_HARD_MAX);
    caps.insert("v_cap".to_string(), v_cap);
    caps
}

fn treadmill_cap(grade: f64, f_max: f64, l_max: f64) -> f64 {
    f_max.max(0.0) * l_max.max(0.0) * grade.max(0.0)
}

fn stairs_cap_double(riser: f64, f_max: f64) -> f64 {
    f_max.max(0.0) * (2.0 * riser.max(0.0))
}

fn energy_cap_from_200m(v200_mps: f64, cost_per_m: f64, efficiency: f64) -> f64 {
    let g = 9.81;
    (efficiency.max(0.0) * cost_per_m.max(0.0) * v200_mps.max(0.0) / g).max(0.0)
}

fn build_wr_anchors(
    profile: &WrProfileConfig,
    wr_min_seconds: f64,
    v_cap: f64,
    overrides: Vec<WrAnchor>,
) -> Vec<WrAnchor> {
    if !overrides.is_empty() {
        return overrides
            .into_iter()
            .filter(|a| a.duration_s > 0.0 && a.gain_m > 0.0)
            .collect();
    }

    let mut anchors: Vec<WrAnchor> = profile
        .anchors
        .iter()
        .map(|(d, g)| WrAnchor {
            duration_s: *d,
            gain_m: g * profile.anchor_scale,
            weight: profile
                .anchor_weights
                .get(&OrderedFloat(*d))
                .copied()
                .unwrap_or(profile.default_anchor_weight),
        })
        .collect();

    // Ensure at least one anchor at wr_min_seconds if above provided anchors.
    if wr_min_seconds > 0.0 {
        let missing = !anchors
            .iter()
            .any(|a| (a.duration_s - wr_min_seconds).abs() < 1e-6);
        if missing {
            anchors.push(WrAnchor {
                duration_s: wr_min_seconds,
                gain_m: v_cap * wr_min_seconds,
                weight: profile.short_anchor_weight,
            });
        }
    }

    anchors.sort_by(|a, b| a.duration_s.partial_cmp(&b.duration_s).unwrap());
    anchors
}

fn fit_wr_model(
    profile: &WrProfileConfig,
    t_vals: &[f64],
    h_vals: &[f64],
    weights: &[f64],
    v_cap: f64,
    wr_min_seconds: f64,
) -> Result<WrModel, HcError> {
    match profile.model {
        WrModelKind::Sbpl2 => fit_sbpl2(profile, t_vals, h_vals, weights, v_cap, wr_min_seconds),
        WrModelKind::Sbpl1 => fit_sbpl1(profile, t_vals, h_vals, weights, v_cap, wr_min_seconds),
    }
}

fn fit_sbpl1(
    profile: &WrProfileConfig,
    t_vals: &[f64],
    h_vals: &[f64],
    weights: &[f64],
    v_cap: f64,
    _wr_min_seconds: f64,
) -> Result<WrModel, HcError> {
    let bounds = &profile.bounds_sbpl1;
    let (s_low, s_high) = bounds.s_inf;
    let (t_low, t_high) = bounds.t_star;
    let (k_low, k_high) = bounds.k;

    let (c_s, c_t, c_k) = profile.coarse_grid_sbpl1;
    let (r_s, r_t, r_k) = profile.refine_grid_sbpl1;

    let mut best_loss = f64::INFINITY;
    let mut best_params = None;

    for s in linspace(s_low, s_high, c_s) {
        for t in logspace(t_low, t_high, c_t) {
            for k in linspace(k_low, k_high, c_k) {
                let loss = sbpl_loss(t_vals, h_vals, weights, v_cap, s, t, k, 500.0);
                if loss < best_loss {
                    best_loss = loss;
                    best_params = Some((s, t, k));
                }
            }
        }
    }

    let (mut s_best, mut t_best, mut k_best) = best_params.ok_or(HcError::MissingWorldRecord)?;

    for s in refine_range(s_best, s_low, s_high, r_s, 0.3) {
        for t in logspace_refine(t_best, t_low, t_high, r_t, 0.4, 2.5) {
            for k in refine_range(k_best, k_low, k_high, r_k, 0.3) {
                let loss = sbpl_loss(t_vals, h_vals, weights, v_cap, s, t, k, 500.0);
                if loss < best_loss {
                    best_loss = loss;
                    s_best = s;
                    t_best = t;
                    k_best = k;
                }
            }
        }
    }

    Ok(WrModel::Sbpl1 {
        v_cap,
        s_inf: s_best,
        t_star: t_best,
        k: k_best,
    })
}

fn fit_sbpl2(
    profile: &WrProfileConfig,
    t_vals: &[f64],
    h_vals: &[f64],
    weights: &[f64],
    v_cap: f64,
    _wr_min_seconds: f64,
) -> Result<WrModel, HcError> {
    let bounds = profile
        .bounds_sbpl2
        .as_ref()
        .ok_or(HcError::MissingWorldRecord)?;
    let (s1_low, s1_high) = bounds.s1;
    let (s2_low, s2_high) = bounds.s2;
    let (t1_low, t1_high) = bounds.t1;
    let (t2_low, t2_high) = bounds.t2;

    let (c_s1, c_s2, c_t1, c_t2) = profile.coarse_grid_sbpl2.unwrap_or((12, 12, 10, 10));
    let (r_s1, r_s2, r_t1, r_t2) = profile.refine_grid_sbpl2.unwrap_or((9, 9, 7, 7));

    let mut best_loss = f64::INFINITY;
    let mut best_params = None;

    for s1 in linspace(s1_low, s1_high, c_s1) {
        for s2 in linspace(s2_low, s2_high, c_s2) {
            if s2 >= s1 - 0.02 {
                continue;
            }
            for t1 in logspace(t1_low, t1_high, c_t1) {
                for t2 in logspace(t2_low, t2_high, c_t2) {
                    if t2 <= t1 * 1.05 {
                        continue;
                    }
                    let loss = sbpl2_loss(
                        t_vals, h_vals, weights, v_cap, s1, s2, t1, t2, profile.k1, profile.k2,
                        5000.0,
                    );
                    if loss < best_loss {
                        best_loss = loss;
                        best_params = Some((s1, s2, t1, t2));
                    }
                }
            }
        }
    }

    let (mut s1_best, mut s2_best, mut t1_best, mut t2_best) =
        best_params.ok_or(HcError::MissingWorldRecord)?;

    for s1 in refine_range(s1_best, s1_low, s1_high, r_s1, 0.25) {
        for s2 in refine_range(s2_best, s2_low, (s1 - 0.01).min(s2_high), r_s2, 0.25) {
            if s2 >= s1 - 0.01 {
                continue;
            }
            for t1 in logspace_refine(t1_best, t1_low, t1_high, r_t1, 0.5, 2.0) {
                for t2 in logspace_refine(t2_best, t2_low, t2_high, r_t2, 0.5, 2.0) {
                    if t2 <= t1 * 1.05 {
                        continue;
                    }
                    let loss = sbpl2_loss(
                        t_vals, h_vals, weights, v_cap, s1, s2, t1, t2, profile.k1, profile.k2,
                        5000.0,
                    );
                    if loss < best_loss {
                        best_loss = loss;
                        s1_best = s1;
                        s2_best = s2;
                        t1_best = t1;
                        t2_best = t2;
                    }
                }
            }
        }
    }

    Ok(WrModel::Sbpl2 {
        v_cap,
        s1: s1_best,
        s2: s2_best,
        t1: t1_best,
        t2: t2_best,
        k1: profile.k1,
        k2: profile.k2,
    })
}

fn sbpl_loss(
    t: &[f64],
    h: &[f64],
    weights: &[f64],
    v_cap: f64,
    s_inf: f64,
    t_star: f64,
    k: f64,
    penalty_lambda: f64,
) -> f64 {
    let pred: Vec<f64> = t
        .iter()
        .map(|&ti| h_sbpl_cap_scalar(ti, v_cap, s_inf, t_star, k))
        .collect();
    let mut loss = 0.0;
    for ((obs, pr), w) in h.iter().zip(pred.iter()).zip(weights.iter()) {
        if *w <= 0.0 {
            continue;
        }
        let denom = obs.max(1.0);
        let residual = (pr - obs) / denom;
        let under = (-residual).max(0.0);
        let over = residual.max(0.0);
        loss += w
            * (residual * residual + penalty_lambda * under * under + penalty_lambda * over * over);
    }
    loss
}

fn sbpl2_loss(
    t: &[f64],
    h: &[f64],
    weights: &[f64],
    v_cap: f64,
    s1: f64,
    s2: f64,
    t1: f64,
    t2: f64,
    k1: f64,
    k2: f64,
    penalty_lambda: f64,
) -> f64 {
    let pred: Vec<f64> = t
        .iter()
        .map(|&ti| h_sbpl_two_break_scalar(ti, v_cap, s1, s2, t1, t2, k1, k2))
        .collect();
    let mut loss = 0.0;
    for ((obs, pr), w) in h.iter().zip(pred.iter()).zip(weights.iter()) {
        if *w <= 0.0 {
            continue;
        }
        let denom = obs.max(1.0);
        let residual = (pr - obs) / denom;
        let under = (-residual).max(0.0);
        let over = residual.max(0.0);
        loss += w
            * (residual * residual + penalty_lambda * under * under + penalty_lambda * over * over);
    }
    loss
}

fn sample_wr_model(
    model: &WrModel,
    anchors: &[f64],
    wr_min_seconds: f64,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let anchor_min = anchors
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min)
        .max(WR_SAMPLE_SECONDS_MIN);
    let w_min = if wr_min_seconds > 0.0 {
        anchor_min.max(wr_min_seconds)
    } else {
        anchor_min
    };

    let anchor_max = anchors.iter().copied().fold(0.0, f64::max);
    let mut w_max = if wr_min_seconds > 0.0 {
        anchor_max.max(wr_min_seconds * 4.0)
    } else {
        anchor_max.max(anchor_min * 4.0)
    };
    if w_max <= w_min {
        w_max = (w_min * 1.01).max(w_min + 1.0);
    }

    let samples = logspace(w_min, w_max, 200);
    let climbs: Vec<f64> = samples.iter().map(|&t| model.evaluate(t)).collect();
    let rates_avg: Vec<f64> = samples
        .iter()
        .zip(climbs.iter())
        .map(|(&t, &c)| if t > 0.0 { c * 3600.0 / t } else { 0.0 })
        .collect();
    let rates_inst: Vec<f64> = samples
        .iter()
        .map(|&t| model.instantaneous_rate(t) * 3600.0)
        .collect();

    (samples, climbs, rates_avg, rates_inst)
}

fn linspace(start: f64, end: f64, count: usize) -> Vec<f64> {
    if count <= 1 {
        return vec![end];
    }
    let mut out = Vec::with_capacity(count);
    let step = (end - start) / (count as f64 - 1.0);
    for i in 0..count {
        out.push(start + step * i as f64);
    }
    out
}

fn logspace(start: f64, end: f64, count: usize) -> Vec<f64> {
    if start <= 0.0 || end <= 0.0 {
        return linspace(start, end, count);
    }
    let log_start = start.ln();
    let log_end = end.ln();
    linspace(log_start, log_end, count)
        .into_iter()
        .map(|x| x.exp())
        .collect()
}

fn refine_range(value: f64, lower: f64, upper: f64, count: usize, scale: f64) -> Vec<f64> {
    let span = (scale * value).max(0.05 * (upper - lower));
    let mut lo = (value - span).max(lower);
    let mut hi = (value + span).min(upper);
    if hi <= lo {
        lo = lower;
        hi = upper;
    }
    linspace(lo, hi, count.max(3))
}

fn logspace_refine(
    value: f64,
    lower: f64,
    upper: f64,
    count: usize,
    lower_scale: f64,
    upper_scale: f64,
) -> Vec<f64> {
    let lo = (value / lower_scale).max(lower);
    let hi = (value * upper_scale).min(upper);
    logspace(lo, hi, count.max(3))
}

fn h_sbpl_cap_scalar(t_s: f64, v_cap: f64, s_inf: f64, t_star: f64, k: f64) -> f64 {
    let t = t_s.max(WR_SAMPLE_SECONDS_MIN);
    let z = (t / t_star.max(WR_SAMPLE_SECONDS_MIN)).powf(k);
    v_cap * t * (1.0 + z).powf((s_inf - 1.0) / k)
}

fn d_h_sbpl_cap_scalar(t_s: f64, v_cap: f64, s_inf: f64, t_star: f64, k: f64) -> f64 {
    let t = t_s.max(WR_SAMPLE_SECONDS_MIN);
    if t_star <= 0.0 {
        return v_cap;
    }
    let z = (t / t_star).powf(k);
    let a = (1.0 + z).powf((s_inf - 1.0) / k);
    let dz_dt = (k / t_star) * (t / t_star).powf(k - 1.0);
    let term = 1.0 + (s_inf - 1.0) * t * dz_dt / (1.0 + z);
    v_cap * a * term
}

fn h_sbpl_two_break_scalar(
    t_s: f64,
    v_cap: f64,
    s_mid: f64,
    s_long: f64,
    t_break1: f64,
    t_break2: f64,
    k1: f64,
    k2: f64,
) -> f64 {
    let t = t_s.max(WR_SAMPLE_SECONDS_MIN);
    let z1 = (t / t_break1.max(WR_SAMPLE_SECONDS_MIN)).powf(k1);
    let z2 = (t / t_break2.max(WR_SAMPLE_SECONDS_MIN)).powf(k2);
    v_cap * t * (1.0 + z1).powf((s_mid - 1.0) / k1) * (1.0 + z2).powf((s_long - s_mid) / k2)
}
