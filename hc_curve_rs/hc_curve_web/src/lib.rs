use leptos::prelude::*;
use leptos::task::spawn_local;

#[cfg(target_arch = "wasm32")]
use leptos::mount::mount_to_body;

const APP_VERSION: &str = env!("CARGO_PKG_VERSION");
const APP_COMMIT: &str = env!("GIT_COMMIT_HASH");
const MIN_DURATION_SECONDS: f64 = 30.0;

#[cfg(feature = "chart_plotly")]
use wasm_bindgen::{closure::Closure, JsCast, JsValue};

#[cfg(feature = "chart_plotly")]
use wasm_bindgen_futures::JsFuture;

#[cfg(feature = "chart_plotly")]
use serde_wasm_bindgen::to_value as to_js;

#[cfg(feature = "chart_plotly")]
use web_sys::{Blob, FileList, HtmlInputElement, HtmlSelectElement};

#[cfg(feature = "chart_plotly")]
use std::cmp::Ordering;

#[cfg(feature = "chart_plotly")]
use std::collections::BTreeSet;

#[cfg(feature = "chart_plotly")]
use hc_curve::{
    compute_ascent_compare, compute_curves, compute_gain_time, list_ascent_algorithms, parse_records,
    AscentAlgorithmConfig, AscentCompareAlgorithmEntry, AscentCompareReport, Curves, GainTimeResult,
    HcError, Params,
};

#[cfg(feature = "chart_plotly")]
#[derive(Clone)]
struct FileBytes {
    name: String,
    ext: String,
    bytes: Vec<u8>,
}

#[cfg(feature = "chart_plotly")]
fn ext_from_name(name: &str) -> String {
    name.rsplit('.').next().unwrap_or("").to_ascii_lowercase()
}

#[cfg(feature = "chart_plotly")]
async fn read_files_from_input(input: &HtmlInputElement) -> Vec<FileBytes> {
    let mut out = Vec::new();
    if let Some(files) = input.files() {
        for i in 0..files.length() {
            if let Some(file) = files.item(i) {
                let name = file.name();
                let ext = ext_from_name(&name);
                match JsFuture::from(file.array_buffer()).await {
                    Ok(buf) => {
                        let u8arr = js_sys::Uint8Array::new(&buf);
                        let mut bytes = vec![0u8; u8arr.length() as usize];
                        u8arr.copy_to(&mut bytes[..]);
                        out.push(FileBytes { name, ext, bytes });
                    }
                    Err(_) => {}
                }
            }
        }
    }
    out
}

#[cfg(feature = "chart_plotly")]
async fn read_files_from_list(list: &FileList) -> Vec<FileBytes> {
    let mut out = Vec::new();
    for i in 0..list.length() {
        if let Some(file) = list.item(i) {
            let name = file.name();
            let ext = ext_from_name(&name);
            match JsFuture::from(file.array_buffer()).await {
                Ok(buf) => {
                    let u8arr = js_sys::Uint8Array::new(&buf);
                    let mut bytes = vec![0u8; u8arr.length() as usize];
                    u8arr.copy_to(&mut bytes[..]);
                    out.push(FileBytes { name, ext, bytes });
                }
                Err(_) => {}
            }
        }
    }
    out
}

#[cfg(feature = "chart_plotly")]
async fn yield_to_browser() {
    let promise = js_sys::Promise::new(&mut |resolve, _reject| {
        if let Some(window) = web_sys::window() {
            let resolve = resolve.clone();
            let cb = Closure::<dyn FnMut()>::new(move || {
                let _ = resolve.call0(&JsValue::NULL);
            });
            let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(
                cb.as_ref().unchecked_ref(),
                0,
            );
            cb.forget();
        } else {
            let _ = resolve.call0(&JsValue::NULL);
        }
    });
    let _ = JsFuture::from(promise).await;
}

#[cfg(feature = "chart_plotly")]
fn get_local_storage() -> Option<web_sys::Storage> {
    web_sys::window()?.local_storage().ok().flatten()
}

#[cfg(feature = "chart_plotly")]
fn load_theme_preference() -> String {
    if let Some(storage) = get_local_storage() {
        if let Ok(Some(value)) = storage.get_item("hc_theme") {
            let normalized = value.trim().to_ascii_lowercase();
            if normalized == "dark" || normalized == "light" {
                return normalized;
            }
        }
    }
    "light".to_string()
}

#[cfg(feature = "chart_plotly")]
fn apply_theme_preference(theme: &str) {
    let normalized = theme.trim().to_ascii_lowercase();
    if let Some(window) = web_sys::window() {
        if let Some(document) = window.document() {
            if let Some(root) = document.document_element() {
                let _ = root.set_attribute("data-theme", &normalized);
            }
        }
    }
    if let Some(storage) = get_local_storage() {
        let _ = storage.set_item("hc_theme", &normalized);
    }
}

#[cfg(feature = "chart_plotly")]
fn css_var(name: &str) -> Option<String> {
    let window = web_sys::window()?;
    let document = window.document()?;
    let root = document.document_element()?;
    let style = window.get_computed_style(&root).ok().flatten()?;
    let value = style.get_property_value(name).ok()?;
    let trimmed = value.trim().to_string();
    if trimmed.is_empty() { None } else { Some(trimmed) }
}

#[cfg(feature = "chart_plotly")]
#[derive(Clone)]
struct PlotTheme {
    fg: String,
    muted: String,
    panel: String,
    grid: String,
    accent: String,
}

#[cfg(feature = "chart_plotly")]
fn current_plot_theme() -> PlotTheme {
    PlotTheme {
        fg: css_var("--fg").unwrap_or_else(|| "#111".to_string()),
        muted: css_var("--muted").unwrap_or_else(|| "#666".to_string()),
        panel: css_var("--panel").unwrap_or_else(|| "#fff".to_string()),
        grid: css_var("--grid").unwrap_or_else(|| "#e8e8e8".to_string()),
        accent: css_var("--accent").unwrap_or_else(|| "#0a7".to_string()),
    }
}

#[cfg(feature = "chart_plotly")]
fn plot_xy(div_id: &str, traces: &js_sys::Array, layout: &JsValue) {
    if let Some(window) = web_sys::window() {
        if let Some(document) = window.document() {
            if let Some(div) = document.get_element_by_id(div_id) {
                let plotly = js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("Plotly"))
                    .unwrap_or(JsValue::UNDEFINED);
                if let Ok(func) = js_sys::Reflect::get(&plotly, &JsValue::from_str("react"))
                    .or_else(|_| js_sys::Reflect::get(&plotly, &JsValue::from_str("newPlot")))
                    .and_then(|v| v.dyn_into::<js_sys::Function>())
                {
                    let div_val = JsValue::from(div);
                    let traces_val = JsValue::from(traces.clone());
                    let config = to_js(&serde_json::json!({
                        "responsive": true,
                        "displaylogo": false,
                        "displayModeBar": "hover"
                    }))
                    .unwrap_or(JsValue::UNDEFINED);
                    let _ = func.call4(&JsValue::NULL, &div_val, &traces_val, layout, &config);
                }
            }
        }
    }
}

#[cfg(feature = "chart_plotly")]
fn build_scatter_trace(name: &str, mode: &str, x: &[f64], y: &[f64], line: Option<serde_json::Value>) -> JsValue {
    let trace = js_sys::Object::new();
    let name_js = JsValue::from_str(name);
    let mode_js = JsValue::from_str(mode);
    js_sys::Reflect::set(&trace, &JsValue::from_str("type"), &JsValue::from_str("scatter")).ok();
    js_sys::Reflect::set(&trace, &JsValue::from_str("name"), &name_js).ok();
    js_sys::Reflect::set(&trace, &JsValue::from_str("mode"), &mode_js).ok();
    let x_arr = js_sys::Array::new();
    for v in x {
        x_arr.push(&JsValue::from_f64(*v));
    }
    let y_arr = js_sys::Array::new();
    for v in y {
        y_arr.push(&JsValue::from_f64(*v));
    }
    js_sys::Reflect::set(&trace, &JsValue::from_str("x"), &x_arr.into()).ok();
    js_sys::Reflect::set(&trace, &JsValue::from_str("y"), &y_arr.into()).ok();
    if let Some(line_spec) = line {
        if let Ok(value) = to_js(&line_spec) {
            js_sys::Reflect::set(&trace, &JsValue::from_str("line"), &value).ok();
        }
    }
    trace.into()
}

#[cfg(feature = "chart_plotly")]
fn build_line_trace(name: &str, x: &[f64], y: &[f64], dash: &str, color: &str) -> JsValue {
    build_scatter_trace(
        name,
        "lines",
        x,
        y,
        Some(serde_json::json!({ "dash": dash, "color": color })),
    )
}

#[cfg(feature = "chart_plotly")]
fn build_iso_line(xmin: f64, xmax: f64, rate: f64, color: &str) -> JsValue {
    let x = [xmin, xmax];
    let y = [xmin / rate * 3600.0, xmax / rate * 3600.0];
    let trace = build_scatter_trace(
        "",
        "lines",
        &x,
        &y,
        Some(serde_json::json!({ "dash": "dot", "color": color, "width": 1 })),
    );
    let obj: js_sys::Object = trace.clone().into();
    js_sys::Reflect::set(&obj, &JsValue::from_str("hoverinfo"), &JsValue::from_str("skip")).ok();
    js_sys::Reflect::set(&obj, &JsValue::from_str("showlegend"), &JsValue::from_bool(false)).ok();
    trace
}

#[cfg(feature = "chart_plotly")]
fn fallback_param_candidates(base: &Params) -> Vec<(String, Params)> {
    let mut candidates = Vec::new();
    candidates.push(("default parameters".to_string(), base.clone()));

    if base.qc_enabled {
        let mut no_qc = base.clone();
        no_qc.qc_enabled = false;
        candidates.push(("disabled QC filtering".to_string(), no_qc));
    }

    if base.source == hc_curve::Source::Auto {
        let mut altitude = base.clone();
        altitude.source = hc_curve::Source::Altitude;
        candidates.push(("forced altitude ascent".to_string(), altitude));
    }

    if base.resample_1hz {
        let mut raw = base.clone();
        raw.resample_1hz = false;
        candidates.push(("raw sampling (no 1 Hz)".to_string(), raw));
    }

    candidates
}

#[cfg(feature = "chart_plotly")]
fn try_compute_curves(
    records_by_file: &Vec<Vec<hc_curve::FitRecord>>,
    base_params: &Params,
) -> Result<(Curves, Params, Vec<String>), HcError> {
    let mut attempts = fallback_param_candidates(base_params);
    let mut last_err: Option<HcError> = None;
    let mut notes: Vec<String> = Vec::new();

    for (label, candidate) in attempts.drain(..) {
        match compute_curves(records_by_file.clone(), &candidate) {
            Ok(curves) => {
                if label != "default parameters" {
                    notes.push(format!("Applied {label}"));
                }
                return Ok((curves, candidate, notes));
            }
            Err(err) => {
                last_err = Some(err);
                continue;
            }
        }
    }

    Err(last_err.unwrap_or_else(|| HcError::InvalidParameter("curve computation failed".into())))
}

#[cfg(feature = "chart_plotly")]
fn blob_url_from_str(s: &str) -> String {
    let arr = js_sys::Array::new();
    arr.push(&JsValue::from_str(s));
    let blob = Blob::new_with_str_sequence(&arr).unwrap();
    web_sys::Url::create_object_url_with_blob(&blob).unwrap()
}

#[cfg(feature = "chart_plotly")]
#[derive(Clone)]
struct MagicPoint {
    duration_s: f64,
    user_gain_m: f64,
    label: String,
    rate_m_per_hr: f64,
}

#[cfg(feature = "chart_plotly")]
fn collect_magic_points(curves: &Curves) -> Vec<MagicPoint> {
    let mut points = Vec::new();
    if let Some(rows) = curves.magic_rows.as_ref() {
        for row in rows {
            let duration = row.get("duration_s").copied().unwrap_or_default();
            let gain = row.get("user_gain_m").copied().unwrap_or_default();
            if duration <= 0.0 || gain <= 0.0 {
                continue;
            }
            let score = row.get("score_pct").copied();
            let label = if let Some(score_val) = score {
                format!("{gain:.0} m ({score_val:.0}%)")
            } else {
                format!("{gain:.0} m")
            };
            let rate = if duration > 0.0 {
                gain * 3600.0 / duration
            } else {
                0.0
            };
            points.push(MagicPoint {
                duration_s: duration,
                user_gain_m: gain,
                label,
                rate_m_per_hr: rate,
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

#[cfg(feature = "chart_plotly")]
fn filter_minutes_and_values(
    durations: &[u64],
    values: &[f64],
    min_seconds: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (&d, &v) in durations.iter().zip(values.iter()) {
        if (d as f64) >= min_seconds {
            xs.push(d as f64 / 60.0);
            ys.push(v);
        }
    }
    (xs, ys)
}

#[cfg(feature = "chart_plotly")]
fn filter_minutes_and_rates(
    durations: &[u64],
    gains: &[f64],
    min_seconds: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for (&d, &g) in durations.iter().zip(gains.iter()) {
        if (d as f64) >= min_seconds && d > 0 {
            xs.push(d as f64 / 60.0);
            ys.push(g * 3600.0 / d as f64);
        }
    }
    (xs, ys)
}

#[cfg(feature = "chart_plotly")]
fn render_curve_plot(
    curves: &Curves,
    mode: &str,
    show_wr: bool,
    show_personal: bool,
    show_sessions: bool,
    show_magic: bool,
    ylog_rate: bool,
    ylog_climb: bool,
) {
    let filtered_points: Vec<&hc_curve::CurvePoint> = curves
        .points
        .iter()
        .filter(|p| (p.duration_s as f64) >= MIN_DURATION_SECONDS)
        .collect();

    if filtered_points.is_empty() {
        let empty = js_sys::Array::new();
        let layout = to_js(&serde_json::json!({
            "title": "No curve points ≥ 30 s",
            "xaxis": { "title": "Duration (min)" },
            "yaxis": { "title": "Climb (m)" }
        }))
        .unwrap();
        plot_xy("curve_plot", &empty, &layout);
        return;
    }

    let durations_min: Vec<f64> = filtered_points
        .iter()
        .map(|p| p.duration_s as f64 / 60.0)
        .collect();
    let climbs: Vec<f64> = filtered_points.iter().map(|p| p.max_climb_m).collect();
    let rates: Vec<f64> = filtered_points
        .iter()
        .map(|p| p.climb_rate_m_per_hr)
        .collect();

    let theme = current_plot_theme();

    let magic_points = if show_magic {
        collect_magic_points(curves)
            .into_iter()
            .filter(|pt| pt.duration_s >= MIN_DURATION_SECONDS)
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };

    let data = js_sys::Array::new();

    match mode {
        "climb" => {
            let climb_trace = build_scatter_trace("Climb (m)", "lines", &durations_min, &climbs, None);
            let climb_obj = js_sys::Object::from(climb_trace.clone());
            if let Ok(line) = to_js(&serde_json::json!({ "width": 2, "color": theme.fg.clone() })) {
                js_sys::Reflect::set(&climb_obj, &JsValue::from_str("line"), &line).ok();
            }
            js_sys::Reflect::set(
                &climb_obj,
                &JsValue::from_str("hovertemplate"),
                &JsValue::from_str("Duration %{x:.1f} min<br>Climb %{y:.1f} m<extra></extra>"),
            )
            .ok();
            data.push(&climb_trace);

            if show_wr {
                if let Some((durs, gains)) = curves.wr_curve.as_ref() {
                    let (wr_minutes, wr_values) =
                        filter_minutes_and_values(durs, gains, MIN_DURATION_SECONDS);
                    if !wr_minutes.is_empty() {
                        let trace = build_line_trace("WR climb", &wr_minutes, &wr_values, "dash", theme.muted.as_str());
                        data.push(&trace);
                    }
                }
            }

            if show_personal {
                if let Some((durs, gains)) = curves.personal_curve.as_ref() {
                    let (minutes, values) =
                        filter_minutes_and_values(durs, gains, MIN_DURATION_SECONDS);
                    if !minutes.is_empty() {
                        let trace = build_line_trace("Personal climb", &minutes, &values, "solid", "#1e90ff");
                        data.push(&trace);
                    }
                }
                if let Some((durs, gains)) = curves.goal_curve.as_ref() {
                    let (minutes, values) =
                        filter_minutes_and_values(durs, gains, MIN_DURATION_SECONDS);
                    if !minutes.is_empty() {
                        let trace = build_line_trace("Goal climb", &minutes, &values, "solid", "#228b22");
                        data.push(&trace);
                    }
                }
            }

            if show_sessions && !curves.session_curves.is_empty() {
                let palette = ["#87cefa", "#fa8072", "#98fb98", "#d8bfd8"];
                for (idx, session) in curves.session_curves.iter().enumerate() {
                    if session.durations.is_empty() || session.climbs.is_empty() {
                        continue;
                    }
                    let (minutes, values) = filter_minutes_and_values(
                        &session.durations,
                        &session.climbs,
                        MIN_DURATION_SECONDS,
                    );
                    if minutes.is_empty() {
                        continue;
                    }
                    let trace = build_scatter_trace(
                        &format!("Session {}", idx + 1),
                        "lines",
                        &minutes,
                        &values,
                        None,
                    );
                    let obj = js_sys::Object::from(trace.clone());
                    if let Ok(line) = to_js(&serde_json::json!({
                        "width": 1.5,
                        "color": palette[idx % palette.len()]
                    })) {
                        js_sys::Reflect::set(&obj, &JsValue::from_str("line"), &line).ok();
                    }
                    js_sys::Reflect::set(&obj, &JsValue::from_str("showlegend"), &JsValue::from_bool(false)).ok();
                    data.push(&trace);
                }
            }

            if !magic_points.is_empty() {
                let trace = js_sys::Object::new();
                let xs = js_sys::Array::new();
                let ys = js_sys::Array::new();
                let texts = js_sys::Array::new();
                for pt in &magic_points {
                    xs.push(&JsValue::from_f64(pt.duration_s / 60.0));
                    ys.push(&JsValue::from_f64(pt.user_gain_m));
                    texts.push(&JsValue::from_str(&pt.label));
                }
                js_sys::Reflect::set(&trace, &JsValue::from_str("type"), &JsValue::from_str("scatter")).ok();
                js_sys::Reflect::set(&trace, &JsValue::from_str("mode"), &JsValue::from_str("markers+text")).ok();
                js_sys::Reflect::set(&trace, &JsValue::from_str("name"), &JsValue::from_str("Magic durations")).ok();
                js_sys::Reflect::set(&trace, &JsValue::from_str("x"), &JsValue::from(xs)).ok();
                js_sys::Reflect::set(&trace, &JsValue::from_str("y"), &JsValue::from(ys)).ok();
                let marker = serde_json::json!({ "size": 8, "color": theme.fg.clone() });
                if let Ok(marker_js) = to_js(&marker) {
                    js_sys::Reflect::set(&trace, &JsValue::from_str("marker"), &marker_js).ok();
                }
                js_sys::Reflect::set(&trace, &JsValue::from_str("text"), &JsValue::from(texts)).ok();
                js_sys::Reflect::set(&trace, &JsValue::from_str("textposition"), &JsValue::from_str("top center")).ok();
                data.push(&JsValue::from(trace));
            }

            let layout = serde_json::json!({
                "title": "Climb vs Duration",
                "hovermode": "x unified",
                "paper_bgcolor": theme.panel.clone(),
                "plot_bgcolor": theme.panel.clone(),
                "font": { "color": theme.fg.clone() },
                "margin": { "l": 56, "r": 24, "t": 54, "b": 80 },
                "xaxis": { "title": "Duration (min)", "gridcolor": theme.grid.clone(), "zerolinecolor": theme.grid.clone() },
                "yaxis": { "title": "Climb (m)", "type": if ylog_climb { "log" } else { "linear" }, "gridcolor": theme.grid.clone(), "zerolinecolor": theme.grid.clone() },
                "legend": { "orientation": "h", "y": -0.25, "x": 0 }
            });
            let layout_js = to_js(&layout).unwrap();
            plot_xy("curve_plot", &data, &layout_js);
        }
        "rate" => {
            let rate_trace = build_scatter_trace("Climb rate (m/h)", "lines", &durations_min, &rates, None);
            let rate_obj = js_sys::Object::from(rate_trace.clone());
            if let Ok(line) = to_js(&serde_json::json!({ "width": 2, "color": "#c00564" })) {
                js_sys::Reflect::set(&rate_obj, &JsValue::from_str("line"), &line).ok();
            }
            js_sys::Reflect::set(
                &rate_obj,
                &JsValue::from_str("hovertemplate"),
                &JsValue::from_str("Duration %{x:.1f} min<br>Rate %{y:.0f} m/h<extra></extra>"),
            )
            .ok();
            data.push(&rate_trace);

            if show_wr {
                if let Some((durs, gains)) = curves.wr_curve.as_ref() {
                    let (wr_minutes, wr_rates) =
                        filter_minutes_and_rates(durs, gains, MIN_DURATION_SECONDS);
                    if !wr_minutes.is_empty() {
                        let trace =
                            build_line_trace("WR rate", &wr_minutes, &wr_rates, "dash", theme.muted.as_str());
                        data.push(&trace);
                    }
                }
            }

            if show_personal {
                if let Some((durs, gains)) = curves.personal_curve.as_ref() {
                    let (minutes, values) =
                        filter_minutes_and_rates(durs, gains, MIN_DURATION_SECONDS);
                    if !minutes.is_empty() {
                        let trace =
                            build_line_trace("Personal rate", &minutes, &values, "solid", "#1e90ff");
                        data.push(&trace);
                    }
                }
                if let Some((durs, gains)) = curves.goal_curve.as_ref() {
                    let (minutes, values) =
                        filter_minutes_and_rates(durs, gains, MIN_DURATION_SECONDS);
                    if !minutes.is_empty() {
                        let trace =
                            build_line_trace("Goal rate", &minutes, &values, "solid", "#228b22");
                        data.push(&trace);
                    }
                }
            }

            if !magic_points.is_empty() {
                let trace = js_sys::Object::new();
                let xs = js_sys::Array::new();
                let ys = js_sys::Array::new();
                let texts = js_sys::Array::new();
                for pt in &magic_points {
                    xs.push(&JsValue::from_f64(pt.duration_s / 60.0));
                    ys.push(&JsValue::from_f64(pt.rate_m_per_hr));
                    texts.push(&JsValue::from_str(&pt.label));
                }
                js_sys::Reflect::set(&trace, &JsValue::from_str("type"), &JsValue::from_str("scatter")).ok();
                js_sys::Reflect::set(&trace, &JsValue::from_str("mode"), &JsValue::from_str("markers+text")).ok();
                js_sys::Reflect::set(&trace, &JsValue::from_str("name"), &JsValue::from_str("Magic rate")).ok();
                js_sys::Reflect::set(&trace, &JsValue::from_str("x"), &JsValue::from(xs)).ok();
                js_sys::Reflect::set(&trace, &JsValue::from_str("y"), &JsValue::from(ys)).ok();
                if let Ok(marker_js) =
                    to_js(&serde_json::json!({ "size": 8, "color": theme.fg.clone() }))
                {
                    js_sys::Reflect::set(&trace, &JsValue::from_str("marker"), &marker_js).ok();
                }
                js_sys::Reflect::set(&trace, &JsValue::from_str("text"), &JsValue::from(texts)).ok();
                js_sys::Reflect::set(&trace, &JsValue::from_str("textposition"), &JsValue::from_str("top center")).ok();
                data.push(&JsValue::from(trace));
            }

            let layout = serde_json::json!({
                "title": "Climb Rate vs Duration",
                "hovermode": "x unified",
                "paper_bgcolor": theme.panel.clone(),
                "plot_bgcolor": theme.panel.clone(),
                "font": { "color": theme.fg.clone() },
                "margin": { "l": 56, "r": 24, "t": 54, "b": 80 },
                "xaxis": { "title": "Duration (min)", "gridcolor": theme.grid.clone(), "zerolinecolor": theme.grid.clone() },
                "yaxis": { "title": "Rate (m/h)", "type": if ylog_rate { "log" } else { "linear" }, "gridcolor": theme.grid.clone(), "zerolinecolor": theme.grid.clone() },
                "legend": { "orientation": "h", "y": -0.25, "x": 0 }
            });
            let layout_js = to_js(&layout).unwrap();
            plot_xy("curve_plot", &data, &layout_js);
        }
        _ => {
            let climb_trace = build_scatter_trace("Climb (m)", "lines", &durations_min, &climbs, None);
            let climb_obj = js_sys::Object::from(climb_trace.clone());
            if let Ok(line) = to_js(&serde_json::json!({ "width": 2, "color": theme.fg.clone() })) {
                js_sys::Reflect::set(&climb_obj, &JsValue::from_str("line"), &line).ok();
            }
            js_sys::Reflect::set(
                &climb_obj,
                &JsValue::from_str("hovertemplate"),
                &JsValue::from_str("Duration %{x:.1f} min<br>Climb %{y:.1f} m<extra></extra>"),
            )
            .ok();
            data.push(&climb_trace);

            let rate_trace = build_scatter_trace("Climb rate (m/h)", "lines", &durations_min, &rates, None);
            let rate_obj = js_sys::Object::from(rate_trace.clone());
            if let Ok(line) = to_js(&serde_json::json!({ "width": 2, "color": "#c00564" })) {
                js_sys::Reflect::set(&rate_obj, &JsValue::from_str("line"), &line).ok();
            }
            js_sys::Reflect::set(&rate_obj, &JsValue::from_str("yaxis"), &JsValue::from_str("y2")).ok();
            js_sys::Reflect::set(
                &rate_obj,
                &JsValue::from_str("hovertemplate"),
                &JsValue::from_str("Duration %{x:.1f} min<br>Rate %{y:.0f} m/h<extra></extra>"),
            )
            .ok();
            data.push(&rate_trace);

            if show_wr {
                if let Some((durs, gains)) = curves.wr_curve.as_ref() {
                    let (wr_minutes, wr_values) =
                        filter_minutes_and_values(durs, gains, MIN_DURATION_SECONDS);
                    if !wr_minutes.is_empty() {
                        let trace = build_line_trace("WR climb", &wr_minutes, &wr_values, "dash", theme.muted.as_str());
                        data.push(&trace);
                    }
                }
            }

            if show_personal {
                if let Some((durs, gains)) = curves.personal_curve.as_ref() {
                    let (minutes, values) =
                        filter_minutes_and_values(durs, gains, MIN_DURATION_SECONDS);
                    if !minutes.is_empty() {
                        let trace =
                            build_line_trace("Personal climb", &minutes, &values, "solid", "#1e90ff");
                        data.push(&trace);
                    }
                }
                if let Some((durs, gains)) = curves.goal_curve.as_ref() {
                    let (minutes, values) =
                        filter_minutes_and_values(durs, gains, MIN_DURATION_SECONDS);
                    if !minutes.is_empty() {
                        let trace =
                            build_line_trace("Goal climb", &minutes, &values, "solid", "#228b22");
                        data.push(&trace);
                    }
                }
            }

            if show_sessions && !curves.session_curves.is_empty() {
                let palette = ["#87cefa", "#fa8072", "#98fb98", "#d8bfd8"];
                for (idx, session) in curves.session_curves.iter().enumerate() {
                    if session.durations.is_empty() || session.climbs.is_empty() {
                        continue;
                    }
                    let (minutes, values) = filter_minutes_and_values(
                        &session.durations,
                        &session.climbs,
                        MIN_DURATION_SECONDS,
                    );
                    if minutes.is_empty() {
                        continue;
                    }
                    let trace = build_scatter_trace(
                        &format!("Session {}", idx + 1),
                        "lines",
                        &minutes,
                        &values,
                        None,
                    );
                    let obj = js_sys::Object::from(trace.clone());
                    if let Ok(line) = to_js(&serde_json::json!({
                        "width": 1.5,
                        "color": palette[idx % palette.len()]
                    })) {
                        js_sys::Reflect::set(&obj, &JsValue::from_str("line"), &line).ok();
                    }
                    js_sys::Reflect::set(&obj, &JsValue::from_str("showlegend"), &JsValue::from_bool(false)).ok();
                    data.push(&trace);
                }
            }

            if !magic_points.is_empty() {
                let trace = js_sys::Object::new();
                let xs = js_sys::Array::new();
                let ys = js_sys::Array::new();
                let texts = js_sys::Array::new();
                for pt in &magic_points {
                    xs.push(&JsValue::from_f64(pt.duration_s / 60.0));
                    ys.push(&JsValue::from_f64(pt.user_gain_m));
                    texts.push(&JsValue::from_str(&pt.label));
                }
                js_sys::Reflect::set(&trace, &JsValue::from_str("type"), &JsValue::from_str("scatter")).ok();
                js_sys::Reflect::set(&trace, &JsValue::from_str("mode"), &JsValue::from_str("markers+text")).ok();
                js_sys::Reflect::set(&trace, &JsValue::from_str("name"), &JsValue::from_str("Magic durations")).ok();
                js_sys::Reflect::set(&trace, &JsValue::from_str("x"), &JsValue::from(xs)).ok();
                js_sys::Reflect::set(&trace, &JsValue::from_str("y"), &JsValue::from(ys)).ok();
                let marker = serde_json::json!({ "size": 8, "color": theme.fg.clone() });
                if let Ok(marker_js) = to_js(&marker) {
                    js_sys::Reflect::set(&trace, &JsValue::from_str("marker"), &marker_js).ok();
                }
                js_sys::Reflect::set(&trace, &JsValue::from_str("text"), &JsValue::from(texts)).ok();
                js_sys::Reflect::set(&trace, &JsValue::from_str("textposition"), &JsValue::from_str("top center")).ok();
                data.push(&JsValue::from(trace));
            }

            let layout = serde_json::json!({
                "title": "Climb + Rate vs Duration",
                "hovermode": "x unified",
                "paper_bgcolor": theme.panel.clone(),
                "plot_bgcolor": theme.panel.clone(),
                "font": { "color": theme.fg.clone() },
                "margin": { "l": 56, "r": 40, "t": 54, "b": 90 },
                "xaxis": { "title": "Duration (min)", "gridcolor": theme.grid.clone(), "zerolinecolor": theme.grid.clone() },
                "yaxis": { "title": "Climb (m)", "type": if ylog_climb { "log" } else { "linear" }, "gridcolor": theme.grid.clone(), "zerolinecolor": theme.grid.clone() },
                "yaxis2": {
                    "title": "Rate (m/h)",
                    "type": if ylog_rate { "log" } else { "linear" },
                    "overlaying": "y",
                    "side": "right",
                    "showgrid": false
                },
                "legend": { "orientation": "h", "y": -0.25, "x": 0 }
            });
            let layout_js = to_js(&layout).unwrap();
            plot_xy("curve_plot", &data, &layout_js);
        }
    }
}

#[cfg(feature = "chart_plotly")]
fn render_ascent_compare_plot(report: &AscentCompareReport, ylog_climb: bool) {
    let theme = current_plot_theme();
    let data = js_sys::Array::new();

    let palette = ["#1e90ff", "#228b22", "#c00564", "#ffa500", "#8a2be2", "#00bcd4"];
    let mut palette_idx = 0usize;

    let push_curve = |label: &str,
                      curve: &[hc_curve::CurvePoint],
                      color: &str,
                      width: f64,
                      dash: &str| {
        let mut xs: Vec<f64> = Vec::new();
        let mut ys: Vec<f64> = Vec::new();
        for p in curve {
            if (p.duration_s as f64) < MIN_DURATION_SECONDS {
                continue;
            }
            xs.push(p.duration_s as f64 / 60.0);
            ys.push(p.max_climb_m);
        }
        if xs.is_empty() {
            return;
        }
        let trace = build_scatter_trace(label, "lines", &xs, &ys, None);
        let obj = js_sys::Object::from(trace.clone());
        if let Ok(line) = to_js(&serde_json::json!({ "width": width, "color": color, "dash": dash })) {
            js_sys::Reflect::set(&obj, &JsValue::from_str("line"), &line).ok();
        }
        js_sys::Reflect::set(
            &obj,
            &JsValue::from_str("hovertemplate"),
            &JsValue::from_str("Duration %{x:.1f} min<br>Climb %{y:.1f} m<extra></extra>"),
        )
        .ok();
        data.push(&trace);
    };

    if let AscentCompareAlgorithmEntry::Ok { name, curve, .. } = &report.baseline {
        push_curve(name.as_str(), curve, theme.fg.as_str(), 2.5, "solid");
    }

    for entry in &report.entries {
        let AscentCompareAlgorithmEntry::Ok { name, curve, .. } = entry else {
            continue;
        };
        let color = palette[palette_idx % palette.len()];
        palette_idx += 1;
        push_curve(name.as_str(), curve, color, 1.8, "solid");
    }

    let layout = serde_json::json!({
        "title": "Algorithm Lab: Climb vs Duration",
        "hovermode": "x unified",
        "paper_bgcolor": theme.panel.clone(),
        "plot_bgcolor": theme.panel.clone(),
        "font": { "color": theme.fg.clone() },
        "margin": { "l": 56, "r": 24, "t": 54, "b": 80 },
        "xaxis": { "title": "Duration (min)", "gridcolor": theme.grid.clone(), "zerolinecolor": theme.grid.clone() },
        "yaxis": { "title": "Climb (m)", "type": if ylog_climb { "log" } else { "linear" }, "gridcolor": theme.grid.clone(), "zerolinecolor": theme.grid.clone() },
        "legend": { "orientation": "h", "y": -0.25, "x": 0 }
    });
    let layout_js = to_js(&layout).unwrap();
    plot_xy("algo_plot", &data, &layout_js);
}

#[cfg(feature = "chart_plotly")]
fn render_gain_plot(
    result: &GainTimeResult,
    mode: &str,
    show_wr: bool,
    show_personal: bool,
    ylog_time: bool,
) {
    if result.curve.is_empty() {
        return;
    }

    let theme = current_plot_theme();

    let gains_m: Vec<f64> = result.curve.iter().map(|p| p.gain_m).collect();
    let gains_display = gains_m.clone();

    let data = js_sys::Array::new();

    match mode {
        "rate" => {
            let rates: Vec<f64> = result
                .curve
                .iter()
                .map(|p| p.avg_rate_m_per_hr)
                .collect();
            let rate_trace =
                build_scatter_trace("Average rate", "lines", &gains_display, &rates, None);
            let rate_obj = js_sys::Object::from(rate_trace.clone());
            if let Ok(line) = to_js(&serde_json::json!({ "width": 2, "color": "#0072b2" })) {
                js_sys::Reflect::set(&rate_obj, &JsValue::from_str("line"), &line).ok();
            }
            js_sys::Reflect::set(
                &rate_obj,
                &JsValue::from_str("hovertemplate"),
                &JsValue::from_str("Gain %{x:.0f} m<br>Rate %{y:.0f} m/h<extra></extra>"),
            )
            .ok();
            data.push(&rate_trace);

            if show_wr {
                if let Some(curve) = result.wr_curve.as_ref() {
                    let wr_x: Vec<f64> = curve.iter().map(|p| p.gain_m).collect();
                    let wr_y: Vec<f64> = curve.iter().map(|p| p.avg_rate_m_per_hr).collect();
                    let trace = build_line_trace("WR rate", &wr_x, &wr_y, "dash", theme.muted.as_str());
                    data.push(&trace);
                }
            }

            if show_personal {
                if let Some(curve) = result.personal_curve.as_ref() {
                    let per_x: Vec<f64> = curve.iter().map(|p| p.gain_m).collect();
                    let per_y: Vec<f64> = curve.iter().map(|p| p.avg_rate_m_per_hr).collect();
                    let trace =
                        build_line_trace("Personal rate", &per_x, &per_y, "solid", "#1e90ff");
                    data.push(&trace);
                }
            }

            if !result.targets.is_empty() {
                let xs = js_sys::Array::new();
                let ys = js_sys::Array::new();
                for target in &result.targets {
                    xs.push(&JsValue::from_f64(target.gain_m));
                    ys.push(&JsValue::from_f64(target.avg_rate_m_per_hr));
                }
                let trace = js_sys::Object::new();
                js_sys::Reflect::set(&trace, &JsValue::from_str("type"), &JsValue::from_str("scatter")).ok();
                js_sys::Reflect::set(&trace, &JsValue::from_str("mode"), &JsValue::from_str("markers")).ok();
                js_sys::Reflect::set(&trace, &JsValue::from_str("name"), &JsValue::from_str("Targets")).ok();
                js_sys::Reflect::set(&trace, &JsValue::from_str("x"), &JsValue::from(xs)).ok();
                js_sys::Reflect::set(&trace, &JsValue::from_str("y"), &JsValue::from(ys)).ok();
                if let Ok(marker) = to_js(&serde_json::json!({ "size": 8, "color": theme.accent.clone() })) {
                    js_sys::Reflect::set(&trace, &JsValue::from_str("marker"), &marker).ok();
                }
                data.push(&JsValue::from(trace));
            }

            let layout = serde_json::json!({
                "title": "Average Rate vs Gain",
                "hovermode": "x unified",
                "paper_bgcolor": theme.panel.clone(),
                "plot_bgcolor": theme.panel.clone(),
                "font": { "color": theme.fg.clone() },
                "margin": { "l": 56, "r": 24, "t": 54, "b": 80 },
                "xaxis": { "title": "Gain (m)", "gridcolor": theme.grid.clone(), "zerolinecolor": theme.grid.clone() },
                "yaxis": { "title": "Rate (m/h)", "type": "linear", "gridcolor": theme.grid.clone(), "zerolinecolor": theme.grid.clone() },
                "legend": { "orientation": "h", "y": -0.25, "x": 0 }
            });
            let layout_js = to_js(&layout).unwrap();
            plot_xy("gain_time_plot", &data, &layout_js);
        }
        _ => {
            let times_minutes: Vec<f64> = result
                .curve
                .iter()
                .map(|p| p.min_time_s / 60.0)
                .collect();
            let time_trace =
                build_scatter_trace("Min time", "lines", &gains_display, &times_minutes, None);
            let time_obj = js_sys::Object::from(time_trace.clone());
            if let Ok(line) = to_js(&serde_json::json!({ "width": 2, "color": "#0072b2" })) {
                js_sys::Reflect::set(&time_obj, &JsValue::from_str("line"), &line).ok();
            }
            js_sys::Reflect::set(
                &time_obj,
                &JsValue::from_str("hovertemplate"),
                &JsValue::from_str("Gain %{x:.0f} m<br>Time %{y:.1f} min<extra></extra>"),
            )
            .ok();
            data.push(&time_trace);

            if show_wr {
                if let Some(curve) = result.wr_curve.as_ref() {
                    let wr_x: Vec<f64> = curve.iter().map(|p| p.gain_m).collect();
                    let wr_y: Vec<f64> = curve.iter().map(|p| p.min_time_s / 60.0).collect();
                    let trace = build_line_trace("WR", &wr_x, &wr_y, "dash", theme.muted.as_str());
                    data.push(&trace);
                }
            }

            if show_personal {
                if let Some(curve) = result.personal_curve.as_ref() {
                    let per_x: Vec<f64> = curve.iter().map(|p| p.gain_m).collect();
                    let per_y: Vec<f64> = curve.iter().map(|p| p.min_time_s / 60.0).collect();
                    let trace = build_line_trace("Personal", &per_x, &per_y, "solid", "#1e90ff");
                    data.push(&trace);
                }
            }

            // Iso-rate guides
            let xmin = gains_display
                .iter()
                .cloned()
                .fold(f64::INFINITY, f64::min)
                .max(0.0);
            let xmax = gains_display
                .iter()
                .cloned()
                .fold(0.0_f64, f64::max)
                .max(1.0);
            let rates = [800.0, 1000.0, 1200.0, 1500.0, 2000.0, 2500.0];
            for rate in rates {
                if rate <= 0.0 {
                    continue;
                }
                let trace = build_iso_line(xmin, xmax, rate, theme.grid.as_str());
                data.push(&trace);
            }

            if !result.targets.is_empty() {
                let xs = js_sys::Array::new();
                let ys = js_sys::Array::new();
                for target in &result.targets {
                    xs.push(&JsValue::from_f64(target.gain_m));
                    ys.push(&JsValue::from_f64(target.min_time_s / 60.0));
                }
                let trace = js_sys::Object::new();
                js_sys::Reflect::set(&trace, &JsValue::from_str("type"), &JsValue::from_str("scatter")).ok();
                js_sys::Reflect::set(&trace, &JsValue::from_str("mode"), &JsValue::from_str("markers")).ok();
                js_sys::Reflect::set(&trace, &JsValue::from_str("name"), &JsValue::from_str("Targets")).ok();
                js_sys::Reflect::set(&trace, &JsValue::from_str("x"), &JsValue::from(xs)).ok();
                js_sys::Reflect::set(&trace, &JsValue::from_str("y"), &JsValue::from(ys)).ok();
                if let Ok(marker) = to_js(&serde_json::json!({ "size": 8, "color": theme.accent.clone() })) {
                    js_sys::Reflect::set(&trace, &JsValue::from_str("marker"), &marker).ok();
                }
                data.push(&JsValue::from(trace));
            }

            let layout = serde_json::json!({
                "title": "Minimum Time vs Gain",
                "hovermode": "x unified",
                "paper_bgcolor": theme.panel.clone(),
                "plot_bgcolor": theme.panel.clone(),
                "font": { "color": theme.fg.clone() },
                "margin": { "l": 56, "r": 24, "t": 54, "b": 80 },
                "xaxis": { "title": "Gain (m)", "gridcolor": theme.grid.clone(), "zerolinecolor": theme.grid.clone() },
                "yaxis": { "title": "Time (min)", "type": if ylog_time { "log" } else { "linear" }, "gridcolor": theme.grid.clone(), "zerolinecolor": theme.grid.clone() },
                "legend": { "orientation": "h", "y": -0.25, "x": 0 }
            });
            let layout_js = to_js(&layout).unwrap();
            plot_xy("gain_time_plot", &data, &layout_js);
        }
    }
}

#[component]
pub fn App() -> impl IntoView {
    let (theme, set_theme) = signal(load_theme_preference());
    Effect::new({
        let theme = theme.clone();
        move |_| {
            apply_theme_preference(&theme.get());
        }
    });

    let (files, set_files) = signal(Vec::<FileBytes>::new());
    let (busy, set_busy) = signal(false);
    let (progress, set_progress) = signal(0.0_f64);
    let (status, set_status) = signal(String::from("No files selected."));
    let (curve_href, set_curve_href) = signal(String::new());
    let (gain_href, set_gain_href) = signal(String::new());
    let (source_sel, set_source_sel) = signal(String::from("auto"));
    let (all_windows, set_all_windows) = signal(false);
    let (step_s, set_step_s) = signal(1u64);
    let (max_dur_s, set_max_dur_s) = signal(0u64); // 0 => None
    let (show_wr, set_show_wr) = signal(true);
    let (show_personal, set_show_personal) = signal(true);
    let (show_sessions, set_show_sessions) = signal(false);
    let (show_magic, set_show_magic) = signal(false);
    let (curve_mode, set_curve_mode) = signal(String::from("combined"));
    let (curve_ylog_rate, set_curve_ylog_rate) = signal(false);
    let (curve_ylog_climb, set_curve_ylog_climb) = signal(false);
    let (gain_mode, set_gain_mode) = signal(String::from("time"));
    let (gain_ylog_time, set_gain_ylog_time) = signal(false);
    let (curve_data, set_curve_data) = signal(Option::<Curves>::None);
    let (gain_data, set_gain_data) = signal(Option::<GainTimeResult>::None);
    let (diag_source, set_diag_source) = signal(String::new());
    let (diag_span_s, set_diag_span_s) = signal(0.0_f64);
    let (diag_gain_m, set_diag_gain_m) = signal(0.0_f64);

    let (records_cache, set_records_cache) = signal(Option::<Vec<Vec<hc_curve::FitRecord>>>::None);
    let (params_cache, set_params_cache) = signal(Option::<Params>::None);

    let default_algo_set: BTreeSet<String> = [
        "strava.altitude.threshold.v1",
        "goldencheetah.altitude.hysteresis.v1",
        "twonav.altitude.min_altitude_increase.v1",
    ]
    .into_iter()
    .map(|id| id.to_string())
    .collect();
    let (algo_lab_enabled, set_algo_lab_enabled) = signal(false);
    let (algo_baseline, set_algo_baseline) = signal(String::from("hc.altitude.canonical.v1"));
    let (algo_selected, set_algo_selected) = signal(default_algo_set);
    let (algo_report, set_algo_report) = signal(Option::<AscentCompareReport>::None);

    fn fmt_hms(secs: f64) -> String {
        let s = if secs.is_finite() && secs >= 0.0 { secs } else { 0.0 };
        let total = s.round() as i64;
        let h = total / 3600; let m = (total % 3600) / 60; let ss = total % 60;
        if h > 0 { format!("{}:{:02}:{:02}", h, m, ss) } else { format!("{}:{:02}", m, ss) }
    }

    // File input handler
    let on_files = move |ev: leptos::ev::Event| {
        let set_files_cb = set_files.clone();
        let set_status_cb = set_status.clone();
        if let Some(target) = ev.target() {
            if let Ok(input) = target.dyn_into::<HtmlInputElement>() {
                let input_clone = input.clone();
                set_status_cb.set("Reading files…".to_string());
                spawn_local(async move {
                    let bytes = read_files_from_input(&input_clone).await;
                    input_clone.set_value("");
                    set_files_cb.set(bytes);
                    set_status_cb.set("Files ready. Click Compute.".to_string());
                });
            }
        }
    };

    let on_toggle_theme = move |_ev: leptos::ev::MouseEvent| {
        let next = if theme.get_untracked() == "dark" { "light" } else { "dark" };
        set_theme.set(next.to_string());
    };

    // Re-render curve chart whenever data or controls change
    Effect::new({
        let curve_mode = curve_mode.clone();
        let show_wr = show_wr.clone();
        let show_personal = show_personal.clone();
        let show_sessions = show_sessions.clone();
        let show_magic = show_magic.clone();
        let curve_ylog_rate = curve_ylog_rate.clone();
        let curve_ylog_climb = curve_ylog_climb.clone();
        let theme = theme.clone();
        move |_| {
            let _ = theme.get();
            if let Some(curves) = curve_data.get() {
                let mode = curve_mode.get();
                render_curve_plot(
                    &curves,
                    mode.as_str(),
                    show_wr.get(),
                    show_personal.get(),
                    show_sessions.get(),
                    show_magic.get(),
                    curve_ylog_rate.get(),
                    curve_ylog_climb.get(),
                );
            }
        }
    });

    // Re-render gain chart when data or controls change
    Effect::new({
        let gain_mode = gain_mode.clone();
        let show_wr = show_wr.clone();
        let show_personal = show_personal.clone();
        let gain_ylog_time = gain_ylog_time.clone();
        let theme = theme.clone();
        move |_| {
            let _ = theme.get();
            if let Some(result) = gain_data.get() {
                let mode = gain_mode.get();
                render_gain_plot(
                    &result,
                    mode.as_str(),
                    show_wr.get(),
                    show_personal.get(),
                    gain_ylog_time.get(),
                );
            }
        }
    });

    // Re-render Algorithm Lab plot when report/theme changes
    Effect::new({
        let algo_report = algo_report.clone();
        let curve_ylog_climb = curve_ylog_climb.clone();
        let theme = theme.clone();
        move |_| {
            let _ = theme.get();
            algo_report.with(|opt| {
                if let Some(report) = opt {
                    render_ascent_compare_plot(report, curve_ylog_climb.get());
                } else {
                    let empty = js_sys::Array::new();
                    let theme = current_plot_theme();
                    let layout = to_js(&serde_json::json!({
                        "title": "Algorithm Lab",
                        "paper_bgcolor": theme.panel.clone(),
                        "plot_bgcolor": theme.panel.clone(),
                        "font": { "color": theme.fg.clone() },
                        "xaxis": { "title": "Duration (min)", "gridcolor": theme.grid.clone(), "zerolinecolor": theme.grid.clone() },
                        "yaxis": { "title": "Climb (m)", "gridcolor": theme.grid.clone(), "zerolinecolor": theme.grid.clone() },
                    }))
                    .unwrap_or(JsValue::UNDEFINED);
                    plot_xy("algo_plot", &empty, &layout);
                }
            });
        }
    });

    // Compute handler
    let on_compute = move |_ev: leptos::ev::MouseEvent| {
        if busy.get_untracked() { return; }
        let files_now = files.get_untracked();
        if files_now.is_empty() {
            set_status.set("Select files first.".to_string());
            return;
        }
        set_busy.set(true);
        set_progress.set(0.0);
        set_status.set("Preparing…".to_string());
        let set_status_cb = set_status.clone();
        let set_busy_cb = set_busy.clone();
        let set_progress_cb = set_progress.clone();
        let set_curve_href_cb = set_curve_href.clone();
        let set_gain_href_cb = set_gain_href.clone();
        let curve_href_get = curve_href.clone();
        let gain_href_get = gain_href.clone();
        let src_now = source_sel.get_untracked();
        let all_now = all_windows.get_untracked();
        let step_now = step_s.get_untracked().max(1);
        let max_now = max_dur_s.get_untracked();
        let set_diag_source_cb = set_diag_source.clone();
        let set_diag_span_cb = set_diag_span_s.clone();
        let set_diag_gain_cb = set_diag_gain_m.clone();
        set_curve_data.set(None);
        set_gain_data.set(None);
        set_records_cache.set(None);
        set_params_cache.set(None);
        set_algo_report.set(None);
        let set_curve_data_cb = set_curve_data.clone();
        let set_gain_data_cb = set_gain_data.clone();
        let set_records_cache_cb = set_records_cache.clone();
        let set_params_cache_cb = set_params_cache.clone();
        let set_algo_report_cb = set_algo_report.clone();
        spawn_local(async move {
            set_progress_cb.set(0.02);
            yield_to_browser().await;

            let mut records_by_file: Vec<Vec<hc_curve::FitRecord>> = Vec::new();
            let mut unsupported: Vec<String> = Vec::new();
            let total_files = files_now.len().max(1);
            for (idx, fb) in files_now.iter().enumerate() {
                set_status_cb.set(format!("Parsing {}/{}: {}", idx + 1, total_files, fb.name));
                yield_to_browser().await;
                match parse_records(&fb.bytes, idx, &fb.ext) {
                    Ok(recs) => { if recs.len() > 2 { records_by_file.push(recs); } },
                    Err(_) => { unsupported.push(fb.name.clone()); }
                }
                set_progress_cb.set(0.05 + 0.35 * ((idx + 1) as f64 / total_files as f64));
                yield_to_browser().await;
            }
            if records_by_file.is_empty() {
                set_status_cb.set("No valid records in uploaded files.".to_string());
                set_progress_cb.set(0.0);
                set_busy_cb.set(false);
                return;
            }

            let mut params = Params::default();
            params.exhaustive = !all_now;
            params.all_windows = all_now;
            params.resample_1hz = true;
            params.qc_enabled = true;
            params.concave_envelope = true;
            params.step_s = step_now;
            params.max_duration_s = if max_now > 0 { Some(max_now) } else { None };
            params.source = match src_now.as_str() {
                "runn" => hc_curve::Source::Runn,
                "altitude" => hc_curve::Source::Altitude,
                _ => hc_curve::Source::Auto,
            };
            set_records_cache_cb.set(Some(records_by_file.clone()));
            set_params_cache_cb.set(Some(params.clone()));
            set_algo_report_cb.set(None);

            // Curves
            set_status_cb.set("Computing curves…".to_string());
            set_progress_cb.set(0.45);
            yield_to_browser().await;
            let (curves, params_used, fallback_notes) = match try_compute_curves(&records_by_file, &params) {
                Ok(res) => res,
                Err(err) => {
                    set_status_cb.set(format!("Curve computation failed: {err}"));
                    set_progress_cb.set(0.0);
                    set_busy_cb.set(false);
                    return;
                }
            };
            set_params_cache_cb.set(Some(params_used.clone()));
            set_curve_data_cb.set(Some(curves.clone()));
            set_status_cb.set(format!(
                "Curve computed: {} durations using {} source.",
                curves.points.len(),
                curves.selected_source
            ));

            // curve.csv
            set_progress_cb.set(0.65);
            yield_to_browser().await;
            let mut curve_csv = String::from("duration_s,max_climb_m,climb_rate_m_per_hr,start_offset_s,end_offset_s,source\n");
            for p in curves.points.iter() {
                curve_csv.push_str(&format!("{},{:.6},{:.6},{:.3},{:.3},{}\n", p.duration_s, p.max_climb_m, p.climb_rate_m_per_hr, p.start_offset_s, p.end_offset_s, curves.selected_source));
            }
            let old_curve = curve_href_get.get_untracked();
            if !old_curve.is_empty() { let _ = web_sys::Url::revoke_object_url(&old_curve); }
            set_curve_href_cb.set(blob_url_from_str(&curve_csv));

            // Gain-time
            set_status_cb.set("Computing gain-time…".to_string());
            set_progress_cb.set(0.75);
            yield_to_browser().await;
            const DEFAULT_GAIN_TARGETS: &[f64] = &[50.0, 100.0, 150.0, 200.0, 300.0, 500.0, 750.0, 1000.0];
            let gt = match compute_gain_time(records_by_file.clone(), &params_used, DEFAULT_GAIN_TARGETS) {
                Ok(r) => r,
                Err(err) => {
                    set_status_cb.set(format!("Gain-time computation failed: {err}"));
                    set_progress_cb.set(0.0);
                    set_busy_cb.set(false);
                    return;
                }
            };
            set_gain_data_cb.set(Some(gt.clone()));
            set_diag_source_cb.set(gt.selected_source.clone());
            set_diag_span_cb.set(gt.total_span_s);
            set_diag_gain_cb.set(gt.total_gain_m);

            let mut gt_csv = String::from("gain_m,min_time_s,avg_rate_m_per_hr,start_offset_s,end_offset_s,note,source\n");
            for p in gt.curve.iter() {
                let so = p.start_offset_s.unwrap_or(f64::NAN);
                let eo = p.end_offset_s.unwrap_or(f64::NAN);
                let note = p.note.clone().unwrap_or_default();
                gt_csv.push_str(&format!("{:.6},{:.6},{:.6},{:.3},{:.3},{},{}\n", p.gain_m, p.min_time_s, p.avg_rate_m_per_hr, so, eo, note, gt.selected_source));
            }
            let old_gain = gain_href_get.get_untracked();
            if !old_gain.is_empty() { let _ = web_sys::Url::revoke_object_url(&old_gain); }
            set_gain_href_cb.set(blob_url_from_str(&gt_csv));

            let mut msg = String::from("Done. Explore the plots and downloads.");
            if !fallback_notes.is_empty() {
                msg.push_str(&format!(" {}.", fallback_notes.join(", ")));
            }
            if !unsupported.is_empty() { msg.push_str(&format!(" Skipped {} unsupported file(s).", unsupported.len())); }
            set_status_cb.set(msg);
            set_progress_cb.set(1.0);
            set_busy_cb.set(false);
        });
    };

    let on_algo_compare = move |_ev: leptos::ev::MouseEvent| {
        if busy.get_untracked() {
            return;
        }
        let Some(records_by_file) = records_cache.get_untracked() else {
            set_status.set("Run Compute first to parse files, then run Algorithm Lab.".to_string());
            return;
        };
        let Some(params_used) = params_cache.get_untracked() else {
            set_status.set("Run Compute first to establish parameters, then run Algorithm Lab.".to_string());
            return;
        };

        let baseline_id = algo_baseline.get_untracked();
        let selected = algo_selected.get_untracked();
        if selected.is_empty() {
            set_status.set("Select at least one algorithm for Algorithm Lab.".to_string());
            return;
        }

        let durations_s = curve_data.get_untracked().map(|curves| {
            curves
                .points
                .iter()
                .map(|p| p.duration_s)
                .collect::<Vec<u64>>()
        });

        set_busy.set(true);
        set_progress.set(0.0);
        set_status.set("Computing Algorithm Lab…".to_string());
        set_algo_report.set(None);

        let set_status_cb = set_status.clone();
        let set_busy_cb = set_busy.clone();
        let set_progress_cb = set_progress.clone();
        let set_algo_report_cb = set_algo_report.clone();

        spawn_local(async move {
            set_progress_cb.set(0.1);
            yield_to_browser().await;

            let baseline_cfg = match AscentAlgorithmConfig::default_for_id(&baseline_id) {
                Some(cfg) => cfg,
                None => {
                    set_status_cb.set(format!("Unknown baseline algorithm '{baseline_id}'."));
                    set_progress_cb.set(0.0);
                    set_busy_cb.set(false);
                    return;
                }
            };

            let mut algo_cfgs: Vec<AscentAlgorithmConfig> = Vec::new();
            for algo_id in selected.iter() {
                if let Some(cfg) = AscentAlgorithmConfig::default_for_id(algo_id) {
                    algo_cfgs.push(cfg);
                }
            }
            if algo_cfgs.is_empty() {
                set_status_cb.set("No valid algorithms selected for Algorithm Lab.".to_string());
                set_progress_cb.set(0.0);
                set_busy_cb.set(false);
                return;
            }

            set_progress_cb.set(0.2);
            yield_to_browser().await;

            match compute_ascent_compare(
                records_by_file,
                &params_used,
                &baseline_cfg,
                &algo_cfgs,
                durations_s,
            ) {
                Ok(report) => {
                    let n_ok = report
                        .entries
                        .iter()
                        .filter(|e| matches!(e, AscentCompareAlgorithmEntry::Ok { .. }))
                        .count();
                    let n_err = report.entries.len().saturating_sub(n_ok);
                    set_algo_report_cb.set(Some(report));
                    if n_err > 0 {
                        set_status_cb.set(format!("Algorithm Lab computed ({n_ok} ok, {n_err} errors)."));
                    } else {
                        set_status_cb.set(format!("Algorithm Lab computed ({n_ok} algorithms)."));
                    }
                    set_progress_cb.set(1.0);
                    set_busy_cb.set(false);
                }
                Err(err) => {
                    set_status_cb.set(format!("Algorithm Lab failed: {err}"));
                    set_progress_cb.set(0.0);
                    set_busy_cb.set(false);
                }
            }
        });
    };

    let file_list_view = move || {
        files.get().into_iter().map(|f| view!{ <li>{f.name}</li> }).collect_view()
    };

    view! {
        <main class="tufte" on:dragover=move |e| { e.prevent_default(); } on:drop=move |e| {
            e.prevent_default();
            if let Ok(de) = e.dyn_into::<web_sys::DragEvent>() {
                if let Some(dt) = de.data_transfer() { if let Some(list) = dt.files() {
                    let set_files_cb = set_files.clone(); let set_status_cb = set_status.clone();
                    spawn_local(async move {
                        let bytes = read_files_from_list(&list).await; set_files_cb.set(bytes);
                        set_status_cb.set("Files ready. Click Compute.".to_string());
                    });
                }}
            }
        }>
            <header>
                <div class="header-row">
                    <div>
                        <h1>"Hillclimb Curves"</h1>
                        <p class="subtitle">"Upload FIT or GPX files to compute hillclimb curves in your browser."</p>
                        <p class="note">{"Web version "}{APP_VERSION}{" ("}{APP_COMMIT}{")"}</p>
                    </div>
                    <button class="btn secondary" on:click=on_toggle_theme>
                        {move || if theme.get() == "dark" { "Light theme" } else { "Dark theme" }}
                    </button>
                </div>
            </header>
            <section class="controls">
                <label class="dropzone">
                    <span>"Drag & drop or click to choose files"</span>
                    <input id="file_input" type="file" multiple on:change=on_files />
                </label>
                <div class="control-row">
                    <label class="note">"Source:"</label>
                    <label><input type="radio" name="src" value="auto" checked on:change=move |ev| { if let Some(t)=ev.target(){ if let Ok(inp)=t.dyn_into::<web_sys::HtmlInputElement>(){ if inp.checked() { set_source_sel.set("auto".to_string()); }}} }/>" auto"</label>
                    <label><input type="radio" name="src" value="runn" on:change=move |ev| { if let Some(t)=ev.target(){ if let Ok(inp)=t.dyn_into::<web_sys::HtmlInputElement>(){ if inp.checked() { set_source_sel.set("runn".to_string()); }}} }/>" runn"</label>
                    <label><input type="radio" name="src" value="altitude" on:change=move |ev| { if let Some(t)=ev.target(){ if let Ok(inp)=t.dyn_into::<web_sys::HtmlInputElement>(){ if inp.checked() { set_source_sel.set("altitude".to_string()); }}} }/>" altitude"</label>
                </div>
                <div class="control-row">
                    <label><input type="checkbox" prop:checked=move || all_windows.get() on:change=move |ev| {
                        if let Some(t) = ev.target() { if let Ok(inp) = t.dyn_into::<web_sys::HtmlInputElement>() { set_all_windows.set(inp.checked()); }}
                    }/>" All windows (exact per-second)"</label>
                    <label>"Step (s): "<input type="number" min="1" value=move || step_s.get().to_string()
                        on:change=move |ev| { if let Some(t)=ev.target(){ if let Ok(inp)=t.dyn_into::<web_sys::HtmlInputElement>(){ if let Ok(v) = inp.value().parse::<u64>() { set_step_s.set(v.max(1)); }}} }/></label>
                    <label>"Max duration (s): "<input type="number" min="0" value=move || max_dur_s.get().to_string()
                        on:change=move |ev| { if let Some(t)=ev.target(){ if let Ok(inp)=t.dyn_into::<web_sys::HtmlInputElement>(){ let v = inp.value().parse::<u64>().unwrap_or(0); set_max_dur_s.set(v); }}}
                    /></label>
                </div>
                <div class="control-row">
                    <label><input type="checkbox" prop:checked=move || show_wr.get() on:change=move |ev| {
                        if let Some(t)=ev.target(){ if let Ok(inp)=t.dyn_into::<web_sys::HtmlInputElement>(){ set_show_wr.set(inp.checked()); }}
                    }/>" WR overlay"</label>
                    <label><input type="checkbox" prop:checked=move || show_personal.get() on:change=move |ev| {
                        if let Some(t)=ev.target(){ if let Ok(inp)=t.dyn_into::<web_sys::HtmlInputElement>(){ set_show_personal.set(inp.checked()); }}
                    }/>" Personal overlays"</label>
                    <label><input type="checkbox" prop:checked=move || show_sessions.get() on:change=move |ev| {
                        if let Some(t)=ev.target(){ if let Ok(inp)=t.dyn_into::<web_sys::HtmlInputElement>(){ set_show_sessions.set(inp.checked()); }}
                    }/>" Session overlays"</label>
                    <label><input type="checkbox" prop:checked=move || show_magic.get() on:change=move |ev| {
                        if let Some(t)=ev.target(){ if let Ok(inp)=t.dyn_into::<web_sys::HtmlInputElement>(){ set_show_magic.set(inp.checked()); }}
                    }/>" Magic markers"</label>
                </div>
                <div class="control-row">
                    <label class="note">"Curve chart:"</label>
                    <select on:change=move |ev| {
                        if let Some(t)=ev.target(){ if let Ok(sel)=t.dyn_into::<HtmlSelectElement>(){ set_curve_mode.set(sel.value()); }}
                    } prop:value=move || curve_mode.get()>
                        <option value="combined">"Combined (climb + rate)"</option>
                        <option value="climb">"Climb only"</option>
                        <option value="rate">"Rate only"</option>
                    </select>
                    <label><input type="checkbox" prop:checked=move || curve_ylog_climb.get() on:change=move |ev| {
                        if let Some(t)=ev.target(){ if let Ok(inp)=t.dyn_into::<web_sys::HtmlInputElement>(){ set_curve_ylog_climb.set(inp.checked()); }}
                    }/>" Climb axis log"</label>
                    <label><input type="checkbox" prop:checked=move || curve_ylog_rate.get() on:change=move |ev| {
                        if let Some(t)=ev.target(){ if let Ok(inp)=t.dyn_into::<web_sys::HtmlInputElement>(){ set_curve_ylog_rate.set(inp.checked()); }}
                    }/>" Rate axis log"</label>
                </div>
                <div class="control-row">
                    <label class="note">"Gain chart:"</label>
                    <select on:change=move |ev| {
                        if let Some(t)=ev.target(){ if let Ok(sel)=t.dyn_into::<HtmlSelectElement>(){ set_gain_mode.set(sel.value()); }}
                    } prop:value=move || gain_mode.get()>
                        <option value="time">"Time vs gain"</option>
                        <option value="rate">"Rate vs gain"</option>
                    </select>
                    <label><input type="checkbox" prop:checked=move || gain_ylog_time.get() on:change=move |ev| {
                        if let Some(t)=ev.target(){ if let Ok(inp)=t.dyn_into::<web_sys::HtmlInputElement>(){ set_gain_ylog_time.set(inp.checked()); }}
                    }/>" Time axis log"</label>
                </div>
                <div class="control-row">
                    <label><input type="checkbox" prop:checked=move || algo_lab_enabled.get() on:change=move |ev| {
                        if let Some(t)=ev.target(){ if let Ok(inp)=t.dyn_into::<web_sys::HtmlInputElement>(){ set_algo_lab_enabled.set(inp.checked()); }}
                    }/>" Algorithm Lab (compare ascent algorithms)"</label>
                </div>
                <button class="btn" on:click=on_compute disabled=move || busy.get()>"Compute"</button>
                <Show when=move || busy.get()>
                    <progress class="progress" max="1" prop:value=move || progress.get().to_string()></progress>
                </Show>
                <div class="status">
                    <Show when=move || busy.get() fallback=|| ()>
                        <span class="spinner" aria-hidden="true"></span>
                    </Show>
                    <span class="note" aria-live="polite">{move || status.get()}</span>
                    <Show when=move || busy.get() fallback=|| ()>
                        <span class="note">{move || format!("{:.0}%", (progress.get() * 100.0).clamp(0.0, 100.0))}</span>
                    </Show>
                </div>
            </section>
            <section class="files">
                <ul>{file_list_view}</ul>
            </section>
            <section class="plots">
                <div id="curve_plot" class="plot"></div>
                <div id="gain_time_plot" class="plot"></div>
                <Show when=move || algo_lab_enabled.get()>
                    <div id="algo_plot" class="plot"></div>
                </Show>
            </section>
            <Show when=move || algo_lab_enabled.get()>
                <section class="controls">
                    <h3>"Algorithm Lab"</h3>
                    <p class="note">"Compare ascent algorithms side-by-side on the same activity. Run Compute first to load your files, then choose a baseline and algorithms to compare."</p>
	                    <div class="control-row">
	                        <label class="note">"Baseline:"</label>
	                        <select on:change=move |ev| {
	                            if let Some(t)=ev.target(){ if let Ok(sel)=t.dyn_into::<HtmlSelectElement>(){ set_algo_baseline.set(sel.value()); }}
	                        } prop:value=move || algo_baseline.get()>
	                            {move || {
	                                list_ascent_algorithms()
	                                    .into_iter()
	                                    .map(|alg| {
	                                        let id = alg.id;
	                                        let label = format!("{} ({})", alg.name, id);
	                                        view! { <option value={id}>{label}</option> }
	                                    })
	                                    .collect_view()
	                            }}
	                        </select>
	                    </div>
	                    <div class="control-row">
	                        <label class="note">"Algorithms:"</label>
	                    </div>
	                    {move || {
	                        let baseline_id = algo_baseline.get();
	                        list_ascent_algorithms()
	                            .into_iter()
	                            .filter(|alg| alg.id.as_str() != baseline_id.as_str())
	                            .map(|alg| {
	                                let id = alg.id;
	                                let id_for_checked = id.clone();
	                                let id_for_update = id.clone();
	                                let name = alg.name;
	                                let desc = alg.description;
	                                let algo_selected = algo_selected.clone();
	                                let set_algo_selected = set_algo_selected.clone();
	                                view! {
	                                    <div class="algo-choice">
	                                        <label>
	                                            <input
	                                                type="checkbox"
	                                                prop:checked=move || algo_selected.get().contains(&id_for_checked)
	                                                on:change=move |ev| {
	                                                    if let Some(t)=ev.target(){
	                                                        if let Ok(inp)=t.dyn_into::<HtmlInputElement>(){
	                                                            let checked = inp.checked();
	                                                            set_algo_selected.update(|set| {
	                                                                if checked { set.insert(id_for_update.clone()); } else { set.remove(&id_for_update); }
	                                                            });
	                                                        }
	                                                    }
	                                                }
	                                            />
	                                            <span>{name}</span>
	                                            <span class="note">" "<code>{id}</code></span>
	                                        </label>
	                                        <div class="note">{desc}</div>
	                                    </div>
	                                }
	                            })
	                            .collect_view()
	                    }}
	                    <button class="btn secondary" on:click=on_algo_compare disabled=move || busy.get()>"Compute Algorithm Lab"</button>
	                    <Show
	                        when=move || algo_report.with(|opt| opt.is_some())
	                        fallback=|| view! { <p class="note">"No Algorithm Lab results yet."</p> }
	                    >
	                        {move || algo_report.with(|opt| {
	                            let report = opt.as_ref().expect("checked by Show");
	
	                            let baseline_row = match &report.baseline {
	                                AscentCompareAlgorithmEntry::Ok {
	                                    algorithm_id,
	                                    name,
	                                    total_gain_m,
	                                    ..
	                                } => view! {
	                                    <tr>
	                                        <td><span class="note">"baseline"</span></td>
	                                        <td><span>{name.clone()}</span><div class="note"><code>{algorithm_id.clone()}</code></div></td>
	                                        <td class="num">{format!("{:.1}", total_gain_m)}</td>
	                                        <td class="num">{format!("{:.1}", 0.0)}</td>
	                                        <td><span class="note">{String::from("ok")}</span></td>
	                                    </tr>
	                                },
	                                AscentCompareAlgorithmEntry::Error {
	                                    algorithm_id,
	                                    name,
	                                    error,
	                                } => view! {
	                                    <tr>
	                                        <td><span class="note">"baseline"</span></td>
	                                        <td><span>{name.clone()}</span><div class="note"><code>{algorithm_id.clone()}</code></div></td>
	                                        <td class="num">{String::from("—")}</td>
	                                        <td class="num">{String::from("—")}</td>
	                                        <td><span class="note">{format!("error: {error}")}</span></td>
	                                    </tr>
	                                },
	                            };
	
	                            let rows = report
	                                .entries
	                                .iter()
	                                .map(|entry| match entry {
	                                    AscentCompareAlgorithmEntry::Ok {
	                                        algorithm_id,
	                                        name,
	                                        total_gain_m,
	                                        delta_total_gain_m,
	                                        ..
	                                    } => view! {
	                                        <tr>
	                                            <td></td>
	                                            <td><span>{name.clone()}</span><div class="note"><code>{algorithm_id.clone()}</code></div></td>
	                                            <td class="num">{format!("{:.1}", total_gain_m)}</td>
	                                            <td class="num">{format!("{:.1}", delta_total_gain_m)}</td>
	                                            <td><span class="note">{String::from("ok")}</span></td>
	                                        </tr>
	                                    },
	                                    AscentCompareAlgorithmEntry::Error {
	                                        algorithm_id,
	                                        name,
	                                        error,
	                                    } => view! {
	                                        <tr>
	                                            <td></td>
	                                            <td><span>{name.clone()}</span><div class="note"><code>{algorithm_id.clone()}</code></div></td>
	                                            <td class="num">{String::from("—")}</td>
	                                            <td class="num">{String::from("—")}</td>
	                                            <td><span class="note">{format!("error: {error}")}</span></td>
	                                        </tr>
	                                    },
	                                })
	                                .collect_view();
	
	                            view! {
	                                <div class="algo-results">
	                                    <table>
	                                        <thead>
	                                            <tr>
	                                                <th></th>
	                                                <th>"Algorithm"</th>
	                                                <th class="num">"Gain (m)"</th>
	                                                <th class="num">"Δ vs baseline (m)"</th>
	                                                <th>"Status"</th>
	                                            </tr>
	                                        </thead>
	                                        <tbody>
	                                            {baseline_row}
	                                            {rows}
	                                        </tbody>
	                                    </table>
	                                </div>
	                            }
	                        })}
	                    </Show>
	                </section>
	            </Show>
            <section class="files">
                <p class="note">{"Source: "}{move || diag_source.get()} {" • Span: "}{move || fmt_hms(diag_span_s.get())} {" • Total gain: "}{move || format!("{:.1} m", diag_gain_m.get())}</p>
                <p class="note">"Nothing leaves your device. All processing happens locally in your browser."</p>
            </section>
            <section class="files">
                <h3>"About & Help"</h3>
                <p class="note">
                    "Upload one or more FIT or GPX files. Choose a data source (auto/runn/altitude) and whether to compute the exact per-second curve (All windows) or the default exhaustive grid. Optionally set step and max duration, then use the chart controls to switch between combined/climb/rate hillclimb views and time/rate gain plots. Overlays for world records, personal goals, sessions, and magic markers can be toggled individually. Enable Algorithm Lab to compare ascent algorithms side-by-side. Use the links below to download CSVs."
                </p>
                <p class="note">
                    "Tips: If a file fails to parse, it will be skipped and noted in the status line. EXPENSIVE modes on very long activities can take time—prefer Max duration and modest Step when experimenting."
                </p>
                <p class="note">
                    "Developers: see the project README and web spec in the repository for details about the algorithms, WR envelopes, and deployment."
                </p>
            </section>
            <section class="downloads">
                <a id="dl_curve" href=move || curve_href.get() download="curve.csv" style=move || if curve_href.get().is_empty() {"display:none;".to_string()} else {"display:inline;".to_string()}>"Download curve.csv"</a>
                <a id="dl_gain" href=move || gain_href.get() download="gain_time.csv" style=move || if gain_href.get().is_empty() {"display:none;".to_string()} else {"display:inline;".to_string()}>"Download gain_time.csv"</a>
            </section>
        </main>
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() {
    #[cfg(feature = "chart_plotly")]
    console_error_panic_hook::set_once();
    mount_to_body(|| view! { <App/> });
}
