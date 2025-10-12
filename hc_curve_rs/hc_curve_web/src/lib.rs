use leptos::*;

#[cfg(feature = "chart_plotly")]
use wasm_bindgen::{JsCast, JsValue};

#[cfg(feature = "chart_plotly")]
use wasm_bindgen_futures::JsFuture;

#[cfg(feature = "chart_plotly")]
use serde::{Deserialize, Serialize};

#[cfg(feature = "chart_plotly")]
use serde_wasm_bindgen::to_value as to_js;

#[cfg(feature = "chart_plotly")]
use web_sys::{Blob, FileList, HtmlInputElement};

#[cfg(feature = "chart_plotly")]
use hc_curve::{compute_curves, compute_gain_time, parse_records, Curves, GainTimeResult, Params};

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
fn plot_xy(div_id: &str, traces: &js_sys::Array, layout: &JsValue) {
    if let Some(window) = web_sys::window() {
        if let Some(document) = window.document() {
            if let Some(div) = document.get_element_by_id(div_id) {
                let plotly = js_sys::Reflect::get(&js_sys::global(), &JsValue::from_str("Plotly"))
                    .unwrap_or(JsValue::UNDEFINED);
                if let Ok(func) = js_sys::Reflect::get(&plotly, &JsValue::from_str("newPlot"))
                    .and_then(|v| v.dyn_into::<js_sys::Function>())
                {
                    let _ = func.call3(&JsValue::NULL, &div.into(), &traces.clone().into(), layout);
                }
            }
        }
    }
}

#[cfg(feature = "chart_plotly")]
fn blob_url_from_str(s: &str) -> String {
    let arr = js_sys::Array::new();
    arr.push(&JsValue::from_str(s));
    let blob = Blob::new_with_str_sequence(&arr).unwrap();
    web_sys::Url::create_object_url_with_blob(&blob).unwrap()
}

#[component]
pub fn App() -> impl IntoView {
    let (files, set_files) = create_signal(Vec::<FileBytes>::new());
    let (busy, set_busy) = create_signal(false);
    let (status, set_status) = create_signal(String::from("No files selected."));
    let (curve_href, set_curve_href) = create_signal(String::new());
    let (gain_href, set_gain_href) = create_signal(String::new());
    let (source_sel, set_source_sel) = create_signal(String::from("auto"));
    let (all_windows, set_all_windows) = create_signal(false);
    let (step_s, set_step_s) = create_signal(1u64);
    let (max_dur_s, set_max_dur_s) = create_signal(0u64); // 0 => None
    let (show_wr, set_show_wr) = create_signal(true);
    let (diag_source, set_diag_source) = create_signal(String::new());
    let (diag_span_s, set_diag_span_s) = create_signal(0.0_f64);
    let (diag_gain_m, set_diag_gain_m) = create_signal(0.0_f64);

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

    // Compute handler
    let on_compute = move |_ev: leptos::ev::MouseEvent| {
        if busy.get_untracked() { return; }
        let files_now = files.get_untracked();
        if files_now.is_empty() {
            set_status.set("Select files first.".to_string());
            return;
        }
        set_busy.set(true);
        set_status.set("Computing… this may take a moment.".to_string());
        let set_status_cb = set_status.clone();
        let set_busy_cb = set_busy.clone();
        let set_curve_href_cb = set_curve_href.clone();
        let set_gain_href_cb = set_gain_href.clone();
        let curve_href_get = curve_href.clone();
        let gain_href_get = gain_href.clone();
        let src_now = source_sel.get_untracked();
        let all_now = all_windows.get_untracked();
        let step_now = step_s.get_untracked().max(1);
        let max_now = max_dur_s.get_untracked();
        let show_wr_now = show_wr.get_untracked();
        let set_diag_source_cb = set_diag_source.clone();
        let set_diag_span_cb = set_diag_span_s.clone();
        let set_diag_gain_cb = set_diag_gain_m.clone();
        spawn_local(async move {
            let mut records_by_file: Vec<Vec<hc_curve::FitRecord>> = Vec::new();
            let mut unsupported: Vec<String> = Vec::new();
            for (idx, fb) in files_now.iter().enumerate() {
                match parse_records(&fb.bytes, idx, &fb.ext) {
                    Ok(recs) => { if recs.len() > 2 { records_by_file.push(recs); } },
                    Err(_) => { unsupported.push(fb.name.clone()); }
                }
            }
            if records_by_file.is_empty() {
                set_status_cb.set("No valid records in uploaded files.".to_string());
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

            // Curves
            let curves = match compute_curves(records_by_file.clone(), &params) {
                Ok(c) => c,
                Err(_) => { set_status_cb.set("Curve computation failed.".to_string()); set_busy_cb.set(false); return; }
            };
            let x: Vec<f64> = curves.points.iter().map(|p| p.duration_s as f64).collect();
            let y: Vec<f64> = curves.points.iter().map(|p| p.max_climb_m).collect();
            let trace = to_js(&serde_json::json!({
                "type": "scatter", "mode": "lines", "name": "Climb (m)", "x": x, "y": y
            })).unwrap();
            let data = js_sys::Array::new(); data.push(&trace);
            if show_wr_now {
                if let Some((dur, gains)) = curves.wr_curve.clone() {
                    let wx: Vec<f64> = dur.into_iter().map(|d| d as f64).collect();
                    let wy: Vec<f64> = gains;
                    let wtrace = to_js(&serde_json::json!({
                        "type":"scatter","mode":"lines","name":"WR","x": wx, "y": wy,
                        "line": {"dash":"dash","color":"#888"}
                    })).unwrap();
                    data.push(&wtrace);
                }
            }
            let layout = to_js(&serde_json::json!({
                "title": "Max Climb vs Duration", "xaxis": {"title": "Duration (s)"}, "yaxis": {"title": "Climb (m)"}
            })).unwrap();
            plot_xy("curve_plot", &data, &layout);

            // curve.csv
            let mut curve_csv = String::from("duration_s,max_climb_m,climb_rate_m_per_hr,start_offset_s,end_offset_s,source\n");
            for p in curves.points.iter() {
                curve_csv.push_str(&format!("{},{:.6},{:.6},{:.3},{:.3},{}\n", p.duration_s, p.max_climb_m, p.climb_rate_m_per_hr, p.start_offset_s, p.end_offset_s, curves.selected_source));
            }
            let old_curve = curve_href_get.get_untracked();
            if !old_curve.is_empty() { let _ = web_sys::Url::revoke_object_url(&old_curve); }
            set_curve_href_cb.set(blob_url_from_str(&curve_csv));

            // Gain-time
            const DEFAULT_GAIN_TARGETS: &[f64] = &[50.0, 100.0, 150.0, 200.0, 300.0, 500.0, 750.0, 1000.0];
            let gt = match compute_gain_time(records_by_file.clone(), &params, DEFAULT_GAIN_TARGETS) {
                Ok(r) => r,
                Err(_) => { set_status_cb.set("Gain-time computation failed.".to_string()); set_busy_cb.set(false); return; }
            };
            set_diag_source_cb.set(gt.selected_source.clone());
            set_diag_span_cb.set(gt.total_span_s);
            set_diag_gain_cb.set(gt.total_gain_m);
            let gx: Vec<f64> = gt.curve.iter().map(|p| p.gain_m).collect();
            let gy: Vec<f64> = gt.curve.iter().map(|p| p.min_time_s).collect();
            let gtrace = to_js(&serde_json::json!({
                "type": "scatter", "mode": "lines+markers", "name": "Min Time (s)", "x": gx, "y": gy
            })).unwrap();
            let gdata = js_sys::Array::new(); gdata.push(&gtrace);
            if show_wr_now {
                if let Some(wr_pts) = gt.wr_curve.clone() {
                    let wx: Vec<f64> = wr_pts.iter().map(|p| p.gain_m).collect();
                    let wy: Vec<f64> = wr_pts.iter().map(|p| p.min_time_s).collect();
                    let wtrace = to_js(&serde_json::json!({
                        "type":"scatter","mode":"lines","name":"WR","x": wx, "y": wy,
                        "line": {"dash":"dash","color":"#888"}
                    })).unwrap();
                    gdata.push(&wtrace);
                }
            }
            // iso-rate guide lines
            let (xmin, xmax) = match (gx.iter().cloned().fold(f64::INFINITY, f64::min), gx.iter().cloned().fold(0.0_f64, f64::max)) {
                (a, b) if a.is_finite() && b > a => (a, b),
                _ => (0.0_f64, 1.0_f64),
            };
            let rates = [800.0, 1000.0, 1200.0, 1500.0, 2000.0, 2500.0];
            for r in rates {
                let xline = vec![xmin, xmax];
                let yline = vec![xmin / r * 3600.0, xmax / r * 3600.0];
                let line = to_js(&serde_json::json!({
                    "type":"scatter","mode":"lines","x": xline, "y": yline,
                    "line": {"dash":"dot","color":"#bbb","width":1},
                    "hoverinfo":"skip","showlegend": false
                })).unwrap();
                gdata.push(&line);
            }
            let glayout = to_js(&serde_json::json!({
                "title": "Minimum Time vs Gain", "xaxis": {"title": "Gain (m)"}, "yaxis": {"title": "Time (s)"}
            })).unwrap();
            plot_xy("gain_time_plot", &gdata, &glayout);

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
            if !unsupported.is_empty() { msg.push_str(&format!(" Skipped {} unsupported file(s).", unsupported.len())); }
            set_status_cb.set(msg);
            set_busy_cb.set(false);
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
                <h1>"Hillclimb Curves"</h1>
                <p class="subtitle">"Upload FIT or GPX files to compute hillclimb curves in your browser."</p>
            </header>
            <section class="controls">
                <label class="dropzone">
                    <span>"Drag & drop or click to choose files"</span>
                    <input id="file_input" type="file" multiple on:change=on_files />
                </label>
                <div>
                    <label class="note">"Source:"</label>
                    <label><input type="radio" name="src" value="auto" checked on:change=move |ev| { if let Some(t)=ev.target(){ if let Ok(inp)=t.dyn_into::<web_sys::HtmlInputElement>(){ if inp.checked() { set_source_sel.set("auto".to_string()); }}} }/>" auto"</label>
                    <label><input type="radio" name="src" value="runn" on:change=move |ev| { if let Some(t)=ev.target(){ if let Ok(inp)=t.dyn_into::<web_sys::HtmlInputElement>(){ if inp.checked() { set_source_sel.set("runn".to_string()); }}} }/>" runn"</label>
                    <label><input type="radio" name="src" value="altitude" on:change=move |ev| { if let Some(t)=ev.target(){ if let Ok(inp)=t.dyn_into::<web_sys::HtmlInputElement>(){ if inp.checked() { set_source_sel.set("altitude".to_string()); }}} }/>" altitude"</label>
                </div>
                <label><input type="checkbox" on:change=move |ev| {
                    if let Some(t) = ev.target() { if let Ok(inp) = t.dyn_into::<web_sys::HtmlInputElement>() { set_all_windows.set(inp.checked()); }}
                }/>" All windows (exact per-second)"</label>
                <label>"Step (s): "<input type="number" min="1" value=move || step_s.get().to_string()
                    on:change=move |ev| { if let Some(t)=ev.target(){ if let Ok(inp)=t.dyn_into::<web_sys::HtmlInputElement>(){ if let Ok(v) = inp.value().parse::<u64>() { set_step_s.set(v.max(1)); }}} }/></label>
                <label>"Max duration (s): "<input type="number" min="0" value=move || max_dur_s.get().to_string()
                    on:change=move |ev| { if let Some(t)=ev.target(){ if let Ok(inp)=t.dyn_into::<web_sys::HtmlInputElement>(){ let v = inp.value().parse::<u64>().unwrap_or(0); set_max_dur_s.set(v); }}}
                /></label>
                <label><input type="checkbox" checked on:change=move |ev| { if let Some(t)=ev.target(){ if let Ok(inp)=t.dyn_into::<web_sys::HtmlInputElement>(){ set_show_wr.set(inp.checked()); }}}
                />" WR overlay"</label>
                <button class="btn" on:click=on_compute disabled=move || busy.get()>"Compute"</button>
                <span class="note">{move || status.get()}</span>
            </section>
            <section class="files">
                <ul>{file_list_view}</ul>
            </section>
            <section class="plots">
                <div id="curve_plot" class="plot"></div>
                <div id="gain_time_plot" class="plot"></div>
            </section>
            <section class="files">
                <p class="note">{"Source: "}{move || diag_source.get()} {" • Span: "}{move || fmt_hms(diag_span_s.get())} {" • Total gain: "}{move || format!("{:.1} m", diag_gain_m.get())}</p>
                <p class="note">"Nothing leaves your device. All processing happens locally in your browser."</p>
            </section>
            <section class="files">
                <h3>"About & Help"</h3>
                <p class="note">
                    "Upload one or more FIT or GPX files. Choose a data source (auto/runn/altitude) and whether to compute the exact per-second curve (All windows) or the default exhaustive grid. Optionally set step and max duration, and toggle the world‑record overlay. Click Compute to see: (1) the maximum climb vs duration curve and (2) the minimum time vs gain plot with iso‑rate guides. Use the links below to download CSVs."
                </p>
                <p class="note">
                    "Tips: If a file fails to parse, it will be skipped and noted in the status line. EXPENSIVE modes on very long activities can take time—prefer Max duration and modest Step when experimenting."
                </p>
                <p class="note">
                    "Developers: see the project README and web spec in the repository for details about the algorithms, WR envelopes, and deployment."
                </p>
            </section>
            <section class="downloads">
                <a id="dl_curve" href=move || curve_href.get() download="curve.csv" style=move || if curve_href.get().is_empty() {"display:none;".into()} else {"display:inline;".into()}>"Download curve.csv"</a>
                <a id="dl_gain" href=move || gain_href.get() download="gain_time.csv" style=move || if gain_href.get().is_empty() {"display:none;".into()} else {"display:inline;".into()}>"Download gain_time.csv"</a>
            </section>
        </main>
    }
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen::prelude::wasm_bindgen(start)]
pub fn start() {
    #[cfg(feature = "chart_plotly")]
    console_error_panic_hook::set_once();
    leptos::mount_to_body(|cx| view! { cx, <App/> });
}
