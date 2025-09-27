use leptos::*;

#[cfg(feature = "chart_plotly")]
use wasm_bindgen::JsCast;

#[component]
pub fn App() -> impl IntoView {
    let (file_names, set_file_names) = create_signal(Vec::<String>::new());

    let on_files = move |ev: leptos::ev::Event| {
        #[cfg(feature = "chart_plotly")]
        if let Some(target) = ev.target() {
            if let Ok(input) = target.dyn_into::<web_sys::HtmlInputElement>() {
                if let Some(list) = input.files() {
                    let mut names = Vec::new();
                    for idx in 0..list.length() {
                        if let Some(file) = list.item(idx) {
                            names.push(file.name());
                        }
                    }
                    set_file_names.set(names);
                }
            }
        }
    };

    let list_view = move || {
        file_names
            .get()
            .into_iter()
            .map(|name| view! { <li>{name}</li> })
            .collect_view()
    };

    view! {
        <main class="tufte">
            <header>
                <h1>"Hillclimb Curves"</h1>
                <p class="subtitle">"Upload FIT or GPX files to compute hillclimb power curves."</p>
            </header>
            <section class="upload">
                <label class="dropzone">
                    <span>"Drag & drop files or browse"</span>
                    <input type="file" multiple on:change=on_files />
                </label>
            </section>
            <section class="files">
                <Show
                    when=move || !file_names.get().is_empty()
                    fallback=move || view! { <p>"No files selected yet."</p> }
                >
                    <ul>{list_view}</ul>
                </Show>
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
