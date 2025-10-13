use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // Re-run if the repository HEAD changes (best-effort; path is relative to crate dir).
    println!("cargo:rerun-if-changed=../../.git/HEAD");

    let hash = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|output| {
            if output.status.success() {
                let value = String::from_utf8_lossy(&output.stdout).trim().to_string();
                if value.is_empty() {
                    None
                } else {
                    Some(value)
                }
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string());

    println!("cargo:rustc-env=GIT_COMMIT_HASH={hash}");
}
