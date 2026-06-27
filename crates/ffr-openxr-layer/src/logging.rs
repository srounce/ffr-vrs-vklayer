//! File-based logging for the OpenXR layer (see the Vulkan layer's copy for
//! rationale). Logs to `$XDG_STATE_HOME/ffr-vrs/ffr-oxr-<pid>.log`, verbosity
//! via `FFR_VRS_LOG`.

use std::sync::Once;

use tracing_subscriber::EnvFilter;

static INIT: Once = Once::new();

pub fn init() {
    INIT.call_once(|| {
        let dir = log_dir();
        let _ = std::fs::create_dir_all(&dir);
        let filename = format!("ffr-oxr-{}.log", std::process::id());
        let appender = tracing_appender::rolling::never(&dir, filename);

        let filter =
            EnvFilter::try_from_env("FFR_VRS_LOG").unwrap_or_else(|_| EnvFilter::new("info"));

        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_ansi(false)
            .with_target(false)
            .with_writer(appender)
            .try_init();
    });
}

fn log_dir() -> String {
    if let Ok(state) = std::env::var("XDG_STATE_HOME") {
        if !state.is_empty() {
            return format!("{state}/ffr-vrs");
        }
    }
    match std::env::var("HOME") {
        Ok(home) if !home.is_empty() => format!("{home}/.local/state/ffr-vrs"),
        _ => "/tmp/ffr-vrs".to_string(),
    }
}
