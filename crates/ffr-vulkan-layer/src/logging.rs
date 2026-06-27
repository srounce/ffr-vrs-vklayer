//! File-based logging for the layer. Layers run inside arbitrary host processes
//! whose stdout/stderr may be closed or redirected, so we always log to a file
//! under `$XDG_STATE_HOME/ffr-vrs/` (one file per pid). Verbosity is controlled
//! by `FFR_VRS_LOG` (a `tracing` `EnvFilter`, default `info`).

use std::sync::Once;

use tracing_subscriber::EnvFilter;

static INIT: Once = Once::new();

/// Initialize the file logger exactly once. Safe to call from every entry point.
pub fn init() {
    INIT.call_once(|| {
        let dir = log_dir();
        let _ = std::fs::create_dir_all(&dir);
        let filename = format!("ffr-vk-{}.log", std::process::id());
        let appender = tracing_appender::rolling::never(&dir, filename);

        let filter = EnvFilter::try_from_env("FFR_VRS_LOG")
            .unwrap_or_else(|_| EnvFilter::new("info"));

        // try_init: tolerate another subscriber already being set in-process.
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_ansi(false)
            .with_target(false)
            .with_writer(appender)
            .try_init();
    });
}

pub(crate) fn log_dir() -> String {
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
