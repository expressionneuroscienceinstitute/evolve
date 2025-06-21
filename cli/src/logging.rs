use clap::ValueEnum;
use tracing_appender::non_blocking;
use tracing_subscriber::{fmt, EnvFilter};

/// Logging verbosity level selected via the `--log` CLI flag (or its aliases).
#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
pub enum LogLevel {
    /// Extremely chatty – every simulation event is emitted.
    Verbose,
    /// Default – important progress updates but not every tick.
    Info,
    /// Only warnings and errors.
    Warn,
    /// Only errors.
    Error,
    /// Disable all log output (errors included).
    None,
}

impl LogLevel {
    fn as_filter_directive(self) -> &'static str {
        match self {
            LogLevel::Verbose => "debug", // `debug` level includes `info` + `warn` + `error`
            LogLevel::Info => "info",
            LogLevel::Warn => "warn",
            LogLevel::Error => "error",
            LogLevel::None => "off",
        }
    }
}

/// Initialise global tracing subscriber based on the chosen `level`.
/// The returned [`WorkerGuard`] must be held for the lifetime of the program
/// to ensure buffered, non-blocking logs are flushed on shutdown.
pub fn init_logging(level: LogLevel, output_json: bool) -> non_blocking::WorkerGuard {
    // Non-blocking writer prevents the simulation loop from stalling even with
    // very verbose output.
    let (writer, guard) = non_blocking(std::io::stderr());

    // Build an EnvFilter string.  For the `None` variant we pass just `off` so
    // that all targets are disabled.  Otherwise we specify per-crate targets
    // to avoid external dependencies spamming the terminal.
    let filter = if level == LogLevel::None {
        EnvFilter::new("off")
    } else {
        let directive = level.as_filter_directive();
        EnvFilter::new(format!(
            "universectl={},universe_sim={},physics_engine={}",
            directive, directive, directive
        ))
    };

    let subscriber_builder = fmt::Subscriber::builder()
        .with_env_filter(filter)
        .with_writer(writer);

    if output_json {
        let subscriber = subscriber_builder.json().finish();
        let _ = tracing::subscriber::set_global_default(subscriber);
    } else {
        let subscriber = subscriber_builder.finish();
        let _ = tracing::subscriber::set_global_default(subscriber);
    }

    // Set LogTracer to capture log crate messages
    let _ = tracing_log::LogTracer::init();

    guard
} 