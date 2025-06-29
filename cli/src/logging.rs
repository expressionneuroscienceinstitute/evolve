use anyhow::Result;
use chrono::{DateTime, Utc};
use std::fs::OpenOptions;
use std::io::Write;
use tracing::{info, warn, error};

pub struct Logger {
    log_file: Option<String>,
    verbose: bool,
}

impl Logger {
    pub fn new(log_file: Option<String>, verbose: bool) -> Self {
        Self { log_file, verbose }
    }

    pub fn log_command(&self, command: &str, args: &[String]) -> Result<()> {
        let timestamp = Utc::now();
        let log_entry = LogEntry {
            timestamp,
            level: LogLevel::Info,
            category: LogCategory::Command,
            message: format!("Executed: {} {}", command, args.join(" ")),
            metadata: None,
        };

        self.write_log_entry(&log_entry)?;
        
        if self.verbose {
            info!("Command executed: {} {}", command, args.join(" "));
        }

        Ok(())
    }

    pub fn log_error(&self, error: &str, context: Option<&str>) -> Result<()> {
        let timestamp = Utc::now();
        let message = if let Some(ctx) = context {
            format!("Error in {}: {}", ctx, error)
        } else {
            format!("Error: {}", error)
        };

        let log_entry = LogEntry {
            timestamp,
            level: LogLevel::Error,
            category: LogCategory::Error,
            message,
            metadata: context.map(|c| c.to_string()),
        };

        self.write_log_entry(&log_entry)?;
        error!("{}", error);

        Ok(())
    }

    pub fn log_performance(&self, operation: &str, duration_ms: u64, success: bool) -> Result<()> {
        let timestamp = Utc::now();
        let status = if success { "SUCCESS" } else { "FAILED" };
        let message = format!("{} completed in {}ms - {}", operation, duration_ms, status);

        let log_entry = LogEntry {
            timestamp,
            level: if success { LogLevel::Info } else { LogLevel::Warn },
            category: LogCategory::Performance,
            message,
            metadata: Some(format!("duration_ms: {}, success: {}", duration_ms, success)),
        };

        self.write_log_entry(&log_entry)?;

        if success {
            info!("{} completed in {}ms", operation, duration_ms);
        } else {
            warn!("{} failed after {}ms", operation, duration_ms);
        }

        Ok(())
    }

    pub fn log_connection(&self, endpoint: &str, success: bool, latency_ms: Option<u64>) -> Result<()> {
        let timestamp = Utc::now();
        let status = if success { "CONNECTED" } else { "FAILED" };
        let latency_info = latency_ms.map_or_else(String::new, |ms| format!(" ({}ms)", ms));
        let message = format!("Connection to {} - {}{}", endpoint, status, latency_info);

        let log_entry = LogEntry {
            timestamp,
            level: if success { LogLevel::Info } else { LogLevel::Error },
            category: LogCategory::Network,
            message,
            metadata: latency_ms.map(|ms| format!("latency_ms: {}", ms)),
        };

        self.write_log_entry(&log_entry)?;

        if success {
            info!("Connected to {}{}", endpoint, latency_info);
        } else {
            error!("Failed to connect to {}", endpoint);
        }

        Ok(())
    }

    pub fn log_simulation_event(&self, event_type: &str, details: &str) -> Result<()> {
        let timestamp = Utc::now();
        let message = format!("Simulation Event [{}]: {}", event_type, details);

        let log_entry = LogEntry {
            timestamp,
            level: LogLevel::Info,
            category: LogCategory::Simulation,
            message,
            metadata: Some(format!("event_type: {}", event_type)),
        };

        self.write_log_entry(&log_entry)?;
        info!("Simulation event [{}]: {}", event_type, details);

        Ok(())
    }

    pub fn log_divine_action(&self, action: &str, parameters: &str, user: &str) -> Result<()> {
        let timestamp = Utc::now();
        let message = format!("DIVINE ACTION by {}: {} with parameters: {}", user, action, parameters);

        let log_entry = LogEntry {
            timestamp,
            level: LogLevel::Warn, // Divine actions are always flagged
            category: LogCategory::Divine,
            message,
            metadata: Some(format!("user: {}, action: {}", user, action)),
        };

        self.write_log_entry(&log_entry)?;
        warn!("Divine action executed by {}: {}", user, action);

        Ok(())
    }

    fn write_log_entry(&self, entry: &LogEntry) -> Result<()> {
        if let Some(ref log_file_path) = self.log_file {
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(log_file_path)?;

            let formatted_entry = self.format_log_entry(entry);
            writeln!(file, "{}", formatted_entry)?;
            file.sync_all()?;
        }

        Ok(())
    }

    fn format_log_entry(&self, entry: &LogEntry) -> String {
        let metadata_str = entry.metadata.as_ref()
            .map_or_else(String::new, |m| format!(" [{}]", m));

        format!(
            "{} {:5} {:12} {}{}",
            entry.timestamp.format("%Y-%m-%d %H:%M:%S%.3f UTC"),
            format!("{:?}", entry.level),
            format!("{:?}", entry.category),
            entry.message,
            metadata_str
        )
    }
}

#[derive(Debug)]
struct LogEntry {
    timestamp: DateTime<Utc>,
    level: LogLevel,
    category: LogCategory,
    message: String,
    metadata: Option<String>,
}

#[derive(Debug)]
enum LogLevel {
    Info,
    Warn,
    Error,
}

#[derive(Debug)]
enum LogCategory {
    Command,
    Error,
    Performance,
    Network,
    Simulation,
    Divine,
}

// CLI audit logging utilities
pub fn log_cli_startup(logger: &Logger, version: &str, args: &[String]) -> Result<()> {
    logger.log_command("universectl_startup", &[format!("version={}", version), format!("args={:?}", args)])?;
    Ok(())
}

pub fn log_cli_shutdown(logger: &Logger, duration_ms: u64) -> Result<()> {
    logger.log_performance("universectl_session", duration_ms, true)?;
    Ok(())
}

pub fn log_rpc_call(logger: &Logger, method: &str, success: bool, duration_ms: u64) -> Result<()> {
    let operation = format!("RPC_{}", method);
    logger.log_performance(&operation, duration_ms, success)?;
    Ok(())
}

pub fn log_file_operation(logger: &Logger, operation: &str, file_path: &str, success: bool, size_bytes: Option<u64>) -> Result<()> {
    let details = if let Some(size) = size_bytes {
        format!("{} {} ({} bytes)", operation, file_path, size)
    } else {
        format!("{} {}", operation, file_path)
    };
    
    logger.log_performance(&details, 0, success)?;
    Ok(())
}

// Session tracking
pub struct Session {
    start_time: DateTime<Utc>,
    commands_executed: u32,
    errors_encountered: u32,
    logger: Logger,
}

impl Session {
    pub fn new(logger: Logger) -> Self {
        Self {
            start_time: Utc::now(),
            commands_executed: 0,
            errors_encountered: 0,
            logger,
        }
    }

    pub fn record_command(&mut self, command: &str, args: &[String]) -> Result<()> {
        self.commands_executed += 1;
        self.logger.log_command(command, args)?;
        Ok(())
    }

    pub fn record_error(&mut self, error: &str, context: Option<&str>) -> Result<()> {
        self.errors_encountered += 1;
        self.logger.log_error(error, context)?;
        Ok(())
    }

    pub fn end_session(&self) -> Result<()> {
        let duration = Utc::now().signed_duration_since(self.start_time);
        let duration_ms = duration.num_milliseconds() as u64;
        
        let session_summary = format!(
            "Session ended: {} commands, {} errors, {} duration",
            self.commands_executed,
            self.errors_encountered,
            humantime::format_duration(std::time::Duration::from_millis(duration_ms))
        );

        self.logger.log_simulation_event("SESSION_END", &session_summary)?;
        log_cli_shutdown(&self.logger, duration_ms)?;
        
        Ok(())
    }
}