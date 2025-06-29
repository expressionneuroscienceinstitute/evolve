use anyhow::Result;
use colored::*;
use std::time::Duration;
use tabled::{Table, Tabled};
use tokio::time;

use crate::rpc::RpcClient;
use crate::data_models::{SimulationStatus, CosmicEra};
use crate::formatters;

pub struct StatusCommand {
    rpc_client: RpcClient,
}

impl StatusCommand {
    pub fn new(socket_path: &str) -> Result<Self> {
        Ok(Self {
            rpc_client: RpcClient::new(socket_path),
        })
    }

    pub async fn execute(&self, refresh_interval: u64, format: &str) -> Result<()> {
        if refresh_interval == 0 {
            // Single shot
            self.display_status(format).await?;
        } else {
            // Continuous refresh
            println!("{}", "Universe Simulation Status Monitor".green().bold());
            println!("{}", "Press Ctrl+C to exit".yellow());
            println!();

            let mut interval = time::interval(Duration::from_secs(refresh_interval));
            loop {
                // Clear screen for continuous mode
                print!("\x1B[2J\x1B[1;1H");
                
                self.display_status(format).await?;
                
                println!("\n{}", format!("Refreshing every {} seconds...", refresh_interval).dimmed());
                interval.tick().await;
            }
        }
        
        Ok(())
    }

    async fn display_status(&self, format: &str) -> Result<()> {
        let status = self.rpc_client.get_status().await?;

        match format {
            "json" => {
                println!("{}", serde_json::to_string_pretty(&status)?);
            }
            "yaml" => {
                println!("{}", serde_yaml::to_string(&status)?);
            }
            "compact" => {
                self.display_compact_status(&status);
            }
            _ => {
                self.display_detailed_status(&status);
            }
        }

        Ok(())
    }

    fn display_detailed_status(&self, status: &SimulationStatus) {
        println!("{}", "UNIVERSE SIMULATION STATUS".cyan().bold());
        println!("{}", "=".repeat(50).cyan());
        println!();

        // Core simulation metrics
        let core_table = vec![
            StatusRow::new("Current Tick", formatters::format_number(status.tick)),
            StatusRow::new("Updates Per Second", format!("{:.1} UPS", status.ups)),
            StatusRow::new("Cosmic Era", format!("{:?}", status.cosmic_era)),
            StatusRow::new("Uptime", status.uptime.clone()),
            StatusRow::new("Save File Age", status.save_file_age.clone()),
        ];

        let table = Table::new(core_table);
        println!("{}", table);
        println!();

        // Entity counts
        println!("{}", "ENTITY COUNTS".yellow().bold());
        let entity_table = vec![
            StatusRow::new("Active Lineages", formatters::format_number(status.lineage_count as u64)),
            StatusRow::new("Planets", formatters::format_number(status.planet_count as u64)),
            StatusRow::new("Star Systems", formatters::format_number(status.star_count as u64)),
        ];

        let table = Table::new(entity_table);
        println!("{}", table);
        println!();

        // Entropy metrics
        println!("{}", "ENTROPY METRICS".red().bold());
        let entropy_table = vec![
            StatusRow::new("Mean Entropy", format!("{:.3}", status.mean_entropy)),
            StatusRow::new("Maximum Entropy", format!("{:.3}", status.max_entropy)),
            StatusRow::new("Entropy Pressure", self.calculate_entropy_pressure(status.mean_entropy, status.max_entropy)),
        ];

        let table = Table::new(entropy_table);
        println!("{}", table);
        println!();

        // Memory usage
        println!("{}", "MEMORY USAGE".blue().bold());
        let memory_color = if status.memory_usage.percentage > 90.0 {
            "red"
        } else if status.memory_usage.percentage > 75.0 {
            "yellow"
        } else {
            "green"
        };

        let memory_table = vec![
            StatusRow::new("Used Memory", format!("{:.1} MB", status.memory_usage.used_mb)),
            StatusRow::new("Available Memory", format!("{:.1} MB", status.memory_usage.available_mb)),
            StatusRow::new("Memory Usage", format!("{:.1}%", status.memory_usage.percentage).color(memory_color).to_string()),
        ];

        let table = Table::new(memory_table);
        println!("{}", table);
        println!();

        // Performance metrics
        println!("{}", "PERFORMANCE METRICS".magenta().bold());
        let perf_table = vec![
            StatusRow::new("CPU Usage", format!("{:.1}%", status.performance_metrics.cpu_usage)),
            StatusRow::new("GPU Usage", 
                match status.performance_metrics.gpu_usage {
                    Some(gpu) => format!("{:.1}%", gpu),
                    None => "N/A".to_string(),
                }
            ),
            StatusRow::new("Avg Tick Time", format!("{:.2} ms", status.performance_metrics.avg_tick_time_ms)),
        ];

        let table = Table::new(perf_table);
        println!("{}", table);
        println!();

        // Network I/O
        println!("{}", "NETWORK I/O".cyan().bold());
        let network_table = vec![
            StatusRow::new("Bytes Sent", formatters::format_bytes(status.performance_metrics.network_io.bytes_sent)),
            StatusRow::new("Bytes Received", formatters::format_bytes(status.performance_metrics.network_io.bytes_received)),
            StatusRow::new("Packets Sent", formatters::format_number(status.performance_metrics.network_io.packets_sent)),
            StatusRow::new("Packets Received", formatters::format_number(status.performance_metrics.network_io.packets_received)),
        ];

        let table = Table::new(network_table);
        println!("{}", table);
        println!();

        // Disk I/O
        println!("{}", "DISK I/O".green().bold());
        let disk_table = vec![
            StatusRow::new("Bytes Read", formatters::format_bytes(status.performance_metrics.disk_io.bytes_read)),
            StatusRow::new("Bytes Written", formatters::format_bytes(status.performance_metrics.disk_io.bytes_written)),
            StatusRow::new("Read Operations", formatters::format_number(status.performance_metrics.disk_io.operations_read)),
            StatusRow::new("Write Operations", formatters::format_number(status.performance_metrics.disk_io.operations_written)),
        ];

        let table = Table::new(disk_table);
        println!("{}", table);
        println!();

        // Status indicators
        self.display_status_indicators(status);
    }

    fn display_compact_status(&self, status: &SimulationStatus) {
        let ups_indicator = if status.ups > 500.0 { "🟢" } else if status.ups > 100.0 { "🟡" } else { "🔴" };
        let memory_indicator = if status.memory_usage.percentage < 75.0 { "🟢" } else if status.memory_usage.percentage < 90.0 { "🟡" } else { "🔴" };
        let entropy_indicator = if status.mean_entropy < 0.5 { "🟢" } else if status.mean_entropy < 0.8 { "🟡" } else { "🔴" };

        println!("{} Tick: {} | UPS: {:.1} {} | Lineages: {} | Entropy: {:.3} {} | Memory: {:.1}% {} | Era: {:?}",
            "UNIVERSE".cyan().bold(),
            formatters::format_number(status.tick),
            status.ups, ups_indicator,
            formatters::format_number(status.lineage_count as u64),
            status.mean_entropy, entropy_indicator,
            status.memory_usage.percentage, memory_indicator,
            status.cosmic_era
        );
    }

    fn display_status_indicators(&self, status: &SimulationStatus) {
        println!("{}", "STATUS INDICATORS".yellow().bold());
        
        let indicators = vec![
            self.get_performance_indicator(status),
            self.get_memory_indicator(status),
            self.get_entropy_indicator(status),
            self.get_cosmic_era_indicator(&status.cosmic_era),
        ];

        for indicator in indicators {
            println!("  {}", indicator);
        }
        println!();
    }

    fn get_performance_indicator(&self, status: &SimulationStatus) -> String {
        let (symbol, color, message) = if status.ups > 500.0 {
            ("🟢", "green", "Excellent performance")
        } else if status.ups > 200.0 {
            ("🟡", "yellow", "Good performance")
        } else if status.ups > 50.0 {
            ("🟠", "orange", "Moderate performance")
        } else {
            ("🔴", "red", "Poor performance")
        };

        format!("{} {}", symbol, message.color(color))
    }

    fn get_memory_indicator(&self, status: &SimulationStatus) -> String {
        let (symbol, color, message) = if status.memory_usage.percentage < 50.0 {
            ("🟢", "green", "Memory usage healthy")
        } else if status.memory_usage.percentage < 75.0 {
            ("🟡", "yellow", "Memory usage moderate")
        } else if status.memory_usage.percentage < 90.0 {
            ("🟠", "orange", "Memory usage high")
        } else {
            ("🔴", "red", "Memory usage critical")
        };

        format!("{} {}", symbol, message.color(color))
    }

    fn get_entropy_indicator(&self, status: &SimulationStatus) -> String {
        let (symbol, color, message) = if status.mean_entropy < 0.3 {
            ("🟢", "green", "Low entropy - high order")
        } else if status.mean_entropy < 0.6 {
            ("🟡", "yellow", "Moderate entropy")
        } else if status.mean_entropy < 0.8 {
            ("🟠", "orange", "High entropy - increasing chaos")
        } else {
            ("🔴", "red", "Critical entropy - approaching heat death")
        };

        format!("{} {}", symbol, message.color(color))
    }

    fn get_cosmic_era_indicator(&self, era: &CosmicEra) -> String {
        let (symbol, color, message) = match era {
            CosmicEra::ParticleSoup => ("⚛️", "blue", "Particle soup - primordial universe"),
            CosmicEra::Starbirth => ("⭐", "yellow", "Starbirth - first stars forming"),
            CosmicEra::PlanetaryAge => ("🪐", "cyan", "Planetary age - worlds coalescing"),
            CosmicEra::Biogenesis => ("🧬", "green", "Biogenesis - life emerging"),
            CosmicEra::DigitalEvolution => ("🤖", "purple", "Digital evolution - AI consciousness"),
            CosmicEra::PostIntelligence => ("🌌", "magenta", "Post-intelligence - transcendent beings"),
        };

        format!("{} {}", symbol, message.color(color))
    }

    fn calculate_entropy_pressure(&self, mean: f64, max: f64) -> String {
        let pressure = mean / max;
        let (color, description) = if pressure < 0.5 {
            ("green", "Low")
        } else if pressure < 0.8 {
            ("yellow", "Moderate")
        } else if pressure < 0.95 {
            ("orange", "High")
        } else {
            ("red", "Critical")
        };

        format!("{} ({:.1}%)", description.color(color), pressure * 100.0)
    }
}

#[derive(Tabled)]
struct StatusRow {
    #[tabled(rename = "Metric")]
    metric: String,
    #[tabled(rename = "Value")]
    value: String,
}

impl StatusRow {
    fn new(metric: &str, value: String) -> Self {
        Self {
            metric: metric.to_string(),
            value,
        }
    }
}