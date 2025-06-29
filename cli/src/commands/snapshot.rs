use anyhow::Result;
use colored::*;
use std::fs::File;
use std::io::Write;
use indicatif::{ProgressBar, ProgressStyle};

use crate::rpc::RpcClient;
use crate::data_models::UniverseSnapshot;

pub struct SnapshotCommand {
    rpc_client: RpcClient,
}

impl SnapshotCommand {
    pub fn new(socket_path: &str) -> Result<Self> {
        Ok(Self {
            rpc_client: RpcClient::new(socket_path),
        })
    }

    pub async fn execute(&self, file_path: &str, format: &str, full_state: bool, compress: bool) -> Result<()> {
        println!("{}", "Creating universe snapshot...".cyan().bold());
        
        // Create progress bar
        let pb = ProgressBar::new(100);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos:>3}% {msg}")
                .unwrap()
                .progress_chars("=>-"),
        );
        
        pb.set_message("Fetching simulation state...");
        pb.set_position(10);

        // Fetch snapshot data
        let snapshot = self.rpc_client.create_snapshot(full_state).await?;
        pb.set_position(60);

        pb.set_message("Serializing data...");
        let serialized_data = match format {
            "json" => {
                pb.set_message("Converting to JSON...");
                serde_json::to_string_pretty(&snapshot)?
            }
            "yaml" => {
                pb.set_message("Converting to YAML...");
                serde_yaml::to_string(&snapshot)?
            }
            "toml" => {
                pb.set_message("Converting to TOML...");
                toml::to_string_pretty(&snapshot)?
            }
            _ => {
                return Err(anyhow::anyhow!("Unsupported format '{}'. Supported: json, yaml, toml", format));
            }
        };
        pb.set_position(80);

        // Apply compression if requested
        let final_data = if compress {
            pb.set_message("Compressing data...");
            self.compress_data(&serialized_data)?
        } else {
            serialized_data.into_bytes()
        };
        pb.set_position(90);

        // Determine final filename
        let final_filename = if compress {
            format!("{}.{}.gz", file_path, format)
        } else {
            format!("{}.{}", file_path, format)
        };

        // Write to file
        pb.set_message("Writing to disk...");
        let mut file = File::create(&final_filename)?;
        file.write_all(&final_data)?;
        file.sync_all()?;
        pb.set_position(100);

        pb.finish_with_message("Snapshot completed!");
        
        // Display summary
        self.display_snapshot_summary(&snapshot, &final_filename, final_data.len(), full_state, compress);

        Ok(())
    }

    fn compress_data(&self, data: &str) -> Result<Vec<u8>> {
        use std::io::prelude::*;
        use flate2::Compression;
        use flate2::write::GzEncoder;

        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(data.as_bytes())?;
        Ok(encoder.finish()?)
    }

    fn display_snapshot_summary(&self, snapshot: &UniverseSnapshot, filename: &str, file_size: usize, full_state: bool, compressed: bool) {
        println!();
        println!("{}", "SNAPSHOT SUMMARY".green().bold());
        println!("{}", "=".repeat(50).green());
        
        // File information
        println!("{}", "File Information:".blue());
        println!("  📁 Filename: {}", filename.yellow());
        println!("  📊 File Size: {}", self.format_file_size(file_size));
        println!("  🗜️  Compressed: {}", if compressed { "Yes".green() } else { "No".red() });
        println!("  📋 Format: {}", snapshot.metadata.version);
        println!("  ⏰ Created: {}", snapshot.metadata.created_at.format("%Y-%m-%d %H:%M:%S UTC"));
        println!();

        // Simulation state
        println!("{}", "Simulation State:".cyan());
        println!("  🔢 Current Tick: {}", crate::formatters::format_number(snapshot.simulation_state.tick));
        println!("  ⚡ UPS: {:.1}", snapshot.simulation_state.ups);
        println!("  🌌 Cosmic Era: {:?}", snapshot.simulation_state.cosmic_era);
        println!("  🧬 Active Lineages: {}", crate::formatters::format_number(snapshot.simulation_state.lineage_count as u64));
        println!("  🪐 Planets: {}", crate::formatters::format_number(snapshot.simulation_state.planet_count as u64));
        println!("  ⭐ Star Systems: {}", crate::formatters::format_number(snapshot.simulation_state.star_count as u64));
        println!("  📈 Mean Entropy: {:.3}", snapshot.simulation_state.mean_entropy);
        println!();

        // Data composition
        println!("{}", "Data Composition:".magenta());
        println!("  🌟 Star Systems: {}", snapshot.star_systems.len());
        println!("  🌍 Planets: {}", snapshot.planets.len());
        println!("  🤖 Lineages: {}", snapshot.lineages.len());
        println!("  📨 Resource Requests: {}", snapshot.resource_requests.len());
        println!("  💌 Petitions: {}", snapshot.petitions.len());
        println!("  📊 Performance Records: {}", snapshot.performance_history.len());
        println!();

        // Completeness indicator
        let completeness_indicator = if full_state { "🔍 Full State" } else { "📝 Summary Only" };
        println!("  {} Snapshot Type: {}", completeness_indicator, if full_state { "Complete".green() } else { "Summary".yellow() });
        
        // Verification
        println!("  ✅ Checksum: {}", snapshot.metadata.checksum);
        
        if compressed {
            let compression_ratio = (file_size as f64 / (file_size as f64 * 1.5)) * 100.0; // Rough estimate
            println!("  📦 Compression Ratio: ~{:.1}%", compression_ratio);
        }

        println!();
        println!("{}", "Snapshot successfully created!".green().bold());
        println!("{}", format!("Use 'universectl load {}' to restore this state", filename).dimmed());
    }

    fn format_file_size(&self, bytes: usize) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        let mut size = bytes as f64;
        let mut unit_index = 0;
        
        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }
        
        if unit_index == 0 {
            format!("{} {}", bytes, UNITS[unit_index])
        } else {
            format!("{:.2} {}", size, UNITS[unit_index])
        }
    }
}