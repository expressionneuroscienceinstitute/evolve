pub mod ascii_viz;
pub mod commands;
pub mod data_models;
pub mod formatters;

pub use data_models::SimulationState;

use anyhow::Result;

/// Load simulation state from a source (file, URL, or mock data)
pub async fn load_simulation_state(source: &str) -> Result<SimulationState> {
    // For now, return mock data regardless of source
    // In a real implementation, this would parse the source and load from file/network
    match source {
        "mock" | "test" | "demo" => {
            Ok(SimulationState::mock())
        }
        _ => {
            // Try to load from file if source looks like a path
            if source.ends_with(".json") || source.contains("/") {
                // Try to load from JSON file
                match tokio::fs::read_to_string(source).await {
                    Ok(content) => {
                        match serde_json::from_str::<SimulationState>(&content) {
                            Ok(state) => Ok(state),
                            Err(_) => {
                                eprintln!("Warning: Could not parse simulation file, using mock data");
                                Ok(SimulationState::mock())
                            }
                        }
                    }
                    Err(_) => {
                        eprintln!("Warning: Could not read simulation file, using mock data");
                        Ok(SimulationState::mock())
                    }
                }
            } else {
                // Default to mock data for any other source
                Ok(SimulationState::mock())
            }
        }
    }
}