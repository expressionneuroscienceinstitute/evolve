use crate::data_models::SimulationState;
use anyhow::Result;
use std::fs::File;
use std::io::Write;
use console::{style, Term};
use indicatif::{ProgressBar, ProgressStyle};

pub async fn execute(
    source: &str,
    output_path: &str,
    include_full_data: bool,
    format: &str,
) -> Result<()> {
    let term = Term::stdout();
    
    // Create progress bar
    let pb = ProgressBar::new(100);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}% {msg}")
            .unwrap()
            .progress_chars("#>-")
    );
    
    pb.set_message("Loading simulation state...");
    pb.set_position(10);
    
    let state = crate::load_simulation_state(source).await?;
    
    pb.set_message("Preparing snapshot data...");
    pb.set_position(30);
    
    let snapshot_data = if include_full_data {
        // Full snapshot with all data
        FullSnapshot {
            metadata: SnapshotMetadata::from_state(&state),
            simulation_state: state.clone(),
            statistics: calculate_statistics(&state),
        }
    } else {
        // Summary snapshot
        FullSnapshot {
            metadata: SnapshotMetadata::from_state(&state),
            simulation_state: create_summary_state(&state),
            statistics: calculate_statistics(&state),
        }
    };
    
    pb.set_message("Formatting output...");
    pb.set_position(60);
    
    let output = match format {
        "json" => serde_json::to_string_pretty(&snapshot_data)?,
        "yaml" => serde_yaml::to_string(&snapshot_data)?,
        "toml" => toml::to_string_pretty(&snapshot_data)?,
        _ => return Err(anyhow::anyhow!("Unsupported format: {}", format)),
    };
    
    pb.set_message("Writing to file...");
    pb.set_position(80);
    
    let mut file = File::create(output_path)?;
    file.write_all(output.as_bytes())?;
    
    pb.set_message("Snapshot complete!");
    pb.set_position(100);
    pb.finish();
    
    term.clear_line()?;
    println!("{} Snapshot saved to: {}", 
        style("✓").green().bold(),
        style(output_path).cyan()
    );
    
    // Print summary statistics
    println!();
    println!("{}", style("Snapshot Summary:").bold());
    println!("  Format:      {}", format);
    println!("  File size:   {:.2} MB", output.len() as f64 / 1_048_576.0);
    println!("  Tick:        {}", snapshot_data.metadata.tick);
    println!("  Sim time:    {:.2} Gyr", snapshot_data.metadata.simulation_time / 1e9);
    println!("  Objects:");
    println!("    Galaxies:  {}", snapshot_data.statistics.galaxy_count);
    println!("    Stars:     {}", snapshot_data.statistics.star_count);
    println!("    Planets:   {}", snapshot_data.statistics.planet_count);
    println!("    Lineages:  {}", snapshot_data.statistics.lineage_count);
    
    Ok(())
}

#[derive(serde::Serialize, serde::Deserialize)]
struct FullSnapshot {
    metadata: SnapshotMetadata,
    simulation_state: SimulationState,
    statistics: SimulationStatistics,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SnapshotMetadata {
    version: String,
    timestamp: chrono::DateTime<chrono::Utc>,
    tick: u64,
    simulation_time: f64,
    box_size: f64,
}

impl SnapshotMetadata {
    fn from_state(state: &SimulationState) -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp: chrono::Utc::now(),
            tick: state.tick,
            simulation_time: state.current_time,
            box_size: state.box_size,
        }
    }
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SimulationStatistics {
    // Object counts
    galaxy_count: usize,
    star_count: usize,
    planet_count: usize,
    habitable_planet_count: usize,
    life_bearing_planet_count: usize,
    lineage_count: usize,
    total_population: u64,
    
    // Physical properties
    total_mass: f64,
    total_stellar_mass: f64,
    total_dark_matter_mass: f64,
    average_temperature: f64,
    total_entropy: f64,
    
    // Evolution metrics
    star_formation_rate: f64,
    average_metallicity: f64,
    spacefaring_civilizations: usize,
}

fn calculate_statistics(state: &SimulationState) -> SimulationStatistics {
    let habitable_planets: Vec<_> = state.planets.iter()
        .filter(|p| p.habitability_score >= 0.5)
        .collect();
    
    let life_bearing_planets: Vec<_> = state.planets.iter()
        .filter(|p| p.has_life)
        .collect();
    
    let total_population: u64 = state.planets.iter()
        .map(|p| p.population)
        .sum();
    
    let total_stellar_mass: f64 = state.galaxies.iter()
        .map(|g| g.stellar_mass)
        .sum();
    
    let total_dark_matter_mass: f64 = state.galaxies.iter()
        .map(|g| g.dark_matter_mass)
        .sum();
    
    let total_sfr: f64 = state.galaxies.iter()
        .map(|g| g.star_formation_rate)
        .sum();
    
    let avg_metallicity = if !state.galaxies.is_empty() {
        state.galaxies.iter().map(|g| g.metallicity).sum::<f64>() / state.galaxies.len() as f64
    } else {
        0.0
    };
    
    let spacefaring = state.lineages.iter()
        .filter(|l| l.has_space_travel)
        .count();
    
    SimulationStatistics {
        galaxy_count: state.galaxies.len(),
        star_count: state.stars.len(),
        planet_count: state.planets.len(),
        habitable_planet_count: habitable_planets.len(),
        life_bearing_planet_count: life_bearing_planets.len(),
        lineage_count: state.lineages.len(),
        total_population,
        total_mass: total_stellar_mass + total_dark_matter_mass,
        total_stellar_mass,
        total_dark_matter_mass,
        average_temperature: state.temperature,
        total_entropy: state.entropy,
        star_formation_rate: total_sfr,
        average_metallicity: avg_metallicity,
        spacefaring_civilizations: spacefaring,
    }
}

fn create_summary_state(state: &SimulationState) -> SimulationState {
    // Create a reduced version of the state for smaller snapshots
    let mut summary = state.clone();
    
    // Keep only a sample of particles
    if summary.particles.len() > 1000 {
        summary.particles.truncate(1000);
    }
    
    if summary.dark_matter.len() > 1000 {
        summary.dark_matter.truncate(1000);
    }
    
    // Keep only major galaxies (top 100 by mass)
    if summary.galaxies.len() > 100 {
        summary.galaxies.sort_by(|a, b| b.mass.partial_cmp(&a.mass).unwrap());
        summary.galaxies.truncate(100);
    }
    
    // Keep only bright stars
    if summary.stars.len() > 500 {
        summary.stars.sort_by(|a, b| b.luminosity.partial_cmp(&a.luminosity).unwrap());
        summary.stars.truncate(500);
    }
    
    // Keep all habitable/life-bearing planets, but limit others
    let mut important_planets: Vec<_> = summary.planets.iter()
        .filter(|p| p.habitability_score >= 0.5 || p.has_life)
        .cloned()
        .collect();
    
    let mut other_planets: Vec<_> = summary.planets.into_iter()
        .filter(|p| p.habitability_score < 0.5 && !p.has_life)
        .take(100)
        .collect();
    
    important_planets.append(&mut other_planets);
    summary.planets = important_planets;
    
    summary
}