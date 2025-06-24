use crate::data_models::{SimulationState, Planet, Star, Galaxy, AgentLineage};
use crate::formatters;
use anyhow::Result;
use console::style;

pub async fn execute(
    source: &str,
    object_type: &str,
    id: &str,
    show_history: bool,
    format: &str,
) -> Result<()> {
    let state = crate::load_simulation_state(source).await?;
    
    match object_type.to_lowercase().as_str() {
        "planet" => inspect_planet(&state, id, show_history, format)?,
        "star" => inspect_star(&state, id, show_history, format)?,
        "galaxy" => inspect_galaxy(&state, id, show_history, format)?,
        "lineage" => inspect_lineage(&state, id, show_history, format)?,
        "system" => inspect_system(&state, id, show_history, format)?,
        _ => return Err(anyhow::anyhow!("Unknown object type: {}", object_type)),
    }
    
    Ok(())
}

fn inspect_planet(state: &SimulationState, id: &str, _show_history: bool, format: &str) -> Result<()> {
    let planet = state.planets.iter()
        .find(|p| p.id == id)
        .ok_or_else(|| anyhow::anyhow!("Planet not found: {}", id))?;
    
    match format {
        "json" => {
            println!("{}", formatters::format_json(planet, true)?);
        }
        "yaml" => {
            println!("{}", formatters::format_yaml(planet)?);
        }
        _ => {
            // Table format
            println!("\n{}", style("═══ PLANET INSPECTION ═══").bold().cyan());
            println!();
            println!("{:<20} {}", style("ID:").bold(), planet.id);
            println!("{:<20} {}", style("Star:").bold(), planet.star_id);
            println!("{:<20} {:.3}, {:.3}, {:.3}", 
                style("Position:").bold(), 
                planet.position.x, planet.position.y, planet.position.z
            );
            println!("{:<20} {} M⊕", style("Mass:").bold(), planet.mass);
            println!("{:<20} {} R⊕", style("Radius:").bold(), planet.radius);
            println!("{:<20} {:.1} K", style("Temperature:").bold(), planet.temperature);
            println!("{:<20} {:.2} atm", style("Atmosphere:").bold(), planet.atmosphere_pressure);
            println!("{:<20} {:.1}%", style("Water:").bold(), planet.water_fraction * 100.0);
            println!("{:<20} {:?}", style("Class:").bold(), planet.planet_class);
            println!("{:<20} {:.2}/1.0", style("Habitability:").bold(), planet.habitability_score);
            
            if planet.has_life {
                println!();
                println!("{}", style("╔═══ LIFE DETECTED ═══╗").bold().green());
                println!("{:<20} {}", style("Population:").bold(), 
                    formatters::format_number(planet.population as f64)
                );
            }
            
            // Atmospheric composition (mock data for now)
            println!();
            println!("{}", style("Atmosphere Composition:").bold());
            println!("  N₂:  78.0%");
            println!("  O₂:  21.0%");
            println!("  Ar:   0.9%");
            println!("  CO₂:  0.1%");
        }
    }
    
    Ok(())
}

fn inspect_star(state: &SimulationState, id: &str, _show_history: bool, format: &str) -> Result<()> {
    let star = state.stars.iter()
        .find(|s| s.id == id)
        .ok_or_else(|| anyhow::anyhow!("Star not found: {}", id))?;
    
    match format {
        "json" => {
            println!("{}", formatters::format_json(star, true)?);
        }
        "yaml" => {
            println!("{}", formatters::format_yaml(star)?);
        }
        _ => {
            println!("\n{}", style("═══ STAR INSPECTION ═══").bold().yellow());
            println!();
            println!("{:<20} {}", style("ID:").bold(), star.id);
            println!("{:<20} {:.3}, {:.3}, {:.3}", 
                style("Position:").bold(), 
                star.position.x, star.position.y, star.position.z
            );
            println!("{:<20} {} M☉", style("Mass:").bold(), star.mass);
            println!("{:<20} {:.0} K", style("Temperature:").bold(), star.temperature);
            println!("{:<20} {} L☉", style("Luminosity:").bold(), star.luminosity);
            println!("{:<20} {:?}", style("Spectral Class:").bold(), star.spectral_class);
            println!("{:<20} {} Gyr", style("Age:").bold(), star.age / 1e9);
            println!("{:<20} {:.3}", style("Metallicity:").bold(), star.metallicity);
            
            // List planets orbiting this star
            let planets: Vec<_> = state.planets.iter()
                .filter(|p| p.star_id == star.id)
                .collect();
            
            if !planets.is_empty() {
                println!();
                println!("{}", style("Planetary System:").bold());
                for (i, planet) in planets.iter().enumerate() {
                    println!("  {}. {} ({:?}, {:.2} AU)", 
                        i + 1, 
                        planet.id, 
                        planet.planet_class,
                        (planet.position - star.position).magnitude() / 1.496e11
                    );
                }
            }
        }
    }
    
    Ok(())
}

fn inspect_galaxy(state: &SimulationState, id: &str, _show_history: bool, format: &str) -> Result<()> {
    let galaxy = state.galaxies.iter()
        .find(|g| g.id == id)
        .ok_or_else(|| anyhow::anyhow!("Galaxy not found: {}", id))?;
    
    match format {
        "json" => {
            println!("{}", formatters::format_json(galaxy, true)?);
        }
        "yaml" => {
            println!("{}", formatters::format_yaml(galaxy)?);
        }
        _ => {
            println!("\n{}", style("═══ GALAXY INSPECTION ═══").bold().magenta());
            println!();
            println!("{:<20} {}", style("ID:").bold(), galaxy.id);
            println!("{:<20} {:?}", style("Type:").bold(), galaxy.galaxy_type);
            println!("{:<20} {:.3}, {:.3}, {:.3} Mpc", 
                style("Position:").bold(), 
                galaxy.position.x, galaxy.position.y, galaxy.position.z
            );
            println!("{:<20} {} M☉", style("Total Mass:").bold(), 
                formatters::format_number(galaxy.mass)
            );
            println!("{:<20} {} M☉", style("Stellar Mass:").bold(), 
                formatters::format_number(galaxy.stellar_mass)
            );
            println!("{:<20} {} M☉", style("Dark Matter:").bold(), 
                formatters::format_number(galaxy.dark_matter_mass)
            );
            println!("{:<20} {:.3}", style("Metallicity:").bold(), galaxy.metallicity);
            println!("{:<20} {} M☉/yr", style("Star Formation:").bold(), galaxy.star_formation_rate);
            println!("{:<20} {:.4}", style("Redshift:").bold(), galaxy.redshift);
            
            // Count stars in this galaxy (within 1 Mpc)
            let star_count = state.stars.iter()
                .filter(|s| (s.position - galaxy.position).magnitude() < 1.0)
                .count();
            
            println!("{:<20} {}", style("Star Count:").bold(), star_count);
        }
    }
    
    Ok(())
}

fn inspect_lineage(state: &SimulationState, id: &str, _show_history: bool, format: &str) -> Result<()> {
    let lineage = state.lineages.iter()
        .find(|l| l.id == id)
        .ok_or_else(|| anyhow::anyhow!("Lineage not found: {}", id))?;
    
    match format {
        "json" => {
            println!("{}", formatters::format_json(lineage, true)?);
        }
        "yaml" => {
            println!("{}", formatters::format_yaml(lineage)?);
        }
        _ => {
            println!("\n{}", style("═══ LINEAGE INSPECTION ═══").bold().green());
            println!();
            println!("{:<20} {}", style("ID:").bold(), lineage.id);
            println!("{:<20} {}", style("Home Planet:").bold(), lineage.planet_id);
            println!("{:<20} {}", style("Generation:").bold(), lineage.generation);
            println!("{:<20} {}", style("Population:").bold(), 
                formatters::format_number(lineage.population as f64)
            );
            println!("{:<20} {:.1}", style("Tech Level:").bold(), lineage.technology_level);
            println!("{:<20} {}", style("Space Travel:").bold(), 
                if lineage.has_space_travel { "Yes" } else { "No" }
            );
            
            if !lineage.colonized_systems.is_empty() {
                println!();
                println!("{}", style("Colonized Systems:").bold());
                for system in &lineage.colonized_systems {
                    println!("  - {}", system);
                }
            }
        }
    }
    
    Ok(())
}

fn inspect_system(state: &SimulationState, id: &str, _show_history: bool, _format: &str) -> Result<()> {
    // Find the star by ID
    let star = state.stars.iter()
        .find(|s| s.id == id)
        .ok_or_else(|| anyhow::anyhow!("System not found: {}", id))?;
    
    println!("\n{}", style("═══ SYSTEM INSPECTION ═══").bold().cyan());
    println!();
    
    // Star information
    println!("{}", style("Central Star:").bold());
    println!("  ID: {}", star.id);
    println!("  Class: {:?} ({:.0} K)", star.spectral_class, star.temperature);
    println!("  Mass: {} M☉", star.mass);
    println!("  Age: {:.2} Gyr", star.age / 1e9);
    
    // Find all planets in this system
    let planets: Vec<_> = state.planets.iter()
        .filter(|p| p.star_id == star.id)
        .collect();
    
    println!();
    println!("{}", style("Planets:").bold());
    
    if planets.is_empty() {
        println!("  No planets detected");
    } else {
        for planet in planets {
            let distance = (planet.position - star.position).magnitude();
            let au_distance = distance / 1.496e11;
            
            println!();
            println!("  {} ({:?})", planet.id, planet.planet_class);
            println!("    Distance: {:.2} AU", au_distance);
            println!("    Mass: {} M⊕", planet.mass);
            println!("    Temperature: {:.0} K", planet.temperature);
            println!("    Habitability: {:.2}", planet.habitability_score);
            
            if planet.has_life {
                println!("    {} Life detected! Population: {}", 
                    style("★").green().bold(),
                    formatters::format_number(planet.population as f64)
                );
            }
        }
    }
    
    Ok(())
}