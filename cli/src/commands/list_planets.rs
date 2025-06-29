use crate::data_models::PlanetClass;
use anyhow::Result;
use comfy_table::{Attribute, Cell, Color, ContentArrangement, Table};

pub async fn execute(
    source: &str,
    class_filter: Option<PlanetClass>,
    habitable_only: bool,
    limit: usize,
    sort_by: &str,
) -> Result<()> {
    let state = crate::load_simulation_state(source).await?;
    
    // Filter planets
    let mut planets: Vec<_> = state.planets.iter()
        .filter(|p| {
            if habitable_only && p.habitability_score < 0.5 {
                return false;
            }
            if let Some(class) = class_filter {
                return p.planet_class == class;
            }
            true
        })
        .collect();
    
    // Sort planets
    match sort_by {
        "name" => planets.sort_by(|a, b| a.id.cmp(&b.id)),
        "class" => planets.sort_by(|a, b| {
            format!("{:?}", a.planet_class).cmp(&format!("{:?}", b.planet_class))
        }),
        "habitability" => planets.sort_by(|a, b| {
            b.habitability_score.partial_cmp(&a.habitability_score).unwrap()
        }),
        "population" => planets.sort_by(|a, b| b.population.cmp(&a.population)),
        _ => return Err(anyhow::anyhow!("Invalid sort field: {}", sort_by)),
    }
    
    // Create table
    let mut table = Table::new();
    table
        .set_content_arrangement(ContentArrangement::Dynamic)
        .set_header(vec![
            Cell::new("ID").add_attribute(Attribute::Bold),
            Cell::new("Class").add_attribute(Attribute::Bold),
            Cell::new("Star").add_attribute(Attribute::Bold),
            Cell::new("Mass").add_attribute(Attribute::Bold),
            Cell::new("Temp (K)").add_attribute(Attribute::Bold),
            Cell::new("Water").add_attribute(Attribute::Bold),
            Cell::new("Habit.").add_attribute(Attribute::Bold),
            Cell::new("Life").add_attribute(Attribute::Bold),
            Cell::new("Population").add_attribute(Attribute::Bold),
        ]);
    
    // Add planet rows
    for planet in planets.iter().take(limit) {
        let class_cell = match planet.planet_class {
            PlanetClass::EarthLike => Cell::new("E").fg(Color::Green),
            PlanetClass::Desert => Cell::new("D").fg(Color::Yellow),
            PlanetClass::Ice => Cell::new("I").fg(Color::Cyan),
            PlanetClass::Toxic => Cell::new("T").fg(Color::Red),
            PlanetClass::GasDwarf => Cell::new("G").fg(Color::Magenta),
        };
        
        let life_cell = if planet.has_life {
            Cell::new("YES").fg(Color::Green).add_attribute(Attribute::Bold)
        } else {
            Cell::new("no").fg(Color::DarkGrey)
        };
        
        let habit_cell = if planet.habitability_score > 0.8 {
            Cell::new(format!("{:.2}", planet.habitability_score)).fg(Color::Green)
        } else if planet.habitability_score > 0.5 {
            Cell::new(format!("{:.2}", planet.habitability_score)).fg(Color::Yellow)
        } else {
            Cell::new(format!("{:.2}", planet.habitability_score)).fg(Color::Red)
        };
        
        table.add_row(vec![
            Cell::new(&planet.id),
            class_cell,
            Cell::new(&planet.star_id),
            Cell::new(format!("{:.2}", planet.mass)),
            Cell::new(format!("{:.0}", planet.temperature)),
            Cell::new(format!("{:.0}%", planet.water_fraction * 100.0)),
            habit_cell,
            life_cell,
            Cell::new(if planet.population > 0 {
                format_population(planet.population)
            } else {
                "-".to_string()
            }),
        ]);
    }
    
    // Print header
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                    PLANETARY CATALOG                         ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    
    // Print filters
    if let Some(class) = class_filter {
        println!("║ Filter: Class = {:?}", class);
    }
    if habitable_only {
        println!("║ Filter: Habitable only (score > 0.5)");
    }
    println!("║ Showing {} of {} planets", planets.len().min(limit), planets.len());
    println!("║ Sorted by: {}", sort_by);
    println!("╚══════════════════════════════════════════════════════════════╝\n");
    
    // Print table
    println!("{}", table);
    
    // Summary statistics
    let total_habitable = planets.iter().filter(|p| p.habitability_score > 0.5).count();
    let total_life = planets.iter().filter(|p| p.has_life).count();
    let total_population: u64 = planets.iter().map(|p| p.population).sum();
    
    println!("\n╔══════════════════════════════════════════════════════════════╗");
    println!("║                         SUMMARY                              ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Total planets shown:    {:>36} ║", planets.len().min(limit));
    println!("║ Habitable planets:      {:>36} ║", total_habitable);
    println!("║ Planets with life:      {:>36} ║", total_life);
    println!("║ Total population:       {:>36} ║", format_population(total_population));
    println!("╚══════════════════════════════════════════════════════════════╝");
    
    Ok(())
}

fn format_population(pop: u64) -> String {
    if pop == 0 {
        "0".to_string()
    } else if pop < 1_000 {
        format!("{}", pop)
    } else if pop < 1_000_000 {
        format!("{:.1}K", pop as f64 / 1_000.0)
    } else if pop < 1_000_000_000 {
        format!("{:.1}M", pop as f64 / 1_000_000.0)
    } else if pop < 1_000_000_000_000 {
        format!("{:.1}B", pop as f64 / 1_000_000_000.0)
    } else {
        format!("{:.1}T", pop as f64 / 1_000_000_000_000.0)
    }
}