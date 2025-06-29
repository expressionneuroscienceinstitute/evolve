use anyhow::Result;
use colored::*;
use rand::prelude::*;

use crate::rpc::RpcClient;
use crate::data_models::{MapData, MapType};

pub async fn render_map(socket_path: &str, map_type: &str, zoom_level: u8, width: u16, height: u16) -> Result<()> {
    let rpc_client = RpcClient::new(socket_path);
    
    // Parse map type
    let parsed_map_type = match map_type.to_lowercase().as_str() {
        "stars" => "stars",
        "entropy" => "entropy", 
        "temperature" => "temperature",
        "density" => "density",
        "lineages" => "lineages",
        "resources" => "resources",
        _ => {
            println!("{}", format!("Unknown map type '{}'. Available: stars, entropy, temperature, density, lineages, resources", map_type).red());
            return Ok(());
        }
    };

    println!("{}", format!("UNIVERSE MAP: {} (Zoom {}x)", map_type.to_uppercase(), zoom_level).cyan().bold());
    println!("{}", "=".repeat(width as usize).cyan());
    println!();

    // Fetch map data
    let map_data = rpc_client.get_map_data(parsed_map_type, zoom_level, width, height).await?;
    
    // Render the map
    render_map_data(&map_data);
    
    // Display legend
    display_legend(&map_data);
    
    // Display map info
    display_map_info(&map_data, zoom_level);

    Ok(())
}

fn render_map_data(map_data: &MapData) {
    for row in &map_data.cells {
        for cell in row {
            let colored_symbol = match map_data.map_type {
                MapType::Stars => colorize_star_symbol(cell.symbol, cell.value),
                MapType::Entropy => colorize_entropy_symbol(cell.symbol, cell.value),
                MapType::Temperature => colorize_temperature_symbol(cell.symbol, cell.value),
                MapType::Density => colorize_density_symbol(cell.symbol, cell.value),
                MapType::Lineages => colorize_lineage_symbol(cell.symbol, cell.value),
                MapType::Resources => colorize_resource_symbol(cell.symbol, cell.value),
            };
            print!("{}", colored_symbol);
        }
        println!();
    }
    println!();
}

fn colorize_star_symbol(symbol: char, value: f64) -> String {
    match symbol {
        '*' => {
            if value > 0.9 { symbol.to_string().bright_white().bold().to_string() }
            else if value > 0.7 { symbol.to_string().yellow().bold().to_string() }
            else if value > 0.5 { symbol.to_string().yellow().to_string() }
            else { symbol.to_string().red().to_string() }
        }
        '·' | '.' => symbol.to_string().blue().dimmed().to_string(),
        '+' => symbol.to_string().cyan().to_string(),
        'o' => symbol.to_string().magenta().to_string(),
        _ => symbol.to_string().dimmed().to_string(),
    }
}

fn colorize_entropy_symbol(symbol: char, value: f64) -> String {
    match symbol {
        '█' | '▓' => {
            if value > 0.8 { symbol.to_string().red().bold().to_string() }
            else if value > 0.6 { symbol.to_string().red().to_string() }
            else if value > 0.4 { symbol.to_string().yellow().to_string() }
            else { symbol.to_string().green().to_string() }
        }
        '▒' | '░' => {
            if value > 0.5 { symbol.to_string().yellow().to_string() }
            else { symbol.to_string().green().dimmed().to_string() }
        }
        _ => symbol.to_string().dimmed().to_string(),
    }
}

fn colorize_temperature_symbol(symbol: char, value: f64) -> String {
    // Value represents temperature in some normalized scale
    match symbol {
        '🔥' | '#' => symbol.to_string().red().bold().to_string(),
        '○' | 'o' => {
            if value > 0.7 { symbol.to_string().yellow().to_string() }
            else if value > 0.4 { symbol.to_string().green().to_string() }
            else { symbol.to_string().blue().to_string() }
        }
        '❄' | '*' => symbol.to_string().cyan().bold().to_string(),
        _ => symbol.to_string().dimmed().to_string(),
    }
}

fn colorize_density_symbol(symbol: char, value: f64) -> String {
    match symbol {
        '█' => symbol.to_string().white().bold().to_string(),
        '▓' => symbol.to_string().bright_white().to_string(),
        '▒' => symbol.to_string().white().to_string(),
        '░' => symbol.to_string().white().dimmed().to_string(),
        '·' => symbol.to_string().black().to_string(),
        _ => symbol.to_string().dimmed().to_string(),
    }
}

fn colorize_lineage_symbol(symbol: char, value: f64) -> String {
    match symbol {
        '🤖' | 'A' => {
            if value > 0.8 { symbol.to_string().magenta().bold().to_string() }
            else if value > 0.6 { symbol.to_string().magenta().to_string() }
            else { symbol.to_string().purple().to_string() }
        }
        '🧬' | 'a' => symbol.to_string().green().to_string(),
        '○' | 'o' => symbol.to_string().cyan().to_string(),
        _ => symbol.to_string().dimmed().to_string(),
    }
}

fn colorize_resource_symbol(symbol: char, value: f64) -> String {
    match symbol {
        '💎' | '$' => symbol.to_string().bright_yellow().bold().to_string(),
        '⚡' | '#' => symbol.to_string().yellow().to_string(),
        '🔶' | '+' => symbol.to_string().truecolor(255, 165, 0).to_string(),
        '⚫' | 'o' => symbol.to_string().black().bold().to_string(),
        _ => symbol.to_string().dimmed().to_string(),
    }
}

fn display_legend(map_data: &MapData) {
    println!("{}", "LEGEND".yellow().bold());
    println!("{}", "-".repeat(30).yellow());
    
    for entry in &map_data.legend {
        let colored_symbol = match map_data.map_type {
            MapType::Stars => colorize_star_symbol(entry.symbol, 0.5),
            MapType::Entropy => colorize_entropy_symbol(entry.symbol, 0.5),
            MapType::Temperature => colorize_temperature_symbol(entry.symbol, 0.5),
            MapType::Density => colorize_density_symbol(entry.symbol, 0.5),
            MapType::Lineages => colorize_lineage_symbol(entry.symbol, 0.5),
            MapType::Resources => colorize_resource_symbol(entry.symbol, 0.5),
        };
        
        let range_text = if let Some((min, max)) = entry.value_range {
            format!(" ({:.2} - {:.2})", min, max)
        } else {
            String::new()
        };
        
        println!("  {} {}{}", colored_symbol, entry.description, range_text);
    }
    println!();
}

fn display_map_info(map_data: &MapData, zoom_level: u8) {
    println!("{}", "MAP INFORMATION".blue().bold());
    println!("{}", "-".repeat(30).blue());
    println!("  📐 Dimensions: {}x{}", map_data.width, map_data.height);
    println!("  🔍 Zoom Level: {}x", zoom_level);
    println!("  🗺️  Map Type: {:?}", map_data.map_type);
    
    // Calculate coverage area
    let area_per_cell = (10.0_f64).powi(zoom_level as i32);
    let total_coverage = map_data.width as f64 * map_data.height as f64 * area_per_cell;
    
    if total_coverage > 1e12 {
        println!("  🌌 Coverage: {:.2} trillion cubic units", total_coverage / 1e12);
    } else if total_coverage > 1e9 {
        println!("  🌌 Coverage: {:.2} billion cubic units", total_coverage / 1e9);
    } else if total_coverage > 1e6 {
        println!("  🌌 Coverage: {:.2} million cubic units", total_coverage / 1e6);
    } else {
        println!("  🌌 Coverage: {:.0} cubic units", total_coverage);
    }
    
    println!();
    
    // Navigation hints
    println!("{}", "NAVIGATION".green().bold());
    println!("{}", "-".repeat(30).green());
    println!("  📍 Use coordinates to inspect specific regions");
    println!("  🔍 Increase zoom for more detail: --zoom {}", (zoom_level + 1).min(10));
    println!("  🔍 Decrease zoom for broader view: --zoom {}", (zoom_level.saturating_sub(1)).max(1));
    println!("  🗺️  Try different map types: stars, entropy, temperature, density, lineages, resources");
}

// Generate mock map data when simulation is not running
pub fn generate_mock_map_data(map_type: &str, zoom_level: u8, width: u16, height: u16) -> MapData {
    let mut rng = thread_rng();
    let mut cells = Vec::new();
    
    let parsed_map_type = match map_type {
        "entropy" => MapType::Entropy,
        "temperature" => MapType::Temperature,
        "density" => MapType::Density,
        "lineages" => MapType::Lineages,
        "resources" => MapType::Resources,
        _ => MapType::Stars,
    };
    
    // Generate cells based on map type
    for y in 0..height {
        let mut row = Vec::new();
        for x in 0..width {
            let base_value = generate_base_value(x, y, width, height, &parsed_map_type, &mut rng);
            let symbol = choose_symbol_for_value(base_value, &parsed_map_type);
            
            row.push(crate::data_models::MapCell {
                value: base_value,
                symbol,
                color: None,
                tooltip: Some(format!("({}, {}) = {:.3}", x, y, base_value)),
            });
        }
        cells.push(row);
    }
    
    // Generate appropriate legend
    let legend = generate_legend_for_type(&parsed_map_type);
    
    MapData {
        width,
        height,
        zoom_level,
        map_type: parsed_map_type,
        cells,
        legend,
    }
}

fn generate_base_value(x: u16, y: u16, width: u16, height: u16, map_type: &MapType, rng: &mut ThreadRng) -> f64 {
    let center_x = width as f64 / 2.0;
    let center_y = height as f64 / 2.0;
    let distance_from_center = ((x as f64 - center_x).powi(2) + (y as f64 - center_y).powi(2)).sqrt();
    let max_distance = (center_x.powi(2) + center_y.powi(2)).sqrt();
    let normalized_distance = distance_from_center / max_distance;
    
    match map_type {
        MapType::Stars => {
            // Stars more concentrated in center (galactic core)
            let base = (1.0 - normalized_distance * 0.8).max(0.1);
            base * rng.gen::<f64>() * rng.gen::<f64>() // Clustered distribution
        }
        MapType::Entropy => {
            // Entropy increases with distance from center
            let base = normalized_distance * 0.7 + 0.2;
            base + rng.gen::<f64>() * 0.3 - 0.15 // Some randomness
        }
        MapType::Temperature => {
            // Temperature decreases with distance
            let base = (1.0 - normalized_distance * 0.9).max(0.0);
            base + rng.gen::<f64>() * 0.4 - 0.2
        }
        MapType::Density => {
            // Matter density in spiral patterns
            let angle = (y as f64 - center_y).atan2(x as f64 - center_x);
            let spiral = (angle * 3.0 + normalized_distance * 10.0).sin() * 0.5 + 0.5;
            spiral * (1.0 - normalized_distance * 0.5) * rng.gen::<f64>()
        }
        MapType::Lineages => {
            // Lineages appear in habitable zones
            if normalized_distance > 0.3 && normalized_distance < 0.7 {
                rng.gen::<f64>() * rng.gen::<f64>() * 0.8 // Sparse but present
            } else {
                rng.gen::<f64>() * 0.1 // Very rare
            }
        }
        MapType::Resources => {
            // Resources scattered but concentrated in certain areas
            let resource_hotspot = ((x as f64 / width as f64 * 7.0).sin() * (y as f64 / height as f64 * 5.0).cos()).abs();
            resource_hotspot * rng.gen::<f64>()
        }
    }
}

fn choose_symbol_for_value(value: f64, map_type: &MapType) -> char {
    match map_type {
        MapType::Stars => {
            if value > 0.8 { '*' }
            else if value > 0.5 { '·' }
            else if value > 0.2 { '.' }
            else { ' ' }
        }
        MapType::Entropy => {
            if value > 0.8 { '█' }
            else if value > 0.6 { '▓' }
            else if value > 0.4 { '▒' }
            else if value > 0.2 { '░' }
            else { ' ' }
        }
        MapType::Temperature => {
            if value > 0.8 { '#' }
            else if value > 0.6 { 'o' }
            else if value > 0.4 { '·' }
            else if value > 0.1 { '.' }
            else { ' ' }
        }
        MapType::Density => {
            if value > 0.8 { '█' }
            else if value > 0.6 { '▓' }
            else if value > 0.4 { '▒' }
            else if value > 0.2 { '░' }
            else { ' ' }
        }
        MapType::Lineages => {
            if value > 0.7 { 'A' }
            else if value > 0.4 { 'a' }
            else if value > 0.1 { 'o' }
            else { ' ' }
        }
        MapType::Resources => {
            if value > 0.8 { '$' }
            else if value > 0.6 { '#' }
            else if value > 0.3 { '+' }
            else if value > 0.1 { '·' }
            else { ' ' }
        }
    }
}

fn generate_legend_for_type(map_type: &MapType) -> Vec<crate::data_models::LegendEntry> {
    match map_type {
        MapType::Stars => vec![
            crate::data_models::LegendEntry {
                symbol: '*',
                description: "High stellar density".to_string(),
                value_range: Some((0.8, 1.0)),
            },
            crate::data_models::LegendEntry {
                symbol: '·',
                description: "Medium stellar density".to_string(),
                value_range: Some((0.5, 0.8)),
            },
            crate::data_models::LegendEntry {
                symbol: '.',
                description: "Low stellar density".to_string(),
                value_range: Some((0.2, 0.5)),
            },
            crate::data_models::LegendEntry {
                symbol: ' ',
                description: "Empty space".to_string(),
                value_range: Some((0.0, 0.2)),
            },
        ],
        MapType::Entropy => vec![
            crate::data_models::LegendEntry {
                symbol: '█',
                description: "Maximum entropy".to_string(),
                value_range: Some((0.8, 1.0)),
            },
            crate::data_models::LegendEntry {
                symbol: '▓',
                description: "High entropy".to_string(),
                value_range: Some((0.6, 0.8)),
            },
            crate::data_models::LegendEntry {
                symbol: '▒',
                description: "Medium entropy".to_string(),
                value_range: Some((0.4, 0.6)),
            },
            crate::data_models::LegendEntry {
                symbol: '░',
                description: "Low entropy".to_string(),
                value_range: Some((0.2, 0.4)),
            },
        ],
        MapType::Temperature => vec![
            crate::data_models::LegendEntry {
                symbol: '#',
                description: "Very hot regions".to_string(),
                value_range: Some((0.8, 1.0)),
            },
            crate::data_models::LegendEntry {
                symbol: 'o',
                description: "Warm regions".to_string(),
                value_range: Some((0.6, 0.8)),
            },
            crate::data_models::LegendEntry {
                symbol: '·',
                description: "Cool regions".to_string(),
                value_range: Some((0.4, 0.6)),
            },
            crate::data_models::LegendEntry {
                symbol: '.',
                description: "Cold regions".to_string(),
                value_range: Some((0.1, 0.4)),
            },
        ],
        MapType::Density => vec![
            crate::data_models::LegendEntry {
                symbol: '█',
                description: "Very high density".to_string(),
                value_range: Some((0.8, 1.0)),
            },
            crate::data_models::LegendEntry {
                symbol: '▓',
                description: "High density".to_string(),
                value_range: Some((0.6, 0.8)),
            },
            crate::data_models::LegendEntry {
                symbol: '▒',
                description: "Medium density".to_string(),
                value_range: Some((0.4, 0.6)),
            },
            crate::data_models::LegendEntry {
                symbol: '░',
                description: "Low density".to_string(),
                value_range: Some((0.2, 0.4)),
            },
        ],
        MapType::Lineages => vec![
            crate::data_models::LegendEntry {
                symbol: 'A',
                description: "Advanced AI lineages".to_string(),
                value_range: Some((0.7, 1.0)),
            },
            crate::data_models::LegendEntry {
                symbol: 'a',
                description: "Developing lineages".to_string(),
                value_range: Some((0.4, 0.7)),
            },
            crate::data_models::LegendEntry {
                symbol: 'o',
                description: "Basic life forms".to_string(),
                value_range: Some((0.1, 0.4)),
            },
        ],
        MapType::Resources => vec![
            crate::data_models::LegendEntry {
                symbol: '$',
                description: "Rare elements".to_string(),
                value_range: Some((0.8, 1.0)),
            },
            crate::data_models::LegendEntry {
                symbol: '#',
                description: "Common metals".to_string(),
                value_range: Some((0.6, 0.8)),
            },
            crate::data_models::LegendEntry {
                symbol: '+',
                description: "Basic materials".to_string(),
                value_range: Some((0.3, 0.6)),
            },
            crate::data_models::LegendEntry {
                symbol: '·',
                description: "Trace elements".to_string(),
                value_range: Some((0.1, 0.3)),
            },
        ],
    }
}