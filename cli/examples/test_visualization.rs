use universectl::data_models::SimulationState;
use universectl::ascii_viz::{AsciiRenderer, MapType};

fn main() -> anyhow::Result<()> {
    // Create a mock simulation state
    let state = SimulationState::mock();
    
    // Create ASCII renderer with 100x40 characters
    let renderer = AsciiRenderer::new(100, 40);
    
    println!("Testing Universe ASCII Visualization\n");
    
    // Test density map at different zoom levels
    println!("=== DENSITY MAP - Full View (Zoom 1x) ===");
    let density_map = renderer.render_map(&state, MapType::Density, 1, None)?;
    println!("{}", density_map);
    
    println!("\n=== DENSITY MAP - Zoomed View (Zoom 3x) ===");
    let density_map_zoomed = renderer.render_map(&state, MapType::Density, 3, Some((0.0, 0.0)))?;
    println!("{}", density_map_zoomed);
    
    println!("\n=== TEMPERATURE MAP ===");
    let temp_map = renderer.render_map(&state, MapType::Temperature, 2, None)?;
    println!("{}", temp_map);
    
    println!("\n=== ENERGY MAP ===");
    let energy_map = renderer.render_map(&state, MapType::Energy, 2, None)?;
    println!("{}", energy_map);
    
    Ok(())
}