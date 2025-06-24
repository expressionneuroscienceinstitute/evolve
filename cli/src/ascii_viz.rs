use crate::data_models::{SimulationState, Galaxy, Star, Planet};
use nalgebra::Vector3;
use std::collections::HashMap;

pub struct AsciiRenderer {
    width: usize,
    height: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum MapType {
    Density,
    Entropy,
    Energy,
    Temperature,
}

impl std::fmt::Display for MapType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MapType::Density => write!(f, "Density"),
            MapType::Entropy => write!(f, "Entropy"),
            MapType::Energy => write!(f, "Energy"),
            MapType::Temperature => write!(f, "Temperature"),
        }
    }
}

impl AsciiRenderer {
    pub fn new(width: usize, height: usize) -> Self {
        Self { width, height }
    }
    
    pub fn render_map(
        &self,
        state: &SimulationState,
        map_type: MapType,
        zoom: u8,
        center: Option<(f64, f64)>,
    ) -> Result<String, anyhow::Error> {
        let mut output = String::new();
        
        // Header with simulation info
        output.push_str(&format!("╔{}╗\n", "═".repeat(self.width - 2)));
        output.push_str(&format!("║ Universe Simulation - {} Map {}║\n", 
            map_type, 
            " ".repeat(self.width - 30 - map_type.to_string().len())
        ));
        output.push_str(&format!("║ Tick: {:10} | Time: {:.2} Gyr | Zoom: {}x {}║\n",
            state.tick,
            state.current_time / 1e9,
            zoom,
            " ".repeat(self.width - 52)
        ));
        output.push_str(&format!("╠{}╣\n", "═".repeat(self.width - 2)));
        
        // Calculate view bounds
        let (center_x, center_y) = center.unwrap_or((0.0, 0.0));
        let view_size = state.box_size / zoom as f64;
        let min_x = center_x - view_size / 2.0;
        let max_x = center_x + view_size / 2.0;
        let min_y = center_y - view_size / 2.0;
        let max_y = center_y + view_size / 2.0;
        
        // Create density map
        let map_data = match map_type {
            MapType::Density => self.render_density_map(state, min_x, max_x, min_y, max_y),
            MapType::Entropy => self.render_entropy_map(state, min_x, max_x, min_y, max_y),
            MapType::Energy => self.render_energy_map(state, min_x, max_x, min_y, max_y),
            MapType::Temperature => self.render_temperature_map(state, min_x, max_x, min_y, max_y),
        };
        
        // Render the map with proper ASCII characters for cosmic structures
        for y in 0..self.height - 8 {
            output.push_str("║");
            for x in 0..self.width - 2 {
                if x < map_data[y].len() {
                    output.push(map_data[y][x]);
                } else {
                    output.push(' ');
                }
            }
            output.push_str("║\n");
        }
        
        // Legend and scale
        output.push_str(&format!("╠{}╣\n", "═".repeat(self.width - 2)));
        output.push_str(&self.render_legend(map_type));
        output.push_str(&format!("║ Scale: {:.1} Mpc/char | Center: ({:.1}, {:.1}) {}║\n",
            view_size / self.width as f64,
            center_x,
            center_y,
            " ".repeat(self.width - 50)
        ));
        output.push_str(&format!("╚{}╝\n", "═".repeat(self.width - 2)));
        
        Ok(output)
    }
    
    fn render_density_map(
        &self,
        state: &SimulationState,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
    ) -> Vec<Vec<char>> {
        let mut map = vec![vec![' '; self.width - 2]; self.height - 8];
        
        // Use different characters for different density levels to show cosmic web
        let density_chars = [' ', '·', '∙', '•', '○', '◉', '◎', '◈', '■', '█'];
        
        // Sample the density field at each point
        for y in 0..map.len() {
            for x in 0..map[0].len() {
                let world_x = min_x + (x as f64 / map[0].len() as f64) * (max_x - min_x);
                let world_y = min_y + (y as f64 / map.len() as f64) * (max_y - min_y);
                let world_z = 0.0; // 2D slice at z=0
                
                // Get density from the field
                let mut density = state.density_field.sample_at(world_x, world_y, world_z);
                
                // Add contribution from galaxies
                for galaxy in &state.galaxies {
                    let dx = galaxy.position.x - world_x;
                    let dy = galaxy.position.y - world_y;
                    let dist_sq = dx * dx + dy * dy;
                    if dist_sq < 25.0 { // Within 5 Mpc
                        density += 10.0 * (-dist_sq / 10.0).exp();
                    }
                }
                
                // Map density to character
                let char_idx = ((density.ln() + 1.0) * 2.0).clamp(0.0, 9.0) as usize;
                map[y][x] = density_chars[char_idx.min(9)];
                
                // Special markers for galaxy clusters
                for galaxy in &state.galaxies {
                    if (galaxy.position.x - world_x).abs() < 0.5 && 
                       (galaxy.position.y - world_y).abs() < 0.5 {
                        map[y][x] = match galaxy.galaxy_type {
                            crate::data_models::GalaxyType::Spiral => '@',
                            crate::data_models::GalaxyType::Elliptical => 'O',
                            crate::data_models::GalaxyType::Irregular => '*',
                            crate::data_models::GalaxyType::Dwarf => 'o',
                        };
                    }
                }
            }
        }
        
        // Add cosmic web filaments
        self.add_filaments(&mut map, min_x, max_x, min_y, max_y);
        
        map
    }
    
    fn render_entropy_map(
        &self,
        state: &SimulationState,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
    ) -> Vec<Vec<char>> {
        let mut map = vec![vec![' '; self.width - 2]; self.height - 8];
        let entropy_chars = [' ', '░', '▒', '▓', '█'];
        
        for y in 0..map.len() {
            for x in 0..map[0].len() {
                let world_x = min_x + (x as f64 / map[0].len() as f64) * (max_x - min_x);
                let world_y = min_y + (y as f64 / map.len() as f64) * (max_y - min_y);
                
                // Simple entropy visualization based on local density variations
                let density = state.density_field.sample_at(world_x, world_y, 0.0);
                let entropy_level = (density.ln() / 3.0).clamp(0.0, 4.0) as usize;
                map[y][x] = entropy_chars[entropy_level.min(4)];
            }
        }
        
        map
    }
    
    fn render_energy_map(
        &self,
        state: &SimulationState,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
    ) -> Vec<Vec<char>> {
        let mut map = vec![vec![' '; self.width - 2]; self.height - 8];
        let energy_chars = ['_', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
        
        for y in 0..map.len() {
            for x in 0..map[0].len() {
                let world_x = min_x + (x as f64 / map[0].len() as f64) * (max_x - min_x);
                let world_y = min_y + (y as f64 / map.len() as f64) * (max_y - min_y);
                
                // Energy based on star density and activity
                let mut energy = 0.0;
                for star in &state.stars {
                    let dx = star.position.x - world_x;
                    let dy = star.position.y - world_y;
                    let dist_sq = dx * dx + dy * dy;
                    if dist_sq < 4.0 { // Within 2 Mpc
                        energy += star.luminosity * (-dist_sq).exp();
                    }
                }
                
                let char_idx = (energy * 5.0).clamp(0.0, 8.0) as usize;
                map[y][x] = energy_chars[char_idx];
            }
        }
        
        map
    }
    
    fn render_temperature_map(
        &self,
        state: &SimulationState,
        min_x: f64,
        max_x: f64,
        min_y: f64,
        max_y: f64,
    ) -> Vec<Vec<char>> {
        let mut map = vec![vec![' '; self.width - 2]; self.height - 8];
        
        // Temperature visualization with color-like ASCII
        let temp_chars = [' ', '.', ':', '+', 'x', 'X', '#', '@', '█'];
        
        for y in 0..map.len() {
            for x in 0..map[0].len() {
                let world_x = min_x + (x as f64 / map[0].len() as f64) * (max_x - min_x);
                let world_y = min_y + (y as f64 / map.len() as f64) * (max_y - min_y);
                
                // Base CMB temperature
                let mut temp = state.temperature;
                
                // Add heating from nearby galaxies and stars
                for galaxy in &state.galaxies {
                    let dx = galaxy.position.x - world_x;
                    let dy = galaxy.position.y - world_y;
                    let dist_sq = dx * dx + dy * dy;
                    if dist_sq < 10.0 {
                        temp += 0.5 * galaxy.star_formation_rate * (-dist_sq / 5.0).exp();
                    }
                }
                
                let char_idx = ((temp - 2.7) * 10.0).clamp(0.0, 8.0) as usize;
                map[y][x] = temp_chars[char_idx];
            }
        }
        
        map
    }
    
    fn add_filaments(&self, map: &mut Vec<Vec<char>>, min_x: f64, max_x: f64, min_y: f64, max_y: f64) {
        // Add cosmic web filaments connecting galaxy clusters
        let filament_chars = ['─', '│', '┌', '┐', '└', '┘', '├', '┤', '┬', '┴', '┼'];
        
        // Simple filament pattern - in real simulation this would come from density field
        for y in 0..map.len() {
            for x in 0..map[0].len() {
                let world_x = min_x + (x as f64 / map[0].len() as f64) * (max_x - min_x);
                let world_y = min_y + (y as f64 / map.len() as f64) * (max_y - min_y);
                
                // Create web-like structure
                let web1 = ((world_x * 0.1).sin() * (world_y * 0.1).cos()).abs();
                let web2 = ((world_x * 0.05).cos() * (world_y * 0.15).sin()).abs();
                
                if web1 < 0.1 || web2 < 0.1 {
                    if map[y][x] == ' ' || map[y][x] == '·' {
                        // Determine filament direction
                        if web1 < web2 {
                            map[y][x] = '─';
                        } else {
                            map[y][x] = '│';
                        }
                    }
                }
            }
        }
    }
    
    fn render_legend(&self, map_type: MapType) -> String {
        match map_type {
            MapType::Density => {
                format!("║ Legend: [' '=void] ['·'=low] ['•'=medium] ['◉'=high] ['@'=galaxy] {}║\n",
                    " ".repeat(self.width - 70))
            }
            MapType::Entropy => {
                format!("║ Legend: [' '=low] ['░'=.] ['▒'=..] ['▓'=...] ['█'=high] {}║\n",
                    " ".repeat(self.width - 62))
            }
            MapType::Energy => {
                format!("║ Legend: ['_'=low] ['▄'=medium] ['█'=high energy] {}║\n",
                    " ".repeat(self.width - 55))
            }
            MapType::Temperature => {
                format!("║ Legend: [' '=2.7K] ['.'=3K] ['+'=5K] ['X'=10K] ['█'=hot] {}║\n",
                    " ".repeat(self.width - 63))
            }
        }
    }
}