use crate::data_models::SimulationState;

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
        // Improved character set for better visual distinction
        let density_chars = [' ', '·', '∙', ':', '•', '○', '◉', '◎', '◈', '■', '█'];
        
        // First pass: Calculate min/max density for normalization
        let mut min_density = f64::MAX;
        let mut max_density = f64::MIN;
        let mut density_samples = vec![vec![0.0; map[0].len()]; map.len()];
        
        for y in 0..map.len() {
            for x in 0..map[0].len() {
                let world_x = min_x + (x as f64 / map[0].len() as f64) * (max_x - min_x);
                let world_y = min_y + (y as f64 / map.len() as f64) * (max_y - min_y);
                let world_z = 0.0; // 2D slice at z=0
                
                // Get density from the field - this already has cosmic web structure
                let density = state.density_field.sample_at(world_x, world_y, world_z);
                density_samples[y][x] = density;
                
                if density > 0.0 {
                    min_density = min_density.min(density);
                    max_density = max_density.max(density);
                }
            }
        }
        
        // Use logarithmic scaling for better visualization of density contrasts
        let log_min = if min_density > 0.0 { min_density.ln() } else { 0.0 };
        let log_max = max_density.ln();
        let log_range = log_max - log_min;
        
        // Second pass: Render the density field
        for y in 0..map.len() {
            for x in 0..map[0].len() {
                let density = density_samples[y][x];
                
                if density > 0.0 {
                    // Logarithmic scaling to better show density contrasts
                    let log_density = density.ln();
                    let normalized = ((log_density - log_min) / log_range).clamp(0.0, 1.0);
                    
                    // Use power scaling to enhance contrast
                    let enhanced = normalized.powf(0.5); // Square root to spread out the range
                    let char_idx = (enhanced * (density_chars.len() - 1) as f64) as usize;
                    map[y][x] = density_chars[char_idx.min(density_chars.len() - 1)];
                }
                
                let world_x = min_x + (x as f64 / map[0].len() as f64) * (max_x - min_x);
                let world_y = min_y + (y as f64 / map.len() as f64) * (max_y - min_y);
                
                // Overlay galaxy markers on top of density field
                for galaxy in &state.galaxies {
                    let dx = (galaxy.position.x - world_x).abs();
                    let dy = (galaxy.position.y - world_y).abs();
                    
                    // Check if galaxy is within this character cell
                    let cell_width = (max_x - min_x) / map[0].len() as f64;
                    let cell_height = (max_y - min_y) / map.len() as f64;
                    
                    if dx < cell_width / 2.0 && dy < cell_height / 2.0 {
                        // Use different markers for different galaxy types
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
        
        // Apply smoothing filter to reduce noise
        self.smooth_map(&mut map);
        
        map
    }
    
    fn smooth_map(&self, map: &mut Vec<Vec<char>>) {
        // Simple smoothing to make cosmic structures more visible
        let density_order = [' ', '·', '∙', ':', '•', '○', '◉', '◎', '◈', '■', '█'];
        let special_chars = ['@', 'O', '*', 'o']; // Galaxy markers
        
        let mut smoothed = map.clone();
        
        for y in 1..map.len()-1 {
            for x in 1..map[0].len()-1 {
                // Skip galaxy markers
                if special_chars.contains(&map[y][x]) {
                    continue;
                }
                
                // Count neighboring density levels
                let mut density_sum = 0;
                let mut count = 0;
                
                for dy in -1..=1 {
                    for dx in -1..=1 {
                        let ny = (y as i32 + dy) as usize;
                        let nx = (x as i32 + dx) as usize;
                        
                        if let Some(pos) = density_order.iter().position(|&c| c == map[ny][nx]) {
                            density_sum += pos;
                            count += 1;
                        }
                    }
                }
                
                if count > 0 {
                    let avg_density = density_sum / count;
                    smoothed[y][x] = density_order[avg_density.min(density_order.len() - 1)];
                }
            }
        }
        
        *map = smoothed;
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