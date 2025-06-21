// Advanced SPH and Sink Particle Shader for Universe Simulation
// High-performance visualization of fluid dynamics, gravitational collapse, and star formation

struct Uniforms {
    view_proj: mat4x4<f32>,
    time: f32,
    scale: f32,
    color_mode: f32,
    filter_threshold: f32,
    scientific_mode: f32,
    sph_enabled: f32,
    sink_enabled: f32,
    _padding: [f32; 2],
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) velocity: vec3<f32>,
    @location(2) mass: f32,
    @location(3) charge: f32,
    @location(4) temperature: f32,
    @location(5) density: f32,
    @location(6) pressure: f32,
    @location(7) internal_energy: f32,
    @location(8) smoothing_length: f32,
    @location(9) particle_type: f32,
    @location(10) interaction_count: f32,
    @location(11) sink_mass: f32,
    @location(12) accretion_rate: f32,
    @builtin(vertex_index) vertex_index: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) size: f32,
    @location(3) intensity: f32,
    @location(4) particle_class: f32,
    @location(5) sph_data: vec4<f32>, // density, pressure, energy, smoothing_length
    @location(6) sink_data: vec4<f32>, // mass, accretion_rate, age, luminosity
}

// Scientific constants for realistic visualization
const SPEED_OF_LIGHT: f32 = 299792458.0;
const PLANCK_CONSTANT: f32 = 6.626e-34;
const BOLTZMANN_CONSTANT: f32 = 1.381e-23;
const STEFAN_BOLTZMANN: f32 = 5.670e-8;
const SOLAR_MASS: f32 = 1.989e30;
const SOLAR_LUMINOSITY: f32 = 3.828e26;

// SPH-specific color palettes
fn get_sph_density_color(density: f32) -> vec3<f32> {
    // Density visualization: transparent -> blue -> white -> red
    let normalized_density = clamp(log(max(density, 1e-30)) / 20.0, 0.0, 1.0);
    
    if (normalized_density < 0.25) {
        // Low density: transparent to blue
        let t = normalized_density * 4.0;
        return vec3<f32>(0.0, 0.0, t);
    } else if (normalized_density < 0.5) {
        // Medium density: blue to cyan
        let t = (normalized_density - 0.25) * 4.0;
        return vec3<f32>(0.0, t, 1.0);
    } else if (normalized_density < 0.75) {
        // High density: cyan to white
        let t = (normalized_density - 0.5) * 4.0;
        return vec3<f32>(t, 1.0, 1.0);
    } else {
        // Very high density: white to red
        let t = (normalized_density - 0.75) * 4.0;
        return vec3<f32>(1.0, 1.0 - t, 1.0 - t);
    }
}

fn get_sph_pressure_color(pressure: f32) -> vec3<f32> {
    // Pressure visualization: green -> yellow -> red (shock waves)
    let normalized_pressure = clamp(log(max(pressure, 1e-30)) / 15.0, 0.0, 1.0);
    
    if (normalized_pressure < 0.5) {
        // Low pressure: green to yellow
        let t = normalized_pressure * 2.0;
        return vec3<f32>(t, 1.0, 0.0);
    } else {
        // High pressure: yellow to red
        let t = (normalized_pressure - 0.5) * 2.0;
        return vec3<f32>(1.0, 1.0 - t, 0.0);
    }
}

fn get_sink_particle_color(sink_mass: f32, accretion_rate: f32) -> vec3<f32> {
    // Sink particle visualization: protostar -> star -> black hole
    let mass_ratio = sink_mass / SOLAR_MASS;
    let accretion_luminosity = accretion_rate * 0.1; // Simplified luminosity calculation
    
    if (mass_ratio < 0.1) {
        // Protostar: red with accretion glow
        let glow = min(accretion_luminosity, 1.0);
        return vec3<f32>(1.0, 0.2 + glow * 0.8, 0.2 + glow * 0.8);
    } else if (mass_ratio < 10.0) {
        // Main sequence star: yellow-white
        let t = (mass_ratio - 0.1) / 9.9;
        return vec3<f32>(1.0, 1.0 - t * 0.3, 1.0 - t * 0.5);
    } else if (mass_ratio < 100.0) {
        // Giant star: orange-red
        let t = (mass_ratio - 10.0) / 90.0;
        return vec3<f32>(1.0, 0.7 - t * 0.5, 0.5 - t * 0.5);
    } else {
        // Black hole: dark with accretion disk glow
        let glow = min(accretion_luminosity * 2.0, 0.8);
        return vec3<f32>(0.1 + glow * 0.9, 0.1 + glow * 0.4, 0.1 + glow * 0.9);
    }
}

fn get_fluid_dynamics_color(input: VertexInput) -> vec3<f32> {
    // Multi-channel fluid dynamics visualization
    let density_color = get_sph_density_color(input.density);
    let pressure_color = get_sph_pressure_color(input.pressure);
    let velocity_color = get_velocity_color(input.velocity);
    
    // Blend based on local conditions
    let density_weight = clamp(input.density / 1e-15, 0.0, 1.0);
    let pressure_weight = clamp(input.pressure / 1e-10, 0.0, 1.0);
    let velocity_weight = clamp(length(input.velocity) / 1000.0, 0.0, 1.0);
    
    var result = density_color * density_weight * 0.4;
    result += pressure_color * pressure_weight * 0.3;
    result += velocity_color * velocity_weight * 0.3;
    
    return normalize(result);
}

fn get_velocity_color(velocity: vec3<f32>) -> vec3<f32> {
    // Doppler shift visualization for fluid flow
    let speed = length(velocity);
    let beta = speed / SPEED_OF_LIGHT;
    let relativistic_factor = clamp(beta * 100.0, 0.0, 1.0);
    
    // Blue shift for approach, red shift for recession
    let velocity_dir = normalize(velocity);
    let doppler_factor = velocity_dir.z; // Assuming z is towards viewer
    
    if (doppler_factor > 0.0) {
        // Approaching - blue shift
        return mix(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(0.0, 0.5, 1.0), relativistic_factor);
    } else {
        // Receding - red shift
        return mix(vec3<f32>(1.0, 1.0, 1.0), vec3<f32>(1.0, 0.0, 0.0), relativistic_factor);
    }
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Create a small quad for each particle (6 vertices per particle)
    let quad_index = input.vertex_index % 6u;
    var offset = vec2<f32>(0.0, 0.0);
    
    // Define quad vertices (two triangles)
    switch quad_index {
        case 0u: { offset = vec2<f32>(-0.01, -0.01); out.uv = vec2<f32>(0.0, 0.0); }
        case 1u: { offset = vec2<f32>( 0.01, -0.01); out.uv = vec2<f32>(1.0, 0.0); }
        case 2u: { offset = vec2<f32>( 0.01,  0.01); out.uv = vec2<f32>(1.0, 1.0); }
        case 3u: { offset = vec2<f32>(-0.01, -0.01); out.uv = vec2<f32>(0.0, 0.0); }
        case 4u: { offset = vec2<f32>( 0.01,  0.01); out.uv = vec2<f32>(1.0, 1.0); }
        default: { offset = vec2<f32>(-0.01,  0.01); out.uv = vec2<f32>(0.0, 1.0); }
    }
    
    // Calculate particle size based on advanced scale mode
    var size_factor: f32;
    if (uniforms.scale == 1.0) {
        // Linear scaling
        size_factor = input.mass / 1e-27;
    } else if (uniforms.scale == 2.0) {
        // Logarithmic scaling
        size_factor = log(max(input.mass, 1e-30)) / log(10.0);
    } else if (uniforms.scale == 3.0) {
        // Energy-based scaling
        let kinetic_energy = 0.5 * input.mass * dot(input.velocity, input.velocity);
        size_factor = log(max(kinetic_energy, 1e-30)) / log(10.0);
    } else if (uniforms.scale == 4.0) {
        // SPH density-based scaling
        size_factor = log(max(input.density, 1e-30)) / log(10.0);
    } else if (uniforms.scale == 5.0) {
        // Pressure-based scaling
        size_factor = log(max(input.pressure, 1e-30)) / log(10.0);
    } else {
        // Custom scaling
        size_factor = uniforms.scale;
    }
    
    out.size = clamp(2.0 + size_factor * 0.5, 1.0, 50.0);
    
    // Scale offset by calculated size
    let size = log(max(input.mass, 1e-30)) * 0.001 + 0.005;
    offset = offset * size;
    
    let world_pos = vec4<f32>(input.position, 1.0);
    let clip_pos = uniforms.view_proj * world_pos;
    
    // Add offset in screen space
    out.clip_position = vec4<f32>(clip_pos.xy + offset * clip_pos.w, clip_pos.z, clip_pos.w);
    
    // Scientific color coding based on mode
    if (uniforms.scientific_mode == 0.0) {
        out.color = get_particle_type_color(input.particle_type);
    } else if (uniforms.scientific_mode == 1.0) {
        out.color = get_charge_color(input.charge);
    } else if (uniforms.scientific_mode == 2.0) {
        out.color = get_temperature_color(input.temperature);
    } else if (uniforms.scientific_mode == 3.0) {
        out.color = get_velocity_color(input.velocity);
    } else if (uniforms.scientific_mode == 4.0) {
        out.color = get_interaction_color(input.interaction_count);
    } else if (uniforms.scientific_mode == 5.0) {
        out.color = get_sph_density_color(input.density);
    } else if (uniforms.scientific_mode == 6.0) {
        out.color = get_sph_pressure_color(input.pressure);
    } else if (uniforms.scientific_mode == 7.0) {
        out.color = get_energy_color(input.internal_energy);
    } else if (uniforms.scientific_mode == 8.0) {
        out.color = get_sink_particle_color(input.sink_mass, input.accretion_rate);
    } else {
        out.color = get_fluid_dynamics_color(input);
    }
    
    // Calculate intensity based on physics
    let relativistic_gamma = 1.0 / sqrt(1.0 - pow(length(input.velocity) / SPEED_OF_LIGHT, 2.0));
    out.intensity = clamp(relativistic_gamma, 1.0, 10.0);
    
    // Enhanced particle classification
    if (input.sink_mass > 0.0) {
        out.particle_class = 5.0; // Sink particles (protostars, stars, black holes)
    } else if (input.mass > 1e20) {
        out.particle_class = 4.0; // Massive objects
    } else if (input.mass > 1e10) {
        out.particle_class = 3.0; // Medium objects
    } else if (input.mass > 1e-25) {
        out.particle_class = 2.0; // Atoms/molecules
    } else if (input.mass > 1e-28) {
        out.particle_class = 1.0; // Nuclei
    } else {
        out.particle_class = 0.0; // Fundamental particles
    }
    
    // Pass SPH data to fragment shader
    out.sph_data = vec4<f32>(input.density, input.pressure, input.internal_energy, input.smoothing_length);
    
    // Pass sink particle data to fragment shader
    let sink_age = uniforms.time - input.particle_type; // Simplified age calculation
    let sink_luminosity = input.accretion_rate * input.sink_mass / SOLAR_MASS;
    out.sink_data = vec4<f32>(input.sink_mass, input.accretion_rate, sink_age, sink_luminosity);
    
    return out;
}

// Helper functions for color generation
fn get_particle_type_color(particle_type: f32) -> vec3<f32> {
    if (particle_type < 1.0) { return vec3<f32>(1.0, 0.0, 0.0); }      // Quarks - Red
    else if (particle_type < 2.0) { return vec3<f32>(0.0, 1.0, 0.0); } // Leptons - Green  
    else if (particle_type < 3.0) { return vec3<f32>(0.0, 0.0, 1.0); } // Gauge bosons - Blue
    else if (particle_type < 4.0) { return vec3<f32>(1.0, 1.0, 0.0); } // Composite particles - Yellow
    else if (particle_type < 5.0) { return vec3<f32>(1.0, 0.0, 1.0); } // Nuclei - Magenta
    else { return vec3<f32>(0.5, 0.5, 0.5); }                          // Unknown - Gray
}

fn get_charge_color(charge: f32) -> vec3<f32> {
    if (charge > 0.0) {
        return vec3<f32>(1.0, 0.2, 0.2); // Positive - Red
    } else if (charge < 0.0) {
        return vec3<f32>(0.2, 0.2, 1.0); // Negative - Blue
    } else {
        return vec3<f32>(0.8, 0.8, 0.8); // Neutral - Gray
    }
}

fn get_temperature_color(temperature: f32) -> vec3<f32> {
    let temp_normalized = clamp(temperature / 10000.0, 0.0, 1.0);
    
    if (temp_normalized < 0.25) {
        return vec3<f32>(temp_normalized * 4.0, 0.0, 0.0);
    } else if (temp_normalized < 0.5) {
        let t = (temp_normalized - 0.25) * 4.0;
        return vec3<f32>(1.0, t, 0.0);
    } else if (temp_normalized < 0.75) {
        let t = (temp_normalized - 0.5) * 4.0;
        return vec3<f32>(1.0, 1.0, t);
    } else {
        let t = (temp_normalized - 0.75) * 4.0;
        return vec3<f32>(1.0 - t * 0.3, 1.0 - t * 0.3, 1.0);
    }
}

fn get_interaction_color(interaction_count: f32) -> vec3<f32> {
    let intensity = clamp(interaction_count / 10.0, 0.0, 1.0);
    
    if (intensity < 0.33) {
        let t = intensity * 3.0;
        return vec3<f32>(0.0, t, 1.0);
    } else if (intensity < 0.67) {
        let t = (intensity - 0.33) * 3.0;
        return vec3<f32>(t, 1.0, 1.0 - t);
    } else {
        let t = (intensity - 0.67) * 3.0;
        return vec3<f32>(1.0, 1.0 - t, 0.0);
    }
}

fn get_energy_color(energy: f32) -> vec3<f32> {
    let energy_normalized = clamp(log(max(energy, 1e-30)) / 20.0, 0.0, 1.0);
    
    if (energy_normalized < 0.33) {
        let t = energy_normalized * 3.0;
        return vec3<f32>(0.0, 0.0, t);
    } else if (energy_normalized < 0.67) {
        let t = (energy_normalized - 0.33) * 3.0;
        return vec3<f32>(0.0, t, 1.0);
    } else {
        let t = (energy_normalized - 0.67) * 3.0;
        return vec3<f32>(t, 1.0, 1.0);
    }
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Create circular particles
    let center = vec2<f32>(0.5, 0.5);
    let dist = length(input.uv - center);
    
    // Enhanced physics-based particle rendering
    var alpha: f32;
    var brightness: f32;
    
    if (input.particle_class >= 5.0) {
        // Sink particles: protostars, stars, black holes
        alpha = smoothstep(1.0, 0.6, dist);
        brightness = 4.0 + input.sink_data.w * 2.0; // Enhanced by luminosity
        
        // Add accretion disk glow for active sink particles
        if (input.sink_data.y > 0.0) {
            let accretion_glow = smoothstep(1.5, 0.5, dist) * input.sink_data.y * 0.5;
            brightness += accretion_glow;
        }
    } else if (input.particle_class >= 4.0) {
        // Massive objects: solid sphere with atmospheric glow
        alpha = smoothstep(1.0, 0.7, dist);
        brightness = 3.0;
    } else if (input.particle_class >= 3.0) {
        // Medium objects: dense core
        alpha = smoothstep(1.0, 0.8, dist);
        brightness = 2.5;
    } else if (input.particle_class >= 2.0) {
        // Atoms: electron cloud visualization
        alpha = smoothstep(1.0, 0.3, dist) * exp(-dist * 2.0);
        brightness = 2.0;
    } else if (input.particle_class >= 1.0) {
        // Nuclei: compact with quantum uncertainty
        alpha = smoothstep(1.0, 0.9, dist);
        brightness = 1.8;
    } else {
        // Fundamental particles: point-like with quantum fuzziness
        alpha = smoothstep(0.5, 0.0, dist);
        brightness = 1.5;
    }
    
    // SPH-specific effects
    if (uniforms.sph_enabled > 0.5) {
        // Add density-based transparency for fluid visualization
        let density_factor = clamp(input.sph_data.x / 1e-15, 0.0, 1.0);
        alpha *= mix(0.3, 1.0, density_factor);
        
        // Add pressure-based glow for shock waves
        let pressure_factor = clamp(input.sph_data.y / 1e-10, 0.0, 1.0);
        brightness += pressure_factor * 2.0;
    }
    
    // Apply relativistic effects
    brightness *= input.intensity;
    
    // Temporal effects for dynamic visualization
    let pulse = sin(uniforms.time * 2.0 + input.particle_class) * 0.1 + 0.9;
    brightness *= pulse;
    
    // Apply filtering threshold
    if (brightness < uniforms.filter_threshold) {
        discard;
    }
    
    // Fallback to simple circular particle if advanced features aren't working
    let basic_alpha = 1.0 - smoothstep(0.3, 0.5, dist);
    
    return vec4<f32>(input.color * brightness, alpha);
} 