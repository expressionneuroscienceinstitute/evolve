// Heavy Mode Scientific Visualization Shader for Universe Simulation
// Advanced particle rendering with physics-based visualization

struct Uniforms {
    view_proj: mat4x4<f32>,
    time: f32,
    scale: f32,
    color_mode: f32,
    filter_threshold: f32,
    _padding: f32,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) velocity: vec3<f32>,
    @location(2) mass: f32,
    @location(3) charge: f32,
    @location(4) temperature: f32,
    @builtin(vertex_index) vertex_index: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) size: f32,
    @location(3) intensity: f32,
    @location(4) particle_class: f32,
};

// Scientific constants for realistic visualization
const SPEED_OF_LIGHT: f32 = 299792458.0;
const PLANCK_CONSTANT: f32 = 6.626e-34;
const BOLTZMANN_CONSTANT: f32 = 1.381e-23;

// Color palettes for scientific visualization
fn get_particle_type_color(particle_type: f32) -> vec3<f32> {
    // Standard Model particle color coding
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
    // Blackbody radiation color based on temperature
    let temp_normalized = clamp(temperature / 10000.0, 0.0, 1.0);
    
    if (temp_normalized < 0.25) {
        // Cool: Black -> Red
        return vec3<f32>(temp_normalized * 4.0, 0.0, 0.0);
    } else if (temp_normalized < 0.5) {
        // Warm: Red -> Yellow
        let t = (temp_normalized - 0.25) * 4.0;
        return vec3<f32>(1.0, t, 0.0);
    } else if (temp_normalized < 0.75) {
        // Hot: Yellow -> White
        let t = (temp_normalized - 0.5) * 4.0;
        return vec3<f32>(1.0, 1.0, t);
    } else {
        // Very hot: White -> Blue
        let t = (temp_normalized - 0.75) * 4.0;
        return vec3<f32>(1.0 - t * 0.3, 1.0 - t * 0.3, 1.0);
    }
}

fn get_velocity_color(velocity: vec3<f32>) -> vec3<f32> {
    // Doppler shift visualization
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

fn get_interaction_color(interaction_count: f32) -> vec3<f32> {
    // Heat map based on interaction frequency
    let intensity = clamp(interaction_count / 10.0, 0.0, 1.0);
    
    if (intensity < 0.33) {
        // Cold - Blue to Cyan
        let t = intensity * 3.0;
        return vec3<f32>(0.0, t, 1.0);
    } else if (intensity < 0.67) {
        // Warm - Cyan to Yellow
        let t = (intensity - 0.33) * 3.0;
        return vec3<f32>(t, 1.0, 1.0 - t);
    } else {
        // Hot - Yellow to Red
        let t = (intensity - 0.67) * 3.0;
        return vec3<f32>(1.0, 1.0 - t, 0.0);
    }
}

fn get_scientific_color(input: VertexInput) -> vec3<f32> {
    // Multi-channel scientific visualization
    let type_color = get_particle_type_color(0.0); // Default particle type for now
    let charge_color = get_charge_color(input.charge);
    let temp_color = get_temperature_color(input.temperature);
    let velocity_color = get_velocity_color(input.velocity);
    let interaction_color = get_interaction_color(0.0); // Default interaction count for now
    
    // Blend multiple channels with physics-based weighting
    let mass_weight = clamp(log(max(input.mass, 1e-30)) / 10.0, 0.0, 1.0);
    let energy_weight = clamp(length(input.velocity) / 1000.0, 0.0, 1.0);
    
    var result = type_color * 0.3;
    result += charge_color * 0.2;
    result += temp_color * 0.2;
    result += velocity_color * energy_weight * 0.2;
    result += interaction_color * mass_weight * 0.1;
    
    return normalize(result);
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
    
    // Calculate particle size based on scale mode
    var size_factor: f32;
    if (uniforms.scale == 1.0) {
        // Linear scaling
        size_factor = input.mass / 1e-27; // Normalize to atomic mass
    } else if (uniforms.scale == 2.0) {
        // Logarithmic scaling for wide mass ranges
        size_factor = log(max(input.mass, 1e-30)) / log(10.0);
    } else if (uniforms.scale == 3.0) {
        // Energy-based scaling
        let kinetic_energy = 0.5 * input.mass * dot(input.velocity, input.velocity);
        size_factor = log(max(kinetic_energy, 1e-30)) / log(10.0);
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
    if (uniforms.color_mode == 0.0) {
        out.color = get_particle_type_color(0.0); // Default particle type
    } else if (uniforms.color_mode == 1.0) {
        out.color = get_charge_color(input.charge);
    } else if (uniforms.color_mode == 2.0) {
        out.color = get_temperature_color(input.temperature);
    } else if (uniforms.color_mode == 3.0) {
        out.color = get_velocity_color(input.velocity);
    } else if (uniforms.color_mode == 4.0) {
        out.color = get_interaction_color(0.0); // Default interaction count
    } else {
        out.color = get_scientific_color(input);
    }
    
    // Modulate by temperature for additional visual feedback
    if (input.temperature > 0.0) {
        let temp_normalized = min(input.temperature / 10000.0, 1.0);
        out.color = mix(out.color, vec3<f32>(1.0, 0.8, 0.2), temp_normalized);
    }
    
    // Calculate intensity based on physics
    let relativistic_gamma = 1.0 / sqrt(1.0 - pow(length(input.velocity) / SPEED_OF_LIGHT, 2.0));
    out.intensity = clamp(relativistic_gamma, 1.0, 10.0);
    
    // Particle classification for shader effects
    if (input.mass > 1e20) {
        out.particle_class = 4.0; // Massive objects (stars, planets)
    } else if (input.mass > 1e10) {
        out.particle_class = 3.0; // Medium objects
    } else if (input.mass > 1e-25) {
        out.particle_class = 2.0; // Atoms/molecules
    } else if (input.mass > 1e-28) {
        out.particle_class = 1.0; // Nuclei
    } else {
        out.particle_class = 0.0; // Fundamental particles
    }
    
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Create circular particles
    let center = vec2<f32>(0.5, 0.5);
    let dist = length(input.uv - center);
    
    // Physics-based particle rendering
    var alpha: f32;
    var brightness: f32;
    
    if (input.particle_class >= 4.0) {
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
    
    return vec4<f32>(input.color * brightness, basic_alpha);
} 