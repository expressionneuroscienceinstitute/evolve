// High-Performance Particle Shader for Universe Simulation
// Optimized for physics engine visualization

struct Uniforms {
    view_proj: mat4x4<f32>,
    time: f32,
    scale: f32,
    color_mode: f32,
    filter_threshold: f32,
    _padding: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) velocity: vec3<f32>,
    @location(2) mass: f32,
    @location(3) charge: f32,
    @location(4) temperature: f32,
    @location(5) particle_type: f32,
    @location(6) interaction_count: f32,
    @location(7) _padding: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) size: f32,
    @location(2) particle_type: f32,
};

// Thermal camera-like temperature color palette (inspired by FLIR Rainbow mode)
fn get_thermal_color(temperature: f32) -> vec3<f32> {
    let temp_normalized = clamp(temperature / 10000.0, 0.0, 1.0);
    
    if (temp_normalized < 0.2) {
        // Very cold: Black to Deep Blue
        let t = temp_normalized * 5.0;
        return vec3<f32>(0.0, 0.0, t * 0.5);
    } else if (temp_normalized < 0.4) {
        // Cold: Deep Blue to Cyan
        let t = (temp_normalized - 0.2) * 5.0;
        return vec3<f32>(0.0, t * 0.8, 0.5 + t * 0.5);
    } else if (temp_normalized < 0.6) {
        // Medium: Cyan to Green
        let t = (temp_normalized - 0.4) * 5.0;
        return vec3<f32>(0.0, 0.8 + t * 0.2, 1.0 - t * 0.5);
    } else if (temp_normalized < 0.8) {
        // Warm: Green to Yellow
        let t = (temp_normalized - 0.6) * 5.0;
        return vec3<f32>(t, 1.0, 0.5 - t * 0.5);
    } else {
        // Hot: Yellow to Red to White
        let t = (temp_normalized - 0.8) * 5.0;
        if (t < 0.5) {
            // Yellow to Red
            return vec3<f32>(1.0, 1.0 - t * 2.0, 0.0);
        } else {
            // Red to White (very hot)
            let white_factor = (t - 0.5) * 2.0;
            return vec3<f32>(1.0, white_factor, white_factor);
        }
    }
}

// White to yellow charge visualization scale
fn get_charge_color(charge: f32) -> vec3<f32> {
    let max_charge = 2.0; // Assume typical range is -2 to +2
    let charge_magnitude = abs(charge);
    let normalized_charge = clamp(charge_magnitude / max_charge, 0.0, 1.0);
    
    if (charge > 0.0) {
        // Positive charge: White to Bright Yellow
        return vec3<f32>(1.0, 1.0, 1.0 - normalized_charge * 0.8);
    } else if (charge < 0.0) {
        // Negative charge: White to Cyan
        return vec3<f32>(1.0 - normalized_charge * 0.8, 1.0, 1.0);
    } else {
        // Neutral: Pure White
        return vec3<f32>(1.0, 1.0, 1.0);
    }
}

// Explicit velocity visualization using direction + magnitude
fn get_velocity_color(velocity: vec3<f32>) -> vec3<f32> {
    let speed = length(velocity);
    let max_speed = 1000.0; // Adjust based on typical speeds in simulation
    let speed_factor = clamp(speed / max_speed, 0.0, 1.0);
    
    if (speed < 0.001) {
        // Nearly stationary: Dark gray
        return vec3<f32>(0.3, 0.3, 0.3);
    }
    
    // Use velocity direction for hue, speed for intensity
    let direction = normalize(velocity);
    
    // Map velocity direction to color components
    // X-axis (left/right): Red channel
    // Y-axis (up/down): Green channel  
    // Z-axis (forward/back): Blue channel
    let dir_color = vec3<f32>(
        abs(direction.x),
        abs(direction.y), 
        abs(direction.z)
    );
    
    // Blend direction color with speed-based brightness
    let base_intensity = 0.2 + speed_factor * 0.8;
    let final_color = dir_color * base_intensity;
    
    // Add speed-based "heat" effect for very fast particles
    if (speed_factor > 0.7) {
        let heat_factor = (speed_factor - 0.7) / 0.3;
        return mix(final_color, vec3<f32>(1.0, 0.8, 0.0), heat_factor * 0.5);
    }
    
    return final_color;
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32, input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Generate quad vertices from vertex index (6 vertices per particle)
    let quad_vertex = vertex_index % 6u;
    var quad_pos: vec2<f32>;
    
    switch quad_vertex {
        case 0u: { quad_pos = vec2<f32>(-1.0, -1.0); }
        case 1u: { quad_pos = vec2<f32>( 1.0, -1.0); }
        case 2u: { quad_pos = vec2<f32>(-1.0,  1.0); }
        case 3u: { quad_pos = vec2<f32>( 1.0, -1.0); }
        case 4u: { quad_pos = vec2<f32>( 1.0,  1.0); }
        case 5u: { quad_pos = vec2<f32>(-1.0,  1.0); }
        default: { quad_pos = vec2<f32>(0.0, 0.0); }
    }
    
    // Calculate particle size based on mass (logarithmic scale)
    let base_size = 0.02; // Smaller base size
    let mass_factor = log(max(input.mass, 1e-30)) / log(10.0);
    let particle_size = base_size * (1.0 + abs(mass_factor) * 0.1);
    
    // Apply quad offset to particle position
    let world_pos = input.position + vec3<f32>(quad_pos * particle_size, 0.0);
    
    // Transform position to clip space
    out.clip_position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    out.size = particle_size;
    
    // Enhanced color modes with realistic palettes
    var color = vec3<f32>(1.0, 1.0, 1.0); // Default white
    
    // Color mode selection
    if (uniforms.color_mode < 0.5) {
        // ParticleType mode (keep existing)
        color = mix(vec3<f32>(0.5, 0.5, 1.0), vec3<f32>(1.0, 0.5, 0.5), input.particle_type / 10.0);
    } else if (uniforms.color_mode < 1.5) {
        // Charge mode: White to Yellow scale
        color = get_charge_color(input.charge);
    } else if (uniforms.color_mode < 2.5) {
        // Temperature mode: Thermal camera palette
        color = get_thermal_color(input.temperature);
    } else {
        // Velocity mode: Explicit direction + magnitude visualization  
        color = get_velocity_color(input.velocity);
    }
    
    out.color = color;
    out.particle_type = input.particle_type;
    
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Simple alpha for now - could add circular falloff later
    let alpha = 0.8;
    
    // Enhanced brightness for different particle types
    var brightness = 1.0;
    if (input.particle_type >= 3.0) {
        brightness = 1.5; // Bright for massive objects
    } else {
        brightness = 1.0; // Normal brightness
    }
    
    // Apply time-based pulsing
    let pulse = sin(uniforms.time * 3.0) * 0.2 + 0.8;
    brightness *= pulse;
    
    return vec4<f32>(input.color * brightness, alpha);
} 