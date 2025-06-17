// High-Performance Particle Shader for Universe Simulation
// Optimized for physics engine visualization

struct Uniforms {
    view_proj: mat4x4<f32>,
    time: f32,
    scale: f32,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) velocity: vec3<f32>,
    @location(2) mass: f32,
    @location(3) charge: f32,
    @location(4) temperature: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) size: f32,
    @location(2) particle_type: f32,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Transform position to clip space
    out.clip_position = uniforms.view_proj * vec4<f32>(input.position, 1.0);
    
    // Calculate particle size based on mass (logarithmic scale)
    let base_size = 2.0;
    let mass_factor = log(max(input.mass, 1e-30)) / log(10.0);
    out.size = base_size + mass_factor * 0.5;
    
    // Color based on particle type and properties
    var color = vec3<f32>(1.0, 1.0, 1.0); // Default white
    
    // Color by charge
    if (input.charge > 0.0) {
        color = vec3<f32>(1.0, 0.2, 0.2); // Red for positive
    } else if (input.charge < 0.0) {
        color = vec3<f32>(0.2, 0.2, 1.0); // Blue for negative
    } else {
        color = vec3<f32>(0.8, 0.8, 0.8); // Gray for neutral
    }
    
    // Modulate by temperature (if available)
    if (input.temperature > 0.0) {
        let temp_normalized = min(input.temperature / 10000.0, 1.0); // Normalize to stellar temps
        color = mix(color, vec3<f32>(1.0, 0.8, 0.2), temp_normalized); // Hot -> yellow/white
    }
    
    // Modulate by velocity magnitude for kinetic energy visualization
    let velocity_mag = length(input.velocity);
    let velocity_factor = min(velocity_mag / 1000.0, 1.0); // Normalize relativistic speeds
    color = mix(color, vec3<f32>(0.0, 1.0, 0.0), velocity_factor * 0.3); // Fast -> green tint
    
    out.color = color;
    
    // Particle type classification for shader branching
    if (input.mass > 1e20) {
        out.particle_type = 3.0; // Massive objects (stars, planets)
    } else if (input.mass > 1e10) {
        out.particle_type = 2.0; // Medium objects (asteroids)
    } else if (input.mass > 1.0) {
        out.particle_type = 1.0; // Small particles
    } else {
        out.particle_type = 0.0; // Subatomic particles
    }
    
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Get fragment position within point sprite
    let coord = input.clip_position.xy;
    
    // Calculate distance from center for circular particles
    let center = vec2<f32>(0.5, 0.5);
    let dist = length(coord - center);
    
    // Circular falloff for particle shape
    let radius = 0.5;
    let alpha = smoothstep(radius, radius * 0.8, dist);
    
    // Enhanced brightness for different particle types
    var brightness = 1.0;
    if (input.particle_type >= 3.0) {
        brightness = 2.0; // Bright for massive objects
    } else if (input.particle_type >= 2.0) {
        brightness = 1.5; // Medium for medium objects
    } else if (input.particle_type >= 1.0) {
        brightness = 1.0; // Normal for small particles
    } else {
        brightness = 0.7; // Dim for subatomic
    }
    
    // Pulsing effect based on time for dynamic visualization
    let pulse = sin(uniforms.time * 2.0) * 0.1 + 0.9;
    brightness *= pulse;
    
    return vec4<f32>(input.color * brightness, alpha);
} 