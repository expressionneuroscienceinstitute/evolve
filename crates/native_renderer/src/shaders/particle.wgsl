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
    
    // Color based on particle properties
    var color = vec3<f32>(1.0, 1.0, 1.0); // Default white
    
    // Color mode selection
    if (uniforms.color_mode < 0.5) {
        // ParticleType mode
        color = mix(vec3<f32>(0.5, 0.5, 1.0), vec3<f32>(1.0, 0.5, 0.5), input.particle_type / 10.0);
    } else if (uniforms.color_mode < 1.5) {
        // Charge mode
        if (input.charge > 0.0) {
            color = vec3<f32>(1.0, 0.3, 0.3); // Red for positive
        } else if (input.charge < 0.0) {
            color = vec3<f32>(0.3, 0.3, 1.0); // Blue for negative
        } else {
            color = vec3<f32>(0.7, 0.7, 0.7); // Gray for neutral
        }
    } else if (uniforms.color_mode < 2.5) {
        // Temperature mode
        let temp_normalized = min(input.temperature / 10000.0, 1.0);
        color = mix(vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 1.0, 0.0), temp_normalized);
    } else {
        // Velocity mode
        let velocity_mag = length(input.velocity);
        let velocity_factor = min(velocity_mag / 1000.0, 1.0);
        color = mix(vec3<f32>(0.2, 0.2, 0.2), vec3<f32>(0.0, 1.0, 0.0), velocity_factor);
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