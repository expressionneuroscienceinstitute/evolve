// Mac-Optimized Particle Shader for Universe Simulation
// Enhanced for Retina displays, Metal backend, and Mac GPU performance

struct Uniforms {
    view_proj: mat4x4<f32>,
    time: f32,
    scale: f32,
    color_mode: f32,
    filter_threshold: f32,
    // Mac-specific uniforms
    retina_scale: f32,           // Retina display scale factor (1.0, 2.0, etc.)
    mac_gpu_type: f32,           // 0=Intel, 1=Apple Silicon, 2=Discrete
    performance_mode: f32,       // 0=Quality, 1=Balanced, 2=Performance
    max_particles: f32,          // Maximum particles for LOD
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VertexInput {
    // Particle data (per instance)
    @location(1) position: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) size: f32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) world_pos: vec3<f32>,
    @location(2) size: f32,
    @location(3) quad_pos: vec2<f32>, // [-1, 1] for the quad
    @location(4) retina_scale: f32,   // Pass retina scale to fragment shader
};

// Mac-specific constants for optimization
const MAC_APPLE_SILICON: f32 = 1.0;
const MAC_INTEL: f32 = 0.0;
const MAC_DISCRETE: f32 = 2.0;

// Performance mode constants
const QUALITY_MODE: f32 = 0.0;
const BALANCED_MODE: f32 = 1.0;
const PERFORMANCE_MODE: f32 = 2.0;

// Mac-optimized particle size calculation
fn calculate_mac_particle_size(base_size: f32, depth: f32, retina_scale: f32) -> f32 {
    // Base size with minimum threshold
    var size = max(base_size, 0.003); // Smaller minimum for Retina displays
    
    // Apply Retina scaling for crisp rendering
    size = size * retina_scale;
    
    // Depth-based LOD for performance
    let depth_factor = clamp(depth / 100.0, 0.1, 1.0);
    size = size * depth_factor;
    
    // Performance mode adjustments
    if (uniforms.performance_mode == PERFORMANCE_MODE) {
        size = size * 0.8; // Smaller particles for performance
    } else if (uniforms.performance_mode == QUALITY_MODE) {
        size = size * 1.2; // Larger particles for quality
    }
    
    // GPU-specific optimizations
    if (uniforms.mac_gpu_type == MAC_APPLE_SILICON) {
        // Apple Silicon can handle more complex rendering
        size = size * 1.1;
    } else if (uniforms.mac_gpu_type == MAC_INTEL) {
        // Intel GPUs need more conservative sizing
        size = size * 0.9;
    }
    
    return clamp(size, 0.001, 0.1); // Clamp to reasonable range
}

@vertex
fn vs_main(
    instance: VertexInput,
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;

    // Optimized quad vertex generation for Metal
    var quad_pos: vec2<f32>;
    switch(vertex_index) {
        case 0u: { quad_pos = vec2<f32>(-1.0, -1.0); }
        case 1u: { quad_pos = vec2<f32>( 1.0, -1.0); }
        case 2u: { quad_pos = vec2<f32>(-1.0,  1.0); }
        case 3u: { quad_pos = vec2<f32>(-1.0,  1.0); }
        case 4u: { quad_pos = vec2<f32>( 1.0, -1.0); }
        default: { quad_pos = vec2<f32>( 1.0,  1.0); }
    }

    // Transform position to clip space
    var clip_pos = uniforms.view_proj * vec4<f32>(instance.position, 1.0);
    
    // Mac-optimized particle sizing
    let depth = length(instance.position);
    let particle_size = calculate_mac_particle_size(instance.size, depth, uniforms.retina_scale);
    
    // Retina-aware billboarding
    let depth_scale = max(0.001, clip_pos.w);
    let offset = quad_pos * particle_size * 0.015 / depth_scale; // Adjusted for Retina
    
    // Apply offset in clip space
    clip_pos.x = clip_pos.x + offset.x;
    clip_pos.y = clip_pos.y + offset.y;
    
    out.clip_position = clip_pos;
    out.color = instance.color;
    out.world_pos = instance.position;
    out.size = particle_size;
    out.quad_pos = quad_pos;
    out.retina_scale = uniforms.retina_scale;
    
    return out;
}

// Mac-optimized lighting calculation
fn calculate_mac_lighting(normal: vec3<f32>, view_dir: vec3<f32>, light_dir: vec3<f32>) -> f32 {
    // Optimized lighting for Mac GPUs
    let diffuse = max(dot(normal, light_dir), 0.0);
    
    // Simplified specular for performance
    let reflect_dir = reflect(-light_dir, normal);
    let specular = pow(max(dot(view_dir, reflect_dir), 0.0), 16.0); // Reduced exponent for performance
    
    return diffuse * 0.7 + specular * 0.3;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Retina-aware circle calculation
    let dist = length(input.quad_pos);
    if (dist > 1.0) {
        discard;
    }
    
    // Calculate normal for lighting
    let normal_z = sqrt(1.0 - dist * dist);
    let normal = vec3<f32>(input.quad_pos.x, input.quad_pos.y, normal_z);
    
    // Mac-optimized lighting
    let light_dir = normalize(vec3<f32>(0.8, 0.8, 0.8));
    let view_dir = normalize(input.world_pos);
    let lighting = calculate_mac_lighting(normal, view_dir, light_dir);
    
    // Ambient lighting
    let ambient = 0.25;
    let total_light = ambient + lighting;
    
    // Apply lighting to color
    let final_color = input.color * total_light;
    
    // Retina-aware rim lighting
    let rim_dot = 1.0 - dot(normal, vec3<f32>(0.0, 0.0, 1.0));
    let rim_intensity = smoothstep(0.6, 1.0, rim_dot) * 0.4; // Adjusted for Retina
    
    let final_lit_color = final_color + vec3<f32>(1.0, 1.0, 1.0) * rim_intensity;
    
    // Performance mode alpha adjustments
    var alpha = 1.0;
    if (uniforms.performance_mode == PERFORMANCE_MODE) {
        alpha = 0.95; // Slightly transparent for performance
    }
    
    return vec4<f32>(final_lit_color, alpha);
} 