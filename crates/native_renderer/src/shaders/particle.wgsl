// High-Performance Particle Shader for Universe Simulation
// Renders particles as 3D-lit spheres for debugging.

struct Uniforms {
    view_proj: mat4x4<f32>,
    time: f32,
    // These are no longer used but kept for struct compatibility
    scale: f32,
    color_mode: f32,
    filter_threshold: f32,
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
};

@vertex
fn vs_main(
    instance: VertexInput,
    @builtin(vertex_index) vertex_index: u32,
) -> VertexOutput {
    var out: VertexOutput;

    var quad_pos: vec2<f32>;
    switch(vertex_index) {
        case 0u: { quad_pos = vec2<f32>(-1.0, -1.0); }
        case 1u: { quad_pos = vec2<f32>( 1.0, -1.0); }
        case 2u: { quad_pos = vec2<f32>(-1.0,  1.0); }
        case 3u: { quad_pos = vec2<f32>(-1.0,  1.0); }
        case 4u: { quad_pos = vec2<f32>( 1.0, -1.0); }
        default: { quad_pos = vec2<f32>( 1.0,  1.0); } // vertex_index == 5u
    }

    // Apply a minimum size to prevent particles from disappearing - scale based on zoom
    let particle_size = max(instance.size, 0.005); // Smaller minimum for better scaling
    
    var clip_pos = uniforms.view_proj * vec4<f32>(instance.position, 1.0);
    
    // Billboarding in clip space with improved scaling for microscope mode
    // Scale the offset to be proportional to the viewport but respect depth
    // Use perspective-correct scaling based on depth
    let depth_scale = max(0.001, clip_pos.w);
    let offset = quad_pos * particle_size * 0.02 / depth_scale; // Enhanced scaling for better visibility
    clip_pos.x = clip_pos.x + offset.x;
    clip_pos.y = clip_pos.y + offset.y;
    
    out.clip_position = clip_pos;
    out.color = instance.color;
    out.world_pos = instance.position;
    out.size = particle_size;
    out.quad_pos = quad_pos;
    
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Use the quad_pos (from -1 to 1) to form a circle
    let dist = length(input.quad_pos);
    if (dist > 1.0) {
        discard;
    }
    
    // Calculate normal for lighting to make it look 3D
    let normal_z = sqrt(1.0 - dist * dist);
    let normal = vec3<f32>(input.quad_pos.x, input.quad_pos.y, normal_z);
    
    // Simple directional lighting
    let light_dir = normalize(vec3<f32>(0.8, 0.8, 0.8));
    let diffuse_intensity = max(dot(normal, light_dir), 0.0);
    
    // Add ambient light to avoid completely black areas
    let ambient_intensity = 0.3;
    let total_light = ambient_intensity + diffuse_intensity * 0.7;
    
    let final_color = input.color * total_light;
    
    // Add a specular highlight for a more "sphere-like" look
    let view_dir = normalize(input.world_pos); // Simplification
    let reflect_dir = reflect(-light_dir, normal);
    var specular_intensity = pow(max(dot(view_dir, reflect_dir), 0.0), 32.0);
    
    // Rim lighting to make the edges pop
    let rim_dot = 1.0 - dot(normal, vec3<f32>(0.0, 0.0, 1.0));
    let rim_intensity = smoothstep(0.5, 1.0, rim_dot) * 0.5;
    
    let final_lit_color = final_color + vec3<f32>(1.0, 1.0, 1.0) * specular_intensity * 0.6 + rim_intensity;
    
    return vec4<f32>(final_lit_color, 1.0);
} 