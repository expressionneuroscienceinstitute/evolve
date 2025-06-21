// GPU Compute Shader for Particle Culling and LOD Selection
// High-performance frustum culling and level-of-detail optimization

struct ParticleData {
    position: vec3<f32>,
    velocity: vec3<f32>,
    mass: f32,
    charge: f32,
    temperature: f32,
    density: f32,
    pressure: f32,
    internal_energy: f32,
    smoothing_length: f32,
    particle_type: f32,
    interaction_count: f32,
    sink_mass: f32,
    accretion_rate: f32,
}

struct CullingUniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    frustum_planes: array<vec4<f32>, 6>, // Near, Far, Left, Right, Top, Bottom
    lod_distances: array<f32, 4>,        // Distance thresholds for LOD levels
    max_particles: u32,
    culling_enabled: u32,
    lod_enabled: u32,
    _padding: array<f32, 2>,
}

@group(0) @binding(0)
var<uniform> uniforms: CullingUniforms;

@group(1) @binding(0)
var<storage, read> input_particles: array<ParticleData>;

@group(1) @binding(1)
var<storage, read_write> output_particles: array<ParticleData>;

@group(1) @binding(2)
var<storage, read_write> visibility_buffer: array<u32>;

@group(1) @binding(3)
var<storage, read_write> lod_buffer: array<u32>;

// Frustum culling test
fn is_in_frustum(position: vec3<f32>) -> bool {
    let pos = vec4<f32>(position, 1.0);
    
    for (var i = 0u; i < 6u; i++) {
        if (dot(uniforms.frustum_planes[i], pos) < 0.0) {
            return false;
        }
    }
    return true;
}

// LOD selection based on distance and particle properties
fn select_lod(position: vec3<f32>, mass: f32, density: f32) -> u32 {
    let distance = length(position - uniforms.camera_pos);
    
    // Base LOD on distance
    var lod = 0u;
    for (var i = 0u; i < 4u; i++) {
        if (distance > uniforms.lod_distances[i]) {
            lod = i + 1u;
        }
    }
    
    // Adjust LOD based on particle importance
    if (mass > 1e20) {
        // Massive objects (stars, planets) - always high detail
        lod = max(0u, lod - 1u);
    } else if (density > 1e-10) {
        // High density particles (SPH) - medium detail
        lod = min(3u, lod + 1u);
    }
    
    return lod;
}

// Distance-based culling for performance
fn should_cull_by_distance(position: vec3<f32>) -> bool {
    let distance = length(position - uniforms.camera_pos);
    let max_view_distance = uniforms.lod_distances[3] * 2.0; // Double the farthest LOD distance
    return distance > max_view_distance;
}

// Size-based culling for very small particles
fn should_cull_by_size(position: vec3<f32>, mass: f32, smoothing_length: f32) -> bool {
    let distance = length(position - uniforms.camera_pos);
    let apparent_size = smoothing_length / distance;
    
    // Cull particles that are too small to see
    return apparent_size < 0.001;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_index = global_id.x;
    
    if (particle_index >= uniforms.max_particles) {
        return;
    }
    
    let particle = input_particles[particle_index];
    var visible = 1u;
    var lod_level = 0u;
    
    // Perform culling tests
    if (uniforms.culling_enabled > 0u) {
        // Frustum culling
        if (!is_in_frustum(particle.position)) {
            visible = 0u;
        }
        
        // Distance-based culling
        if (should_cull_by_distance(particle.position)) {
            visible = 0u;
        }
        
        // Size-based culling
        if (should_cull_by_size(particle.position, particle.mass, particle.smoothing_length)) {
            visible = 0u;
        }
    }
    
    // LOD selection
    if (uniforms.lod_enabled > 0u && visible > 0u) {
        lod_level = select_lod(particle.position, particle.mass, particle.density);
    }
    
    // Write results
    visibility_buffer[particle_index] = visible;
    lod_buffer[particle_index] = lod_level;
    
    // Copy visible particles to output buffer
    if (visible > 0u) {
        output_particles[particle_index] = particle;
    }
}

// Additional compute shader for particle sorting by depth
@compute @workgroup_size(64)
fn sort_particles(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_index = global_id.x;
    
    if (particle_index >= uniforms.max_particles) {
        return;
    }
    
    // Simple depth-based sorting for transparency
    let particle = input_particles[particle_index];
    let depth = dot(uniforms.view_proj[2], vec4<f32>(particle.position, 1.0));
    
    // Store depth for sorting (simplified - full sorting would require multiple passes)
    // This is a placeholder for more sophisticated depth sorting
}

// Compute shader for particle clustering (for LOD optimization)
@compute @workgroup_size(64)
fn cluster_particles(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_index = global_id.x;
    
    if (particle_index >= uniforms.max_particles) {
        return;
    }
    
    // Group nearby particles for LOD optimization
    // This would be used to render particle clusters at lower detail levels
    // Implementation depends on specific clustering algorithm
} 