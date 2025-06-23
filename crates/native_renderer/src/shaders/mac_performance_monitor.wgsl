// Mac Performance Monitoring Shader
// Provides real-time performance metrics for Mac GPU optimization

struct PerformanceUniforms {
    frame_time: f32,
    gpu_utilization: f32,
    memory_usage: f32,
    particle_count: f32,
    mac_gpu_type: f32,
    retina_scale: f32,
    performance_mode: f32,
    target_fps: f32,
    current_fps: f32,
    buffer_pool_stats: vec4<f32>, // active, retired, active_memory, total_allocated
};

@group(0) @binding(0)
var<uniform> perf_uniforms: PerformanceUniforms;

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(input.position, 0.0, 1.0);
    out.uv = input.uv;
    return out;
}

// Mac-specific performance color coding
fn get_performance_color(metric: f32, threshold_low: f32, threshold_high: f32) -> vec3<f32> {
    if (metric < threshold_low) {
        return vec3<f32>(0.0, 1.0, 0.0); // Green - Good
    } else if (metric < threshold_high) {
        return vec3<f32>(1.0, 1.0, 0.0); // Yellow - Warning
    } else {
        return vec3<f32>(1.0, 0.0, 0.0); // Red - Critical
    }
}

// GPU type detection color
fn get_gpu_type_color(gpu_type: f32) -> vec3<f32> {
    if (gpu_type == 1.0) {
        return vec3<f32>(0.0, 0.8, 1.0); // Apple Silicon - Blue
    } else if (gpu_type == 0.0) {
        return vec3<f32>(1.0, 0.5, 0.0); // Intel - Orange
    } else {
        return vec3<f32>(0.8, 0.0, 1.0); // Discrete - Purple
    }
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let uv = input.uv;
    
    // Create performance visualization based on UV coordinates
    var color = vec3<f32>(0.1, 0.1, 0.1); // Dark background
    
    // Frame time indicator (top left)
    if (uv.x < 0.3 && uv.y > 0.7) {
        let frame_time_norm = clamp(perf_uniforms.frame_time / 16.67, 0.0, 2.0); // Normalize to 60fps
        color = get_performance_color(frame_time_norm, 0.8, 1.2);
    }
    
    // GPU utilization indicator (top right)
    if (uv.x > 0.7 && uv.y > 0.7) {
        color = get_performance_color(perf_uniforms.gpu_utilization, 0.7, 0.9);
    }
    
    // Memory usage indicator (bottom left)
    if (uv.x < 0.3 && uv.y < 0.3) {
        let memory_norm = clamp(perf_uniforms.memory_usage / 1024.0, 0.0, 1.0); // Normalize to 1GB
        color = get_performance_color(memory_norm, 0.6, 0.8);
    }
    
    // Particle count indicator (bottom right)
    if (uv.x > 0.7 && uv.y < 0.3) {
        let particle_norm = clamp(perf_uniforms.particle_count / 1000000.0, 0.0, 1.0); // Normalize to 1M
        color = get_performance_color(particle_norm, 0.7, 0.9);
    }
    
    // FPS indicator (center)
    if (uv.x > 0.4 && uv.x < 0.6 && uv.y > 0.4 && uv.y < 0.6) {
        let fps_ratio = perf_uniforms.current_fps / perf_uniforms.target_fps;
        color = get_performance_color(fps_ratio, 0.8, 0.95);
    }
    
    // GPU type indicator (center top)
    if (uv.x > 0.4 && uv.x < 0.6 && uv.y > 0.8) {
        color = get_gpu_type_color(perf_uniforms.mac_gpu_type);
    }
    
    // Retina scale indicator (center bottom)
    if (uv.x > 0.4 && uv.x < 0.6 && uv.y < 0.2) {
        if (perf_uniforms.retina_scale > 1.0) {
            color = vec3<f32>(0.0, 1.0, 0.5); // Green-cyan for Retina
        } else {
            color = vec3<f32>(0.5, 0.5, 0.5); // Gray for standard
        }
    }
    
    // Performance mode indicator (left center)
    if (uv.x < 0.2 && uv.y > 0.4 && uv.y < 0.6) {
        if (perf_uniforms.performance_mode == 0.0) {
            color = vec3<f32>(1.0, 0.0, 0.0); // Red for Quality
        } else if (perf_uniforms.performance_mode == 1.0) {
            color = vec3<f32>(1.0, 1.0, 0.0); // Yellow for Balanced
        } else {
            color = vec3<f32>(0.0, 1.0, 0.0); // Green for Performance
        }
    }
    
    // Buffer pool stats visualization (right center)
    if (uv.x > 0.8 && uv.y > 0.4 && uv.y < 0.6) {
        let active_ratio = perf_uniforms.buffer_pool_stats.x / 100.0; // Normalize
        color = get_performance_color(active_ratio, 0.5, 0.8);
    }
    
    return vec4<f32>(color, 0.8); // Semi-transparent overlay
} 