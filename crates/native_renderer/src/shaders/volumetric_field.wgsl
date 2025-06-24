// Volumetric Field Visualization Shader
// Advanced 3D field rendering using ray-casting and volumetric techniques
// Based on 2025 WebGPU best practices for scientific visualization

struct Uniforms {
    view_proj: mat4x4<f32>,
    camera_pos: vec3<f32>,
    time: f32,
    field_density: f32,
    iso_value: f32,
    step_size: f32,
    max_steps: f32,
    // HDR parameters
    exposure: f32,
    gamma: f32,
    hdr_scale: f32,
    // Volumetric parameters
    absorption_coefficient: f32,
    scattering_coefficient: f32,
    phase_function_g: f32,
    light_direction: vec3<f32>,
    light_intensity: f32,
    // Advanced visualization modes
    visualization_mode: f32,    // 0=density, 1=temperature, 2=velocity, 3=pressure
    color_map_mode: f32,        // 0=scientific, 1=artistic, 2=medical
    transfer_function_alpha: f32,
    _padding: [f32; 2],
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var field_texture: texture_3d<f32>;

@group(0) @binding(2)
var field_sampler: sampler;

@group(0) @binding(3)
var transfer_function_texture: texture_1d<f32>;

@group(0) @binding(4)
var transfer_function_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) ray_direction: vec3<f32>,
    @location(2) camera_position: vec3<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    
    // Transform to screen space
    out.clip_position = uniforms.view_proj * vec4<f32>(input.position, 1.0);
    out.uv = input.uv;
    
    // Calculate ray direction for ray-casting
    let world_pos = input.position;
    out.ray_direction = normalize(world_pos - uniforms.camera_pos);
    out.camera_position = uniforms.camera_pos;
    
    return out;
}

// Scientific color mapping functions
fn get_scientific_color(value: f32, mode: f32) -> vec3<f32> {
    if (mode < 0.5) {
        // Density visualization (blue-white-red)
        if (value < 0.33) {
            let t = value * 3.0;
            return vec3<f32>(0.0, t, 1.0);
        } else if (value < 0.67) {
            let t = (value - 0.33) * 3.0;
            return vec3<f32>(t, 1.0, 1.0 - t);
        } else {
            let t = (value - 0.67) * 3.0;
            return vec3<f32>(1.0, 1.0 - t, 0.0);
        }
    } else if (mode < 1.5) {
        // Temperature visualization (blackbody radiation)
        return blackbody_color(value * 10000.0); // Scale to temperature range
    } else if (mode < 2.5) {
        // Velocity visualization (Doppler shift)
        return doppler_color(value);
    } else {
        // Pressure visualization (green-yellow-red)
        if (value < 0.5) {
            let t = value * 2.0;
            return vec3<f32>(t, 1.0, 0.0);
        } else {
            let t = (value - 0.5) * 2.0;
            return vec3<f32>(1.0, 1.0 - t, 0.0);
        }
    }
}

fn blackbody_color(temperature: f32) -> vec3<f32> {
    // Simplified blackbody radiation color calculation
    let temp = clamp(temperature, 1000.0, 40000.0);
    
    var r: f32;
    var g: f32;
    var b: f32;
    
    // Red component
    if (temp < 6600.0) {
        r = 1.0;
    } else {
        r = 1.292936 * pow(temp / 100.0 - 60.0, -0.1332047);
        r = clamp(r, 0.0, 1.0);
    }
    
    // Green component
    if (temp < 6600.0) {
        g = -0.114876 + 0.4944 * log(temp / 100.0) - 0.1 * pow(log(temp / 100.0), 2.0);
    } else {
        g = 1.29293 * pow(temp / 100.0 - 60.0, -0.0755148);
    }
    g = clamp(g, 0.0, 1.0);
    
    // Blue component
    if (temp > 6600.0) {
        b = 1.0;
    } else if (temp < 1900.0) {
        b = 0.0;
    } else {
        b = 0.543206 * log(temp / 100.0 - 10.0) - 1.19625;
        b = clamp(b, 0.0, 1.0);
    }
    
    return vec3<f32>(r, g, b);
}

fn doppler_color(velocity_factor: f32) -> vec3<f32> {
    // Doppler shift visualization: blue for approach, red for recession
    let beta = velocity_factor * 2.0 - 1.0; // Map [0,1] to [-1,1]
    
    if (beta > 0.0) {
        // Red shift (receding)
        let intensity = beta;
        return vec3<f32>(1.0, 1.0 - intensity, 1.0 - intensity);
    } else {
        // Blue shift (approaching)
        let intensity = -beta;
        return vec3<f32>(1.0 - intensity, 1.0 - intensity, 1.0);
    }
}

// Advanced transfer function with scientific accuracy
fn apply_transfer_function(value: f32) -> vec4<f32> {
    let color = textureSample(transfer_function_texture, transfer_function_sampler, value);
    
    // Apply scientific color mapping
    let scientific_color = get_scientific_color(value, uniforms.color_map_mode);
    
    // Blend transfer function with scientific mapping
    let final_color = mix(color.rgb, scientific_color, 0.5);
    
    // Alpha based on value and transfer function
    let alpha = color.a * uniforms.transfer_function_alpha * smoothstep(0.01, 0.99, value);
    
    return vec4<f32>(final_color, alpha);
}

// Volumetric ray-casting implementation
fn ray_cast_volume(ray_start: vec3<f32>, ray_dir: vec3<f32>) -> vec4<f32> {
    var color = vec3<f32>(0.0);
    var alpha = 0.0;
    var position = ray_start;
    
    let step_vector = ray_dir * uniforms.step_size;
    let max_iterations = i32(uniforms.max_steps);
    
    for (var i = 0; i < max_iterations; i = i + 1) {
        // Check if we're still inside the volume
        if (any(position < vec3<f32>(0.0)) || any(position > vec3<f32>(1.0))) {
            break;
        }
        
        // Sample the 3D field
        let field_value = textureSample(field_texture, field_sampler, position).r;
        
        // Apply transfer function
        let sample_color = apply_transfer_function(field_value);
        
        // Volumetric lighting calculation
        let lighting = calculate_volumetric_lighting(position, uniforms.light_direction);
        
        // Accumulate color and alpha (front-to-back blending)
        let attenuated_color = sample_color.rgb * sample_color.a * lighting;
        color = color + attenuated_color * (1.0 - alpha);
        alpha = alpha + sample_color.a * (1.0 - alpha);
        
        // Early ray termination for opaque regions
        if (alpha > 0.99) {
            break;
        }
        
        position = position + step_vector;
    }
    
    return vec4<f32>(color, alpha);
}

fn calculate_volumetric_lighting(position: vec3<f32>, light_dir: vec3<f32>) -> f32 {
    // Sample gradient for normal calculation
    let eps = 0.01;
    let gradient_x = textureSample(field_texture, field_sampler, position + vec3<f32>(eps, 0.0, 0.0)).r -
                     textureSample(field_texture, field_sampler, position - vec3<f32>(eps, 0.0, 0.0)).r;
    let gradient_y = textureSample(field_texture, field_sampler, position + vec3<f32>(0.0, eps, 0.0)).r -
                     textureSample(field_texture, field_sampler, position - vec3<f32>(0.0, eps, 0.0)).r;
    let gradient_z = textureSample(field_texture, field_sampler, position + vec3<f32>(0.0, 0.0, eps)).r -
                     textureSample(field_texture, field_sampler, position - vec3<f32>(0.0, 0.0, eps)).r;
    
    let normal = normalize(vec3<f32>(gradient_x, gradient_y, gradient_z));
    
    // Lambertian diffuse lighting
    let diffuse = max(dot(normal, normalize(light_dir)), 0.0);
    
    // Add ambient lighting
    let ambient = 0.2;
    
    return ambient + diffuse * uniforms.light_intensity;
}

// HDR tone mapping with scientific accuracy
fn tone_map_hdr(color: vec3<f32>) -> vec3<f32> {
    // Apply exposure
    let exposed = color * uniforms.exposure;
    
    // Reinhard tone mapping for scientific visualization
    let mapped = exposed / (exposed + vec3<f32>(1.0));
    
    // Apply gamma correction
    return pow(mapped, vec3<f32>(1.0 / uniforms.gamma));
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Calculate ray entry and exit points
    let ray_start = input.camera_position;
    let ray_dir = normalize(input.ray_direction);
    
    // Perform volumetric ray-casting
    let volume_color = ray_cast_volume(ray_start, ray_dir);
    
    // Apply HDR tone mapping for scientific accuracy
    let tone_mapped = tone_map_hdr(volume_color.rgb * uniforms.hdr_scale);
    
    // Return final color with scientific HDR support
    return vec4<f32>(tone_mapped, volume_color.a);
} 