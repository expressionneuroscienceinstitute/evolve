// HDR Tonemapping Shader for Scientific Visualization
// Publication-quality color accuracy and dynamic range handling
// Supports multiple tonemapping operators for different scientific use cases

struct HDRUniforms {
    // Tonemapping parameters
    exposure: f32,
    gamma: f32,
    white_point: f32,
    contrast: f32,
    // Operator selection
    tonemap_operator: f32,    // 0=Linear, 1=Reinhard, 2=ACES, 3=Uncharted2, 4=Scientific
    color_space: f32,         // 0=sRGB, 1=Rec2020, 2=P3, 3=Scientific
    adaptation_factor: f32,
    bloom_strength: f32,
    // Scientific color accuracy
    preserve_scientific_data: f32,  // 0=false, 1=true
    color_temperature: f32,
    tint: f32,
    saturation: f32,
    // Advanced features
    local_adaptation: f32,
    highlight_recovery: f32,
    shadow_recovery: f32,
    _padding: f32,
}

@group(0) @binding(0)
var<uniform> uniforms: HDRUniforms;

@group(0) @binding(1)
var hdr_texture: texture_2d<f32>;

@group(0) @binding(2)
var hdr_sampler: sampler;

@group(0) @binding(3)
var bloom_texture: texture_2d<f32>;

@group(0) @binding(4)
var adaptation_texture: texture_2d<f32>;  // For local adaptation

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(input.position, 0.0, 1.0);
    out.uv = input.uv;
    return out;
}

// Linear tonemapping (preserves scientific accuracy)
fn tonemap_linear(color: vec3<f32>) -> vec3<f32> {
    return color * uniforms.exposure;
}

// Reinhard tonemapping (good for general scientific visualization)
fn tonemap_reinhard(color: vec3<f32>) -> vec3<f32> {
    let exposed = color * uniforms.exposure;
    return exposed / (1.0 + exposed);
}

// ACES tonemapping (cinematic, good for presentation)
fn tonemap_aces(color: vec3<f32>) -> vec3<f32> {
    let exposed = color * uniforms.exposure;
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((exposed * (a * exposed + b)) / (exposed * (c * exposed + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// Uncharted 2 tonemapping (good dynamic range)
fn tonemap_uncharted2(color: vec3<f32>) -> vec3<f32> {
    let A = 0.15;
    let B = 0.50;
    let C = 0.10;
    let D = 0.20;
    let E = 0.02;
    let F = 0.30;
    
    let x = color * uniforms.exposure;
    let numerator = x * (A * x + C * B) + D * E;
    let denominator = x * (A * x + B) + D * F;
    return numerator / denominator - E / F;
}

// Scientific tonemapping (preserves data relationships)
fn tonemap_scientific(color: vec3<f32>) -> vec3<f32> {
    let exposed = color * uniforms.exposure;
    
    // Preserve scientific data relationships
    if (uniforms.preserve_scientific_data > 0.5) {
        // Use log mapping to preserve dynamic range while maintaining data accuracy
        let log_color = log(exposed + 1.0) / log(uniforms.white_point + 1.0);
        return log_color;
    } else {
        // Standard scientific visualization with gamma correction
        return pow(exposed / uniforms.white_point, vec3<f32>(1.0 / uniforms.gamma));
    }
}

// Color space conversion functions
fn srgb_to_linear(color: vec3<f32>) -> vec3<f32> {
    return select(
        pow((color + 0.055) / 1.055, vec3<f32>(2.4)),
        color / 12.92,
        color <= vec3<f32>(0.04045)
    );
}

fn linear_to_srgb(color: vec3<f32>) -> vec3<f32> {
    return select(
        1.055 * pow(color, vec3<f32>(1.0 / 2.4)) - 0.055,
        12.92 * color,
        color <= vec3<f32>(0.0031308)
    );
}

// Rec2020 color space (wider gamut for scientific data)
fn linear_to_rec2020(color: vec3<f32>) -> vec3<f32> {
    // Simplified conversion matrix for Rec2020
    let r = color.r * 1.7166511 - color.g * 0.3556708 - color.b * 0.2533663;
    let g = color.r * -0.6666844 + color.g * 1.6164812 + color.b * 0.0157685;
    let b = color.r * 0.0176399 - color.g * 0.0427706 + color.b * 0.9421031;
    return vec3<f32>(r, g, b);
}

// P3 color space (good for displays)
fn linear_to_p3(color: vec3<f32>) -> vec3<f32> {
    // Simplified conversion matrix for Display P3
    let r = color.r * 1.2249401 - color.g * 0.2249401 + color.b * 0.0000000;
    let g = color.r * -0.0420569 + color.g * 1.0420569 + color.b * 0.0000000;
    let b = color.r * -0.0196376 - color.g * 0.0786361 + color.b * 1.0982735;
    return vec3<f32>(r, g, b);
}

// Color temperature adjustment for scientific accuracy
fn adjust_color_temperature(color: vec3<f32>, temperature: f32) -> vec3<f32> {
    // Simplified color temperature adjustment
    let temp_scale = temperature / 6500.0; // Normalize to daylight
    
    var adjusted: vec3<f32>;
    if (temp_scale > 1.0) {
        // Warmer (more red)
        adjusted.r = color.r;
        adjusted.g = color.g * (1.0 / temp_scale);
        adjusted.b = color.b * (1.0 / (temp_scale * temp_scale));
    } else {
        // Cooler (more blue)
        adjusted.r = color.r * temp_scale;
        adjusted.g = color.g * sqrt(temp_scale);
        adjusted.b = color.b;
    }
    
    return adjusted;
}

// Local adaptation for enhanced detail in scientific data
fn apply_local_adaptation(uv: vec2<f32>, color: vec3<f32>) -> vec3<f32> {
    if (uniforms.local_adaptation < 0.1) {
        return color;
    }
    
    let adaptation = textureSample(adaptation_texture, hdr_sampler, uv).r;
    let adapted_exposure = uniforms.exposure * (1.0 + uniforms.local_adaptation * adaptation);
    
    return color * adapted_exposure;
}

// Bloom contribution for high-energy phenomena
fn add_bloom(uv: vec2<f32>, base_color: vec3<f32>) -> vec3<f32> {
    if (uniforms.bloom_strength < 0.1) {
        return base_color;
    }
    
    let bloom = textureSample(bloom_texture, hdr_sampler, uv).rgb;
    return base_color + bloom * uniforms.bloom_strength;
}

// Highlight and shadow recovery for scientific detail preservation
fn recover_details(color: vec3<f32>) -> vec3<f32> {
    let luminance = dot(color, vec3<f32>(0.299, 0.587, 0.114));
    
    // Highlight recovery
    let highlight_factor = 1.0 - smoothstep(0.8, 1.0, luminance);
    let recovered_highlights = color * (1.0 + uniforms.highlight_recovery * highlight_factor);
    
    // Shadow recovery
    let shadow_factor = 1.0 - smoothstep(0.0, 0.2, luminance);
    let recovered_shadows = recovered_highlights * (1.0 + uniforms.shadow_recovery * shadow_factor);
    
    return recovered_shadows;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    // Sample HDR color
    var hdr_color = textureSample(hdr_texture, hdr_sampler, input.uv).rgb;
    
    // Apply local adaptation
    hdr_color = apply_local_adaptation(input.uv, hdr_color);
    
    // Add bloom if enabled
    hdr_color = add_bloom(input.uv, hdr_color);
    
    // Apply tonemapping operator based on selection
    var tonemapped: vec3<f32>;
    if (uniforms.tonemap_operator < 0.5) {
        tonemapped = tonemap_linear(hdr_color);
    } else if (uniforms.tonemap_operator < 1.5) {
        tonemapped = tonemap_reinhard(hdr_color);
    } else if (uniforms.tonemap_operator < 2.5) {
        tonemapped = tonemap_aces(hdr_color);
    } else if (uniforms.tonemap_operator < 3.5) {
        tonemapped = tonemap_uncharted2(hdr_color);
    } else {
        tonemapped = tonemap_scientific(hdr_color);
    }
    
    // Apply color temperature adjustment
    tonemapped = adjust_color_temperature(tonemapped, uniforms.color_temperature);
    
    // Apply contrast and saturation
    let contrast_adjusted = pow(tonemapped, vec3<f32>(uniforms.contrast));
    let gray = vec3<f32>(dot(contrast_adjusted, vec3<f32>(0.299, 0.587, 0.114)));
    let saturated = mix(gray, contrast_adjusted, uniforms.saturation);
    
    // Apply highlight/shadow recovery for scientific detail
    let detailed = recover_details(saturated);
    
    // Convert to target color space
    var final_color: vec3<f32>;
    if (uniforms.color_space < 0.5) {
        final_color = linear_to_srgb(detailed);
    } else if (uniforms.color_space < 1.5) {
        final_color = linear_to_rec2020(detailed);
    } else if (uniforms.color_space < 2.5) {
        final_color = linear_to_p3(detailed);
    } else {
        // Scientific color space (linear, preserves data)
        final_color = detailed;
    }
    
    // Apply gamma correction if not in scientific mode
    if (uniforms.preserve_scientific_data < 0.5) {
        final_color = pow(final_color, vec3<f32>(1.0 / uniforms.gamma));
    }
    
    // Apply tint adjustment
    final_color.r = final_color.r * (1.0 + uniforms.tint);
    final_color.g = final_color.g * (1.0 - abs(uniforms.tint));
    final_color.b = final_color.b * (1.0 - uniforms.tint);
    
    return vec4<f32>(clamp(final_color, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
} 