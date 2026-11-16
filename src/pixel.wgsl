// Draws the pixel layer: flat colored quads lying on the globe, blended over the hexes.
struct Uniforms {
    mvp: mat4x4<f32>,
    light: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexIn {
    @location(0) pos: vec3<f32>,
    @location(1) color: vec4<f32>,   // rgb + opacity
    @location(2) vis: f32,           // < 0 means behind the horizon
};

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) vis: f32,
};

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.clip = u.mvp * vec4<f32>(v.pos, 1.0);
    out.color = v.color;
    out.vis = v.vis;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    if in.vis < 0.0 {
        discard;
    }
    return in.color;
}
