// Runs on the GPU. Two small programs: one per vertex, one per pixel.

struct Uniforms {
    mvp: mat4x4<f32>, // model * view * projection, rebuilt every frame from the camera
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexIn {
    @location(0) pos: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec3<f32>,
};

// Per vertex: where on the screen does this 3D point land?
@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.clip = u.mvp * vec4<f32>(v.pos, 1.0);
    out.color = v.color;
    return out;
}

// Per pixel: what color is it? (Flat cells, so the blend between vertices is one color.)
@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
