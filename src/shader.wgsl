// Runs on the GPU. Two small programs: one per vertex, one per pixel.

struct Uniforms {
    mvp: mat4x4<f32>,   // globe coordinates -> screen, rebuilt every frame from the camera
    light: vec4<f32>,   // xyz = direction to the light (globe coordinates), w = 1 to shade, 0 for flat colors
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexIn {
    @location(0) pos: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) world: vec3<f32>,
};

// Per vertex: where on the screen does this 3D point land?
@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    var out: VertexOut;
    out.clip = u.mvp * vec4<f32>(v.pos, 1.0);
    out.color = v.color;
    out.world = v.pos;
    return out;
}

// Per pixel: what color is it?
@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // The slope of this tiny patch of surface: from how the position changes between neighbouring pixels.
    let normal = normalize(cross(dpdx(in.world), dpdy(in.world)));
    // Which way is "facing us" depends on winding; flip so lit side faces the outside of the globe.
    let outward = select(-normal, normal, dot(normal, in.world) > 0.0);
    let shade = 0.45 + 0.55 * max(dot(outward, normalize(u.light.xyz)), 0.0);
    let lit = in.color * shade;
    return vec4<f32>(mix(in.color, lit, u.light.w), 1.0);
}
