// The ground, shaded per pixel. The mesh only carries position, a smooth normal, the true height and the
// coarse "range" height; color, rock/snow edges, shore line, light and haze are worked out for every pixel,
// so edges are crisp curves instead of zig-zags along triangles. Keep in step with `ground::shade` (Rust)
// and `terrain::color_in_range`, which do the same thing on the CPU (tests, software preview).

struct Uniforms {
    mvp: mat4x4<f32>,     // (globe position - origin) -> screen
    sun: vec4<f32>,       // xyz: toward the light, globe coordinates; w: focal length in pixels
    eye: vec4<f32>,       // xyz: camera position relative to origin; w: distance to the horizon (haze scale)
    origin: vec4<f32>,    // xyz: the origin point, globe coordinates
    sky: vec4<f32>,       // rgb: sky color; w: haze amount (0 = no haze)
    refm: vec4<f32>,      // xyz: origin in meters, wrapped to 0..4096 (anchors the ground detail); w: meters per globe radius
};
@group(0) @binding(0) var<uniform> u: Uniforms;

struct VertexIn {
    @location(0) pos: vec3<f32>,   // relative to the patch center
    @location(1) nrm: vec4<f32>,   // xyz: shading normal; w: steepness * 1.6
    @location(2) hr: vec2<f32>,    // x: height (m, negative = sea floor); y: range height (m)
    @location(3) inst: vec4<f32>,  // per patch: xyz its center relative to the origin; w its width (radians)
    @location(4) morph: f32,       // radial move onto the parent patch's half-resolution mesh
    @location(5) nrm_p: vec4<f32>, // the normal that mesh has there
};

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) nrm: vec3<f32>,
    @location(1) steep: f32,
    @location(2) hr: vec2<f32>,
    @location(3) rel: vec3<f32>,
};

@vertex
fn vs_main(v: VertexIn) -> VertexOut {
    var out: VertexOut;
    var p = v.pos + v.inst.xyz;
    // Geomorphing: a patch is split when it grows past the limit `lim` below, so its children start at about
    // half of it (0.5). The slide happens only between 0.54 and 0.66 of it (the far end of the patch's
    // range, like CDLOD's last 30%): a patch is its sharp self for the rest of its life. Between those two the points slide from the parent's coarser shape to their own, so a level change
    // never pops (the same blend in reverse when moving away).
    let dist = max(distance(u.eye.xyz, p), 1e-9);
    let px = u.sun.w * v.inst.w / dist;
    // The split limit grows with distance (ground::px_limit); children are born at half of it.
    let gain = select(0.0, 1.0, u.sky.w > 0.5); // inside only (sky.w = haze flag)
    let lim = select(56.0, 112.0, u.sky.w > 0.5) * (1.0 + clamp((dist - 10000.0 / 6371000.0) / (10000.0 / 6371000.0), 0.0, gain));
    let k = 1.0 - smooth_(0.54 * lim, 0.66 * lim, px);
    p = p + normalize(u.origin.xyz + p) * (v.morph * k);
    out.clip = u.mvp * vec4<f32>(p, 1.0);
    out.nrm = normalize(mix(v.nrm.xyz, v.nrm_p.xyz, k));
    out.steep = v.nrm.w / 1.6;
    out.hr = v.hr;
    out.rel = p;
    return out;
}

fn smooth_(a: f32, b: f32, x: f32) -> f32 {
    let t = clamp((x - a) / (b - a), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

fn seg(a: vec3<f32>, b: vec3<f32>, x0: f32, x1: f32, x: f32) -> vec3<f32> {
    return mix(a, b, clamp((x - x0) / (x1 - x0), 0.0, 1.0));
}

// Water by depth in meters.
fn water(d: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.05, 0.40, 0.52);
    let c1 = vec3<f32>(0.03, 0.26, 0.48);
    let c2 = vec3<f32>(0.015, 0.14, 0.36);
    let c3 = vec3<f32>(0.008, 0.07, 0.24);
    let c4 = vec3<f32>(0.004, 0.03, 0.14);
    let c5 = vec3<f32>(0.002, 0.01, 0.07);
    if d <= 0.0 { return c0; }
    if d <= 30.0 { return seg(c0, c1, 0.0, 30.0, d); }
    if d <= 200.0 { return seg(c1, c2, 30.0, 200.0, d); }
    if d <= 1000.0 { return seg(c2, c3, 200.0, 1000.0, d); }
    if d <= 4000.0 { return seg(c3, c4, 1000.0, 4000.0, d); }
    if d <= 11000.0 { return seg(c4, c5, 4000.0, 11000.0, d); }
    return c5;
}

// Land by height in meters (sand, grass, forest, rock).
fn land(h: f32) -> vec3<f32> {
    let c0 = vec3<f32>(0.62, 0.50, 0.26);
    let c1 = vec3<f32>(0.58, 0.47, 0.24);
    let c2 = vec3<f32>(0.22, 0.42, 0.08);
    let c3 = vec3<f32>(0.15, 0.34, 0.06);
    let c4 = vec3<f32>(0.07, 0.22, 0.06);
    let c5 = vec3<f32>(0.18, 0.27, 0.09);
    let c6 = vec3<f32>(0.33, 0.30, 0.25);
    let c7 = vec3<f32>(0.42, 0.38, 0.33);
    if h <= 0.0 { return c0; }
    if h <= 6.0 { return seg(c0, c1, 0.0, 6.0, h); }
    if h <= 40.0 { return seg(c1, c2, 6.0, 40.0, h); }
    if h <= 500.0 { return seg(c2, c3, 40.0, 500.0, h); }
    if h <= 1200.0 { return seg(c3, c4, 500.0, 1200.0, h); }
    if h <= 2000.0 { return seg(c4, c5, 1200.0, 2000.0, h); }
    if h <= 2600.0 { return seg(c5, c6, 2000.0, 2600.0, h); }
    if h <= 3200.0 { return seg(c6, c7, 2600.0, 3200.0, h); }
    return c7;
}

// ---- ground detail: noise anchored to the world (in meters), so it does not swim when the camera moves ----
fn wrap(c: vec3<f32>) -> vec3<f32> {
    return c - 4096.0 * floor(c / 4096.0);
}

fn hash3(c: vec3<f32>) -> f32 {
    var q = fract(wrap(c) * 0.1031);
    q = q + dot(q, q.yzx + 33.33);
    return fract((q.x + q.y) * q.z);
}

fn vnoise(p: vec3<f32>) -> f32 {
    let c = floor(p);
    let f = fract(p);
    let w = f * f * (3.0 - 2.0 * f);
    let x00 = mix(hash3(c), hash3(c + vec3<f32>(1.0, 0.0, 0.0)), w.x);
    let x10 = mix(hash3(c + vec3<f32>(0.0, 1.0, 0.0)), hash3(c + vec3<f32>(1.0, 1.0, 0.0)), w.x);
    let x01 = mix(hash3(c + vec3<f32>(0.0, 0.0, 1.0)), hash3(c + vec3<f32>(1.0, 0.0, 1.0)), w.x);
    let x11 = mix(hash3(c + vec3<f32>(0.0, 1.0, 1.0)), hash3(c + vec3<f32>(1.0, 1.0, 1.0)), w.x);
    return mix(mix(x00, x10, w.y), mix(x01, x11, w.y), w.z);
}

// Small-scale relief used only for the bump of the lighting: two scales.
fn relief(p: vec3<f32>) -> f32 {
    return vnoise(p * 1.1) * 0.6 + vnoise(p * 0.3) * 0.9 + vnoise(p * 0.07) * 1.2;
}

// Light on a surface color (same numbers as `lit` in ground.rs).
fn lit(base: vec3<f32>, n_dot_sun: f32, n_dot_up: f32) -> vec3<f32> {
    let ndl = max(n_dot_sun, 0.0);
    let k = 0.5 + 0.5 * n_dot_up;
    let sun = vec3<f32>(1.30, 1.10, 0.85) * ndl;
    let sky = vec3<f32>(0.14, 0.12, 0.09);
    let top = vec3<f32>(0.26, 0.36, 0.55);
    let c = base * (sun + sky + (top - sky) * k);
    let l = dot(c, vec3<f32>(0.299, 0.587, 0.114));
    return max(vec3<f32>(l) + (c - vec3<f32>(l)) * 1.18, vec3<f32>(0.0));
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    let h = in.hr.x;
    let rel = in.rel;
    let up = normalize(u.origin.xyz + rel);
    let to_eye = u.eye.xyz - rel;
    let dist_m = length(to_eye) * u.refm.w;
    let view = normalize(to_eye);
    // Detail fades with distance (it would only shimmer): fine grain first, then medium, then broad.
    let f_fine = 1.0 - smooth_(30.0, 400.0, dist_m);
    let f_mid = 1.0 - smooth_(300.0, 3000.0, dist_m);
    let f_far = 1.0 - smooth_(2000.0, 30000.0, dist_m);
    let pm = u.refm.xyz + rel * u.refm.w;
    let nB = vnoise(pm * 0.2);
    var n = normalize(in.nrm);
    var base: vec3<f32>;
    var spec = vec3<f32>(0.0);
    var foam = 0.0;
    // Where the shore is: the height wobbles a little so the waterline is not a perfect curve.
    let edge = h + (nB - 0.5) * 3.0 * f_mid;
    if h < 0.0 {
        // Water: flat sphere with gentle ripples, glints of sun, brighter toward grazing views.
        let rp = pm * 0.5;
        let e = 0.2;
        let r0 = vnoise(rp);
        let g = vec3<f32>(vnoise(rp + vec3<f32>(e, 0.0, 0.0)) - r0, vnoise(rp + vec3<f32>(0.0, e, 0.0)) - r0, vnoise(rp + vec3<f32>(0.0, 0.0, e)) - r0);
        n = normalize(up - (g - up * dot(g, up)) * (0.9 * f_fine));
        base = water(-h);
        let fres = pow(1.0 - max(dot(n, view), 0.0), 4.0);
        base = mix(base, vec3<f32>(0.35, 0.55, 0.85), fres * 0.7);
        let half = normalize(u.sun.xyz + view);
        let glint = pow(max(dot(n, half), 0.0), 220.0) * 2.5 + pow(max(dot(n, half), 0.0), 24.0) * 0.12;
        spec = vec3<f32>(1.0, 0.92, 0.75) * glint * step(0.0, dot(up, u.sun.xyz) + 0.05);
        foam = (1.0 - smooth_(0.0, 1.4, -edge)) * f_mid;
    } else {
        let steep = in.steep;
        let rockiness = smooth_(0.07, 0.34, steep) * 0.85;
        let rock = vec3<f32>(0.24, 0.20, 0.16);
        let c = mix(land(h), rock, rockiness);
        // Snow settles on high, gentler ground and slides off steep faces.
        let snow = smooth_(3600.0, 4900.0, h) * (1.0 - smooth_(0.08, 0.36, steep));
        let snowed = mix(c, vec3<f32>(0.90, 0.93, 1.00), snow);
        // A grey hint over whole mountain ranges, the same at any zoom.
        let snowy = smooth_(3600.0, 4700.0, h);
        let m = smooth_(1300.0, 3000.0, in.hr.y) * 0.7 * (1.0 - snowy);
        base = mix(snowed, vec3<f32>(0.36, 0.32, 0.28), m);
        // Texture: blotches of lighter / darker / drier ground at several sizes.
        let nA = vnoise(pm * 0.8);
        let nC = vnoise(pm * 0.03);
        let nD = vnoise(pm * 0.004);
        let v = (nA - 0.5) * 0.35 * f_fine + (nB - 0.5) * 0.40 * f_fine + (nC - 0.5) * 0.50 * f_mid + (nD - 0.5) * 0.45 * f_far;
        let dry = clamp((nC * 0.5 + nD * 0.5 - 0.45) * 2.5, 0.0, 1.0) * (1.0 - rockiness) * (1.0 - snow) * f_mid * 0.45;
        base = base * (1.0 + v);
        base = mix(base, base * vec3<f32>(1.45, 1.10, 0.55), dry);
        // Wet sand along the shore.
        base = base * mix(0.65, 1.0, smooth_(0.0, 2.5, edge));
        // Bump: the lighting normal gets small crinkles (more on rock, softer on snow).
        if f_mid > 0.0 {
            let e = 0.25;
            let r0 = relief(pm);
            let g = vec3<f32>(relief(pm + vec3<f32>(e, 0.0, 0.0)) - r0, relief(pm + vec3<f32>(0.0, e, 0.0)) - r0, relief(pm + vec3<f32>(0.0, 0.0, e)) - r0) / e;
            let k = mix(0.07, 0.22, rockiness) * (1.0 - 0.6 * snow) * f_mid;
            n = normalize(n - (g - n * dot(g, n)) * k);
        }
    }
    var color = lit(base, dot(n, u.sun.xyz), dot(n, up));
    color = color + spec;
    color = mix(color, vec3<f32>(0.92, 0.96, 1.0) * (0.35 + 0.65 * max(dot(up, u.sun.xyz), 0.0)), foam * 0.85);
    if u.sky.w > 0.0 {
        let t = clamp(length(rel - u.eye.xyz) / u.eye.w, 0.0, 1.0);
        let f = 1.0 * u.sky.w * (t * t * (3.0 - 2.0 * t));
        color = mix(color, u.sky.rgb, f);
    }
    return vec4<f32>(color, 1.0);
}
