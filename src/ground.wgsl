// The ground, shaded per pixel. The mesh only carries position, a smooth normal, the true height and the
// coarse "range" height; color, rock/snow edges, shore line, light and haze are worked out for every pixel,
// so edges are crisp curves instead of zig-zags along triangles. Keep in step with `ground::shade` (Rust)
// and `terrain::color_in_range`, which do the same thing on the CPU (tests, software preview).

struct Uniforms {
    mvp: mat4x4<f32>,     // (globe position - origin) -> screen
    sun: vec4<f32>,       // xyz: toward the light, globe coordinates; w: focal length in pixels
    eye: vec4<f32>,       // xyz: camera position relative to origin; w: distance to the horizon (haze scale)
    origin: vec4<f32>,    // xyz: the origin point, globe coordinates; w: meters added to heights for colour (sea level)
    sky: vec4<f32>,       // rgb: sky color; w: haze amount (0 = no haze)
    refm: vec4<f32>,      // xyz: origin in meters, wrapped to 0..4096 (anchors the ground detail); w: meters per globe radius
    dbg: vec4<f32>,       // x: debug bits (F1..F6 in the app): 1 no noise, 2 no dry/wet, 4 plain beach, 8 no detail fades, 16 default cover, 32 no bump, 64 all sand
    ov: array<vec4<f32>, 6>, // the stroke being made (two windows of 3): [center - origin, half size m], [t1, on], [t2, 0]
};

fn dbg_on(bit: u32) -> bool {
    return (u32(u.dbg.x) & bit) != 0u;
}
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var ov_tex: texture_2d_array<f32>; // meters added to the ground by the stroke in progress

struct Ov { h: f32, g: vec3<f32>, th: f32 };

// What the stroke being made adds at the point `p` (relative to the origin): the height, and the slope of it as a
// vector (to tilt the normal). The meshes are only remade when the stroke ends; until then this shows it.
fn ov_at(layer: i32, p: vec3<f32>) -> Ov {
    var o: Ov;
    o.h = 0.0;
    o.g = vec3<f32>(0.0);
    o.th = 0.0;
    let a = u.ov[layer * 3];
    let b = u.ov[layer * 3 + 1];
    let c = u.ov[layer * 3 + 2];
    if b.w < 0.5 { return o; }
    let d = p - a.xyz;
    let x = dot(d, b.xyz) * u.refm.w;
    let y = dot(d, c.xyz) * u.refm.w;
    let fx = (x + a.w) / (2.0 * a.w) * 512.0 - 0.5;
    let fy = (y + a.w) / (2.0 * a.w) * 512.0 - 0.5;
    if fx < 0.0 || fy < 0.0 || fx >= 511.0 || fy >= 511.0 { return o; }
    let ix = i32(floor(fx));
    let iy = i32(floor(fy));
    let tx = fx - floor(fx);
    let ty = fy - floor(fy);
    let v00 = textureLoad(ov_tex, vec2<i32>(ix, iy), layer, 0).x;
    let v10 = textureLoad(ov_tex, vec2<i32>(ix + 1, iy), layer, 0).x;
    let v01 = textureLoad(ov_tex, vec2<i32>(ix, iy + 1), layer, 0).x;
    let v11 = textureLoad(ov_tex, vec2<i32>(ix + 1, iy + 1), layer, 0).x;
    o.h = mix(mix(v00, v10, tx), mix(v01, v11, tx), ty);
    let texel = 2.0 * a.w / 512.0;
    let gx = mix(v10 - v00, v11 - v01, ty) / texel;
    let gy = mix(v01 - v00, v11 - v10, tx) / texel;
    o.g = b.xyz * gx + c.xyz * gy;
    o.th = o.h * c.w;
    return o;
}

struct VertexIn {
    @location(0) pos: vec3<f32>,   // relative to the patch center
    @location(1) nrm: vec4<f32>,   // xyz: shading normal; w: steepness * 1.6
    @location(2) hr: vec2<f32>,    // x: height (m, negative = sea floor); y: range height (m)
    @location(3) inst: vec4<f32>,  // per patch: xyz its center relative to the origin; w its width (radians)
    @location(4) morph: f32,       // radial move onto the parent patch's half-resolution mesh
    @location(5) nrm_p: vec4<f32>, // the normal that mesh has there
    @location(6) cover: vec4<f32>, // rgb: the biome's ground cover colour; a: how sandy its beaches are
    @location(7) shore: f32,       // beach weight: 1 sand/stone, 0 grass (smoothed height, then the neighbour rule)
    @location(8) ovf: vec4<f32>,   // per patch: x, y = 1 when stroke window 0 / 1 is to be added (the mesh does not show it yet)
};

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) nrm: vec3<f32>,
    @location(1) steep: f32,
    @location(2) hr: vec2<f32>,
    @location(3) rel: vec3<f32>,
    @location(4) cover: vec4<f32>,
    @location(5) shore: f32,
    @location(6) ovh: f32,
    @location(7) psea: vec3<f32>, // the point with its height taken off: what the ground detail is anchored to
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
    var o0: Ov;
    var o1: Ov;
    o0.h = 0.0; o0.g = vec3<f32>(0.0); o0.th = 0.0;
    o1.h = 0.0; o1.g = vec3<f32>(0.0); o1.th = 0.0;
    if v.ovf.x > 0.5 { o0 = ov_at(0, p); }
    if v.ovf.y > 0.5 { o1 = ov_at(1, p); }
    let oh = o0.h + o1.h;
    let og = o0.g + o1.g;
    p = p + normalize(u.origin.xyz + p) * (oh / u.refm.w);
    out.clip = u.mvp * vec4<f32>(p, 1.0);
    out.nrm = normalize(mix(v.nrm.xyz, v.nrm_p.xyz, k) - og);
    out.steep = v.nrm.w / 1.6 + (1.0 - 1.0 / sqrt(1.0 + dot(og, og)));
    // The colours follow the height the point has at this moment: the same slide as the position (the point moves
    // radially by `morph * k`), so a patch changing detail level changes its colours smoothly instead of popping.
    let hm = v.hr.x + v.morph * k * u.refm.w + oh;
    out.hr = vec2<f32>(hm, v.hr.y);
    out.rel = p;
    out.cover = v.cover;
    out.shore = v.shore;
    out.ovh = o0.th + o1.th;
    // The texture is pinned to the spot on the sea-level sphere under the point, so it does not slide when the ground
    // is raised or lowered (a stroke being shown, an edit made real, a mountain growing).
    let radial_m = max(v.hr.x, 0.0) + v.morph * k * u.refm.w + oh;
    out.psea = p - normalize(u.origin.xyz + p) * (radial_m / u.refm.w);
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

// Land by height in meters (sand, ground cover, forest line, rock). `cover` is the biome's ground cover colour;
// the plain forest green (0.22, 0.42, 0.08) gives the original ramp.
fn land(h: f32, beach: f32, cover: vec3<f32>, sandy: f32) -> vec3<f32> {
    let lum = vec3<f32>(0.299, 0.587, 0.114);
    let k = vec3<f32>(1.0) * (1.0 / dot(lum, vec3<f32>(0.22, 0.42, 0.08)));
    // Beaches: sand where it is warm, stone (a little of the biome's colour) where it is cold.
    let stone = vec3<f32>(0.40, 0.40, 0.41) * 0.9 + cover * 0.1;
    let c0 = mix(stone, vec3<f32>(0.62, 0.50, 0.26), sandy);
    let c1 = c0 * 0.94;
    let c2 = cover;
    let c3 = cover * dot(lum, vec3<f32>(0.15, 0.34, 0.06)) * k.x;
    let c4 = cover * dot(lum, vec3<f32>(0.07, 0.22, 0.06)) * k.x;
    let c5 = cover * dot(lum, vec3<f32>(0.18, 0.27, 0.09)) * k.x;
    let c6 = vec3<f32>(0.33, 0.30, 0.25);
    let c7 = vec3<f32>(0.42, 0.38, 0.33);
    // The beach follows `beach`, worked out on the CPU from a smoothed height and then cleaned with the neighbours
    // (1 = sand/stone, 0 = grass), not the exact height: that has metre-scale detail that comes and goes with the view.
    if h <= 120.0 {
        let upper = seg(c2, c3, 40.0, 500.0, max(h, 40.0));
        // Cliffs right at the coast (exact height well above the smoothed one) are not beach.
        let w = max(1.0 - beach, smooth_(40.0, 120.0, h));
        return mix(c0, upper, w);
    }
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
fn relief(p: vec3<f32>, f1: f32, f2: f32, f3: f32) -> f32 {
    return vnoise(p * 1.1) * 0.6 * f1 + vnoise(p * 0.3) * 0.9 * f2 + vnoise(p * 0.07) * 1.2 * f3;
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
    let pm = u.refm.xyz + in.psea * u.refm.w;
    let view = normalize(to_eye);
    // Detail fades with distance (it would only shimmer): fine grain first, then medium, then broad.
    // Each size of detail is blended out by how big its features are on the screen, not by a fixed distance: noise
    // with `s` cells per metre is shown fully while a cell is more than ~10 pixels, gone when it is ~3 pixels (finer
    // would only crawl and shimmer as the camera moves, like TV static). `fp` is the size of a pixel on the ground.
    let pmdx = dpdx(pm);
    let pmdy = dpdy(pm);
    let fp = max(length(pmdx), length(pmdy));
    var f_fine = 1.0 - smooth_(0.10, 0.35, fp * 1.1);
    var f_low = 1.0 - smooth_(0.10, 0.35, fp * 0.3);
    var f_mid = 1.0 - smooth_(0.10, 0.35, fp * 0.07);
    var f_far = (1.0 - smooth_(0.10, 0.35, fp * 0.004)) * (1.0 - smooth_(2000.0, 30000.0, dist_m));
    if dbg_on(8u) { f_fine = 1.0; f_low = 1.0; f_mid = 1.0; f_far = 1.0; }
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
        // Colours judge the height from the planet's sea level, so lowered or raised land keeps its look.
        let ch = max(h + u.origin.w, 0.0);
        // Coasts are not all alike: the sand gives way to stone in places (worked out per mesh point on the CPU).
        let sandy = in.cover.a;
        var cover = in.cover.rgb;
        var sandy2 = sandy;
        if dbg_on(16u) { cover = vec3<f32>(0.22, 0.42, 0.08); }
        // Same as the old F7 test: beaches are plain sand for now (the per-vertex sand variation showed as green speckle).
        sandy2 = 1.0;
        if dbg_on(64u) { sandy2 = sandy; }
        var beach = in.shore;
        if dbg_on(4u) { beach = 1.0 - smooth_(4.0, 40.0, ch); }
        let c = mix(land(ch, beach, cover, sandy2), rock, rockiness);
        // Snow settles on high, gentler ground and slides off steep faces.
        let snow = smooth_(3600.0, 4900.0, ch) * (1.0 - smooth_(0.08, 0.36, steep));
        let snowed = mix(c, vec3<f32>(0.88, 0.90, 0.93), snow);
        // A grey hint over whole mountain ranges, the same at any zoom.
        let snowy = smooth_(3600.0, 4700.0, ch);
        let m = smooth_(1300.0, 3000.0, max(in.hr.y + u.origin.w, 0.0)) * 0.6 * (1.0 - snowy);
        base = mix(snowed, vec3<f32>(0.33, 0.33, 0.31), m);
        // Texture: blotches of lighter / darker / drier ground at several sizes.
        let nA = vnoise(pm * 0.8);
        let nC = vnoise(pm * 0.03);
        let nD = vnoise(pm * 0.004);
        var v = (nA - 0.5) * 0.35 * f_fine + (nB - 0.5) * 0.40 * f_low + (nC - 0.5) * 0.50 * f_mid + (nD - 0.5) * 0.45 * f_far;
        // Dry patches are for grass, not for the beach (they made orange blotches that came and went with distance).
        if dbg_on(1u) { v = 0.0; }
        let on_grass = 1.0 - beach;
        let dry = clamp((nC * 0.5 + nD * 0.5 - 0.45) * 2.5, 0.0, 1.0) * (1.0 - rockiness) * (1.0 - snow) * f_mid * 0.45 * on_grass;
        base = base * (1.0 + v);
        if !dbg_on(2u) {
            base = mix(base, base * vec3<f32>(1.45, 1.10, 0.55), dry);
            // Wet sand along the shore.
            base = base * mix(0.65, 1.0, smooth_(0.0, 2.5, edge));
        }
        // Bump: the lighting normal gets small crinkles (more on rock, softer on snow).
        if f_mid > 0.0 && !dbg_on(32u) {
            let e = 0.25;
            let r0 = relief(pm, f_fine, f_low, f_mid);
            let g = vec3<f32>(relief(pm + vec3<f32>(e, 0.0, 0.0), f_fine, f_low, f_mid) - r0, relief(pm + vec3<f32>(0.0, e, 0.0), f_fine, f_low, f_mid) - r0, relief(pm + vec3<f32>(0.0, 0.0, e), f_fine, f_low, f_mid) - r0) / e;
            let k = mix(0.07, 0.22, rockiness) * (1.0 - 0.6 * snow) * f_mid;
            n = normalize(n - (g - n * dot(g, n)) * k);
        }
    }
    var color = lit(base, dot(n, u.sun.xyz), dot(n, up));
    color = color + spec;
    // The stroke being made is tinted (warm where it raises, cool where it lowers) until the real ground replaces it.
    let tw = smooth_(0.0, 4.0, abs(in.ovh)) * 0.3;
    color = mix(color, select(vec3<f32>(0.30, 0.65, 1.0), vec3<f32>(1.0, 0.72, 0.25), in.ovh > 0.0), tw);
    color = mix(color, vec3<f32>(0.92, 0.96, 1.0) * (0.35 + 0.65 * max(dot(up, u.sun.xyz), 0.0)), foam * 0.85);
    if u.sky.w > 0.0 {
        let t = clamp(length(rel - u.eye.xyz) / u.eye.w, 0.0, 1.0);
        let f = 1.0 * u.sky.w * (t * t * (3.0 - 2.0 * t));
        color = mix(color, u.sky.rgb, f);
    }
    return vec4<f32>(color, 1.0);
}
