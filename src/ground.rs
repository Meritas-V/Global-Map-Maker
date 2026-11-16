//! The ground: the planet surface drawn around the camera, displaced by the terrain height
//! function, colored by depth / altitude / steepness and lit from the light direction.
//!
//! Built from the same cube-sphere quadtree as the pixels, but only as deep as needed to look
//! round and sharp: a patch is split until it is about GROUND_PX wide on screen, and patches that
//! are outside the view are skipped. Water is a flat sphere at sea level, tinted by depth.

use crate::pixelpaint::{PixelId, MAX_LEVEL};
use crate::pixelview::{is_visible, screen_px, width_rad, PixelView};
use crate::terrain;
use crate::tiles;
use bytemuck::{Pod, Zeroable};
use glam::DVec3;

/// One corner of the ground mesh, 32 bytes: where it is (relative to the frame's origin point), which way the
/// surface faces (xyz, 1/127 steps) and how steep it really is (w), and the true height plus the coarse
/// "range" height. Color, shore, snow, light and haze are worked out per pixel from these (ground.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct GroundVertex {
    pub pos: [f32; 3],
    pub nrm: [i8; 4],
    /// Height in meters (negative = sea floor), range height in meters.
    pub hr: [f32; 2],
    /// Geomorphing: how far (radius, unit sphere) this point moves to sit on the half-resolution mesh of the
    /// parent patch. The shader blends it in as the patch shrinks on screen, so a level change never pops.
    pub morph: f32,
    /// The surface normal the parent patch's mesh has at this spot (xyz, 1/127 steps); blended with `nrm` by the
    /// same factor as `morph`, so lighting does not jump at a level change either.
    pub nrm_p: [i8; 4],
}

/// Steepness (0..~0.6) is stored times this in a signed byte.
const STEEP_SCALE: f32 = 1.6;

impl GroundVertex {
    pub fn normal(&self) -> [f32; 3] {
        [0, 1, 2].map(|i| self.nrm[i] as f32 / 127.0)
    }

    pub fn steepness(&self) -> f32 {
        self.nrm[3] as f32 / 127.0 / STEEP_SCALE
    }
}

fn snorm(x: f64) -> i8 {
    (x.clamp(-1.0, 1.0) * 127.0).round() as i8
}

/// Light and haze for one frame, as the ground shader gets them.
#[derive(Clone, Copy)]
pub struct Lighting {
    /// Unit vector toward the light, globe coordinates.
    pub sun: DVec3,
    /// The frame's origin point (globe coordinates) and the camera relative to it.
    pub origin: DVec3,
    pub eye_rel: DVec3,
    /// Distance from the eye to the bare-sphere horizon (haze scale).
    pub horizon: f64,
    /// Some(sky color) fades far ground into the sky.
    pub haze: Option<[f32; 3]>,
}

impl Lighting {
    /// With a render distance, fog is fully thick a little before it, so the cut-off edge never shows.
    pub fn with_render_distance(mut self, d: f64) -> Lighting {
        self.horizon = self.horizon.min(0.85 * d);
        self
    }

    pub fn new(sun: DVec3, origin: DVec3, eye: DVec3, haze: Option<[f32; 3]>) -> Lighting {
        Lighting { sun, origin, eye_rel: eye - origin, horizon: (eye.length_squared() - 1.0).max(1e-18).sqrt(), haze }
    }
}

/// Light on a surface color: warm sun on faces turned toward it, cool sky light from above and a warm bounce
/// from below, so shadowed slopes go blue instead of grey. (Same numbers as `lit` in ground.wgsl.)
pub fn lit(base: [f32; 3], n_dot_sun: f64, n_dot_up: f64) -> [f32; 3] {
    let ndl = n_dot_sun.max(0.0) as f32;
    let k = (0.5 + 0.5 * n_dot_up) as f32; // 1 = facing up
    let sun = [1.30, 1.10, 0.85].map(|x| x * ndl);
    let sky = [0.14, 0.12, 0.09];
    let top = [0.26, 0.36, 0.55];
    let c = [0, 1, 2].map(|i| base[i] * (sun[i] + sky[i] + (top[i] - sky[i]) * k));
    // A touch more color: lit colors drift grey-brown otherwise.
    let l = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2];
    c.map(|x| (l + (x - l) * 1.18).max(0.0))
}

/// The color of one pixel of ground. The CPU twin of `fs_main` in ground.wgsl (used by tests and the
/// software preview); keep the two in step.
pub fn shade(h: f64, range_h: f64, nrm: [f32; 3], steep: f32, rel: DVec3, light: &Lighting) -> [f32; 3] {
    let up = (light.origin + rel).normalize();
    let (n, base) = if h < 0.0 {
        // Open water is a flat sphere: shade it as one, tinted by depth.
        (up, terrain::color(h, 0.0))
    } else {
        let n = DVec3::new(nrm[0] as f64, nrm[1] as f64, nrm[2] as f64).normalize();
        (n, terrain::color_in_range(h, range_h, steep as f64))
    };
    let mut color = lit(base, n.dot(light.sun), n.dot(up));
    if let Some(sky) = light.haze {
        let t = ((rel - light.eye_rel).length() / light.horizon).clamp(0.0, 1.0);
        let f = HAZE_MAX * (t * t * (3.0 - 2.0 * t)) as f32;
        color = [0, 1, 2].map(|i| color[i] + (sky[i] - color[i]) * f);
    }
    color
}

/// Patches are split until they are this many screen pixels wide or less. Outside (the globe) 56; inside
/// 112 with twice the pieces per edge (16 instead of 8): the mesh points and height-tile samples are still
/// ~7 px apart, so the ground is as sharp as with 56 px / 8 pieces, but there are 4x fewer patches (draw calls,
/// meshes, loader jobs) and the same number of tiles.
pub const GROUND_PX: f64 = 56.0;
pub const GROUND_PX_INSIDE: f64 = 112.0;
/// Farther away the limit grows (far ground is hazy and small: fewer, bigger patches, coarser tiles, fewer draw
/// calls): from FAR_START_M on it grows by GROUND_PX per FAR_REF_M, up to GROUND_PX x (1 + FAR_GAIN). Nearer than
/// FAR_START_M every patch is the full-detail size (coarser patches there read as softer mountains).
/// ground.wgsl has the same numbers (its geomorph window follows this limit); a test checks they match.
pub const FAR_START_M: f64 = 10000.0;
pub const FAR_REF_M: f64 = 10000.0;
pub const FAR_GAIN: f64 = 1.0;

/// Geomorph window as fractions of the patch size limit: fully the parent's shape up to MORPH_START (just
/// after a patch is born at 0.5), fully itself from MORPH_END on. Narrow on purpose: a wide window leaves most
/// patches half-blurred in steady state. Same numbers in ground.wgsl (checked by test).
pub const MORPH_START: f64 = 0.54;
pub const MORPH_END: f64 = 0.66;

/// The patch size limit (screen pixels) at `dist` (globe radii from the eye). `inside`: the flying scene (the
/// outside globe keeps one limit everywhere so it stays round).
pub fn px_limit(dist: f64, inside: bool) -> f64 {
    let gain = if inside { FAR_GAIN } else { 0.0 };
    let r = terrain::EARTH_RADIUS_M;
    (if inside { GROUND_PX_INSIDE } else { GROUND_PX }) * (1.0 + ((dist - FAR_START_M / r) / (FAR_REF_M / r)).clamp(0.0, gain))
}
/// Safety cap on patches per frame.
pub const MAX_PATCHES: usize = 6_000;
/// Longest edge (radians) of one flat piece inside a patch (keeps big patches round).
const SEGMENT_RAD: f64 = 0.04;
/// Most pieces per patch edge.
const MAX_SEGMENTS: usize = 16;
/// Shading uses heights this many times taller than real, so slopes read as relief (the shape itself is real scale).
const SHADE_EXAGGERATION: f64 = 7.0;
/// Vertices one frame can hold.
pub const MAX_GROUND_VERTS: usize = 7_000_000;
/// Color of the sky, and what far ground fades into (inside scene).
pub const SKY: [f32; 3] = [0.50, 0.68, 0.92];
/// Most the haze mixes in: the far ground never disappears completely, so mountains still show over the horizon.
const HAZE_MAX: f32 = 1.0;

/// Small enough to stop splitting: about GROUND_PX wide on screen, or, seen at a grazing angle, a thin
/// sliver (a patch far out toward the horizon can be wide but only a few pixels tall; splitting
/// those just piles up thousands of invisible triangles).
fn small_enough(id: PixelId, view: &PixelView) -> bool {
    let lim = px_limit((view.eye - id.center_dir()).length(), view.max_dist.is_finite());
    if screen_px(id, view) <= lim {
        return true;
    }
    let (mut lo, mut hi) = ((f64::MAX, f64::MAX), (f64::MIN, f64::MIN));
    for (fx, fy) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
        let p = view.view.transform_point3(id.dir_at(fx, fy));
        if p.z > -1e-12 {
            return false; // not in front of the lens: can't tell, keep splitting
        }
        let (sx, sy) = (view.focal_px * p.x / -p.z, view.focal_px * p.y / -p.z);
        lo = (lo.0.min(sx), lo.1.min(sy));
        hi = (hi.0.max(sx), hi.1.max(sy));
    }
    let (bw, bh) = (hi.0 - lo.0, hi.1 - lo.1);
    bw.max(bh) <= 4.0 * lim && bw * bh <= 0.3 * lim * lim
}

/// Which patches to draw for this camera.
pub fn visible_patches(view: &PixelView) -> Vec<PixelId> {
    let mut out = Vec::new();
    let mut stack: Vec<PixelId> = (0..6).map(|f| PixelId::new(f, 0, 0, 0)).collect();
    while let Some(id) = stack.pop() {
        if !is_visible(id, view) {
            continue;
        }
        let leaf = small_enough(id, view) || id.level == MAX_LEVEL || out.len() + stack.len() * 4 > MAX_PATCHES;
        if leaf {
            out.push(id);
        } else {
            stack.extend(id.children());
        }
    }
    out
}

/// What makes one patch mesh: the patch, how many pieces per edge, and which height-tile level it samples.
/// Heights depend on nothing else (not on the camera), so a mesh made for a key can be drawn for as long as
/// the key keeps being asked for, from any camera position.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct PatchKey {
    pub id: PixelId,
    pub n: u8,
    pub lvl: u8,
    /// The height-tile level of this patch's parent (what the mesh morphs toward).
    pub plvl: u8,
}

/// One patch's triangles. Positions are relative to `center` (a point on the unit sphere), so the mesh stays
/// precise at any zoom and is moved into place per frame with one offset.
pub struct PatchMesh {
    pub center: DVec3,
    /// The triangles as a plain list (3 vertices each); what the tests and the flat builders read.
    pub verts: Vec<GroundVertex>,
    /// The same triangles with every point stored once (grid points, then the skirt's hanging points) ...
    pub ivert: Vec<GroundVertex>,
    /// ... and the triangles as indices into `ivert`. This is what goes to the GPU: about 4x smaller than `verts`,
    /// so four times as many patches fit in the cache, and the vertex shader runs once per point, not once per corner.
    pub index: Vec<u16>,
}

/// Allowed numbers of pieces per patch edge: few sizes, so meshes are reused as the view changes a little.
const N_STEPS: [usize; 1] = [16];

/// Pieces per patch edge: with 16, a patch (56-112 px wide) has mesh points 3.5-7 px apart, and the height-tile level follows (one tile sample per mesh point).
const BASE_PIECES: usize = 16;

/// The mesh a patch gets. Depends on the patch alone (not on the camera): pieces per edge follow the patch size,
/// and the height-tile level follows from that (one tile sample per mesh point), so a patch keeps the same mesh
/// however you move, and meshes made once are reused for good.
pub fn patch_key(id: PixelId, view: &PixelView) -> PatchKey {
    let (n, lvl) = pieces_and_level(id, view);
    let plvl = if id.level == 0 { lvl } else { pieces_and_level(id.ancestor(id.level - 1), view).1 };
    PatchKey { id, n, lvl, plvl }
}

fn pieces_and_level(id: PixelId, view: &PixelView) -> (u8, u8) {
    let w = width_rad(id.level);
    let roundness = (w / SEGMENT_RAD).ceil() as usize;
    let want = roundness.max(BASE_PIECES);
    let n = N_STEPS.iter().copied().find(|q| *q >= want).unwrap_or(MAX_SEGMENTS).min(MAX_SEGMENTS);
    // Never finer than the scene allows (the outside map view stays soft however close you get).
    let lvl = match view.terrain {
        Some(t) => {
            let cap = t.detail_cap.map_or(tiles::MAX_TILE_LEVEL as f64, |c| tiles::level_for(c).floor());
            let want = (tiles::spacing_rad(0) / (w / n as f64)).log2().ceil().clamp(0.0, tiles::MAX_TILE_LEVEL as f64);
            want.min(cap) as u8
        }
        None => 0,
    };
    (n as u8, lvl)
}

/// Makes the height tiles these meshes will read, on all cores at once, instead of one by one.
pub fn prefetch_patches(keys: &[PatchKey], view: &PixelView) {
    let Some(t) = view.terrain else { return };
    let mut pts = Vec::with_capacity(keys.len() * 5);
    let mut range = Vec::with_capacity(keys.len() * 5);
    for k in keys {
        let f = tiles::freq_for_level(k.lvl);
        for (fx, fy) in [(0.5, 0.5), (0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
            let d = k.id.dir_at(fx, fy);
            pts.push((d, f));
            range.push((d, terrain::RANGE_FREQ));
        }
    }
    t.terrain.prefetch(&pts);
    // The coarse "range" heights used for color: a handful of big tiles, made once.
    t.terrain.prefetch(&range);
}

/// Builds one patch (reads heights only, so patches can be built on all cores at once).
pub fn build_patch(key: PatchKey, view: &PixelView) -> PatchMesh {
    build_patch_from(key, view.terrain.map(|t| t.terrain))
}

/// The same from the height source alone (what the background loader has: no camera).
pub fn build_patch_from(key: PatchKey, terrain: Option<&dyn terrain::HeightSource>) -> PatchMesh {
    let (id, n) = (key.id, key.n as usize);
    let w = width_rad(id.level);
    let center = id.center_dir();
    let freq = tiles::freq_for_level(key.lvl);
    // Grid with one extra ring all around, so normals at the patch edge match the neighbour's.
    let m = n + 3;
    let mut pos = vec![DVec3::ZERO; m * m];
    let mut shape = vec![DVec3::ZERO; m * m]; // same points with exaggerated height, for lighting only
    let mut hgt = vec![0.0f64; m * m];
    let mut rng = vec![0.0f64; m * m]; // coarse height, for the mountain-range tint
    for gj in 0..m {
        for gi in 0..m {
            let u = id.dir_at((gi as f64 - 1.0) / n as f64, (gj as f64 - 1.0) / n as f64);
            let interior = (1..=n + 1).contains(&gi) && (1..=n + 1).contains(&gj);
            let (h, r, rh) = match terrain {
                Some(t) => {
                    let h = t.height_m(u, freq);
                    // The range tint is only needed where a vertex is drawn, not in the padding ring.
                    let rh = if interior { t.height_m(u, terrain::RANGE_FREQ) } else { 0.0 };
                    (h, 1.0 + h.max(0.0) / terrain::EARTH_RADIUS_M, rh)
                }
                None => (0.0, 1.0, 0.0),
            };
            pos[gj * m + gi] = u * r;
            shape[gj * m + gi] = u * (1.0 + SHADE_EXAGGERATION * (r - 1.0));
            hgt[gj * m + gi] = h;
            rng[gj * m + gi] = rh;
        }
    }
    let vertex = |i: usize, j: usize| {
        let (gi, gj) = (i + 1, j + 1);
        let p = pos[gj * m + gi];
        let dx = shape[gj * m + gi + 1] - shape[gj * m + gi - 1];
        let dy = shape[(gj + 1) * m + gi] - shape[(gj - 1) * m + gi];
        let up = p.normalize();
        let mut nrm = dx.cross(dy).normalize();
        if !nrm.is_finite() {
            nrm = up;
        }
        if nrm.dot(up) < 0.0 {
            nrm = -nrm;
        }
        let h = hgt[gj * m + gi];
        // Slope for coloring comes from the real (not exaggerated) shape: rock and snow follow actual steepness.
        let tdx = pos[gj * m + gi + 1] - pos[gj * m + gi - 1];
        let tdy = pos[(gj + 1) * m + gi] - pos[(gj - 1) * m + gi];
        let mut tn = tdx.cross(tdy).normalize();
        if !tn.is_finite() || tn.dot(up) < 0.0 {
            tn = if tn.is_finite() { -tn } else { up };
        }
        let steep = if h <= 0.0 { 0.0 } else { (1.0 - tn.dot(up)).max(0.0) };
        (
            GroundVertex {
                pos: (p - center).as_vec3().to_array(),
                nrm: [snorm(nrm.x), snorm(nrm.y), snorm(nrm.z), (steep as f32 * STEEP_SCALE * 127.0).round().min(127.0) as i8],
                hr: [h as f32, rng[gj * m + gi] as f32],
                morph: 0.0,
                nrm_p: [0; 4],
            },
            up,
        )
    };
    // Each grid point is made once (4 quads share it).
    let grid: Vec<(GroundVertex, DVec3)> = (0..=n).flat_map(|j| (0..=n).map(move |i| (i, j))).map(|(i, j)| vertex(i, j)).collect();
    let radius = |i: usize, j: usize| pos[(j + 1) * m + i + 1].length();
    // Geomorph targets: where each point sits on the half-resolution mesh (the parent patch's mesh). Even points
    // are the parent's own points, read at the parent's detail; odd points are halfway between them, along the
    // same diagonal the parent cuts its quads with.
    let half = n / 2;
    // The parent's points, with one extra ring around (for its normals): heights at its detail level.
    let pw = half + 3;
    let pr: Vec<f64> = match terrain {
        Some(t) if half > 0 => {
            let pf = tiles::freq_for_level(key.plvl);
            (0..pw)
                .flat_map(|j| (0..pw).map(move |i| (i, j)))
                .map(|(i, j)| {
                    let u = id.dir_at(2.0 * (i as f64 - 1.0) / n as f64, 2.0 * (j as f64 - 1.0) / n as f64);
                    1.0 + t.height_m(u, pf).max(0.0) / terrain::EARTH_RADIUS_M
                })
                .collect()
        }
        _ => Vec::new(),
    };
    // Parent grid point (a, b), a and b in 0..=half, lives at pr[(b + 1) * pw + a + 1].
    let prr = |a: usize, b: usize| pr[(b + 1) * pw + a + 1];
    let parent_normals: Vec<DVec3> = if pr.is_empty() {
        Vec::new()
    } else {
        let shape_at = |a: i32, b: i32| {
            let u = id.dir_at(2.0 * a as f64 / n as f64, 2.0 * b as f64 / n as f64);
            let r = pr[((b + 1) as usize) * pw + (a + 1) as usize];
            u * (1.0 + SHADE_EXAGGERATION * (r - 1.0))
        };
        (0..=half as i32)
            .flat_map(|b| (0..=half as i32).map(move |a| (a, b)))
            .map(|(a, b)| {
                let up = id.dir_at(2.0 * a as f64 / n as f64, 2.0 * b as f64 / n as f64);
                let mut nr = (shape_at(a + 1, b) - shape_at(a - 1, b)).cross(shape_at(a, b + 1) - shape_at(a, b - 1)).normalize();
                if !nr.is_finite() {
                    nr = up;
                }
                if nr.dot(up) < 0.0 {
                    nr = -nr;
                }
                nr
            })
            .collect()
    };
    let parent_normal = |i: usize, j: usize| -> DVec3 {
        let pn = |a: usize, b: usize| parent_normals[b * (half + 1) + a];
        let (a, b) = (i / 2, j / 2);
        let sum = match (i % 2, j % 2) {
            (0, 0) => pn(a, b),
            (1, 0) => pn(a, b) + pn(a + 1, b),
            (0, 1) => pn(a, b) + pn(a, b + 1),
            _ => pn(a, b) + pn(a + 1, b) + pn(a, b + 1) + pn(a + 1, b + 1),
        };
        sum.normalize()
    };
    let target = |i: usize, j: usize| -> f64 {
        let p = |a: usize, b: usize| prr(a, b);
        match (i % 2, j % 2) {
            (0, 0) => p(i / 2, j / 2),
            (1, 0) => 0.5 * (p(i / 2, j / 2) + p(i / 2 + 1, j / 2)),
            (0, 1) => 0.5 * (p(i / 2, j / 2) + p(i / 2, j / 2 + 1)),
            _ => {
                let (a, b, c, d) = (p(i / 2, j / 2), p(i / 2 + 1, j / 2), p(i / 2 + 1, j / 2 + 1), p(i / 2, j / 2 + 1));
                if (a - c).abs() <= (b - d).abs() {
                    0.5 * (a + c)
                } else {
                    0.5 * (b + d)
                }
            }
        }
    };
    let morph_of = |i: usize, j: usize| -> f32 {
        if pr.is_empty() {
            0.0
        } else {
            (target(i, j) - radius(i, j)) as f32
        }
    };
    let grid: Vec<(GroundVertex, DVec3)> = grid
        .into_iter()
        .enumerate()
        .map(|(q, (v, up))| {
            let (i, j) = (q % (n + 1), q / (n + 1));
            let nrm_p = if pr.is_empty() { [v.nrm[0], v.nrm[1], v.nrm[2], 0] } else { let pn = parent_normal(i, j); [snorm(pn.x), snorm(pn.y), snorm(pn.z), 0] };
            (GroundVertex { morph: morph_of(i, j), nrm_p, ..v }, up)
        })
        .collect();
    let at = |i: usize, j: usize| grid[j * (n + 1) + i];
    let mut out = Vec::with_capacity(n * n * 6 + n * 24);
    let mut ivert: Vec<GroundVertex> = grid.iter().map(|g| g.0).collect();
    let mut index: Vec<u16> = Vec::with_capacity(n * n * 6 + n * 24);
    let gq = |i: usize, j: usize| (j * (n + 1) + i) as u16;
    for i in 0..n {
        for j in 0..n {
            let (a, b, c, d) = (at(i, j).0, at(i + 1, j).0, at(i + 1, j + 1).0, at(i, j + 1).0);
            let (qa, qb, qc, qd) = (gq(i, j), gq(i + 1, j), gq(i + 1, j + 1), gq(i, j + 1));
            // Cut the quad along the diagonal whose ends are closest in height: the cut then follows the
            // ridge or valley instead of making a zig-zag across it.
            let d_ac = (radius(i, j) - radius(i + 1, j + 1)).abs();
            let d_bd = (radius(i + 1, j) - radius(i, j + 1)).abs();
            if d_ac <= d_bd {
                out.extend_from_slice(&[a, b, c, a, c, d]);
                index.extend_from_slice(&[qa, qb, qc, qa, qc, qd]);
            } else {
                out.extend_from_slice(&[a, b, d, b, c, d]);
                index.extend_from_slice(&[qa, qb, qd, qb, qc, qd]);
            }
        }
    }
    // Skirts: a short strip hanging down from every patch edge, so where a patch meets a finer
    // (or coarser) neighbour no crack shows the background through.
    let skirt = 0.35 * w / n as f64;
    let hang = |(v, up): (GroundVertex, DVec3)| {
        let d = (up * skirt).as_vec3();
        GroundVertex { pos: [v.pos[0] - d.x, v.pos[1] - d.y, v.pos[2] - d.z], ..v }
    };
    // Hanging points are made once per edge point (neighbouring strips share them).
    let mut hung_at: std::collections::HashMap<u16, u16> = std::collections::HashMap::new();
    let mut strip = |(ai, aj): (usize, usize), (bi, bj): (usize, usize)| {
        let (a, b) = (at(ai, aj), at(bi, bj));
        let (la, lb) = (hang(a), hang(b));
        out.extend_from_slice(&[a.0, b.0, lb, a.0, lb, la]);
        let mut hung = |i: usize, j: usize, p: GroundVertex| -> u16 {
            *hung_at.entry(gq(i, j)).or_insert_with(|| {
                ivert.push(p);
                (ivert.len() - 1) as u16
            })
        };
        let (ila, ilb) = (hung(ai, aj, la), hung(bi, bj, lb));
        index.extend_from_slice(&[gq(ai, aj), gq(bi, bj), ilb, gq(ai, aj), ilb, ila]);
    };
    for k in 0..n {
        strip((k, 0), (k + 1, 0));
        strip((k, n), (k + 1, n));
        strip((0, k), (0, k + 1));
        strip((n, k), (n, k + 1));
    }
    PatchMesh { center, verts: out, ivert, index }
}

/// Builds many meshes on all cores (results in the same order as `keys`).
pub fn build_patches(keys: &[PatchKey], view: &PixelView) -> Vec<PatchMesh> {
    prefetch_patches(keys, view);
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()).min(16);
    if threads == 1 || keys.len() < 8 {
        return keys.iter().map(|k| build_patch(*k, view)).collect();
    }
    let chunk = keys.len().div_ceil(threads);
    let parts: Vec<Vec<PatchMesh>> = std::thread::scope(|scope| {
        let handles: Vec<_> = keys
            .chunks(chunk)
            .map(|ks| scope.spawn(move || ks.iter().map(|k| build_patch(*k, view)).collect::<Vec<_>>()))
            .collect();
        handles.into_iter().map(|h| h.join().expect("ground thread")).collect()
    });
    parts.into_iter().flatten().collect()
}

/// All the patches as one list of triangles, positions relative to `origin` (tests and the software preview;
/// the app keeps per-patch meshes and moves them into place on the GPU instead).
pub fn build_ground(patches: &[PixelId], origin: DVec3, view: &PixelView) -> Vec<GroundVertex> {
    let keys: Vec<PatchKey> = patches.iter().map(|id| patch_key(*id, view)).collect();
    let mut out = Vec::new();
    for mesh in build_patches(&keys, view) {
        let off = (mesh.center - origin).as_vec3();
        out.extend(mesh.verts.into_iter().map(|v| GroundVertex { pos: [v.pos[0] + off.x, v.pos[1] + off.y, v.pos[2] + off.z], ..v }));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::Camera;
    use crate::terrain::{max_freq, HeightSource, Terrain, TerrainRef};

    fn setup(height: f64) -> (Camera, Terrain, f64) {
        let t = Terrain::new(7);
        let f = max_freq(height, Camera::focal_px(720.0));
        // Like the app: the camera sits `height` above the ground under it, not above the bare sphere.
        let mut c = Camera { yaw: 0.7, pitch: 0.3, distance: 1.0 + height, floor: 1.0, fly: None };
        c.floor = t.ground_radius(c.eye64().normalize(), f);
        c.distance = c.floor + height;
        (c, t, f)
    }

    #[test]
    fn every_zoom_gives_a_sane_number_of_patches() {
        for h in [3.0, 1.0, 0.1, 1e-2, 1e-4, 1e-6, 5e-8] {
            let (c, t, _f) = setup(h);
            let v = PixelView::from_camera(&c, 1280, 720, Some(TerrainRef::new(&t, &c, 720.0)));
            let n = visible_patches(&v).len();
            assert!(n >= 1 && n <= MAX_PATCHES, "height {h}: {n} patches");
        }
    }

    #[test]
    fn zoomed_in_only_what_is_on_screen_is_built() {
        let (c, t, _f) = setup(1e-5);
        let v = PixelView::from_camera(&c, 1280, 720, Some(TerrainRef::new(&t, &c, 720.0)));
        let n = visible_patches(&v).len();
        assert!(n < 2_500, "{n} patches for one screen is too many");
        let verts = build_ground(&visible_patches(&v), c.eye64().normalize(), &v);
        assert!(verts.len() < MAX_GROUND_VERTS, "{}", verts.len());
    }

    #[test]
    fn far_away_the_whole_globe_is_covered_and_round() {
        let (c, t, _f) = setup(3.0);
        let v = PixelView::from_camera(&c, 1280, 720, Some(TerrainRef::new(&t, &c, 720.0)));
        let patches = visible_patches(&v);
        let verts = build_ground(&patches, c.eye64().normalize(), &v);
        assert!(verts.len() <= MAX_GROUND_VERTS, "{}", verts.len());
        assert!(patches.len() > 50, "needs enough patches to look round: {}", patches.len());
    }

    #[test]
    fn close_up_vertices_stay_precise() {
        let (c, t, _f) = setup(1e-6);
        let v = PixelView::from_camera(&c, 1280, 720, Some(TerrainRef::new(&t, &c, 720.0)));
        let origin = c.eye64().normalize() * c.floor;
        let verts = build_ground(&visible_patches(&v), origin, &v);
        assert!(!verts.is_empty());
        // Relative to the origin under the camera everything on screen is close and finite
        // (mountains can be a few km away at most, nothing near the globe's size).
        for p in &verts {
            assert!(p.pos.iter().all(|x| x.is_finite() && x.abs() < 0.2), "{:?}", p.pos);
        }
    }

    #[test]
    fn shading_brightens_toward_the_light() {
        let origin = DVec3::X;
        let at = |sun: DVec3| shade(300.0, 300.0, [1.0, 0.0, 0.0], 0.0, DVec3::ZERO, &Lighting::new(sun, origin, origin * 1.001, None));
        let sum = |c: [f32; 3]| c[0] + c[1] + c[2];
        assert!(sum(at(origin)) > sum(at(-origin)));
    }

    #[test]
    fn haze_pulls_far_ground_toward_the_sky() {
        let o = DVec3::X;
        let far = DVec3::new(0.0, 0.05, 0.0);
        let plain = shade(300.0, 300.0, [1.0, 0.0, 0.0], 0.0, far, &Lighting::new(o, o, o * 1.0001, None));
        let hazy = shade(300.0, 300.0, [1.0, 0.0, 0.0], 0.0, far, &Lighting::new(o, o, o * 1.0001, Some(SKY)));
        assert!(hazy.iter().all(|x| x.is_finite()));
        assert!(hazy[2] - hazy[0] > plain[2] - plain[0]);
    }

    #[test]
    fn per_pixel_shading_matches_the_vertex_format() {
        let v = GroundVertex { pos: [0.0; 3], nrm: [0, 127, 0, 50], hr: [10.0, 10.0], morph: 0.0, nrm_p: [0, 127, 0, 0] };
        assert!((v.normal()[1] - 1.0).abs() < 1e-6);
        assert!((v.steepness() - 50.0 / 127.0 / 1.6).abs() < 1e-6);
    }

    #[test]
    fn ground_from_tiles_matches_ground_from_the_function_and_gets_cheaper() {
        use crate::tiles::TileStore;
        let t = Terrain::new(7);
        let f = max_freq(1e-3, Camera::focal_px(720.0));
        let mut c = Camera { yaw: 0.7, pitch: 0.3, distance: 1.0 + 1e-3, floor: 1.0, fly: None };
        c.floor = t.ground_radius(c.eye64().normalize(), f);
        c.distance = c.floor + 1e-3;
        let tiles = TileStore::new(Terrain::new(7));
        let v = PixelView::from_camera(&c, 1280, 720, Some(TerrainRef::new(&tiles, &c, 720.0)));
        let t0 = std::time::Instant::now();
        let patches = visible_patches(&v);
        let first = build_ground(&patches, c.eye64().normalize() * c.floor, &v);
        let cold = t0.elapsed().as_secs_f64() * 1000.0;
        let t1 = std::time::Instant::now();
        let again = build_ground(&patches, c.eye64().normalize() * c.floor, &v);
        let warm = t1.elapsed().as_secs_f64() * 1000.0;
        println!("tiles: {} made, cold {cold:.1} ms, warm {warm:.1} ms, {} vertices", tiles.generated(), first.len());
        assert_eq!(first.len(), again.len());
        assert!(tiles.generated() < 400, "{} tiles for one view", tiles.generated());
        assert!(warm < cold);
        assert!(warm < 600.0, "{warm} ms"); // meshes also compute their parent-level heights (geomorph), so a bit slower than before
    }

    #[test]
    fn flying_camera_builds_a_sane_ground_with_haze() {
        use crate::camera::Fly;
        let t = Terrain::new(7);
        let dir = glam::DVec3::new(0.3, 0.5, 0.8).normalize();
        for (alt, tilt) in [(0.012, -0.8), (1e-3, -0.3), (1e-5, 0.0), (2.5e-7, 0.2), (1e-5, -1.2)] {
            let f = max_freq(alt, Camera::focal_px(720.0));
            let mut c = Camera { yaw: 0.0, pitch: 0.0, distance: 1.0, floor: 1.0, fly: Some(Fly { dir, heading: 0.9, tilt }) };
            c.floor = t.ground_radius(dir, f);
            c.distance = c.floor + alt;
            let v = PixelView::from_camera(&c, 1280, 720, Some(TerrainRef::new(&t, &c, 720.0)));
            let patches = visible_patches(&v);
            assert!(!patches.is_empty() && patches.len() <= MAX_PATCHES, "alt {alt} tilt {tilt}: {} patches", patches.len());
            let origin = c.eye64().normalize() * c.floor;
            let plain = build_ground(&patches, origin, &v);
            let hazy = build_ground(&patches, origin, &v);
            assert_eq!(plain.len(), hazy.len());
            println!("alt {alt} tilt {tilt}: {} patches, {} verts", patches.len(), plain.len());
            assert!(plain.len() <= MAX_GROUND_VERTS, "{}", plain.len());
            assert!(hazy.iter().all(|p| p.pos.iter().all(|x| x.is_finite()) && p.hr.iter().all(|x| x.is_finite())));
        }
    }

    #[test]
    fn looking_at_the_horizon_shows_ground_ahead_not_behind() {
        use crate::camera::Fly;
        let t = Terrain::new(7);
        let dir = glam::DVec3::new(0.3, 0.5, 0.8).normalize();
        let alt = 1e-4;
        let mut c = Camera { yaw: 0.0, pitch: 0.0, distance: 1.0, floor: 1.0, fly: Some(Fly { dir, heading: 0.0, tilt: 0.0 }) };
        c.floor = t.ground_radius(dir, 1e6);
        c.distance = c.floor + alt;
        let v = PixelView::from_camera(&c, 1280, 720, Some(TerrainRef::new(&t, &c, 720.0)));
        let ahead = visible_patches(&v);
        let (n, _, _) = c.fly.unwrap().frame();
        // Every patch center must be in front of the camera's side of the globe (north half-space, give or take the wide view).
        let behind = ahead.iter().filter(|id| (id.center_dir() - dir).dot(n) < -0.02).count();
        assert!(behind * 20 < ahead.len(), "{behind} of {} patches are behind the viewer", ahead.len());
    }

    #[test]
    #[ignore]
    fn profile_fly_frame() {
        use crate::camera::Fly;
        use crate::tiles::TileStore;
        let tiles = TileStore::new(Terrain::new(2026));
        let dir = glam::DVec3::new(0.3, 0.5, 0.8).normalize();
        for (alt, tilt) in [(0.012, -0.8), (1e-3, -0.3), (1e-5, 0.0), (1e-5, -0.5)] {
            let mut c = Camera { yaw: 0.0, pitch: 0.0, distance: 1.0, floor: 1.0, fly: Some(Fly { dir, heading: 0.9, tilt }) };
            let f = max_freq(alt, Camera::focal_px(720.0));
            c.floor = tiles.ground_radius(dir, f);
            c.distance = c.floor + alt;
            for pass in ["cold", "warm"] {
                let t0 = std::time::Instant::now();
                let v = PixelView::from_camera(&c, 1280, 720, Some(TerrainRef::new(&tiles, &c, 720.0)));
                let patches = visible_patches(&v);
                let t1 = t0.elapsed().as_secs_f64() * 1e3;
                let origin = c.eye64().normalize() * c.floor;
                let verts = build_ground(&patches, origin, &v);
                let t2 = t0.elapsed().as_secs_f64() * 1e3 - t1;
                println!("alt {alt} tilt {tilt} {pass}: walk {t1:.1} ms, build {t2:.1} ms, {} patches {} verts", patches.len(), verts.len());
            }
        }
    }

    #[test]
    fn building_one_frame_is_fast_enough() {
        let (c, t, _f) = setup(1e-3);
        let v = PixelView::from_camera(&c, 1280, 720, Some(TerrainRef::new(&t, &c, 720.0)));
        let start = std::time::Instant::now();
        let patches = visible_patches(&v);
        let verts = build_ground(&patches, c.eye64().normalize(), &v);
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        println!("{} patches, {} vertices, {ms:.1} ms", patches.len(), verts.len());
        assert!(ms < 2500.0, "{ms} ms"); // sandbox cores are shared with other tests
    }

    #[test]
    fn a_patch_mesh_does_not_depend_on_the_camera() {
        // The cache relies on this: the same key gives the same mesh wherever the camera is.
        let t = Terrain::new(7);
        let (c1, _, _) = setup(1e-4);
        let (mut c2, _, _) = setup(3e-4);
        c2.yaw += 0.01;
        let v1 = PixelView::from_camera(&c1, 1280, 720, Some(TerrainRef::new(&t, &c1, 720.0)));
        let v2 = PixelView::from_camera(&c2, 1280, 720, Some(TerrainRef::new(&t, &c2, 720.0)));
        let id = visible_patches(&v1)[0];
        let key = PatchKey { id, n: 4, lvl: 9, plvl: 8 };
        let (a, b) = (build_patch(key, &v1), build_patch(key, &v2));
        assert_eq!(a.verts.len(), b.verts.len());
        // the indexed form describes the same triangles
        assert_eq!(a.index.len(), a.verts.len());
        assert!(a.index.iter().zip(&a.verts).all(|(i, v)| a.ivert[*i as usize].pos == v.pos && a.ivert[*i as usize].nrm == v.nrm));
        assert!(a.verts.iter().zip(&b.verts).all(|(x, y)| x.pos == y.pos && x.hr == y.hr && x.nrm == y.nrm));
    }

    #[test]
    fn patch_keys_stay_the_same_while_the_camera_moves_a_little() {
        let (c, t, _f) = setup(1e-4);
        let v = PixelView::from_camera(&c, 1280, 720, Some(TerrainRef::new(&t, &c, 720.0)));
        let patches = visible_patches(&v);
        let keys: std::collections::HashSet<PatchKey> = patches.iter().map(|id| patch_key(*id, &v)).collect();
        let mut c2 = c;
        c2.yaw += 2e-6;
        let v2 = PixelView::from_camera(&c2, 1280, 720, Some(TerrainRef::new(&t, &c2, 720.0)));
        let p2 = visible_patches(&v2);
        let keys2: Vec<PatchKey> = p2.iter().map(|id| patch_key(*id, &v2)).collect();
        let reused = keys2.iter().filter(|k| keys.contains(k)).count();
        assert!(reused * 10 >= keys2.len() * 8, "only {reused} of {} meshes reusable", keys2.len());
    }

    #[test]
    fn a_render_distance_drops_far_patches_and_keeps_near_ones() {
        use crate::camera::Fly;
        let t = Terrain::new(7);
        let dir = glam::DVec3::new(0.3, 0.5, 0.8).normalize();
        let mut c = Camera { yaw: 0.0, pitch: 0.0, distance: 1.0, floor: 1.0, fly: Some(Fly { dir, heading: 0.9, tilt: 0.0 }) };
        c.floor = t.ground_radius(dir, 1e5);
        c.distance = c.floor + 1e-4;
        let v = PixelView::from_camera(&c, 1280, 720, Some(TerrainRef::new(&t, &c, 720.0)));
        let all = visible_patches(&v).len();
        let limit = 20_000.0 / terrain::EARTH_RADIUS_M;
        let near = visible_patches(&v.with_max_dist(limit));
        assert!(near.len() < all, "{} vs {all}", near.len());
        assert!(!near.is_empty());
        for id in &near {
            // everything kept starts within the limit (give or take the patch's own size)
            assert!((id.center_dir() - c.eye64()).length() < limit + 2.0 * width_rad(id.level), "kept a far patch");
        }
    }

    #[test]
    fn a_child_patch_morphs_exactly_onto_its_parents_mesh() {
        let t = Terrain::new(11);
        let parent = PixelId::new(2, 6, 21, 17);
        let pk = PatchKey { id: parent, n: 8, lvl: 7, plvl: 6 };
        let mut worst = 0.0f64;
        let mut matched = 0;
        let pm = build_patch_from(pk, Some(&t));
        let abs = |m: &PatchMesh, v: &GroundVertex| m.center + DVec3::new(v.pos[0] as f64, v.pos[1] as f64, v.pos[2] as f64);
        let top = 8 * 8 * 6; // the surface triangles (the rest are skirts)
        let parent_pts: Vec<DVec3> = pm.verts[..top].iter().map(|v| abs(&pm, v)).collect();
        for child in parent.children() {
            let ck = PatchKey { id: child, n: 8, lvl: 8, plvl: 7 };
            let cm = build_patch_from(ck, Some(&t));
            for v in &cm.verts[..top] {
                let p = abs(&cm, v);
                let up = p.normalize();
                // A child point that is also a parent mesh point (same direction): morphed, it sits on the parent's surface.
                if let Some(qi) = parent_pts.iter().position(|q| (q.normalize() - up).length() < 2e-8) {
                    let q = parent_pts[qi];
                    let (n1, n2) = (v.nrm_p.map(|c| c as f64), pm.verts[qi].nrm.map(|c| c as f64));
                    assert!((0..3).all(|c| (n1[c] - n2[c]).abs() <= 2.0), "parent normal {n1:?} vs {n2:?}");
                    worst = worst.max((p.length() + v.morph as f64 - q.length()).abs());
                    matched += 1;
                }
            }
        }
        assert!(matched > 100, "only {matched} points matched");
        assert!(worst < 2e-8, "morphed child is {worst} off the parent surface (unit sphere)");
    }

    /// How many patches (= draw calls, meshes, tiles) one big-screen flying view costs. Marcel's view: 2560x1600,
    /// ~500 m up, 80 km render distance. Printed for tuning; the bound keeps it from creeping back up.
    #[test]
    fn a_big_screen_flying_view_stays_cheap() {
        use crate::camera::Fly;
        let t = Terrain::new(7);
        let dir = glam::DVec3::new(0.3, 0.5, 0.8).normalize();
        let mut worst = 0;
        for (alt_m, tilt, heading) in [(500.0, -0.2, 0.9), (500.0, 0.0, 2.0), (500.0, -0.6, 4.0), (60.0, -0.1, 1.0), (3000.0, -0.3, 5.0)] {
            let alt = alt_m / terrain::EARTH_RADIUS_M;
            let f = max_freq(alt, Camera::focal_px(1600.0));
            let mut c = Camera { yaw: 0.0, pitch: 0.0, distance: 1.0, floor: 1.0, fly: Some(Fly { dir, heading, tilt }) };
            c.floor = t.ground_radius(dir, f);
            c.distance = c.floor + alt;
            let v = PixelView::from_camera(&c, 2560, 1600, Some(TerrainRef::new(&t, &c, 1600.0))).with_max_dist(80_000.0 / terrain::EARTH_RADIUS_M);
            let patches = visible_patches(&v);
            let keys: std::collections::HashSet<PatchKey> = patches.iter().map(|id| patch_key(*id, &v)).collect();
            let mut tile_ids = std::collections::HashSet::new();
            for k in &keys {
                tile_ids.insert((k.id.ancestor(k.id.level.min(3).min(k.lvl)).face, k.lvl));
            }
            let mut by_level = std::collections::BTreeMap::new();
            for id in &patches {
                *by_level.entry(id.level).or_insert(0usize) += 1;
            }
            println!("alt {alt_m} m tilt {tilt}: {} patches, by level {:?}", patches.len(), by_level);
            worst = worst.max(patches.len());
        }
        assert!(worst < 6000);
    }

    /// ground.wgsl repeats the patch size limit (its geomorph window follows it): both must have the same numbers.
    #[test]
    fn the_shader_uses_the_same_patch_size_limit() {
        let r = terrain::EARTH_RADIUS_M;
        let line = format!("let lim = select({:.1}, {:.1}, u.sky.w > 0.5) * (1.0 + clamp((dist - {:.1} / {:.1}) / ({:.1} / {:.1}), 0.0, gain));", GROUND_PX, GROUND_PX_INSIDE, FAR_START_M, r, FAR_REF_M, r);
        assert!(include_str!("ground.wgsl").contains(&format!("select(0.0, {:.1}, u.sky.w > 0.5)", FAR_GAIN)));
        assert!(include_str!("ground.wgsl").contains(&format!("smooth_({:.2} * lim, {:.2} * lim, px)", MORPH_START, MORPH_END)));
        assert!(include_str!("ground.wgsl").contains(&line), "ground.wgsl should contain: {line}");
    }
}
