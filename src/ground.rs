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

/// One corner of the ground mesh, 24 bytes: where it is (relative to the frame's origin point), which way the
/// surface faces (xyz, 1/127 steps) and how steep it really is (w), and the true height plus the coarse
/// "range" height. Color, shore, snow, light and haze are worked out per pixel from these (ground.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct GroundVertex {
    pub pos: [f32; 3],
    pub nrm: [i8; 4],
    /// Height in meters (negative = sea floor), range height in meters.
    pub hr: [f32; 2],
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

/// Patches are split until they are this many screen pixels wide or less.
pub const GROUND_PX: f64 = 56.0;
/// Target spacing of mesh points on screen, in pixels.
const SPACING_PX: f64 = 8.0;
/// Safety cap on patches per frame.
pub const MAX_PATCHES: usize = 6_000;
/// Above this many patches the mesh spacing grows (see build_ground).
const CROWD_PATCHES: f64 = 900.0;
/// Longest edge (radians) of one flat piece inside a patch (keeps big patches round).
const SEGMENT_RAD: f64 = 0.04;
/// Most pieces per patch edge.
const MAX_SEGMENTS: usize = 12;
/// Shading uses heights this many times taller than real, so slopes read as relief (the shape itself is real scale).
const SHADE_EXAGGERATION: f64 = 7.0;
/// Vertices one frame can hold.
pub const MAX_GROUND_VERTS: usize = 800_000;
/// Color of the sky, and what far ground fades into (inside scene).
pub const SKY: [f32; 3] = [0.50, 0.68, 0.92];
/// Most the haze mixes in: the far ground never disappears completely, so mountains still show over the horizon.
const HAZE_MAX: f32 = 0.8;

/// Small enough to stop splitting: about GROUND_PX wide on screen, or, seen at a grazing angle, a thin
/// sliver (a patch far out toward the horizon can be wide but only a few pixels tall; splitting
/// those just piles up thousands of invisible triangles).
fn small_enough(id: PixelId, view: &PixelView) -> bool {
    if screen_px(id, view) <= GROUND_PX {
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
    bw.max(bh) <= 4.0 * GROUND_PX && bw * bh <= 0.3 * GROUND_PX * GROUND_PX
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
}

/// One patch's triangles. Positions are relative to `center` (a point on the unit sphere), so the mesh stays
/// precise at any zoom and is moved into place per frame with one offset.
pub struct PatchMesh {
    pub center: DVec3,
    pub verts: Vec<GroundVertex>,
}

/// Allowed numbers of pieces per patch edge: few sizes, so meshes are reused as the view changes a little.
const N_STEPS: [usize; 7] = [1, 2, 3, 4, 6, 8, 12];

/// Spread factor for busy views (flying low, looking toward the horizon): more patches, so coarser spacing.
pub fn crowd(patch_count: usize) -> f64 {
    (patch_count as f64 / CROWD_PATCHES).max(1.0).sqrt()
}

/// The mesh a patch should have for this view.
pub fn patch_key(id: PixelId, view: &PixelView, crowd: f64) -> PatchKey {
    let w = width_rad(id.level);
    let roundness = (w / SEGMENT_RAD).ceil() as usize;
    let max_n = ((MAX_SEGMENTS as f64 / crowd).ceil() as usize).max(3).max(roundness).min(MAX_SEGMENTS);
    let want = ((screen_px(id, view) / (SPACING_PX * crowd)).ceil() as usize).max(roundness).clamp(1, max_n);
    let n = N_STEPS.iter().copied().find(|q| *q >= want).unwrap_or(MAX_SEGMENTS).min(MAX_SEGMENTS);
    // Height tiles about as fine as the mesh (the cubic sampling fills in between), never finer than the scene allows.
    let lvl = match view.terrain {
        Some(t) => {
            let cap = tiles::level_for(t.max_freq).floor();
            let want = (tiles::spacing_rad(0) / (w / n as f64)).log2().ceil().clamp(0.0, tiles::MAX_TILE_LEVEL as f64);
            want.min(cap) as u8
        }
        None => 0,
    };
    PatchKey { id, n: n as u8, lvl }
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
            let (h, r, rh) = match view.terrain {
                Some(t) => {
                    let h = t.terrain.height_m(u, freq);
                    // The range tint is only needed where a vertex is drawn, not in the padding ring.
                    let rh = if interior { t.terrain.height_m(u, terrain::RANGE_FREQ) } else { 0.0 };
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
            },
            up,
        )
    };
    // Each grid point is made once (4 quads share it).
    let grid: Vec<(GroundVertex, DVec3)> = (0..=n).flat_map(|j| (0..=n).map(move |i| (i, j))).map(|(i, j)| vertex(i, j)).collect();
    let at = |i: usize, j: usize| grid[j * (n + 1) + i];
    let radius = |i: usize, j: usize| pos[(j + 1) * m + i + 1].length();
    let mut out = Vec::with_capacity(n * n * 6 + n * 24);
    for i in 0..n {
        for j in 0..n {
            let (a, b, c, d) = (at(i, j).0, at(i + 1, j).0, at(i + 1, j + 1).0, at(i, j + 1).0);
            // Cut the quad along the diagonal whose ends are closest in height: the cut then follows the
            // ridge or valley instead of making a zig-zag across it.
            let d_ac = (radius(i, j) - radius(i + 1, j + 1)).abs();
            let d_bd = (radius(i + 1, j) - radius(i, j + 1)).abs();
            if d_ac <= d_bd {
                out.extend_from_slice(&[a, b, c, a, c, d]);
            } else {
                out.extend_from_slice(&[a, b, d, b, c, d]);
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
    let mut strip = |a: (GroundVertex, DVec3), b: (GroundVertex, DVec3)| {
        let (la, lb) = (hang(a), hang(b));
        out.extend_from_slice(&[a.0, b.0, lb, a.0, lb, la]);
    };
    for k in 0..n {
        strip(at(k, 0), at(k + 1, 0));
        strip(at(k, n), at(k + 1, n));
        strip(at(0, k), at(0, k + 1));
        strip(at(n, k), at(n, k + 1));
    }
    PatchMesh { center, verts: out }
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
    let c = crowd(patches.len());
    let keys: Vec<PatchKey> = patches.iter().map(|id| patch_key(*id, view, c)).collect();
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
        let v = GroundVertex { pos: [0.0; 3], nrm: [0, 127, 0, 50], hr: [10.0, 10.0] };
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
        assert!(warm < 60.0, "{warm} ms");
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
        assert!(ms < 400.0, "{ms} ms");
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
        let key = PatchKey { id, n: 4, lvl: 9 };
        let (a, b) = (build_patch(key, &v1), build_patch(key, &v2));
        assert_eq!(a.verts.len(), b.verts.len());
        assert!(a.verts.iter().zip(&b.verts).all(|(x, y)| x.pos == y.pos && x.hr == y.hr && x.nrm == y.nrm));
    }

    #[test]
    fn patch_keys_stay_the_same_while_the_camera_moves_a_little() {
        let (c, t, _f) = setup(1e-4);
        let v = PixelView::from_camera(&c, 1280, 720, Some(TerrainRef::new(&t, &c, 720.0)));
        let patches = visible_patches(&v);
        let cr = crowd(patches.len());
        let keys: std::collections::HashSet<PatchKey> = patches.iter().map(|id| patch_key(*id, &v, cr)).collect();
        let mut c2 = c;
        c2.yaw += 2e-6;
        let v2 = PixelView::from_camera(&c2, 1280, 720, Some(TerrainRef::new(&t, &c2, 720.0)));
        let p2 = visible_patches(&v2);
        let cr2 = crowd(p2.len());
        let keys2: Vec<PatchKey> = p2.iter().map(|id| patch_key(*id, &v2, cr2)).collect();
        let reused = keys2.iter().filter(|k| keys.contains(k)).count();
        assert!(reused * 10 >= keys2.len() * 8, "only {reused} of {} meshes reusable", keys2.len());
    }
}
