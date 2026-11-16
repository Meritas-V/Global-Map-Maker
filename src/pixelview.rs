//! Seeing and using the pixel layer: which pixels to draw for the current camera, how they turn
//! into triangles, which pixel is under the mouse, and the brush shapes.

use crate::camera::Camera;
use crate::hexpaint;
use crate::hexview::{pick_ray, visible_radius};
use crate::pixelpaint::{PixelId, PixelStore, MAX_LEVEL};
use crate::terrain::TerrainRef;
use bytemuck::{Pod, Zeroable};
use glam::{DMat4, DVec3};
use std::f64::consts::FRAC_PI_2;

/// One corner of a pixel quad: position (relative to the camera-side origin) and rgba.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct PixelVertex {
    pub pos: [f32; 3],
    pub color: [f32; 4],
    /// How far the point is on the visible side of the horizon (dot(point, eye) - 1, worked out in f64).
    /// Negative = behind the horizon, the shader drops it. Always positive on terrain (depth test instead).
    pub vis: f32,
}

/// Pixels smaller than this on screen are merged into their parent's average color.
pub const LEAF_PX: f64 = 3.0;
/// Most quads one frame may produce (further merging kicks in beyond this).
pub const MAX_QUADS: usize = 120_000;
/// With terrain, a painted pixel bigger than this on screen is cut into smaller pieces that follow the relief.
pub const PIECE_PX: f64 = 56.0;
/// Painted pixels sit this many screen pixels above the terrain, so the depth test never hides them.
pub const LIFT_PX: f64 = 4.0;

pub struct Quad {
    pub id: PixelId,
    pub color: [f32; 3],
    pub alpha: f32,
}

/// What the walk needs to know about the camera.
pub struct PixelView<'a> {
    pub eye: DVec3,
    /// Angle (radians) of the visible cap of the globe (with room for mountains over the horizon).
    pub radius: f64,
    pub focal_px: f64,
    /// Globe coordinates -> camera space.
    pub view: DMat4,
    pub tan_y: f64,
    pub tan_x: f64,
    pub terrain: Option<TerrainRef<'a>>,
}

impl<'a> PixelView<'a> {
    pub fn from_camera(camera: &Camera, width: u32, height: u32, terrain: Option<TerrainRef<'a>>) -> PixelView<'a> {
        let (w, h) = (width.max(1) as f64, height.max(1) as f64);
        let tan_y = (Camera::fov_y() * 0.5).tan();
        // Mountains can be seen over the horizon of the bare sphere: about acos(1 / (1 + 8.8 km / R)) more.
        let margin = if terrain.is_some() { 0.056 } else { 0.0 };
        PixelView {
            eye: camera.eye64(),
            radius: visible_radius(camera, w / h) + margin,
            focal_px: Camera::focal_px(h),
            view: camera.view64(),
            tan_y,
            tan_x: tan_y * w / h,
            terrain,
        }
    }
}

/// Angular width of a pixel at the middle of its face.
pub fn width_rad(level: u8) -> f64 {
    FRAC_PI_2 / PixelId::size(level)
}

/// Is any of this pixel possibly on screen? (Horizon cap, then the view frustum around its true position.)
pub fn is_visible(id: PixelId, view: &PixelView) -> bool {
    let c = id.center_dir();
    let w = width_rad(id.level);
    let ang = c.dot(view.eye.normalize()).clamp(-1.0, 1.0).acos();
    if ang > view.radius + 1.6 * w {
        return false;
    }
    let r = view.terrain.map_or(1.0, |t| t.radius_peek(c));
    let v = view.view.transform_point3(c * r);
    // Size of the sphere that surely holds the pixel: its corners plus how far the relief can lift it.
    let relief = if view.terrain.is_some() { (0.7 * w).min(1.5e-3) } else { 0.0 };
    let rho = 1.1 * w + relief;
    let z = -v.z;
    if z + rho <= 0.0 {
        return false;
    }
    v.x.abs() <= z * view.tan_x + rho * (1.0 + view.tan_x * view.tan_x).sqrt()
        && v.y.abs() <= z * view.tan_y + rho * (1.0 + view.tan_y * view.tan_y).sqrt()
}

pub fn screen_px(id: PixelId, view: &PixelView) -> f64 {
    let dist = (view.eye - id.center_dir()).length().max(1e-12);
    width_rad(id.level) * view.focal_px / dist
}

/// Everything painted that is on screen, as quads. Painted pixels come out at their own size;
/// anything that would be smaller than LEAF_PX is merged into one quad with the average color
/// and a see-through amount equal to how much of it is bare.
pub fn visible_pixels(store: &PixelStore, view: &PixelView) -> Vec<Quad> {
    let mut out: Vec<Quad> = Vec::new();
    let mut stack: Vec<(PixelId, hexpaint::Paint)> = (0..6).map(|f| (PixelId::new(f, 0, 0, 0), 0)).collect();
    while let Some((id, under)) = stack.pop() {
        if !is_visible(id, view) {
            continue;
        }
        let eff = store.own(id).unwrap_or(under);
        if !store.has_finer(id) {
            if eff != 0 {
                out.push(Quad { id, color: hexpaint::color(eff), alpha: 1.0 });
            }
            continue;
        }
        let merge = screen_px(id, view) <= LEAF_PX || id.level == MAX_LEVEL || out.len() + stack.len() * 4 > MAX_QUADS;
        if !merge {
            for c in id.children() {
                stack.push((c, eff));
            }
            continue;
        }
        let comp = store.composition(id, eff);
        let alpha: f64 = comp.iter().skip(1).map(|c| c.max(0.0)).sum();
        if alpha < 0.003 {
            continue;
        }
        let mut col = [0.0f64; 3];
        for (v, share) in comp.iter().enumerate().skip(1) {
            let c = hexpaint::color(v as u8);
            for k in 0..3 {
                col[k] += c[k] as f64 * share.max(0.0) / alpha;
            }
        }
        out.push(Quad { id, color: [col[0] as f32, col[1] as f32, col[2] as f32], alpha: alpha.min(1.0) as f32 });
    }
    out
}

/// On terrain, cut big pixels into on-screen pieces of at most PIECE_PX so they can follow the relief.
pub fn split_for_relief(quads: Vec<Quad>, view: &PixelView) -> Vec<Quad> {
    let mut out = Vec::with_capacity(quads.len());
    let mut stack = quads;
    while let Some(q) = stack.pop() {
        if screen_px(q.id, view) <= PIECE_PX || q.id.level >= MAX_LEVEL || out.len() + stack.len() > MAX_QUADS {
            out.push(q);
            continue;
        }
        for c in q.id.children() {
            if is_visible(c, view) {
                stack.push(Quad { id: c, color: q.color, alpha: q.alpha });
            }
        }
    }
    out
}

/// Where painted points sit: on the bare unit sphere, or on the terrain a little above it.
#[derive(Clone, Copy)]
pub struct Surface<'a> {
    pub terrain: Option<TerrainRef<'a>>,
    /// How far above the ground (in screen pixels) to float, so the depth test never swallows paint.
    pub lift_px: f64,
    pub focal_px: f64,
}

impl<'a> Surface<'a> {
    pub fn flat() -> Surface<'static> {
        Surface { terrain: None, lift_px: 0.0, focal_px: 1.0 }
    }

    fn lift_at(&self, p: DVec3, eye: DVec3) -> f64 {
        self.lift_px * (p - eye).length() / self.focal_px
    }

    fn point(&self, u: DVec3, eye: DVec3, radius: f64) -> DVec3 {
        let p = u * radius;
        if self.terrain.is_some() {
            p + u * self.lift_at(p, eye)
        } else {
            p
        }
    }
}

/// Triangles for the quads. Big pixels are cut into a few pieces so they follow the curve of the
/// globe (and the relief). Without terrain the pixels are drawn without a depth test (they would
/// lose a depth fight with flat hexes), so hiding what is behind the horizon is done by `vis`.
pub fn build_vertices(quads: &[Quad], origin: DVec3, eye: DVec3, surface: &Surface) -> Vec<PixelVertex> {
    let mut out = Vec::with_capacity(quads.len() * 6);
    for q in quads {
        let w = width_rad(q.id.level);
        let color = [q.color[0], q.color[1], q.color[2], q.alpha];
        let (n, center_only) = match surface.terrain {
            Some(t) => {
                // How many finest-detail wavelengths wide is this piece?
                let waves = w * t.freq_at(q.id.center_dir());
                if waves < 0.2 {
                    (1, true)
                } else {
                    (((waves / 0.18).ceil() as usize).clamp(1, 10).max((w / 0.04).ceil() as usize).min(10), false)
                }
            }
            None => (((w / 0.03).ceil() as usize).clamp(1, 8), false),
        };
        let center_radius = match (surface.terrain, center_only) {
            (Some(t), true) => Some(t.radius(q.id.center_dir())),
            _ => None,
        };
        let at = |i: usize, j: usize| {
            let u = q.id.dir_at(i as f64 / n as f64, j as f64 / n as f64);
            let r = match (surface.terrain, center_radius) {
                (_, Some(r)) => r,
                (Some(t), None) => t.radius(u),
                (None, None) => 1.0,
            };
            let p = surface.point(u, eye, r);
            let vis = if surface.terrain.is_some() { 1.0 } else { (u.dot(eye) - 1.0) as f32 };
            PixelVertex { pos: (p - origin).as_vec3().to_array(), color, vis }
        };
        for i in 0..n {
            for j in 0..n {
                let (a, b, c, d) = (at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1));
                out.extend_from_slice(&[a, b, c, a, c, d]);
            }
        }
    }
    out
}

/// Translucent white quads, for showing where the brush would paint.
pub fn highlight_vertices(ids: &[PixelId], origin: DVec3, eye: DVec3, surface: &Surface, view: &PixelView) -> Vec<PixelVertex> {
    let mut quads: Vec<Quad> = ids.iter().map(|id| Quad { id: *id, color: [1.0, 1.0, 1.0], alpha: 0.45 }).collect();
    if surface.terrain.is_some() {
        // Big pixels (seen from high up) are cut into pieces that follow the relief, like the paint is.
        // Without this a far-away highlight is one flat quad stretched across a valley.
        quads = split_for_relief(quads, view);
    }
    let lifted = Surface { lift_px: surface.lift_px * 1.6, ..*surface };
    build_vertices(&quads, origin, eye, &lifted)
}

/// The pixel level whose pixels are 2^-k of a hex wide (`hex_width` in radians).
pub fn pixel_level(hex_width: f64, k: u8) -> u8 {
    let l0 = (FRAC_PI_2 / hex_width).log2().round();
    (l0 + k as f64).clamp(0.0, MAX_LEVEL as f64) as u8
}

/// The pixel at `level` under a screen position (on the terrain if there is one).
pub fn pick_pixel(
    camera: &Camera,
    width: u32,
    height: u32,
    x: f64,
    y: f64,
    level: u8,
    terrain: Option<TerrainRef>,
) -> Option<PixelId> {
    let (origin, dir) = pick_ray(camera, width, height, x, y);
    let p = match terrain {
        Some(t) => t.pick(origin, dir)?,
        None => {
            let b = origin.dot(dir);
            let disc = b * b - (origin.dot(origin) - 1.0);
            if disc < 0.0 {
                return None;
            }
            let t = -b - disc.sqrt();
            if t <= 0.0 {
                return None;
            }
            origin + dir * t
        }
    };
    Some(PixelId::from_dir(p, level))
}

/// A round brush `r` pixels out from the middle one (r = 0 is a single pixel). Stays on the face.
pub fn disk(center: PixelId, r: u32) -> Vec<PixelId> {
    let n = 1i64 << center.level;
    let r = r as i64;
    let mut out = Vec::new();
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy > r * r + r {
                continue;
            }
            let (x, y) = (center.x as i64 + dx, center.y as i64 + dy);
            if (0..n).contains(&x) && (0..n).contains(&y) {
                out.push(PixelId::new(center.face, center.level, x as u32, y as u32));
            }
        }
    }
    out
}

/// A line of pixels from a to b (both included), following the shortest way over the globe, so it
/// has no gaps even when a and b sit on different cube faces or the mouse moved far between events.
pub fn line(a: PixelId, b: PixelId) -> Vec<PixelId> {
    let level = b.level;
    let (da, db) = (a.center_dir(), b.center_dir());
    let angle = da.dot(db).clamp(-1.0, 1.0).acos();
    if angle < 1e-12 || angle > 3.0 {
        return vec![b];
    }
    let steps = ((angle / (0.7 * width_rad(level))).ceil() as usize).clamp(1, 20_000);
    let mut out: Vec<PixelId> = Vec::with_capacity(steps + 1);
    for k in 0..=steps {
        let t = k as f64 / steps as f64;
        let p = (da * ((1.0 - t) * angle).sin() + db * (t * angle).sin()) / angle.sin();
        let id = if k == 0 { PixelId::from_dir(da, level) } else if k == steps { b } else { PixelId::from_dir(p, level) };
        if out.last() != Some(&id) {
            out.push(id);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hexpaint::RED;
    use crate::hexview::cell_width;

    fn cam(height: f64) -> Camera {
        Camera { yaw: 0.3, pitch: 0.4, distance: 1.0 + height, floor: 1.0, fly: None }
    }

    #[test]
    fn pixel_size_follows_the_brush_step_and_the_hex_level() {
        let w12 = cell_width(12);
        let l4 = pixel_level(w12, 4);
        // 4 steps finer than the hex: pixel is about 1/16 of a hex wide.
        let ratio = w12 / width_rad(l4);
        assert!((ratio - 16.0).abs() < 8.0, "{ratio}");
        assert!(pixel_level(w12, 9) > pixel_level(w12, 2));
        assert!(pixel_level(cell_width(3), 4) < l4, "coarser hexes -> coarser pixels");
        assert!(pixel_level(cell_width(12), 10) <= MAX_LEVEL);
    }

    #[test]
    fn a_painted_pixel_under_the_camera_is_drawn_full_color() {
        let c = cam(1e-5); // close enough that a level-24 pixel is about 10 px on screen
        let level = 24;
        let id = PixelId::from_dir(c.eye64().normalize(), level);
        let mut s = PixelStore::default();
        s.begin_stroke();
        s.paint(id, RED);
        s.end_stroke();
        let quads = visible_pixels(&s, &PixelView::from_camera(&c, 1280, 720, None));
        assert_eq!(quads.len(), 1);
        assert_eq!(quads[0].id, id);
        assert_eq!(quads[0].alpha, 1.0);
        assert_eq!(quads[0].color, hexpaint::color(RED));
    }

    #[test]
    fn far_away_the_same_pixel_fades_into_a_tiny_average() {
        let c = cam(2.0);
        let id = PixelId::from_dir(c.eye64().normalize(), 24);
        let mut s = PixelStore::default();
        s.begin_stroke();
        s.paint(id, RED);
        s.end_stroke();
        let quads = visible_pixels(&s, &PixelView::from_camera(&c, 1280, 720, None));
        assert!(quads.len() <= 2, "{}", quads.len());
        if let Some(q) = quads.first() {
            assert!(q.alpha < 0.01 && q.alpha > 0.0, "{}", q.alpha);
            for k in 0..3 {
                assert!((q.color[k] - hexpaint::color(RED)[k]).abs() < 1e-3);
            }
        }
    }

    #[test]
    fn a_hole_in_a_big_paint_shows_up_as_a_gap() {
        let c = cam(1e-3);
        let eye = c.eye64().normalize();
        let big = PixelId::from_dir(eye, 18);
        let hole = PixelId::from_dir(eye, 24);
        let mut s = PixelStore::default();
        s.begin_stroke();
        s.paint(big, RED);
        s.paint(hole, 0);
        s.end_stroke();
        let quads = visible_pixels(&s, &PixelView::from_camera(&c, 1280, 720, None));
        assert!(quads.len() > 3, "the big pixel must be split around the hole");
        assert!(quads.iter().all(|q| q.id != hole), "the hole itself is not drawn");
        // Total painted area = big minus hole.
        let area: f64 = quads.iter().map(|q| q.alpha as f64 * 0.25f64.powi(q.id.level as i32)).sum();
        let want = 0.25f64.powi(18) - 0.25f64.powi(24);
        assert!((area - want).abs() / want < 1e-6, "{area} vs {want}");
    }

    #[test]
    fn vertices_are_triangles_and_stay_near_the_origin() {
        let c = cam(1e-4);
        let id = PixelId::from_dir(c.eye64().normalize(), 24);
        let q = [Quad { id, color: [1.0, 0.0, 0.0], alpha: 1.0 }];
        let v = build_vertices(&q, c.eye64().normalize(), c.eye64(), &Surface::flat());
        assert_eq!(v.len(), 6);
        for p in &v {
            assert!(p.pos.iter().all(|x| x.is_finite() && x.abs() < 1e-5), "{:?}", p.pos);
        }
        // A pixel of this size is about 1.5e-7 wide; the quad should have roughly that extent.
        let xs: Vec<f32> = v.iter().map(|p| p.pos[0]).collect();
        assert!(xs.iter().cloned().fold(f32::MIN, f32::max) - xs.iter().cloned().fold(f32::MAX, f32::min) < 1e-6);
    }

    #[test]
    fn horizon_flag_is_positive_in_front_and_negative_behind() {
        let c = cam(0.5);
        let eye = c.eye64();
        let front = PixelId::from_dir(eye.normalize(), 10);
        let back = PixelId::from_dir(-eye.normalize(), 10);
        let q = |id| [Quad { id, color: [1.0; 3], alpha: 1.0 }];
        assert!(build_vertices(&q(front), eye.normalize(), eye, &Surface::flat()).iter().all(|v| v.vis > 0.0));
        assert!(build_vertices(&q(back), eye.normalize(), eye, &Surface::flat()).iter().all(|v| v.vis < 0.0));
    }

    #[test]
    fn big_pixels_get_cut_up_to_follow_the_curve() {
        let q = [Quad { id: PixelId::new(0, 3, 2, 2), color: [0.5; 3], alpha: 1.0 }];
        assert!(build_vertices(&q, DVec3::ZERO, DVec3::new(0.0, 0.0, 3.0), &Surface::flat()).len() > 6);
    }

    #[test]
    fn picking_the_middle_of_the_screen_finds_the_pixel_under_the_camera() {
        let c = cam(1e-4);
        let got = pick_pixel(&c, 1280, 720, 640.0, 360.0, 24, None).unwrap();
        let want = PixelId::from_dir(c.eye64().normalize(), 24);
        assert_eq!(got.face, want.face);
        assert!((got.x as i64 - want.x as i64).abs() <= 2 && (got.y as i64 - want.y as i64).abs() <= 2);
    }

    #[test]
    fn brush_disk_sizes_and_edges() {
        let mid = PixelId::new(1, 10, 500, 500);
        assert_eq!(disk(mid, 0).len(), 1);
        assert_eq!(disk(mid, 1).len(), 9);
        assert!(disk(mid, 4).len() > 45 && disk(mid, 4).len() < 81);
        let corner = PixelId::new(1, 10, 0, 0);
        assert!(disk(corner, 2).iter().all(|p| p.x < 1024 && p.y < 1024));
        assert!(disk(corner, 2).len() < disk(mid, 2).len());
    }

    #[test]
    fn line_has_no_gaps_and_both_ends() {
        let a = PixelId::new(2, 12, 100, 100);
        let b = PixelId::new(2, 12, 130, 112);
        let l = line(a, b);
        assert_eq!(l.first(), Some(&a));
        assert_eq!(l.last(), Some(&b));
        assert!(l.len() >= 31);
        for w in l.windows(2) {
            assert!((w[0].x as i64 - w[1].x as i64).abs() <= 1 && (w[0].y as i64 - w[1].y as i64).abs() <= 1, "{:?}", w);
        }
    }

    #[test]
    fn line_across_cube_faces_is_connected() {
        let a = PixelId::from_dir(DVec3::new(0.9, 0.1, 0.1).normalize(), 12);
        let b = PixelId::from_dir(DVec3::new(0.1, 0.9, 0.2).normalize(), 12);
        assert_ne!(a.face, b.face);
        let l = line(a, b);
        assert!(l.len() > 100);
        for w in l.windows(2) {
            let d = w[0].center_dir().dot(w[1].center_dir()).clamp(-1.0, 1.0).acos();
            assert!(d < 2.5 * width_rad(12), "gap {d}");
        }
    }
}
