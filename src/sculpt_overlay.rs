//! A stand-in for the ground while a stroke is going on.
//!
//! Remaking meshes for every frame of a stroke is too much work, so while the button is down the stroke is only
//! *shown*: its running total is kept in a small height grid around the brush (`Window`), which the ground's vertex
//! shader adds to the meshes that are already there (moving the points, tilting the normals). When the stroke
//! ends the real meshes are remade in the background and swapped in all at once, and the window is dropped.

use crate::terrain::EARTH_RADIUS_M;
use glam::DVec3;

/// Grid points along one side of a window.
pub const N: usize = 512;
/// Half the side of a window, in brush radii.
pub const HALF_R: f64 = 8.0;

pub struct Window {
    pub center: DVec3,
    t1: DVec3,
    t2: DVec3,
    pub radius_m: f64,
    pub half_m: f64,
    /// Tint what the window shows (raise / lower do, smooth and flatten keep the natural colours).
    pub tint: bool,
    /// The edit version before this window's first change: meshes made at or before it show none of it.
    pub v_first: u64,
    /// Meters added to the ground, row by row.
    pub data: Vec<f32>,
    /// Grid points changed since the last upload: x0, y0, x1, y1 (end exclusive).
    pub dirty: Option<(usize, usize, usize, usize)>,
    /// Everything the stroke touched so far, same form.
    touched: Option<(usize, usize, usize, usize)>,
}

fn tangents(c: DVec3) -> (DVec3, DVec3) {
    let helper = if c.z.abs() < 0.9 { DVec3::Z } else { DVec3::X };
    let t1 = c.cross(helper).normalize();
    (t1, c.cross(t1))
}

fn union(a: Option<(usize, usize, usize, usize)>, b: (usize, usize, usize, usize)) -> Option<(usize, usize, usize, usize)> {
    Some(match a {
        None => b,
        Some(a) => (a.0.min(b.0), a.1.min(b.1), a.2.max(b.2), a.3.max(b.3)),
    })
}

impl Window {
    pub fn new(center: DVec3, radius_m: f64) -> Window {
        let (t1, t2) = tangents(center);
        // The whole grid goes to the GPU the first time.
        Window { center, t1, t2, radius_m, half_m: HALF_R * radius_m, tint: true, v_first: 0, data: vec![0.0; N * N], dirty: Some((0, 0, N, N)), touched: None }
    }

    fn texel_m(&self) -> f64 {
        2.0 * self.half_m / N as f64
    }

    pub fn ground_dist(&self, p: DVec3) -> f64 {
        self.center.dot(p).clamp(-1.0, 1.0).acos() * EARTH_RADIUS_M
    }

    /// Can a brush of this size at `p` still be drawn into this window (with room around it)?
    pub fn fits(&self, p: DVec3, radius_m: f64) -> bool {
        radius_m <= self.radius_m * 1.25 && radius_m >= self.radius_m * 0.5 && self.ground_dist(p) + 1.3 * radius_m <= self.half_m
    }

    /// The direction of grid point (i, j).
    fn dir(&self, i: usize, j: usize) -> DVec3 {
        let x = (i as f64 + 0.5) * self.texel_m() - self.half_m;
        let y = (j as f64 + 0.5) * self.texel_m() - self.half_m;
        (self.center + (self.t1 * x + self.t2 * y) / EARTH_RADIUS_M).normalize()
    }

    /// Adds `amount(direction, distance from p)` (meters) to every grid point within `radius_m` of `p`. The amount is
    /// worked out only at every `STRIDE`th grid point and filled in between (the brush is smooth, and tools that read
    /// the ground make this the expensive part).
    pub fn add(&mut self, p: DVec3, radius_m: f64, amount: impl Fn(DVec3, f64) -> f64) {
        const STRIDE: usize = 3;
        let tm = self.texel_m();
        let off = p - self.center;
        let (cx, cy) = (off.dot(self.t1) * EARTH_RADIUS_M, off.dot(self.t2) * EARTH_RADIUS_M);
        let to_i = |v: f64| (((v + self.half_m) / tm).floor() as i64).clamp(0, N as i64 - 1) as usize;
        let (i0, i1) = (to_i(cx - radius_m - tm), to_i(cx + radius_m + tm));
        let (j0, j1) = (to_i(cy - radius_m - tm), to_i(cy + radius_m + tm));
        // Nodes on a lattice that starts at a multiple of STRIDE, covering the box.
        let (ni0, nj0) = (i0 / STRIDE, j0 / STRIDE);
        let (nw, nh) = (i1 / STRIDE + 2 - ni0, j1 / STRIDE + 2 - nj0);
        let mut nodes = vec![0.0f64; nw * nh];
        for nj in 0..nh {
            for ni in 0..nw {
                let (i, j) = (((ni0 + ni) * STRIDE).min(N - 1), ((nj0 + nj) * STRIDE).min(N - 1));
                let d = self.dir(i, j);
                let dist = p.dot(d).clamp(-1.0, 1.0).acos() * EARTH_RADIUS_M;
                if dist < radius_m {
                    let a = amount(d, dist);
                    if a.is_finite() {
                        nodes[nj * nw + ni] = a;
                    }
                }
            }
        }
        for j in j0..=j1 {
            for i in i0..=i1 {
                let (fx, fy) = ((i as f64 / STRIDE as f64) - ni0 as f64, (j as f64 / STRIDE as f64) - nj0 as f64);
                let (x0, y0) = (fx.floor() as usize, fy.floor() as usize);
                let (tx, ty) = (fx - x0 as f64, fy - y0 as f64);
                let n = |x: usize, y: usize| nodes[(y.min(nh - 1)) * nw + x.min(nw - 1)];
                let a = (n(x0, y0) * (1.0 - tx) + n(x0 + 1, y0) * tx) * (1.0 - ty) + (n(x0, y0 + 1) * (1.0 - tx) + n(x0 + 1, y0 + 1) * tx) * ty;
                if a != 0.0 {
                    self.data[j * N + i] += a as f32;
                }
            }
        }
        let rect = (i0, j0, i1 + 1, j1 + 1);
        self.dirty = union(self.dirty, rect);
        self.touched = union(self.touched, rect);
    }

    #[allow(dead_code)]
    fn add_exact(&mut self, p: DVec3, radius_m: f64, amount: impl Fn(DVec3, f64) -> f64) {
        let tm = self.texel_m();
        let off = p - self.center;
        let (cx, cy) = (off.dot(self.t1) * EARTH_RADIUS_M, off.dot(self.t2) * EARTH_RADIUS_M);
        let to_i = |v: f64| (((v + self.half_m) / tm).floor() as i64).clamp(0, N as i64 - 1) as usize;
        let (i0, i1) = (to_i(cx - radius_m - tm), to_i(cx + radius_m + tm));
        let (j0, j1) = (to_i(cy - radius_m - tm), to_i(cy + radius_m + tm));
        for j in j0..=j1 {
            for i in i0..=i1 {
                let d = self.dir(i, j);
                let dist = p.dot(d).clamp(-1.0, 1.0).acos() * EARTH_RADIUS_M;
                if dist >= radius_m {
                    continue;
                }
                let a = amount(d, dist);
                if a.is_finite() && a != 0.0 {
                    self.data[j * N + i] += a as f32;
                }
            }
        }
        let rect = (i0, j0, i1 + 1, j1 + 1);
        self.dirty = union(self.dirty, rect);
        self.touched = union(self.touched, rect);
    }

    /// A circle (center, radius in meters) around everything the stroke touched so far.
    pub fn touched_disc(&self) -> Option<(DVec3, f64)> {
        let (x0, y0, x1, y1) = self.touched?;
        let tm = self.texel_m();
        let mx = (x0 + x1) as f64 * 0.5 * tm - self.half_m;
        let my = (y0 + y1) as f64 * 0.5 * tm - self.half_m;
        let r = 0.5 * ((x1 - x0) as f64).hypot((y1 - y0) as f64) * tm;
        let c = (self.center + (self.t1 * mx + self.t2 * my) / EARTH_RADIUS_M).normalize();
        Some((c, r))
    }

    /// The rows to upload: (x, y, width, height, values), and clears the changed mark.
    pub fn take_dirty(&mut self) -> Option<(usize, usize, usize, usize, Vec<f32>)> {
        let (x0, y0, x1, y1) = self.dirty.take()?;
        let mut out = Vec::with_capacity((x1 - x0) * (y1 - y0));
        for j in y0..y1 {
            out.extend_from_slice(&self.data[j * N + x0..j * N + x1]);
        }
        Some((x0, y0, x1 - x0, y1 - y0, out))
    }

    /// Three vec4 for the shader: [center - origin, half size in m], [t1, 1 = on], [t2, 0].
    pub fn uniform(&self, origin: DVec3) -> [f32; 12] {
        let c = self.center - origin;
        [
            c.x as f32, c.y as f32, c.z as f32, self.half_m as f32,
            self.t1.x as f32, self.t1.y as f32, self.t1.z as f32, 1.0,
            self.t2.x as f32, self.t2.y as f32, self.t2.z as f32, if self.tint { 1.0 } else { 0.0 },
        ]
    }
}

/// Where the brush dabs go as the pointer moves from `last` (the previous dab) to `now`: one dab every `spacing_m`
/// along the way, like Blender's spaced strokes. Returns the new dabs and the position of the newest one (the
/// distance not covered yet carries over to the next call, so splitting a drag into more or fewer frames makes the
/// same dabs). A jump of more than `max` spacings paints nothing along the way: one dab at the new place.
pub fn dabs_along(last: DVec3, now: DVec3, spacing_m: f64, max: usize) -> (Vec<DVec3>, DVec3) {
    let theta = 2.0 * ((last - now).length() * 0.5).clamp(0.0, 1.0).asin(); // exact for tiny angles, unlike acos
    let dist = theta * EARTH_RADIUS_M;
    if dist < spacing_m || theta < 1e-12 {
        return (Vec::new(), last);
    }
    let n = (dist / spacing_m).floor() as usize;
    if n > max {
        return (vec![now], now);
    }
    let at = |t: f64| ((last * ((1.0 - t) * theta).sin() + now * (t * theta).sin()) / theta.sin()).normalize();
    let out: Vec<DVec3> = (1..=n).map(|k| at(k as f64 * spacing_m / dist)).collect();
    let newest = *out.last().unwrap();
    (out, newest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dabs_are_evenly_spaced_and_do_not_depend_on_how_the_drag_is_split_into_frames() {
        let a = DVec3::Z;
        let b = DVec3::new((1000.5 / EARTH_RADIUS_M).sin(), 0.0, (1000.5 / EARTH_RADIUS_M).cos());
        // One frame.
        let (all, _) = dabs_along(a, b, 100.0, 20);
        assert_eq!(all.len(), 10);
        // The same drag in 7 frames.
        let mut last = a;
        let mut count = 0;
        for k in 1..=7 {
            let now = DVec3::new((1000.5 * k as f64 / 7.0 / EARTH_RADIUS_M).sin(), 0.0, (1000.5 * k as f64 / 7.0 / EARTH_RADIUS_M).cos());
            let (d, l) = dabs_along(last, now, 100.0, 20);
            count += d.len();
            last = l;
        }
        assert_eq!(count, 10);
        // Standing still makes none; a huge jump makes one at the new place.
        assert!(dabs_along(a, a, 100.0, 20).0.is_empty());
        let (j, l) = dabs_along(a, b, 10.0, 20);
        assert_eq!(j.len(), 1);
        assert!(l.dot(b) > 0.9999999);
    }

    #[test]
    fn a_stroke_lands_under_the_brush_and_nowhere_else() {
        let c = DVec3::new(0.3, 0.5, 0.8).normalize();
        let mut w = Window::new(c, 100.0);
        w.add(c, 100.0, |_, dist| 10.0 * (1.0 - dist / 100.0));
        let mid = w.data[(N / 2) * N + N / 2];
        assert!(mid > 9.0 && mid <= 10.0, "mid {mid}");
        assert_eq!(w.data[0], 0.0);
        assert_eq!(w.data[(N / 2) * N + N / 2 + 100], 0.0); // 100 grid points = 156 m away
        let (cc, r) = w.touched_disc().unwrap();
        assert!(cc.dot(c) > 0.999999 && r > 100.0 && r < 400.0, "{r}");
    }

    #[test]
    fn the_window_gives_up_when_the_brush_nears_its_edge() {
        let c = DVec3::Z;
        let w = Window::new(c, 100.0);
        assert!(w.fits(c, 100.0));
        let far = DVec3::new((700.0 / EARTH_RADIUS_M).sin(), 0.0, (700.0 / EARTH_RADIUS_M).cos());
        assert!(!w.fits(far, 100.0)); // 700 + 130 > 800
        assert!(!w.fits(c, 300.0)); // much bigger brush
    }

    #[test]
    fn only_the_changed_rows_are_taken_for_upload() {
        let mut w = Window::new(DVec3::Z, 100.0);
        assert!(w.take_dirty().is_some());
        assert!(w.take_dirty().is_none());
        w.add(DVec3::Z, 100.0, |_, _| 1.0);
        let (x, y, ww, hh, v) = w.take_dirty().unwrap();
        assert!(ww < N && hh < N && x > 0 && y > 0);
        assert_eq!(v.len(), ww * hh);
    }
}
