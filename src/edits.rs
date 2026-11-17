//! Hand edits of the ground: a sparse layer of height offsets on top of the generated terrain.
//!
//! The terrain is a function of position, so an edit cannot live on mesh vertices (meshes are rebuilt at every
//! zoom). Instead each stroke writes meters of height into small grids (the same 64 x 64 tiles on the
//! cube-sphere the terrain uses), at the tile level that suits the brush: a big brush writes coarse tiles, a
//! small one fine tiles. Every vertex then reads terrain + edits. Only touched places cost memory.
//!
//! Edits also feed a pyramid of coarser grids (`mips`), the way the terrain's own tiles are low-passed versions of
//! each other: a mountain raised by hand is a smaller bump on a coarser mesh, not nothing. Without it the globe
//! (made of coarse meshes) never showed what was made inside the planet.
//!
//! All tools boil down to "add this many meters here": raise and lower add a fixed amount, smooth and flatten add
//! the difference to a target. Each lattice point's amount depends on its direction only, so two tiles sharing
//! an edge agree on it without talking to each other.

use crate::pixelpaint::PixelId;
use crate::terrain::EARTH_RADIUS_M;
use crate::tiles::{spacing_rad, Tile, MAX_TILE_LEVEL, SAMPLES, TILE_N};
use glam::DVec3;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::RwLock;

/// Edit cells across a brush radius: smaller cells would be wasted, bigger ones would look blocky.
pub const CELLS_PER_RADIUS: f64 = 14.0;

/// The edit tile level for a brush of this radius (meters).
pub fn level_for_radius(radius_m: f64) -> u8 {
    let want_cell = (radius_m / CELLS_PER_RADIUS).max(1e-3);
    let level0_cell = spacing_rad(0) * EARTH_RADIUS_M;
    (level0_cell / want_cell).log2().ceil().clamp(0.0, MAX_TILE_LEVEL as f64) as u8
}

/// Distance on the ground between two unit directions, meters.
fn ground_dist(a: DVec3, b: DVec3) -> f64 {
    2.0 * ((a - b).length() * 0.5).clamp(0.0, 1.0).asin() * EARTH_RADIUS_M
}

#[derive(Default)]
pub struct EditLayer {
    tiles: RwLock<HashMap<PixelId, Tile>>,
    /// Bit `l` is set when any tile of level `l` exists.
    levels: AtomicU32,
    /// Level `c` of the pyramid holds the low-passed sum of every edit made at a level `c + 2` or finer.
    mips: RwLock<HashMap<PixelId, Tile>>,
    mip_levels: AtomicU32,
    /// Detail dampers (see `stroke_damp`): per tile level, how much of the detail finer than that level is gone, 0..1.
    damp: RwLock<HashMap<PixelId, Tile>>,
    damp_levels: AtomicU32,
    version: AtomicU64,
}

impl EditLayer {
    pub fn is_empty(&self) -> bool {
        self.levels.load(Ordering::Relaxed) == 0 && self.mip_levels.load(Ordering::Relaxed) == 0 && self.damp_levels.load(Ordering::Relaxed) == 0
    }

    /// Changes with every stroke.
    pub fn version(&self) -> u64 {
        self.version.load(Ordering::Relaxed)
    }

    /// Meters added to the ground at a direction, for a mesh whose detail is tile level `l` (fractional):
    /// edits much finer than the mesh can show are left out, and fade in over one level.
    pub fn at(&self, d: DVec3, l: f64) -> f64 {
        let mask = self.levels.load(Ordering::Relaxed);
        let mip_mask = self.mip_levels.load(Ordering::Relaxed);
        if mask == 0 && mip_mask == 0 {
            return 0.0;
        }
        let mut sum = 0.0;
        // Edits finer than the mesh come from the pyramid (the mesh's own level, or between two levels).
        if mip_mask != 0 {
            let l0 = l.floor().max(0.0);
            let t = l - l0;
            let mips = self.mips.read().unwrap();
            for (c, w) in [(l0 as i64, 1.0 - t), (l0 as i64 + 1, t)] {
                if w <= 1e-9 || !(0..=MAX_TILE_LEVEL as i64).contains(&c) || mip_mask >> c & 1 == 0 {
                    continue;
                }
                let id = PixelId::from_dir(d, c as u8);
                if let Some(tile) = mips.get(&id) {
                    let (fx, fy) = id.frac_in(d);
                    sum += w * tile.cubic(fx, fy);
                }
            }
        }
        if mask == 0 {
            return sum;
        }
        let tiles = self.tiles.read().unwrap();
        for e in 0..=MAX_TILE_LEVEL {
            if mask >> e & 1 == 0 {
                continue;
            }
            let w = (l + 2.0 - e as f64).clamp(0.0, 1.0);
            if w <= 0.0 {
                continue;
            }
            let id = PixelId::from_dir(d, e);
            if let Some(t) = tiles.get(&id) {
                let (fx, fy) = id.frac_in(d);
                sum += w * t.cubic(fx, fy);
            }
        }
        sum
    }

    /// Detail dampers: how much of the ground's detail finer than tile level `level` is gone here (0 = all of it
    /// still there, 1 = flat down to that level's scale). This is how a smoothed place stays smooth when you zoom in:
    /// the terrain's own fine detail is turned down, not just its coarse shape evened out.
    pub fn damp_at(&self, d: DVec3, level: u8) -> f64 {
        if self.damp_levels.load(Ordering::Relaxed) >> level & 1 == 0 {
            return 0.0;
        }
        let id = PixelId::from_dir(d, level);
        let (fx, fy) = id.frac_in(d);
        self.damp.read().unwrap().get(&id).map_or(0.0, |t| t.cubic(fx, fy).clamp(0.0, 1.0))
    }

    pub fn has_damp(&self) -> bool {
        self.damp_levels.load(Ordering::Relaxed) != 0
    }

    /// The levels that have a damper, finest last.
    pub fn damp_level_list(&self) -> Vec<u8> {
        let m = self.damp_levels.load(Ordering::Relaxed);
        (0..=MAX_TILE_LEVEL).filter(|l| m >> l & 1 == 1).collect()
    }

    /// Adds `amount(direction, distance)` (0..1 steps) to the damper of tile level `level` within `radius_m` of `center`.
    pub fn stroke_damp(&self, level: u8, center: DVec3, radius_m: f64, amount: impl Fn(DVec3, f64) -> f64) {
        let c = center.normalize();
        let cell_m = spacing_rad(level) * EARTH_RADIUS_M;
        let helper = if c.z.abs() < 0.9 { DVec3::Z } else { DVec3::X };
        let t1 = c.cross(helper).normalize();
        let t2 = c.cross(t1);
        let n = ((radius_m + 3.0 * cell_m) / cell_m).ceil() as i32;
        let mut ids = HashSet::new();
        for iy in -n..=n {
            for ix in -n..=n {
                let off = (t1 * (ix as f64 * cell_m) + t2 * (iy as f64 * cell_m)) / EARTH_RADIUS_M;
                ids.insert(PixelId::from_dir((c + off).normalize(), level));
            }
        }
        let mut writes: Vec<(PixelId, usize, f32)> = Vec::new();
        for id in ids {
            for j in 0..SAMPLES {
                for i in 0..SAMPLES {
                    let d = id.dir_at(i as f64 / TILE_N as f64, j as f64 / TILE_N as f64);
                    let dist = ground_dist(d, c);
                    if dist < radius_m {
                        let a = amount(d, dist);
                        if a > 0.0 && a.is_finite() {
                            writes.push((id, j * SAMPLES + i, a as f32));
                        }
                    }
                }
            }
        }
        if writes.is_empty() {
            return;
        }
        let mut damp = self.damp.write().unwrap();
        for (id, idx, a) in writes {
            damp.entry(id).or_insert_with(Tile::zeros).values_mut()[idx] += a;
        }
        drop(damp);
        self.damp_levels.fetch_or(1 << level, Ordering::Relaxed);
        self.version.fetch_add(1, Ordering::Relaxed);
    }

    /// One brush application: every lattice point of level `level` within `radius_m` of `center` gets
    /// `amount(direction, distance_m)` meters added. All amounts are worked out first (so tools that read the
    /// ground, like smooth, see it as it was), then written.
    pub fn stroke(&self, level: u8, center: DVec3, radius_m: f64, mut amount: impl FnMut(DVec3, f64) -> f64) {
        let c = center.normalize();
        let cell_m = spacing_rad(level) * EARTH_RADIUS_M;
        // Tiles with a lattice point inside the brush: look at a grid of points over the brush plus a margin.
        let helper = if c.z.abs() < 0.9 { DVec3::Z } else { DVec3::X };
        let t1 = c.cross(helper).normalize();
        let t2 = c.cross(t1);
        let n = ((radius_m + 3.0 * cell_m) / cell_m).ceil() as i32;
        let mut ids = HashSet::new();
        for iy in -n..=n {
            for ix in -n..=n {
                let off = (t1 * (ix as f64 * cell_m) + t2 * (iy as f64 * cell_m)) / EARTH_RADIUS_M;
                ids.insert(PixelId::from_dir((c + off).normalize(), level));
            }
        }
        let mut writes: Vec<(PixelId, usize, f32)> = Vec::new();
        for id in ids {
            for j in 0..SAMPLES {
                for i in 0..SAMPLES {
                    let d = id.dir_at(i as f64 / TILE_N as f64, j as f64 / TILE_N as f64);
                    let dist = ground_dist(d, c);
                    if dist >= radius_m {
                        continue;
                    }
                    let a = amount(d, dist);
                    if a != 0.0 && a.is_finite() {
                        writes.push((id, j * SAMPLES + i, a as f32));
                    }
                }
            }
        }
        if writes.is_empty() {
            return;
        }
        // What was written, looked up by place: the pyramid reads the fine result instead of working the tool out
        // again (smooth and flatten read the ground for every point, which made big brushes crawl).
        let by_place: HashMap<(PixelId, usize), f32> = writes.iter().map(|(id, idx, a)| ((*id, *idx), *a)).collect();
        let fine_at = |p: DVec3| -> f64 {
            let id = PixelId::from_dir(p, level);
            let (fx, fy) = id.frac_in(p);
            let (i, j) = ((fx * TILE_N as f64).round() as usize, (fy * TILE_N as f64).round() as usize);
            by_place.get(&(id, j.min(TILE_N) * SAMPLES + i.min(TILE_N))).copied().unwrap_or(0.0) as f64
        };
        // The pyramid: the same stroke, low-passed onto each coarser level whose cells are no bigger than about the
        // brush (a bigger cell would hardly notice it). Each coarse lattice point takes a tent-weighted average of the
        // brush over its neighbourhood; like the fine ones it depends on the direction only, so tiles agree.
        let mut mip_writes: Vec<(PixelId, usize, f32)> = Vec::new();
        let mut mip_set = 0u32;
        for cl in (0..level.saturating_sub(1)).rev() {
            let cell_c = spacing_rad(cl) * EARTH_RADIUS_M;
            if cell_c > 1.5 * radius_m {
                break; // coarser levels only get coarser
            }
            let n_c = ((radius_m + 3.0 * cell_c) / cell_c).ceil() as i32;
            let mut cids = HashSet::new();
            for iy in -n_c..=n_c {
                for ix in -n_c..=n_c {
                    let off = (t1 * (ix as f64 * cell_c) + t2 * (iy as f64 * cell_c)) / EARTH_RADIUS_M;
                    cids.insert(PixelId::from_dir((c + off).normalize(), cl));
                }
            }
            for id in cids {
                for j in 0..SAMPLES {
                    for i in 0..SAMPLES {
                        let q = id.dir_at(i as f64 / TILE_N as f64, j as f64 / TILE_N as f64);
                        if ground_dist(q, c) >= radius_m + cell_c {
                            continue;
                        }
                        let qh = if q.z.abs() < 0.9 { DVec3::Z } else { DVec3::X };
                        let (q1, q2) = (q.cross(qh).normalize(), q.cross(q.cross(qh).normalize()));
                        let (mut sum, mut wsum) = (0.0, 0.0);
                        for sy in -2i32..=2 {
                            for sx in -2i32..=2 {
                                let (ox, oy) = (sx as f64 * 0.5, sy as f64 * 0.5); // in cells
                                let w = (1.0 - ox.abs() * 0.5).max(0.0) * (1.0 - oy.abs() * 0.5).max(0.0);
                                wsum += w;
                                let p = (q + (q1 * ox + q2 * oy) * cell_c / EARTH_RADIUS_M).normalize();
                                let dist = ground_dist(p, c);
                                if dist < radius_m {
                                    sum += w * fine_at(p);
                                }
                            }
                        }
                        let a = sum / wsum;
                        if a != 0.0 {
                            mip_writes.push((id, j * SAMPLES + i, a as f32));
                            mip_set |= 1 << cl;
                        }
                    }
                }
            }
        }
        let mut tiles = self.tiles.write().unwrap();
        for (id, idx, a) in writes {
            tiles.entry(id).or_insert_with(Tile::zeros).values_mut()[idx] += a;
        }
        drop(tiles);
        if !mip_writes.is_empty() {
            let mut mips = self.mips.write().unwrap();
            for (id, idx, a) in mip_writes {
                mips.entry(id).or_insert_with(Tile::zeros).values_mut()[idx] += a;
            }
            self.mip_levels.fetch_or(mip_set, Ordering::Relaxed);
        }
        self.levels.fetch_or(1 << level, Ordering::Relaxed);
        self.version.fetch_add(1, Ordering::Relaxed);
    }
}

/// How strongly the brush acts at `t` (0 = middle, 1 = edge), for each falloff.
pub fn falloff_weight(kind: Falloff, t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    match kind {
        Falloff::Soft => 0.5 * (1.0 + (std::f64::consts::PI * t).cos()),
        Falloff::Linear => 1.0 - t,
        Falloff::Hard => {
            if t < 0.8 {
                1.0
            } else {
                // A smooth tail, not a straight ramp: no crease where it starts or at the very edge.
                let u = (1.0 - t) / 0.2;
                u * u * (3.0 - 2.0 * u)
            }
        }
    }
}

pub use crate::ribbon::Falloff;

#[cfg(test)]
mod tests {
    use super::*;

    fn spot() -> DVec3 {
        DVec3::new(0.3, 0.5, 0.8).normalize()
    }

    #[test]
    fn brush_level_follows_radius() {
        assert!(level_for_radius(1.0) > level_for_radius(100.0));
        assert!(level_for_radius(100.0) > level_for_radius(100_000.0));
        // The cell is never much bigger than a tenth of the radius, unless the coarsest level is already finer.
        for r in [5.0, 80.0, 3_000.0, 90_000.0] {
            let cell = spacing_rad(level_for_radius(r)) * EARTH_RADIUS_M;
            assert!(cell <= r / CELLS_PER_RADIUS * 1.001 || level_for_radius(r) == 0 || level_for_radius(r) == MAX_TILE_LEVEL, "r {r} cell {cell}");
        }
    }

    #[test]
    fn a_big_edit_shows_on_the_coarse_meshes_the_globe_is_made_of() {
        let layer = EditLayer::default();
        let level = level_for_radius(30_000.0);
        layer.stroke(level, spot(), 30_000.0, |_, dist| 800.0 * falloff_weight(Falloff::Soft, dist / 30_000.0));
        let fine = layer.at(spot(), level as f64);
        assert!((fine - 800.0).abs() < 60.0, "fine {fine}");
        for l in 2..level.saturating_sub(1) {
            let v = layer.at(spot(), l as f64);
            assert!(v > 50.0 && v <= 801.0, "level {l}: {v}");
        }
        // Between two levels it is between the two.
        let (a, b) = (layer.at(spot(), 3.0), layer.at(spot(), 4.0));
        let mid = layer.at(spot(), 3.5);
        assert!(mid >= a.min(b) - 1.0 && mid <= a.max(b) + 1.0);
        // And not far away.
        assert!(layer.at(DVec3::new(-0.3, 0.5, -0.8).normalize(), 3.0).abs() < 1e-6);
    }

    #[test]
    fn a_raise_stroke_shows_up_in_the_middle_and_not_far_away() {
        let layer = EditLayer::default();
        assert!(layer.is_empty());
        let level = level_for_radius(200.0);
        layer.stroke(level, spot(), 200.0, |_, dist| 5.0 * falloff_weight(Falloff::Soft, dist / 200.0));
        assert!(!layer.is_empty());
        let mid = layer.at(spot(), level as f64);
        assert!((mid - 5.0).abs() < 0.4, "middle {mid}");
        let far = layer.at(DVec3::new(0.3, 0.5, 0.8 + 0.01).normalize(), level as f64);
        assert!(far.abs() < 1e-6, "far {far}");
        // A mesh much coarser than the edit sees it low-passed (smaller, wider), not at full height; far coarser still
        // (cells much bigger than the brush) hardly at all.
        let coarse = layer.at(spot(), (level as f64) - 3.0);
        assert!(coarse > 0.0 && coarse < 5.0, "coarse {coarse}");
        assert!(layer.at(spot(), 0.0).abs() < 0.01);
    }

    #[test]
    fn strokes_add_up_and_a_stroke_across_a_tile_edge_has_no_seam() {
        let layer = EditLayer::default();
        // A direction exactly on a tile edge at the brush's level: the brush sits on the seam.
        let level = level_for_radius(300.0);
        let id = PixelId::from_dir(spot(), level);
        let on_edge = id.dir_at(1.0, 0.5);
        layer.stroke(level, on_edge, 300.0, |_, dist| 2.0 * falloff_weight(Falloff::Linear, dist / 300.0));
        layer.stroke(level, on_edge, 300.0, |_, dist| 2.0 * falloff_weight(Falloff::Linear, dist / 300.0));
        // Walk across the edge in small steps: the height must be continuous.
        let step = 1.0 / EARTH_RADIUS_M;
        let (t1, _) = (on_edge.cross(DVec3::Z).normalize(), 0);
        let mut prev = layer.at(on_edge - t1 * step * 60.0, level as f64);
        let mut biggest_jump: f64 = 0.0;
        for k in -59..60 {
            let h = layer.at((on_edge + t1 * step * k as f64 * 1.0).normalize(), level as f64);
            biggest_jump = biggest_jump.max((h - prev).abs());
            prev = h;
        }
        assert!(layer.at(on_edge, level as f64) > 3.5, "two strokes of 2 m should add up");
        assert!(biggest_jump < 0.2, "jump {biggest_jump}");
    }
}
