//! The pixel layer: tiny square pixels that ignore the hex borders.
//!
//! The globe is split like a cube: 6 faces, each a square grid of 2^level x 2^level pixels
//! (equal-angle warp, so pixels stay close to square). Every pixel splits into exactly 4
//! children, so painting works the same way as on hexes ("one level at a time"), but the
//! pyramid is exact: a coarse pixel is precisely the sum of its 4 children.
//!
//! Storage is sparse: one sorted map per level, keyed by (face, Morton code). Morton order
//! keeps everything inside a pixel in one contiguous key range, which makes "wipe the finer
//! paint inside this pixel" a range delete. Each coarser pixel also keeps a small running
//! tally of what is painted inside it (for the zoomed-out average).

use crate::hexpaint::{Paint, PALETTE_LEN};
use glam::DVec3;
use std::collections::{BTreeMap, HashMap};
use std::f64::consts::FRAC_PI_4;

/// Finest level: 2^31 pixels along a face edge (about 5 mm on an Earth-sized globe).
pub const MAX_LEVEL: u8 = 31;
const LEVELS: usize = MAX_LEVEL as usize + 1;

/// (face normal, "u" axis, "v" axis) for the 6 cube faces.
const FACES: [(DVec3, DVec3, DVec3); 6] = [
    (DVec3::X, DVec3::Y, DVec3::Z),
    (DVec3::NEG_X, DVec3::NEG_Y, DVec3::Z),
    (DVec3::Y, DVec3::NEG_X, DVec3::Z),
    (DVec3::NEG_Y, DVec3::X, DVec3::Z),
    (DVec3::Z, DVec3::X, DVec3::Y),
    (DVec3::NEG_Z, DVec3::X, DVec3::NEG_Y),
];

/// Direction on the globe for face coordinates u, v in -1..1.
pub fn face_dir(face: u8, u: f64, v: f64) -> DVec3 {
    let (n, a, b) = FACES[face as usize];
    (n + a * (u * FRAC_PI_4).tan() + b * (v * FRAC_PI_4).tan()).normalize()
}

fn spread(v: u32) -> u64 {
    let mut x = v as u64;
    x = (x | x << 16) & 0x0000_FFFF_0000_FFFF;
    x = (x | x << 8) & 0x00FF_00FF_00FF_00FF;
    x = (x | x << 4) & 0x0F0F_0F0F_0F0F_0F0F;
    x = (x | x << 2) & 0x3333_3333_3333_3333;
    x = (x | x << 1) & 0x5555_5555_5555_5555;
    x
}

fn compact(mut x: u64) -> u32 {
    x &= 0x5555_5555_5555_5555;
    x = (x | x >> 1) & 0x3333_3333_3333_3333;
    x = (x | x >> 2) & 0x0F0F_0F0F_0F0F_0F0F;
    x = (x | x >> 4) & 0x00FF_00FF_00FF_00FF;
    x = (x | x >> 8) & 0x0000_FFFF_0000_FFFF;
    x = (x | x >> 16) & 0x0000_0000_FFFF_FFFF;
    x as u32
}

/// One pixel: which face, how fine, and which column / row on that face.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
pub struct PixelId {
    pub face: u8,
    pub level: u8,
    pub x: u32,
    pub y: u32,
}

type Key = (u8, u64);

impl PixelId {
    pub fn new(face: u8, level: u8, x: u32, y: u32) -> PixelId {
        PixelId { face, level, x, y }
    }

    /// Pixels along one face edge at this level.
    pub fn size(level: u8) -> f64 {
        (1u64 << level) as f64
    }

    fn morton(self) -> u64 {
        spread(self.x) | spread(self.y) << 1
    }

    fn key(self) -> Key {
        (self.face, self.morton())
    }

    fn from_key(level: u8, key: Key) -> PixelId {
        PixelId { face: key.0, level, x: compact(key.1), y: compact(key.1 >> 1) }
    }

    /// The pixel at a coarser level that contains this one.
    pub fn ancestor(self, level: u8) -> PixelId {
        let s = self.level - level;
        PixelId { face: self.face, level, x: self.x >> s, y: self.y >> s }
    }

    pub fn children(self) -> [PixelId; 4] {
        let (l, x, y) = (self.level + 1, self.x * 2, self.y * 2);
        [
            PixelId::new(self.face, l, x, y),
            PixelId::new(self.face, l, x + 1, y),
            PixelId::new(self.face, l, x, y + 1),
            PixelId::new(self.face, l, x + 1, y + 1),
        ]
    }

    /// Face coordinates (-1..1) of a point inside the pixel; fx, fy in 0..1 (0.5 = middle).
    pub fn uv(self, fx: f64, fy: f64) -> (f64, f64) {
        let n = Self::size(self.level);
        ((self.x as f64 + fx) / n * 2.0 - 1.0, (self.y as f64 + fy) / n * 2.0 - 1.0)
    }

    pub fn dir_at(self, fx: f64, fy: f64) -> DVec3 {
        let (u, v) = self.uv(fx, fy);
        face_dir(self.face, u, v)
    }

    pub fn center_dir(self) -> DVec3 {
        self.dir_at(0.5, 0.5)
    }

    /// The pixel at `level` that contains a direction.
    /// Where a direction sits inside this pixel, as fractions (0..1 across x and y), measured on this pixel's own face.
    pub fn frac_in(self, dir: DVec3) -> (f64, f64) {
        let (n, a, b) = FACES[self.face as usize];
        let d = dir.normalize();
        let c = d.dot(n);
        let (u, v) = ((d.dot(a) / c).atan() / FRAC_PI_4, (d.dot(b) / c).atan() / FRAC_PI_4);
        let size = Self::size(self.level);
        (((u + 1.0) * 0.5 * size - self.x as f64), ((v + 1.0) * 0.5 * size - self.y as f64))
    }

    pub fn from_dir(dir: DVec3, level: u8) -> PixelId {
        let d = dir.normalize();
        let (mut face, mut best) = (0u8, f64::MIN);
        for (i, (n, _, _)) in FACES.iter().enumerate() {
            let c = d.dot(*n);
            if c > best {
                best = c;
                face = i as u8;
            }
        }
        let (n, a, b) = FACES[face as usize];
        let c = d.dot(n);
        let (u, v) = ((d.dot(a) / c).atan() / FRAC_PI_4, (d.dot(b) / c).atan() / FRAC_PI_4);
        let size = Self::size(level);
        let cell = |t: f64| (((t + 1.0) * 0.5 * size).floor().max(0.0) as u64).min((1u64 << level) - 1) as u32;
        PixelId { face, level, x: cell(u), y: cell(v) }
    }
}

#[derive(Clone, Copy)]
struct Entry {
    value: Paint,
    /// What showed here just before this entry was painted (the coarser paint above it).
    above: Paint,
}

/// Running tally for a pixel: how many painted entries lie strictly inside it, and the
/// net change they make to its composition (fraction of each paint type).
#[derive(Clone, Copy)]
struct Tally {
    count: u32,
    delta: [f64; PALETTE_LEN],
}

#[derive(Default)]
struct Stroke {
    old: HashMap<PixelId, Option<Paint>>,
}

pub struct PixelStore {
    levels: Vec<BTreeMap<Key, Entry>>,
    tallies: Vec<HashMap<Key, Tally>>,
    undo: Vec<Stroke>,
    current: Option<Stroke>,
}

impl Default for PixelStore {
    fn default() -> Self {
        PixelStore {
            levels: (0..LEVELS).map(|_| BTreeMap::new()).collect(),
            tallies: (0..LEVELS).map(|_| HashMap::new()).collect(),
            undo: Vec::new(),
            current: None,
        }
    }
}

impl PixelStore {
    pub fn begin_stroke(&mut self) {
        self.current = Some(Stroke::default());
    }

    pub fn end_stroke(&mut self) {
        if let Some(s) = self.current.take() {
            if !s.old.is_empty() {
                self.undo.push(s);
            }
        }
    }

    pub fn undo_depth(&self) -> usize {
        self.undo.len()
    }

    fn remember(&mut self, id: PixelId, old: Option<Paint>) {
        if let Some(s) = self.current.as_mut() {
            s.old.entry(id).or_insert(old);
        }
    }

    fn entry(&self, id: PixelId) -> Option<&Entry> {
        self.levels[id.level as usize].get(&id.key())
    }

    /// Paint from strictly coarser pixels (0 if none).
    fn value_above(&self, id: PixelId) -> Paint {
        let m = id.morton();
        for l in (0..id.level).rev() {
            if self.levels[l as usize].is_empty() {
                continue;
            }
            if let Some(e) = self.levels[l as usize].get(&(id.face, m >> (2 * (id.level - l)))) {
                return e.value;
            }
        }
        0
    }

    /// What this pixel shows from its own entry or a coarser one (0 = bare).
    pub fn effective(&self, id: PixelId) -> Paint {
        self.entry(id).map(|e| e.value).unwrap_or_else(|| self.value_above(id))
    }

    /// The pixel's own entry only.
    pub fn own(&self, id: PixelId) -> Option<Paint> {
        self.entry(id).map(|e| e.value)
    }

    /// Is anything painted strictly inside this pixel?
    pub fn has_finer(&self, id: PixelId) -> bool {
        self.tallies[id.level as usize].contains_key(&id.key())
    }

    /// Fractions (summing to 1) of each paint type that this pixel is made of, given what it
    /// shows on its own (`eff`, from its entry or coarser ones) and what is painted inside it.
    pub fn composition(&self, id: PixelId, eff: Paint) -> [f64; PALETTE_LEN] {
        let mut comp = [0.0; PALETTE_LEN];
        comp[(eff as usize).min(PALETTE_LEN - 1)] = 1.0;
        if let Some(t) = self.tallies[id.level as usize].get(&id.key()) {
            for (c, d) in comp.iter_mut().zip(t.delta.iter()) {
                *c += d;
            }
        }
        comp
    }

    fn apply_tally(&mut self, id: PixelId, value: Paint, above: Paint, add: bool) {
        let m = id.morton();
        let (v, a) = ((value as usize).min(PALETTE_LEN - 1), (above as usize).min(PALETTE_LEN - 1));
        for l in 0..id.level {
            let share = 0.25f64.powi((id.level - l) as i32) * if add { 1.0 } else { -1.0 };
            let key = (id.face, m >> (2 * (id.level - l)));
            let map = &mut self.tallies[l as usize];
            let t = map.entry(key).or_insert(Tally { count: 0, delta: [0.0; PALETTE_LEN] });
            t.delta[v] += share;
            t.delta[a] -= share;
            if add {
                t.count += 1;
            } else {
                t.count -= 1;
            }
            if t.count == 0 {
                map.remove(&key);
            }
        }
    }

    fn insert_entry(&mut self, id: PixelId, value: Paint) {
        let above = self.value_above(id);
        self.levels[id.level as usize].insert(id.key(), Entry { value, above });
        self.apply_tally(id, value, above, true);
    }

    fn remove_entry(&mut self, id: PixelId) -> Option<Paint> {
        let e = self.levels[id.level as usize].remove(&id.key())?;
        self.apply_tally(id, e.value, e.above, false);
        Some(e.value)
    }

    /// Paint one pixel at its own level (0 erases). Anything painted inside it is wiped.
    pub fn paint(&mut self, id: PixelId, value: Paint) {
        let m = id.morton();
        for f in (id.level + 1)..=MAX_LEVEL {
            if self.levels[f as usize].is_empty() {
                continue;
            }
            let k = 2 * (f - id.level) as u32;
            let (lo, hi) = ((id.face, m << k), (id.face, (m + 1) << k));
            let doomed: Vec<Key> = self.levels[f as usize].range(lo..hi).map(|(k, _)| *k).collect();
            for key in doomed {
                let did = PixelId::from_key(f, key);
                if let Some(old) = self.remove_entry(did) {
                    self.remember(did, Some(old));
                }
            }
        }
        let old_exact = self.own(id);
        let above = self.value_above(id);
        if value == above {
            if let Some(old) = old_exact {
                self.remove_entry(id);
                self.remember(id, Some(old));
            }
        } else if old_exact != Some(value) {
            if old_exact.is_some() {
                self.remove_entry(id);
            }
            self.insert_entry(id, value);
            self.remember(id, old_exact);
        }
    }

    /// Undo the last stroke. Returns false if there was nothing to undo.
    pub fn undo(&mut self) -> bool {
        let Some(stroke) = self.undo.pop() else { return false };
        for id in stroke.old.keys() {
            self.remove_entry(*id);
        }
        // Coarser first, so each restored entry sees the right paint above it.
        let mut restore: Vec<(PixelId, Paint)> = stroke.old.iter().filter_map(|(id, o)| o.map(|v| (*id, v))).collect();
        restore.sort_by_key(|(id, _)| id.level);
        for (id, v) in restore {
            self.insert_entry(id, v);
        }
        true
    }

    pub fn painted_pixels(&self) -> usize {
        self.levels.iter().map(|m| m.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hexpaint::RED;

    fn px(level: u8) -> PixelId {
        PixelId::from_dir(DVec3::new(0.3, -0.5, 0.8), level)
    }

    #[test]
    fn morton_roundtrips() {
        for (x, y) in [(0u32, 0u32), (1, 0), (0, 1), (123_456, 7_890_123), (u32::MAX >> 1, 5)] {
            let m = spread(x) | spread(y) << 1;
            assert_eq!((compact(m), compact(m >> 1)), (x, y));
        }
    }

    #[test]
    fn center_maps_back_to_the_same_pixel_on_every_face() {
        for level in [0u8, 1, 5, 17, 31] {
            for face in 0..6u8 {
                let n = 1u64 << level;
                for (x, y) in [(0u64, 0u64), (n - 1, n - 1), (n / 3, n / 2)] {
                    let id = PixelId::new(face, level, x as u32, y as u32);
                    assert_eq!(PixelId::from_dir(id.center_dir(), level), id, "{id:?}");
                }
            }
        }
    }

    #[test]
    fn children_sit_inside_their_parent_and_corners_are_unit_vectors() {
        let p = px(12);
        for c in p.children() {
            assert_eq!(c.ancestor(12), p);
            assert_eq!(PixelId::from_dir(c.center_dir(), 12), p);
        }
        for (fx, fy) in [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)] {
            assert!((p.dir_at(fx, fy).length() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn random_directions_land_in_a_pixel_that_is_close() {
        let mut s = 12345u64;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (s >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        };
        for _ in 0..2000 {
            let d = DVec3::new(next(), next(), next()).normalize();
            let id = PixelId::from_dir(d, 10);
            let ang = d.dot(id.center_dir()).clamp(-1.0, 1.0).acos();
            assert!(ang < 1.6 * (std::f64::consts::FRAC_PI_2 / 1024.0), "{ang}");
        }
    }

    #[test]
    fn pixels_are_close_to_square() {
        // Edge lengths of one pixel at several spots: ratio should stay modest.
        for d in [DVec3::X, DVec3::new(1.0, 1.0, 1.0), DVec3::new(1.0, 0.9, 0.0), DVec3::new(0.1, 0.2, 1.0)] {
            let p = PixelId::from_dir(d, 12);
            let a = (p.dir_at(1.0, 0.0) - p.dir_at(0.0, 0.0)).length();
            let b = (p.dir_at(0.0, 1.0) - p.dir_at(0.0, 0.0)).length();
            let r = a.max(b) / a.min(b);
            assert!(r < 1.5, "{r}");
        }
    }

    #[test]
    fn painting_coarse_wipes_fine_and_undo_restores_everything() {
        let mut s = PixelStore::default();
        let coarse = px(10);
        let fine: Vec<PixelId> = coarse.children().iter().flat_map(|c| c.children()).collect();
        s.begin_stroke();
        for f in &fine {
            s.paint(*f, RED);
        }
        s.end_stroke();
        assert_eq!(s.painted_pixels(), 16);
        s.begin_stroke();
        s.paint(coarse, 2);
        s.end_stroke();
        assert_eq!(s.painted_pixels(), 1);
        assert_eq!(s.effective(fine[5]), 2);
        assert!(!s.has_finer(coarse));
        assert!(s.undo());
        assert_eq!(s.painted_pixels(), 16);
        assert_eq!(s.effective(fine[5]), RED);
        assert!(s.has_finer(coarse));
        assert!(s.undo());
        assert_eq!(s.painted_pixels(), 0);
        assert!(!s.has_finer(coarse), "tallies are cleaned up too");
    }

    #[test]
    fn composition_adds_up_and_handles_holes_and_nesting() {
        let mut s = PixelStore::default();
        let big = px(8);
        s.begin_stroke();
        s.paint(big, RED); // whole pixel red
        let kids = big.children();
        s.paint(kids[0], 0); // a hole: a quarter is bare again
        s.paint(kids[1].children()[2], 2); // a sixteenth is grass, inside the red
        s.end_stroke();
        let comp = s.composition(big, s.effective(big));
        assert!((comp.iter().sum::<f64>() - 1.0).abs() < 1e-12, "{comp:?}");
        assert!((comp[0] - 0.25).abs() < 1e-12, "{comp:?}");
        assert!((comp[2] - 1.0 / 16.0).abs() < 1e-12, "{comp:?}");
        assert!((comp[RED as usize] - (0.75 - 1.0 / 16.0)).abs() < 1e-12, "{comp:?}");
    }

    #[test]
    fn same_color_over_same_color_stores_nothing_and_erasing_bare_stores_nothing() {
        let mut s = PixelStore::default();
        s.begin_stroke();
        s.paint(px(8), RED);
        s.paint(px(12), RED);
        s.paint(PixelId::from_dir(DVec3::new(-0.3, 0.5, -0.8), 15), 0); // bare ground far away
        s.end_stroke();
        assert_eq!(s.painted_pixels(), 1);
    }

    #[test]
    fn undo_in_a_nested_history_leaves_no_drift() {
        let mut s = PixelStore::default();
        for (i, level) in [6u8, 9, 14, 20].into_iter().enumerate() {
            s.begin_stroke();
            s.paint(px(level), (i as u8 % 8) + 1);
            s.end_stroke();
        }
        while s.undo() {}
        assert_eq!(s.painted_pixels(), 0);
        for l in 0..LEVELS {
            assert!(s.tallies[l].is_empty(), "level {l} still has tallies");
        }
    }
}
