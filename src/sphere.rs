//! The globe's cell structure. THIS IS THE PART YOU WRITE.
//!
//! The idea: the planet is a handful of base faces. Each face is a quadtree.
//! A cell at `level` splits into 4 cells at `level + 1`.
//! Coordinates inside a face are (x, y) in 0..2^level.
//!
//! Layout: HEALPix (nested scheme). 12 base faces, equal-area cells.
//! Faces 0-3 wrap around the north, 4-7 around the equator, 8-11 around the south.
//! The formulas follow the standard HEALPix reference; the tests at the bottom
//! check the properties we care about (unit length, round trips, full coverage).

/// How many base faces the planet starts from. Change this when we pick a layout.
pub const BASE_FACES: u8 = 12;

use std::f64::consts::PI;

// Per-face lookup tables from the HEALPix reference.
// JRLL: which ring band a face starts in. JPLL: where it starts around the globe.
const JRLL: [i64; 12] = [2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4];
const JPLL: [i64; 12] = [1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7];

/// A unit direction from the planet's center: [x, y, z], length 1.
pub type Dir = [f64; 3];

/// The heart of the layout: a spot on a face -> a direction on the sphere.
///
/// (u, v) are positions on the face's grid, in cell units, 0..2^level.
/// Whole numbers are cell CORNERS; x + 0.5 is a cell's middle.
fn point(face: u8, level: u8, u: f64, v: f64) -> Dir {
    let nside = (1_u64 << level) as f64; // cells per face edge
    let f = face as usize;

    // Which horizontal ring is this spot on? 0 = north pole ... 4*nside = south pole.
    let jr = JRLL[f] as f64 * nside - u - v;

    // z (height) and nr (ring size, in units of 4 cells).
    let third = 3.0 * nside * nside;
    let (z, nr) = if jr < nside {
        // north polar cap
        (1.0 - jr * jr / third, jr)
    } else if jr > 3.0 * nside {
        // south polar cap
        let nr = 4.0 * nside - jr;
        (nr * nr / third - 1.0, nr)
    } else {
        // equatorial belt
        ((2.0 * nside - jr) * 2.0 / (3.0 * nside), nside)
    };

    // phi (longitude): the face's starting spot, shifted by how far along the diagonal we are.
    let mut tmp = JPLL[f] as f64 * nr + u - v;
    if tmp < 0.0 {
        tmp += 8.0 * nr;
    }
    // At a pole the ring has zero width and every longitude is the same point.
    let phi = if nr > 0.0 { (PI / 4.0) * tmp / nr } else { 0.0 };

    let sin_theta = (1.0 - z * z).max(0.0).sqrt();
    [sin_theta * phi.cos(), sin_theta * phi.sin(), z.clamp(-1.0, 1.0)]
}

/// One cell on the globe.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellId {
    pub face: u8,
    pub level: u8,
    pub x: u32,
    pub y: u32,
}

impl CellId {
    /// The 4 cells this one splits into at level + 1.
    pub fn children(&self) -> [CellId; 4] {
        let (face, level) = (self.face, self.level + 1);
        let (x, y) = (self.x * 2, self.y * 2); // top-left child, in the finer grid
        [
            CellId { face, level, x,         y },
            CellId { face, level, x: x + 1,  y },
            CellId { face, level, x,         y: y + 1 },
            CellId { face, level, x: x + 1,  y: y + 1 },
        ]
    }

    /// The cell one level up that contains this one. None at level 0.
    pub fn parent(&self) -> Option<CellId> {
        if self.level == 0 {
            return None;
        }
        // Integer division throws away the remainder: 2 and 3 both become 1.
        Some(CellId { face: self.face, level: self.level - 1, x: self.x / 2, y: self.y / 2 })
    }

    /// The cell at a coarser `level` that contains this one (itself if `level` equals ours).
    pub fn ancestor_at(&self, level: u8) -> CellId {
        assert!(level <= self.level);
        let shift = self.level - level;
        CellId { face: self.face, level, x: self.x >> shift, y: self.y >> shift }
    }

    /// Direction from the planet's center to the middle of this cell.
    pub fn center(&self) -> Dir {
        point(self.face, self.level, self.x as f64 + 0.5, self.y as f64 + 0.5)
    }

    /// The 4 corners of this cell as directions, going around the cell:
    /// (x, y), (x+1, y), (x+1, y+1), (x, y+1). Neighbouring cells share corners exactly.
    pub fn corners(&self) -> [Dir; 4] {
        let (x, y) = (self.x as f64, self.y as f64);
        let p = |u, v| point(self.face, self.level, u, v);
        [p(x, y), p(x + 1.0, y), p(x + 1.0, y + 1.0), p(x, y + 1.0)]
    }

    /// The cell at `level` that contains direction `dir`.
    pub fn from_dir(dir: Dir, level: u8) -> CellId {
        let nside = 1_i64 << level;
        let z = dir[2];
        let za = z.abs();
        let phi = dir[1].atan2(dir[0]).rem_euclid(2.0 * PI);
        let tt = phi * 2.0 / PI; // longitude in "quarter turns", 0..4

        let (face, ix, iy);
        if za <= 2.0 / 3.0 {
            // equatorial belt: two families of diagonal lines cross each cell
            let t1 = nside as f64 * (0.5 + tt);
            let t2 = nside as f64 * z * 0.75;
            let jp = (t1 - t2) as i64; // ascending edge line
            let jm = (t1 + t2) as i64; // descending edge line
            let ifp = jp / nside;
            let ifm = jm / nside;
            face = if ifp == ifm {
                ifp | 4
            } else if ifp < ifm {
                ifp
            } else {
                ifm + 8
            };
            ix = jm & (nside - 1);
            iy = nside - (jp & (nside - 1)) - 1;
        } else {
            // polar caps
            let ntt = (tt as i64).min(3);
            let tp = tt - ntt as f64;
            let sin_theta = (1.0 - z * z).max(0.0).sqrt();
            let tmp = if za < 0.99 {
                nside as f64 * (3.0 * (1.0 - za)).sqrt()
            } else {
                nside as f64 * sin_theta / ((1.0 + za) / 3.0).sqrt()
            };
            let jp = ((tp * tmp) as i64).min(nside - 1);
            let jm = (((1.0 - tp) * tmp) as i64).min(nside - 1);
            if z >= 0.0 {
                face = ntt;
                ix = nside - jm - 1;
                iy = nside - jp - 1;
            } else {
                face = ntt + 8;
                ix = jp;
                iy = jm;
            }
        }
        CellId { face: face as u8, level, x: ix as u32, y: iy as u32 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn len(d: Dir) -> f64 {
        (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
    }

    // ---- step 1 ----
    #[test]
    fn children_have_this_cell_as_parent() {
        let c = CellId { face: 3, level: 2, x: 1, y: 3 };
        for ch in c.children() {
            assert_eq!(ch.level, 3);
            assert_eq!(ch.parent(), Some(c));
        }
    }

    #[test]
    fn four_distinct_children() {
        let c = CellId { face: 0, level: 0, x: 0, y: 0 };
        let kids = c.children();
        for i in 0..4 {
            for j in (i + 1)..4 {
                assert_ne!(kids[i], kids[j]);
            }
        }
    }

    #[test]
    fn level_zero_has_no_parent() {
        assert_eq!(CellId { face: 5, level: 0, x: 0, y: 0 }.parent(), None);
    }

    // ---- step 2 ----
    #[test]
    fn centers_are_unit_length() {
        for face in 0..BASE_FACES {
            let c = CellId { face, level: 3, x: 2, y: 5 };
            assert!((len(c.center()) - 1.0).abs() < 1e-9);
        }
    }

    // ---- step 3 ----
    #[test]
    fn center_roundtrips() {
        for face in 0..BASE_FACES {
            for (x, y) in [(0, 0), (3, 1), (7, 7), (2, 6)] {
                let c = CellId { face, level: 3, x, y };
                assert_eq!(CellId::from_dir(c.center(), 3), c);
            }
        }
    }

    #[test]
    fn coarser_lookup_matches_parent() {
        let c = CellId { face: 1, level: 4, x: 9, y: 6 };
        assert_eq!(CellId::from_dir(c.center(), 3), c.parent().unwrap());
    }

    // ---- extra checks ----
    #[test]
    fn every_cell_roundtrips_at_level_3() {
        for face in 0..BASE_FACES {
            for x in 0..8 {
                for y in 0..8 {
                    let c = CellId { face, level: 3, x, y };
                    assert_eq!(CellId::from_dir(c.center(), 3), c, "{:?}", c);
                }
            }
        }
    }

    #[test]
    fn random_points_land_in_a_valid_cell_near_the_point() {
        // tiny deterministic pseudo-random generator, no extra crates needed
        let mut s: u64 = 12345;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / ((1u64 << 53) as f64)
        };
        for _ in 0..5000 {
            let z = 2.0 * next() - 1.0;
            let phi = 2.0 * PI * next();
            let r = (1.0 - z * z).sqrt();
            let d = [r * phi.cos(), r * phi.sin(), z];
            let c = CellId::from_dir(d, 5);
            assert!(c.face < BASE_FACES && c.x < 32 && c.y < 32);
            let m = c.center();
            let dot = d[0] * m[0] + d[1] * m[1] + d[2] * m[2];
            assert!(dot > 0.99, "point too far from its cell center: {:?}", c);
        }
    }

    // ---- corners ----
    #[test]
    fn corners_are_unit_length() {
        for face in 0..BASE_FACES {
            let c = CellId { face, level: 2, x: 1, y: 2 };
            for p in c.corners() {
                assert!((len(p) - 1.0).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn neighbours_in_a_face_share_corners() {
        let a = CellId { face: 6, level: 3, x: 2, y: 4 };
        let right = CellId { x: 3, ..a };
        let below = CellId { y: 5, ..a };
        let close = |p: Dir, q: Dir| (0..3).all(|i| (p[i] - q[i]).abs() < 1e-12);
        assert!(close(a.corners()[1], right.corners()[0]));
        assert!(close(a.corners()[2], right.corners()[3]));
        assert!(close(a.corners()[3], below.corners()[0]));
        assert!(close(a.corners()[2], below.corners()[1]));
    }

    #[test]
    fn whole_globe_has_no_gaps_or_overlaps() {
        // A closed surface cut into 12*n*n quads has exactly 12*n*n + 2 distinct corner points.
        // Any seam between faces that doesn't line up would add extra points.
        use std::collections::HashSet;
        for level in 0..=4u8 {
            let n = 1u32 << level;
            let mut pts = HashSet::new();
            for face in 0..BASE_FACES {
                for x in 0..n {
                    for y in 0..n {
                        for p in (CellId { face, level, x, y }).corners() {
                            let key = p.map(|c| (c * 1e8).round() as i64);
                            pts.insert(key);
                        }
                    }
                }
            }
            assert_eq!(pts.len() as u32, 12 * n * n + 2, "level {level}");
        }
    }
}
