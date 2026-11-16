//! The hex globe as triangles: every hex is a fan around its middle.

use crate::hexpaint::{self, PaintStore, PALETTE_LEN};
use crate::mesh::Vertex;
use glam::DVec3;
use h3o::{CellIndex, LatLng, Resolution};
use std::collections::{HashMap, HashSet};

pub fn to_dir64(ll: LatLng) -> DVec3 {
    let (lat, lng) = (ll.lat_radians(), ll.lng_radians());
    // z is north, same as the square-cell globe.
    DVec3::new(lat.cos() * lng.cos(), lat.cos() * lng.sin(), lat.sin())
}

/// Position relative to `origin`, worked out in f64, handed to the GPU as f32.
fn rel(p: DVec3, origin: DVec3) -> [f32; 3] {
    (p - origin).as_vec3().to_array()
}

fn scramble(v: u64) -> u64 {
    let mut h = v.wrapping_add(0x9E37_79B9_7F4A_7C15);
    h = (h ^ (h >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^ (h >> 31)
}

/// Unpainted ground: a dark, slightly bluish gray. Each level nudges it a little from its parent's
/// tone (so the grid stays readable and zoomed-out cells look like an average of what's inside).
/// Kept dull on purpose so every palette color stands out against it.
pub fn base_color(cell: CellIndex, res: Resolution) -> [f32; 3] {
    let mut col = [0.17f32, 0.20, 0.25];
    for level in 0..=u8::from(res) {
        let Some(anc) = Resolution::try_from(level).ok().and_then(|r| cell.parent(r)) else { continue };
        let h = scramble(u64::from(anc));
        let unit = (h & 0xFF) as f32 / 255.0 - 0.5;
        let step = if level == 0 { 0.08 } else { 0.03 };
        for (i, c) in col.iter_mut().enumerate() {
            *c = (*c + unit * step * (1.0 + 0.1 * i as f32)).clamp(0.08, 0.34);
        }
    }
    col
}

fn mix(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
}

/// The color a hex is drawn with: its own paint (or a coarser cell's), mixed with the average of
/// whatever finer paint lies inside it, lightened if the brush is over it.
pub fn cell_color(
    cell: CellIndex,
    res: Resolution,
    store: &PaintStore,
    coverage: &HashMap<CellIndex, [f32; PALETTE_LEN]>,
    highlighted: &HashSet<CellIndex>,
) -> [f32; 3] {
    let base = base_color(cell, res);
    let under = store.effective(cell);
    let mut col = if under == 0 { base } else { hexpaint::color(under) };
    if let Some(cov) = coverage.get(&cell) {
        let total: f32 = cov.iter().sum();
        let norm = total.max(1.0);
        let mut mixed = [0.0f32; 3];
        for (i, share) in cov.iter().enumerate() {
            if *share > 0.0 {
                let c = if i == 0 { base } else { hexpaint::color(i as u8) };
                for k in 0..3 {
                    mixed[k] += c[k] * share / norm;
                }
            }
        }
        let t = (total / norm).min(1.0);
        col = [col[0] * (1.0 - t) + mixed[0], col[1] * (1.0 - t) + mixed[1], col[2] * (1.0 - t) + mixed[2]];
    }
    if highlighted.contains(&cell) {
        col = mix(col, [1.0, 1.0, 1.0], 0.45);
    }
    col
}

/// Triangles for the given hexes.
pub fn build_hex_cells(
    cells: &[CellIndex],
    res: Resolution,
    store: &PaintStore,
    coverage: &HashMap<CellIndex, [f32; PALETTE_LEN]>,
    highlighted: &HashSet<CellIndex>,
    origin: DVec3,
) -> Vec<Vertex> {
    let mut out = Vec::with_capacity(cells.len() * 18);
    for &cell in cells {
        let color = cell_color(cell, res, store, coverage, highlighted);
        let center = rel(to_dir64(LatLng::from(cell)), origin);
        let ring: Vec<[f32; 3]> = cell.boundary().iter().map(|ll| rel(to_dir64(*ll), origin)).collect();
        for i in 0..ring.len() {
            let (a, b) = (ring[i], ring[(i + 1) % ring.len()]);
            for pos in [center, a, b] {
                out.push(Vertex { pos, color });
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn triangles_are_on_the_sphere_and_ground_is_dull() {
        let res = Resolution::try_from(1).unwrap();
        let cells: Vec<CellIndex> = CellIndex::base_cells().flat_map(|b| b.children(res)).collect();
        assert_eq!(cells.len() as u64, 2 + 120 * 7);
        let store = PaintStore::default();
        let v = build_hex_cells(&cells, res, &store, &HashMap::new(), &HashSet::new(), DVec3::ZERO);
        assert_eq!(v.len() % 3, 0);
        assert!(v.len() >= cells.len() * 15);
        for p in &v {
            let l = (p.pos[0].powi(2) + p.pos[1].powi(2) + p.pos[2].powi(2)).sqrt();
            assert!((l - 1.0).abs() < 1e-3, "{l}");
            assert!(p.color.iter().all(|c| *c < 0.4), "base ground should be dark: {:?}", p.color);
        }
    }

    #[test]
    fn painted_cell_has_its_palette_color_and_highlight_is_lighter() {
        let res = Resolution::try_from(3).unwrap();
        let cell = LatLng::new(10.0, 20.0).unwrap().to_cell(res);
        let mut store = PaintStore::default();
        store.begin_stroke();
        store.paint(cell, hexpaint::RED);
        store.end_stroke();
        let none = HashSet::new();
        let cov = store.coverage(res);
        assert_eq!(cell_color(cell, res, &store, &cov, &none), hexpaint::color(hexpaint::RED));
        let hi: HashSet<_> = [cell].into_iter().collect();
        assert!(cell_color(cell, res, &store, &cov, &hi)[1] > 0.2);
    }

    #[test]
    fn zoomed_out_cell_is_the_average_of_what_is_inside() {
        let res = Resolution::try_from(3).unwrap();
        let parent = LatLng::new(10.0, 20.0).unwrap().to_cell(res);
        let mut store = PaintStore::default();
        store.begin_stroke();
        let kids: Vec<CellIndex> = parent.children(Resolution::try_from(4).unwrap()).collect();
        for k in &kids {
            store.paint(*k, hexpaint::RED);
        }
        store.end_stroke();
        let cov = store.coverage(res);
        let c = cell_color(parent, res, &store, &cov, &HashSet::new());
        let want = hexpaint::color(hexpaint::RED);
        for i in 0..3 {
            assert!((c[i] - want[i]).abs() < 0.02, "{c:?} vs {want:?}");
        }
        // Half painted: halfway between ground and red.
        let mut store2 = PaintStore::default();
        store2.begin_stroke();
        for k in kids.iter().take(kids.len() / 2) {
            store2.paint(*k, hexpaint::RED);
        }
        store2.end_stroke();
        let c2 = cell_color(parent, res, &store2, &store2.coverage(res), &HashSet::new());
        assert!(c2[0] > base_color(parent, res)[0] + 0.1 && c2[0] < want[0]);
    }
}
