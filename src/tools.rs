//! Paint tools on the hex grid: shapes and fill. Pure functions, no window needed.

use crate::hexpaint::PaintStore;
use h3o::CellIndex;
use std::collections::{HashSet, VecDeque};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tool {
    Pencil,
    Eraser,
    Line,
    Circle,
    Fill,
    Eyedropper,
}

impl Tool {
    pub fn name(self) -> &'static str {
        match self {
            Tool::Pencil => "pencil",
            Tool::Eraser => "eraser",
            Tool::Line => "line",
            Tool::Circle => "circle",
            Tool::Fill => "fill",
            Tool::Eyedropper => "eyedropper",
        }
    }
}

/// Every cell within `rings` steps of any of the given cells (the brush thickness).
pub fn expand(cells: impl IntoIterator<Item = CellIndex>, rings: u32) -> HashSet<CellIndex> {
    let mut out = HashSet::new();
    for c in cells {
        if rings == 0 {
            out.insert(c);
        } else {
            out.extend(c.grid_disk::<Vec<_>>(rings));
        }
    }
    out
}

/// Straight line of hexes from a to b (inclusive). Falls back to just b if the grid can't draw it.
pub fn line_cells(a: CellIndex, b: CellIndex) -> Vec<CellIndex> {
    if a.resolution() != b.resolution() {
        return vec![b];
    }
    match a.grid_path_cells(b) {
        Ok(it) => {
            let v: Vec<CellIndex> = it.filter_map(Result::ok).collect();
            if v.is_empty() {
                vec![b]
            } else {
                v
            }
        }
        Err(_) => vec![b],
    }
}

/// Number of steps between two hexes, if the grid can tell.
pub fn distance(a: CellIndex, b: CellIndex) -> Option<u32> {
    if a.resolution() != b.resolution() {
        return None;
    }
    a.grid_distance(b).ok().map(|d| d.max(0) as u32)
}

/// Filled disk (a hexagon-ish circle) of hexes.
pub fn disk_cells(center: CellIndex, radius: u32) -> Vec<CellIndex> {
    center.grid_disk::<Vec<_>>(radius)
}

/// Just the outline of that disk.
pub fn ring_cells(center: CellIndex, radius: u32) -> Vec<CellIndex> {
    if radius == 0 {
        return vec![center];
    }
    let inner: HashSet<CellIndex> = center.grid_disk::<Vec<_>>(radius - 1).into_iter().collect();
    center.grid_disk::<Vec<_>>(radius).into_iter().filter(|c| !inner.contains(c)).collect()
}

/// Bucket fill: every connected hex (inside `allowed`) with the same paint as `start`.
/// Stops after `cap` hexes so a fill on open ground can't run away.
pub fn flood_fill(start: CellIndex, allowed: &HashSet<CellIndex>, store: &PaintStore, cap: usize) -> Vec<CellIndex> {
    let target = store.effective(start);
    let mut seen: HashSet<CellIndex> = HashSet::new();
    let mut out = Vec::new();
    let mut queue = VecDeque::new();
    seen.insert(start);
    queue.push_back(start);
    while let Some(c) = queue.pop_front() {
        out.push(c);
        if out.len() >= cap {
            break;
        }
        for n in c.grid_disk::<Vec<_>>(1) {
            if n != c && allowed.contains(&n) && !seen.contains(&n) && store.effective(n) == target {
                seen.insert(n);
                queue.push_back(n);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hexpaint::RED;
    use h3o::{LatLng, Resolution};

    fn cell() -> CellIndex {
        LatLng::new(10.0, 20.0).unwrap().to_cell(Resolution::try_from(9).unwrap())
    }

    #[test]
    fn line_has_both_ends_and_no_gaps() {
        let a = cell();
        let b = a.grid_disk::<Vec<_>>(6).into_iter().find(|c| a.grid_distance(*c) == Ok(6)).unwrap();
        let line = line_cells(a, b);
        assert_eq!(line.first(), Some(&a));
        assert_eq!(line.last(), Some(&b));
        assert_eq!(line.len(), 7);
        for w in line.windows(2) {
            assert_eq!(w[0].grid_distance(w[1]), Ok(1));
        }
    }

    #[test]
    fn ring_and_disk_sizes() {
        let c = cell();
        assert_eq!(disk_cells(c, 3).len(), 37);
        assert_eq!(ring_cells(c, 3).len(), 18);
        assert_eq!(expand([c], 2).len(), 19);
    }

    #[test]
    fn fill_stops_at_a_painted_outline() {
        let c = cell();
        let mut store = PaintStore::default();
        store.begin_stroke();
        for r in ring_cells(c, 3) {
            store.paint(r, RED);
        }
        store.end_stroke();
        let allowed: HashSet<CellIndex> = disk_cells(c, 8).into_iter().collect();
        let inside = flood_fill(c, &allowed, &store, 10_000);
        assert_eq!(inside.len(), 37 - 18, "only the 19 cells inside the outline");
        // From outside, the fill takes everything allowed that is bare ground.
        let outside_start = disk_cells(c, 8).into_iter().find(|x| c.grid_distance(*x) == Ok(8)).unwrap();
        let outside = flood_fill(outside_start, &allowed, &store, 10_000);
        assert_eq!(outside.len(), disk_cells(c, 8).len() - 37);
    }

    #[test]
    fn fill_respects_the_cap() {
        let c = cell();
        let allowed: HashSet<CellIndex> = disk_cells(c, 10).into_iter().collect();
        assert_eq!(flood_fill(c, &allowed, &PaintStore::default(), 50).len(), 50);
    }
}
