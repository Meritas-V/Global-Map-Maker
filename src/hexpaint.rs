//! Painting on the hex globe: what is painted, and undo.
//!
//! Paint is stored sparsely: one map per H3 resolution, holding only the cells that were
//! actually painted. Painting a cell at resolution R wipes everything painted *inside* it at
//! finer resolutions (you paint one level at a time). Every stroke remembers the old values
//! of what it changed, so Ctrl+Z puts it all back.
//!
//! Value 0 is an explicit "hole": bare ground showing through a coarser paint. That is how
//! erasing a small spot inside a big painted cell works without splitting the big cell.

use h3o::{CellIndex, Resolution};
use std::collections::HashMap;

/// What a cell is painted with: an index into PALETTE. 0 = bare ground.
pub type Paint = u8;
pub const RED: Paint = 1;

/// Colors you can paint with. Index 0 is a placeholder for bare ground (the map's base color).
pub const PALETTE: [([f32; 3], &str); 10] = [
    ([0.0, 0.0, 0.0], "erase"),
    ([0.62, 0.02, 0.05], "deep red"),
    ([0.30, 0.62, 0.20], "grass"),
    ([0.08, 0.36, 0.14], "forest"),
    ([0.86, 0.78, 0.50], "sand"),
    ([0.45, 0.45, 0.47], "rock"),
    ([0.95, 0.96, 1.00], "snow"),
    ([0.45, 0.30, 0.16], "dirt"),
    ([0.22, 0.55, 0.80], "shallow water"),
    ([0.05, 0.15, 0.45], "deep water"),
];
pub const PALETTE_LEN: usize = PALETTE.len();

pub fn color(p: Paint) -> [f32; 3] {
    PALETTE.get(p as usize).map(|c| c.0).unwrap_or([1.0, 0.0, 1.0])
}

pub fn name(p: Paint) -> &'static str {
    PALETTE.get(p as usize).map(|c| c.1).unwrap_or("?")
}

const RES_COUNT: usize = 16;

fn res_of(r: usize) -> Resolution {
    Resolution::try_from(r as u8).unwrap()
}

fn level(cell: CellIndex) -> usize {
    usize::from(u8::from(cell.resolution()))
}

/// What one stroke changed: for each touched cell, its entry *before* the stroke.
#[derive(Default)]
struct Stroke {
    old: HashMap<CellIndex, Option<Paint>>,
}

pub struct PaintStore {
    levels: Vec<HashMap<CellIndex, Paint>>,
    undo: Vec<Stroke>,
    current: Option<Stroke>,
}

impl Default for PaintStore {
    fn default() -> Self {
        PaintStore {
            levels: (0..RES_COUNT).map(|_| HashMap::new()).collect(),
            undo: Vec::new(),
            current: None,
        }
    }
}

impl PaintStore {
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

    fn remember(&mut self, cell: CellIndex, old: Option<Paint>) {
        if let Some(s) = self.current.as_mut() {
            s.old.entry(cell).or_insert(old); // only the first (original) value counts
        }
    }

    /// The paint coming from strictly coarser cells (what shows if this cell had no entry of its own).
    fn value_above(&self, cell: CellIndex) -> Paint {
        for l in (0..level(cell)).rev() {
            if self.levels[l].is_empty() {
                continue;
            }
            if let Some(v) = cell.parent(res_of(l)).and_then(|c| self.levels[l].get(&c)) {
                return *v;
            }
        }
        0
    }

    /// Paint one cell at its own resolution (0 erases). Anything painted inside it (finer) is wiped.
    pub fn paint(&mut self, cell: CellIndex, value: Paint) {
        let r = level(cell);
        for f in (r + 1)..RES_COUNT {
            if self.levels[f].is_empty() {
                continue;
            }
            let fres = res_of(f);
            let map = &self.levels[f];
            let doomed: Vec<CellIndex> = if (map.len() as u64) > cell.children_count(fres) {
                cell.children(fres).filter(|c| map.contains_key(c)).collect()
            } else {
                map.keys().filter(|k| k.parent(cell.resolution()) == Some(cell)).copied().collect()
            };
            for d in doomed {
                let old = self.levels[f].remove(&d);
                self.remember(d, old);
            }
        }
        let old_exact = self.levels[r].get(&cell).copied();
        if value == self.value_above(cell) {
            // The coarser paint already shows this: no entry needed.
            if old_exact.is_some() {
                self.levels[r].remove(&cell);
                self.remember(cell, old_exact);
            }
        } else if old_exact != Some(value) {
            self.levels[r].insert(cell, value);
            self.remember(cell, old_exact);
        }
    }

    pub fn undo_depth(&self) -> usize {
        self.undo.len()
    }

    /// Undo the last stroke. Returns false if there was nothing to undo.
    pub fn undo(&mut self) -> bool {
        let Some(stroke) = self.undo.pop() else { return false };
        for (cell, old) in stroke.old {
            let r = level(cell);
            match old {
                Some(v) => self.levels[r].insert(cell, v),
                None => self.levels[r].remove(&cell),
            };
        }
        true
    }

    /// The finest entry at or above this cell (Some(0) = a hole in coarser paint).
    pub fn value_at(&self, cell: CellIndex) -> Option<Paint> {
        for l in (0..=level(cell)).rev() {
            if self.levels[l].is_empty() {
                continue;
            }
            if let Some(v) = cell.parent(res_of(l)).and_then(|c| self.levels[l].get(&c)) {
                return Some(*v);
            }
        }
        None
    }

    /// What this cell is painted with, 0 if nothing.
    pub fn effective(&self, cell: CellIndex) -> Paint {
        self.value_at(cell).unwrap_or(0)
    }

    /// For zoomed-out views: for each coarse cell, how much of it is covered by finer paint,
    /// per paint type (fractions of the cell, 0..1). Index 0 counts holes. This is the
    /// "average" you see from space.
    pub fn coverage(&self, res: Resolution) -> HashMap<CellIndex, [f32; PALETTE_LEN]> {
        let r = usize::from(u8::from(res));
        let mut out: HashMap<CellIndex, [f32; PALETTE_LEN]> = HashMap::new();
        for f in (r + 1)..RES_COUNT {
            let share = 7f32.powi(-((f - r) as i32));
            for (cell, v) in &self.levels[f] {
                if let Some(anc) = cell.parent(res) {
                    out.entry(anc).or_insert([0.0; PALETTE_LEN])[(*v as usize).min(PALETTE_LEN - 1)] += share;
                }
            }
        }
        out
    }

    pub fn painted_cells(&self) -> usize {
        self.levels.iter().map(|m| m.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use h3o::LatLng;

    fn cell(res: u8) -> CellIndex {
        LatLng::new(55.0, 9.8).unwrap().to_cell(Resolution::try_from(res).unwrap())
    }

    #[test]
    fn painting_coarse_wipes_fine_inside_and_undo_restores() {
        let mut s = PaintStore::default();
        let coarse = cell(5);
        let fine: Vec<CellIndex> = coarse.children(Resolution::try_from(7).unwrap()).take(10).collect();
        s.begin_stroke();
        for f in &fine {
            s.paint(*f, RED);
        }
        s.end_stroke();
        assert_eq!(s.painted_cells(), 10);

        s.begin_stroke();
        s.paint(coarse, RED);
        s.end_stroke();
        assert_eq!(s.painted_cells(), 1, "fine cells inside should be gone");
        assert_eq!(s.value_at(fine[3]), Some(RED), "still shows as painted through its parent");

        assert!(s.undo());
        assert_eq!(s.painted_cells(), 10, "undo brings the fine cells back");
        assert_eq!(s.value_at(coarse), None);
        assert!(s.undo());
        assert_eq!(s.painted_cells(), 0);
        assert!(!s.undo());
    }

    #[test]
    fn stroke_remembers_original_value_only_once() {
        let mut s = PaintStore::default();
        let c = cell(6);
        s.begin_stroke();
        s.paint(c, RED);
        s.paint(c, 2);
        s.end_stroke();
        assert_eq!(s.effective(c), 2);
        assert!(s.undo());
        assert_eq!(s.value_at(c), None, "back to unpainted, not to the middle value");
    }

    #[test]
    fn coverage_is_a_fraction_of_the_parent_per_color() {
        let mut s = PaintStore::default();
        let parent = cell(4);
        let kids: Vec<CellIndex> = parent.children(Resolution::try_from(5).unwrap()).collect();
        s.begin_stroke();
        for (i, k) in kids.iter().enumerate() {
            s.paint(*k, if i < 3 { RED } else { 2 });
        }
        s.end_stroke();
        let cov = s.coverage(Resolution::try_from(4).unwrap())[&parent];
        assert!((cov[RED as usize] - 3.0 / 7.0).abs() < 1e-4, "{cov:?}");
        assert!((cov[2] - (kids.len() as f32 - 3.0) / 7.0).abs() < 1e-4, "{cov:?}");
    }

    #[test]
    fn erasing_inside_a_big_paint_makes_a_hole_and_undo_closes_it() {
        let mut s = PaintStore::default();
        let big = cell(4);
        let small = big.children(Resolution::try_from(7).unwrap()).nth(100).unwrap();
        s.begin_stroke();
        s.paint(big, RED);
        s.end_stroke();
        s.begin_stroke();
        s.paint(small, 0);
        s.end_stroke();
        assert_eq!(s.effective(small), 0, "hole");
        assert_eq!(s.effective(big), RED, "rest of the big cell untouched");
        assert!(s.undo());
        assert_eq!(s.effective(small), RED);
    }

    #[test]
    fn painting_the_same_color_over_existing_paint_stores_nothing_new() {
        let mut s = PaintStore::default();
        let big = cell(4);
        let small = big.children(Resolution::try_from(8).unwrap()).next().unwrap();
        s.begin_stroke();
        s.paint(big, RED);
        s.paint(small, RED);
        s.end_stroke();
        assert_eq!(s.painted_cells(), 1);
    }

    #[test]
    fn erasing_bare_ground_stores_nothing() {
        let mut s = PaintStore::default();
        s.begin_stroke();
        s.paint(cell(6), 0);
        s.end_stroke();
        assert_eq!(s.painted_cells(), 0);
        assert!(!s.undo(), "nothing changed, nothing to undo");
    }
}
