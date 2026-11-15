//! Turns cells into triangles the GPU can draw. No GPU code in here, so it's testable.

use crate::sphere::{CellId, BASE_FACES};

/// One point sent to the GPU: where it is and what color it is.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub color: [f32; 3],
}

fn scramble(c: &CellId) -> u64 {
    // Mix the cell's numbers into one 64-bit value (splitmix64-style scrambling).
    let mut h = (c.face as u64) << 56 ^ (c.level as u64) << 48 ^ (c.x as u64) << 24 ^ c.y as u64;
    h = h.wrapping_add(0x9E37_79B9_7F4A_7C15);
    h = (h ^ (h >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h = (h ^ (h >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^ (h >> 31)
}

/// A stable pseudo-random color for a cell: the same cell always gets the same color.
///
/// Each cell's color is its parent's color plus a small random nudge. So zoomed out,
/// a big cell is roughly the average of the little ones inside it, and zooming in
/// only adds variation around the same hue (instead of everything changing at once).
pub fn cell_color(c: &CellId) -> [f32; 3] {
    let root = scramble(&c.ancestor_at(0));
    let mut col = [0u32, 8, 16].map(|sh| 0.3 + 0.6 * (((root >> sh) & 0xFF) as f32 / 255.0));
    for level in 1..=c.level {
        let h = scramble(&c.ancestor_at(level));
        for (i, sh) in [0u32, 8, 16].into_iter().enumerate() {
            let nudge = (((h >> sh) & 0xFF) as f32 / 255.0 - 0.5) * 0.18;
            col[i] = (col[i] + nudge).clamp(0.2, 1.0);
        }
    }
    col
}

/// Two triangles per cell (6 vertices), flat color per cell.
pub fn build_cells(cells: &[CellId]) -> Vec<Vertex> {
    let mut out = Vec::with_capacity(cells.len() * 6);
    for cell in cells {
        let color = cell_color(cell);
        let [a, b, c, d] = cell.corners().map(|p| [p[0] as f32, p[1] as f32, p[2] as f32]);
        // Cut the quad a-b-c-d into triangles (a,b,c) and (a,c,d).
        for pos in [a, b, c, a, c, d] {
            out.push(Vertex { pos, color });
        }
    }
    out
}

/// Every cell at `level`.
pub fn build_globe(level: u8) -> Vec<Vertex> {
    let n = 1u32 << level;
    let mut cells = Vec::with_capacity(BASE_FACES as usize * (n * n) as usize);
    for face in 0..BASE_FACES {
        for x in 0..n {
            for y in 0..n {
                cells.push(CellId { face, level, x, y });
            }
        }
    }
    build_cells(&cells)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn six_vertices_per_cell() {
        assert_eq!(build_globe(2).len(), 12 * 16 * 6);
    }

    #[test]
    fn colors_are_stable_and_in_range() {
        let c = CellId { face: 3, level: 4, x: 5, y: 6 };
        assert_eq!(cell_color(&c), cell_color(&c));
        assert!(cell_color(&c).iter().all(|v| (0.2..=1.0).contains(v)));
    }
}

#[cfg(test)]
mod color_tests {
    use super::*;

    #[test]
    fn child_color_stays_near_parent_color() {
        let parent = CellId { face: 2, level: 5, x: 7, y: 9 };
        let pc = cell_color(&parent);
        for ch in parent.children() {
            let cc = cell_color(&ch);
            for i in 0..3 {
                assert!((pc[i] - cc[i]).abs() <= 0.09 + 1e-6);
            }
        }
    }
}
