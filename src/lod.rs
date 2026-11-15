//! Level of detail: decides which cells to draw for a given camera.
//!
//! Start from the coarse "root" cells. A cell that looks bigger than TARGET_PX on screen is
//! replaced by its 4 children (and those are checked the same way). Cells on the far side
//! of the globe or outside the window are skipped. Far away that gives a few thousand big
//! cells; up close, many small ones, but still only a few thousand.

use crate::sphere::{CellId, BASE_FACES};
use glam::{Mat4, Vec3, Vec4};

/// Detail level the drawing starts from. 12 * 4^ROOT_LEVEL root faces (level 1 = 48).
pub const ROOT_LEVEL: u8 = 1;
/// Finest level. Each root face is 2^(MAX_LEVEL - ROOT_LEVEL) cells wide (level 11 = 1024).
pub const MAX_LEVEL: u8 = 11;
/// A cell is split while it covers more than this many screen pixels (across its bounding circle).
pub const TARGET_PX: f32 = 14.0;
/// Safety cap on how many cells we ever draw at once.
pub const MAX_CELLS: usize = 60_000;

/// Everything about the camera that the detail decision needs.
pub struct View {
    /// Camera position, in the globe's own coordinates (the globe has radius 1).
    pub eye: Vec3,
    /// Globe coordinates -> screen clip space.
    pub mvp: Mat4,
    /// Pixels per unit of size at distance 1 (how "zoomed" the lens is).
    pub focal_px: f32,
    /// Window size in pixels.
    pub viewport: (f32, f32),
}

/// A cell's middle and the radius of a ball around that middle that contains the whole cell.
fn bounds(c: &CellId) -> (Vec3, f32) {
    let center = Vec3::from(c.center().map(|v| v as f32));
    let radius = c
        .corners()
        .iter()
        .map(|p| (Vec3::from(p.map(|v| v as f32)) - center).length())
        .fold(0.0, f32::max);
    (center, radius)
}

/// Could any part of this cell be on screen? (Conservative: when unsure, says yes.)
fn on_screen(view: &View, center: Vec3, radius: f32) -> bool {
    // Horizon: a point p on the unit globe can be seen from `eye` if p . eye >= 1.
    if center.dot(view.eye) + radius * view.eye.length() < 1.0 {
        return false;
    }
    // Window edges.
    let clip = view.mvp * Vec4::new(center.x, center.y, center.z, 1.0);
    if clip.w <= radius {
        return true; // too close to the camera to project reliably, keep it
    }
    let (vw, vh) = view.viewport;
    let r_ndc_x = radius * view.focal_px / clip.w / (vw * 0.5);
    let r_ndc_y = radius * view.focal_px / clip.w / (vh * 0.5);
    (clip.x / clip.w).abs() <= 1.0 + r_ndc_x && (clip.y / clip.w).abs() <= 1.0 + r_ndc_y
}

/// The cells to draw for this camera. Never overlap, never contain each other.
pub fn visible_cells(view: &View) -> Vec<CellId> {
    let n = 1u32 << ROOT_LEVEL;
    let mut stack = Vec::new();
    for face in 0..BASE_FACES {
        for x in 0..n {
            for y in 0..n {
                stack.push(CellId { face, level: ROOT_LEVEL, x, y });
            }
        }
    }

    let mut out = Vec::new();
    while let Some(cell) = stack.pop() {
        let (center, radius) = bounds(&cell);
        if !on_screen(view, center, radius) {
            continue;
        }
        let dist = (view.eye - center).length().max(1e-6);
        let size_px = 2.0 * radius * view.focal_px / dist;
        let room = out.len() + stack.len() + 4 < MAX_CELLS;
        if size_px > TARGET_PX && cell.level < MAX_LEVEL && room {
            stack.extend(cell.children());
        } else {
            out.push(cell);
        }
    }
    out
}
