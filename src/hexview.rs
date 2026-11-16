//! Which hex resolution to show for the current zoom, which cells are on screen,
//! and which cell is under the mouse.

use crate::camera::Camera;
use glam::DVec3;
use h3o::{CellIndex, LatLng, Resolution};

/// Finest hex level (H3 res 12: hexes about 20 m wide on an Earth-sized globe). Finer detail comes from the pixel layer.
pub const MAX_PAINT_RES: u8 = 12;
/// Aim for hexagons about this wide on screen.
pub const TARGET_HEX_PX: f64 = 22.0;
/// Most hexes drawn at once (18 vertices each; must fit the shared vertex buffer).
pub const MAX_HEX_CELLS: usize = 20_000;

const FOV_HALF_Y: f64 = 22.5 * std::f64::consts::PI / 180.0;

fn total_cells(res: u8) -> u64 {
    2 + 120 * 7u64.pow(res as u32)
}

/// Approximate width of one hex (flat to flat) on the unit sphere.
pub fn cell_width(res: u8) -> f64 {
    let area = 4.0 * std::f64::consts::PI / total_cells(res) as f64;
    1.0746 * area.sqrt()
}

/// Angular radius (radians) of the part of the globe the camera can see.
pub fn visible_radius(camera: &Camera, aspect: f64) -> f64 {
    let d = camera.distance;
    let horizon = (1.0 / d).acos();
    if camera.fly.is_some() {
        // Looking around freely: anything up to the horizon can be in view.
        return horizon;
    }
    let alpha = (FOV_HALF_Y.tan() * (1.0 + aspect * aspect).sqrt()).atan(); // screen corner
    let s = d * alpha.sin();
    if s >= 1.0 {
        horizon
    } else {
        ((s.asin() - alpha).max(0.0)).min(horizon)
    }
}

/// Ring count needed to cover `radius` radians around a cell.
fn rings_for(res: u8, radius: f64) -> u32 {
    (radius / (0.866 * cell_width(res))).ceil() as u32 + 2
}

fn estimated_count(res: u8, camera: &Camera, aspect: f64) -> u64 {
    let n = total_cells(res);
    if n <= MAX_HEX_CELLS as u64 {
        return n;
    }
    let k = rings_for(res, visible_radius(camera, aspect)) as u64;
    3 * k * (k + 1) + 1
}

/// Pick the resolution for this zoom: the finest one whose hexes are still at least
/// TARGET_HEX_PX wide, but never so fine that too many hexes land on screen.
pub fn choose_res(camera: &Camera, width: u32, height: u32) -> u8 {
    let (w, h) = (width.max(1) as f64, height.max(1) as f64);
    let aspect = w / h;
    let focal_px = Camera::focal_px(h);
    let above_surface = camera.height().max(1e-9);
    let mut res = 0;
    for r in 0..=MAX_PAINT_RES {
        if cell_width(r) * focal_px / above_surface >= TARGET_HEX_PX {
            res = r;
        }
    }
    while res > 0 && estimated_count(res, camera, aspect) > MAX_HEX_CELLS as u64 {
        res -= 1;
    }
    res
}

pub fn dir_to_latlng(p: DVec3) -> LatLng {
    let p = p.normalize();
    LatLng::from_radians(p.z.clamp(-1.0, 1.0).asin(), p.y.atan2(p.x)).unwrap()
}

/// All hexes at `res` that could be on screen.
pub fn visible_cells(res: u8, camera: &Camera, width: u32, height: u32) -> Vec<CellIndex> {
    let aspect = width.max(1) as f64 / height.max(1) as f64;
    let r = Resolution::try_from(res).unwrap();
    if total_cells(res) <= MAX_HEX_CELLS as u64 {
        return CellIndex::base_cells().flat_map(|b| b.children(r)).collect();
    }
    let center = dir_to_latlng(camera.eye64()).to_cell(r);
    let k = rings_for(res, visible_radius(camera, aspect));
    center.grid_disk::<Vec<_>>(k)
}

/// The ray (start, unit direction) through a screen position, in the globe's own coordinates.
pub fn pick_ray(camera: &Camera, width: u32, height: u32, x: f64, y: f64) -> (DVec3, DVec3) {
    let (w, h) = (width.max(1) as f64, height.max(1) as f64);
    let (nx, ny) = (2.0 * x / w - 1.0, 1.0 - 2.0 * y / h);
    // Straight from the lens: the ray in camera space, turned into globe coordinates (no projection matrix, so no
    // precision lost to the millimeter-near plane).
    let tan_y = (Camera::fov_y() * 0.5).tan();
    let d = DVec3::new(nx * tan_y * w / h, ny * tan_y, -1.0);
    let dir = camera.view64().inverse().transform_vector3(d).normalize();
    (camera.eye64(), dir)
}

/// The point on the globe under a screen position (in the globe's own coordinates), if any.
pub fn pick_point(camera: &Camera, width: u32, height: u32, x: f64, y: f64) -> Option<DVec3> {
    let (origin, dir) = pick_ray(camera, width, height, x, y);
    // Ray vs unit sphere.
    let b = origin.dot(dir);
    let c = origin.dot(origin) - 1.0;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let t = -b - disc.sqrt();
    (t > 0.0).then(|| origin + dir * t)
}

/// The hex at `res` under a screen position.
pub fn pick_cell(res: u8, camera: &Camera, width: u32, height: u32, x: f64, y: f64) -> Option<CellIndex> {
    let p = pick_point(camera, width, height, x, y)?;
    Some(dir_to_latlng(p).to_cell(Resolution::try_from(res).unwrap()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cam(distance: f64) -> Camera {
        Camera { yaw: 0.7, pitch: 0.3, distance, floor: 1.0, fly: None }
    }

    #[test]
    fn closer_means_finer_and_never_too_many_cells() {
        let mut last = 0;
        for h in [3.0f64, 1.5, 0.5, 0.1, 0.02, 0.005, 0.001, 0.0003, 0.00002, 0.000006] {
            let c = cam(1.0 + h);
            let res = choose_res(&c, 1280, 720);
            assert!(res >= last, "res should not drop when zooming in: {res} < {last} at {h}");
            last = res;
            let cells = visible_cells(res, &c, 1280, 720);
            assert!(cells.len() <= MAX_HEX_CELLS, "{} cells at height {h}", cells.len());
        }
        assert_eq!(last, MAX_PAINT_RES, "closest zoom should reach the finest hex level");
    }

    #[test]
    fn cell_under_screen_center_is_among_visible_cells() {
        for h in [1.5f64, 0.1, 0.005, 0.0003, 0.000006] {
            let c = cam(1.0 + h);
            let res = choose_res(&c, 1280, 720);
            let cells = visible_cells(res, &c, 1280, 720);
            let mid = pick_cell(res, &c, 1280, 720, 640.0, 360.0).expect("center should hit the globe");
            assert!(cells.contains(&mid), "center cell missing at height {h}");
            // Corners of the window, when they hit the globe, should be covered too.
            for (x, y) in [(5.0, 5.0), (1275.0, 5.0), (5.0, 715.0), (1275.0, 715.0)] {
                if let Some(cell) = pick_cell(res, &c, 1280, 720, x, y) {
                    assert!(cells.contains(&cell), "corner ({x},{y}) missing at height {h}");
                }
            }
        }
    }

    #[test]
    fn center_pick_matches_point_under_camera() {
        let c = cam(2.0);
        let p = pick_point(&c, 1000, 800, 500.0, 400.0).unwrap();
        let e = c.eye64().normalize();
        assert!(p.dot(e) > 0.9999, "{p:?} vs {e:?}");
    }

    #[test]
    fn center_pick_is_exact_even_at_the_closest_zoom() {
        let c = cam(1.0 + 0.000006);
        let res = choose_res(&c, 1280, 720);
        let want = dir_to_latlng(c.eye64()).to_cell(Resolution::try_from(res).unwrap());
        let got = pick_cell(res, &c, 1280, 720, 640.0, 360.0).unwrap();
        let near = want.grid_disk::<Vec<_>>(1);
        assert!(near.contains(&got), "picked a cell more than 1 hex off at res {res}");
    }

    #[test]
    fn missing_the_globe_gives_none() {
        assert!(pick_point(&cam(3.0), 1000, 800, 2.0, 2.0).is_none());
    }
}
