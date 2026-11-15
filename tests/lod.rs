//! Checks the level-of-detail choice without a window.
use global_map_maker::camera::Camera;
use global_map_maker::lod::{visible_cells, MAX_CELLS, MAX_LEVEL, ROOT_LEVEL};
use global_map_maker::sphere::CellId;
use std::collections::HashSet;

fn cells_for(distance: f32, yaw: f32, pitch: f32) -> (Camera, Vec<CellId>) {
    let cam = Camera { yaw, pitch, distance };
    let cells = visible_cells(&cam.view(1000, 800));
    (cam, cells)
}

fn deepest(cells: &[CellId]) -> u8 {
    cells.iter().map(|c| c.level).max().unwrap()
}

#[test]
fn far_and_near_both_draw_a_sane_number_of_cells() {
    for (d, label) in [(3.2, "far"), (1.5, "mid"), (1.05, "close"), (1.004, "very close")] {
        let (_, cells) = cells_for(d, 0.4, 0.3);
        println!("{label}: distance {d}: {} cells, finest level {}", cells.len(), deepest(&cells));
        assert!(cells.len() >= 20, "{label}: too few cells ({})", cells.len());
        assert!(cells.len() < MAX_CELLS, "{label}: hit the safety cap");
    }
}

#[test]
fn zooming_in_makes_cells_finer() {
    let (_, far) = cells_for(3.2, 0.4, 0.3);
    let (_, near) = cells_for(1.02, 0.4, 0.3);
    assert!(deepest(&near) > deepest(&far) + 2, "far {} near {}", deepest(&far), deepest(&near));
    assert!(deepest(&near) <= MAX_LEVEL);
    assert!(far.iter().all(|c| c.level >= ROOT_LEVEL));
}

#[test]
fn no_cell_contains_another() {
    for d in [3.2, 1.3, 1.02] {
        let (_, cells) = cells_for(d, 1.1, -0.4);
        let set: HashSet<CellId> = cells.iter().copied().collect();
        for c in &cells {
            for l in ROOT_LEVEL..c.level {
                assert!(!set.contains(&c.ancestor_at(l)), "{:?} sits inside another drawn cell", c);
            }
        }
    }
}

#[test]
fn the_middle_of_the_view_is_fully_covered() {
    use global_map_maker::sphere::CellId as C;
    for d in [3.2, 1.3, 1.02] {
        let (cam, cells) = cells_for(d, 0.7, 0.2);
        let set: HashSet<CellId> = cells.iter().copied().collect();
        let levels: HashSet<u8> = cells.iter().map(|c| c.level).collect();
        let e = cam.eye().normalize();
        // Random points on the globe within a few degrees of the spot right under the camera.
        let mut s: u64 = 99;
        let mut next = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let up = if e.z.abs() < 0.9 { glam::Vec3::Z } else { glam::Vec3::X };
        let a = e.cross(up).normalize();
        let b = e.cross(a);
        // Sample only what the camera can actually see: the patch gets smaller the closer we are.
        let spread = ((d - 1.0) * 0.6).min(1.0);
        for _ in 0..500 {
            let p = (e + a * ((next() as f32 - 0.5) * spread) + b * ((next() as f32 - 0.5) * spread)).normalize();
            let hit = levels
                .iter()
                .any(|&l| set.contains(&C::from_dir([p.x as f64, p.y as f64, p.z as f64], l)));
            assert!(hit, "distance {d}: point {p:?} isn't inside any drawn cell");
        }
    }
}
