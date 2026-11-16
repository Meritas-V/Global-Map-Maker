//! Flies a camera for a while with the same loading logic as the app (minus the GPU) and checks the loader keeps up:
//! no frame needs more than a few patches made on the spot, and when the camera stops everything gets loaded.
use global_map_maker::camera::{Camera, Fly};
use global_map_maker::ground::{build_patches, patch_key, visible_patches, PatchKey};
use global_map_maker::pixelpaint::PixelId;
use global_map_maker::pixelview::PixelView;
use global_map_maker::stream::{plan_frame, Streamer};
use global_map_maker::terrain::{max_freq, HeightSource, Terrain, TerrainRef};
use global_map_maker::tiles::TileStore;
use glam::DVec3;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

#[test]
fn flying_keeps_the_loader_ahead_of_the_camera() {
    let tiles = Arc::new(TileStore::new(Terrain::new(2026)));
    let streamer = Streamer::new(tiles.clone(), 2);
    let (w, h) = (1280u32, 720u32);
    let alt = 1e-4;
    let dir0 = DVec3::new(0.3, 0.5, 0.8).normalize();
    let mut cam = Camera { yaw: 0.0, pitch: 0.0, distance: 1.0, floor: 1.0, fly: Some(Fly { dir: dir0, heading: 0.3, tilt: -0.2 }) };
    cam.floor = tiles.ground_radius(dir0, max_freq(alt, Camera::focal_px(h as f64)));
    cam.distance = cam.floor + alt;
    let limit = 60_000.0 / global_map_maker::terrain::EARTH_RADIUS_M;

    let mut ready: HashSet<PatchKey> = HashSet::new();
    let mut by_id: HashMap<PixelId, PatchKey> = HashMap::new();
    let mut ahead: Vec<(f64, PatchKey)> = Vec::new();
    let mut slow_later = 0.0f64;
    let (mut worst_sync, mut worst_plan_ms, mut standins_total, mut first_sync) = (0usize, 0.0f64, 0usize, 0usize);
    let mut final_missing = usize::MAX;
    let mut standin_frames = 0usize;
    let mut min_ratio = 1.0f64; // drawn patches / wanted patches, after warm-up: a collapse means one coarse mesh hides many ready ones
    let (mut walk_ms, mut plan_ms): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
    for frame in 0..400 {
        // Fly forward for 300 frames at ~60 fps, then stop and let the loader finish.
        if frame < 300 {
            let f = cam.fly.as_mut().unwrap();
            let (n, e, _) = f.frame();
            let (sh, ch) = f.heading.sin_cos();
            f.dir = (f.dir + (n * ch + e * sh) * alt * 0.8 / 60.0).normalize();
        }
        let t0 = Instant::now();
        for (k, _m) in streamer.take_done(200) {
            ready.insert(k);
            by_id.insert(k.id, k);
        }
        let view = PixelView::from_camera(&cam, w, h, Some(TerrainRef::new(&*tiles, &cam, h as f64))).with_max_dist(limit);
        let tw = Instant::now();
        let patches = visible_patches(&view);
        walk_ms.push(tw.elapsed().as_secs_f64() * 1e3);
        let keys: Vec<PatchKey> = patches.iter().map(|id| patch_key(*id, &view)).collect();
        let tp = Instant::now();
        let key_of = |id: PixelId| patch_key(id, &view);
        let mut plan = plan_frame(&keys, &|k| ready.contains(k), &by_id, &key_of);
        let eye = cam.eye64();
        let n_now = plan.holes.len();
        if n_now > 0 {
            let now = plan.holes.clone();
            let _ = build_patches(&now, &view);
            for k in &now {
                ready.insert(*k);
                by_id.insert(k.id, *k);
            }
            plan = plan_frame(&keys, &|k| ready.contains(k), &by_id, &key_of);
        }
        plan_ms.push(tp.elapsed().as_secs_f64() * 1e3);
        let mut wanted: Vec<(f64, PatchKey)> = plan.chain.iter().filter(|k| !ready.contains(k)).map(|k| (k.id.level as f64 + (k.id.center_dir() - eye).length() - 100.0, *k)).collect();
        wanted.extend(plan.missing.iter().map(|k| ((k.id.center_dir() - eye).length(), *k)));
        if frame < 300 {
            if frame % 4 == 0 {
                ahead.clear();
                let mut ac = cam.clone();
                let f = ac.fly.as_mut().unwrap();
                let (n, e, _) = f.frame();
                let (sh, ch) = f.heading.sin_cos();
                f.dir = (f.dir + (n * ch + e * sh) * alt * 0.8 * 1.5).normalize();
                let va = PixelView::from_camera(&ac, w, h, Some(TerrainRef::new(&*tiles, &ac, h as f64))).with_max_dist(limit);
                let pa = visible_patches(&va);
                for id in pa {
                    let k = patch_key(id, &va);
                    ahead.push(((k.id.center_dir() - eye).length() + 5.0, k));
                }
            }
            wanted.extend(ahead.iter().filter(|(_, k)| !ready.contains(k)).copied());
        }
        streamer.request(wanted);
        worst_plan_ms = worst_plan_ms.max(t0.elapsed().as_secs_f64() * 1e3);
        if frame > 0 && frame < 300 {
            slow_later = slow_later.max(t0.elapsed().as_secs_f64() * 1e3);
        }
        if frame == 0 {
            first_sync = n_now;
        } else if frame < 300 {
            worst_sync = worst_sync.max(n_now);
        }
        if frame > 30 && frame < 300 {
            min_ratio = min_ratio.min(plan.draw.len() as f64 / keys.len().max(1) as f64);
        }
        let coarse = plan.draw.iter().filter(|k| !keys.contains(k)).count();
        standins_total += coarse;
        if frame < 300 && coarse > 0 {
            standin_frames += 1;
        }
        if frame >= 300 {
            final_missing = plan.missing.len();
            if final_missing == 0 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(20));
        } else {
            std::thread::sleep(std::time::Duration::from_millis(8));
        }
    }
    let pct = |v: &mut Vec<f64>, q: f64| { v.sort_by(|a, b| a.partial_cmp(b).unwrap()); v[((v.len() - 1) as f64 * q) as usize] };
    println!("visible_patches ms: median {:.2} p99 {:.2}; plan_frame ms: median {:.2} p99 {:.2}", pct(&mut walk_ms, 0.5), pct(&mut walk_ms, 0.99), pct(&mut plan_ms, 0.5), pct(&mut plan_ms, 0.99));
    println!("distinct meshes made: {} (cache {}), tiles generated: {}, frames with stand-ins: {standin_frames}/300", ready.len(), by_id.len(), tiles.generated());
    println!("first frame made {first_sync} on the spot; worst later frame {worst_sync}; stand-in draws {standins_total}; slowest frame {worst_plan_ms:.1} ms, slowest after the first {slow_later:.1} ms");
    println!("lowest drawn/wanted ratio while flying: {min_ratio:.2}");
    assert!(min_ratio > 0.3, "drawn patches collapsed to {min_ratio:.2} of the wanted ones: a coarse mesh is hiding ready ones");
    assert_eq!(final_missing, 0, "the loader never caught up once the camera stopped");
    assert!(worst_sync <= 24, "{worst_sync} patches had to be made on the spot in one frame while flying");
}
