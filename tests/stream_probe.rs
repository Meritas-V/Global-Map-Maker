use global_map_maker::camera::{Camera, Fly};
use global_map_maker::ground::{patch_key, visible_patches};
use global_map_maker::pixelview::PixelView;
use global_map_maker::stream::Streamer;
use global_map_maker::terrain::{max_freq, HeightSource, Terrain, TerrainRef};
use global_map_maker::tiles::TileStore;
use glam::DVec3;
use std::sync::Arc;
/// Timing probe for the background loader (not a check): cargo test --release --test stream_probe -- --ignored --nocapture
#[test]
#[ignore]
fn probe() {
    let tiles = Arc::new(TileStore::new(Terrain::new(2026)));
    let st = Streamer::new(tiles.clone(), 2);
    let dir0 = DVec3::new(0.3, 0.5, 0.8).normalize();
    let alt = 1e-4;
    let mut cam = Camera { yaw: 0.0, pitch: 0.0, distance: 1.0, floor: 1.0, fly: Some(Fly { dir: dir0, heading: 0.3, tilt: -0.2 }) };
    cam.floor = tiles.ground_radius(dir0, max_freq(alt, Camera::focal_px(720.0)));
    cam.distance = cam.floor + alt;
    let v = PixelView::from_camera(&cam, 1280, 720, Some(TerrainRef::new(&*tiles, &cam, 720.0))).with_max_dist(60_000.0 / 6_371_000.0);
    let p = visible_patches(&v);
    let keys: Vec<_> = p.iter().map(|id| patch_key(*id, &v)).collect();
    println!("{} keys", keys.len());
    st.request(keys.iter().enumerate().map(|(i, k)| (i as f64, *k)).collect());
    let t0 = std::time::Instant::now();
    let mut n = 0;
    while n < keys.len() && t0.elapsed().as_secs() < 60 {
        n += st.take_done(1000).len();
        std::thread::sleep(std::time::Duration::from_millis(50));
        println!("{:?} ms: {} done, {} tiles", t0.elapsed().as_millis(), n, tiles.generated());
    }
}
