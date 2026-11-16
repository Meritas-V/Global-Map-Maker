//! How long one height tile takes to make (to size a "pre-generate the planet" idea):
//! cargo test --release --test tile_cost -- --ignored --nocapture
use global_map_maker::pixelpaint::PixelId;
use global_map_maker::terrain::Terrain;
use global_map_maker::tiles::generate;

#[test]
#[ignore]
fn tile_cost() {
    let t = Terrain::new(2026);
    for level in [4u8, 7, 10, 13] {
        let t0 = std::time::Instant::now();
        let n = 40;
        for i in 0..n {
            let _ = generate(&t, PixelId::new(0, level, (i * 7919) % (1 << level), (i * 104729) % (1 << level)));
        }
        println!("level {level}: {:.2} ms per tile", t0.elapsed().as_secs_f64() * 1e3 / n as f64);
    }
}
