use global_map_maker::planet::{Biome, BiomeMode, PlanetSettings};
use global_map_maker::terrain::{HeightSource, Terrain};
use glam::DVec3;

fn dirs() -> Vec<DVec3> {
    (0..200).map(|i| {
        let k = i as f64 + 0.5;
        let z = 1.0 - 2.0 * k / 200.0;
        let a = k * 2.399963229728653;
        let r = (1.0 - z * z).sqrt();
        DVec3::new(r * a.cos(), r * a.sin(), z)
    }).collect()
}

#[test]
fn sea_level_shifts_every_height() {
    let a = Terrain::with_settings(&PlanetSettings::default());
    let b = Terrain::with_settings(&PlanetSettings { sea_level_m: 255.0, ..PlanetSettings::default() });
    for d in dirs() {
        let (ha, hb) = (a.height_m(d, 5_000.0), b.height_m(d, 5_000.0));
        assert!((ha - hb - 255.0).abs() < 1e-6, "{ha} {hb}");
    }
}

#[test]
fn single_desert_is_sand_coloured() {
    let t = Terrain::with_settings(&PlanetSettings { biomes: BiomeMode::Single(Biome::Desert), ..PlanetSettings::default() });
    let want = Biome::Desert.cover();
    for d in dirs().into_iter().take(20) {
        let c = t.cover_at(d, 50.0);
        for i in 0..3 {
            assert!((c[i] - want[i]).abs() < 0.02, "{c:?} vs {want:?}");
        }
    }
}

#[test]
fn settings_change_the_terrain() {
    let a = Terrain::with_settings(&PlanetSettings::default());
    let b = Terrain::with_settings(&PlanetSettings { mountains: 2.0, land: 0.8, ..PlanetSettings::default() });
    assert!(dirs().iter().any(|&d| (a.height_m(d, 5_000.0) - b.height_m(d, 5_000.0)).abs() > 1.0));
}

/// Share of land per biome on the default planet. Run: cargo test --release --test settings biome_shares -- --ignored --nocapture
#[test]
#[ignore]
fn biome_shares() {
    for seed in [2026u64, 4000, 7] {
        let t = Terrain::with_settings(&PlanetSettings { seed, ..PlanetSettings::default() });
        let mut share = [0.0f64; 7];
        let mut land = 0.0;
        let n = 40_000;
        for i in 0..n {
            let k = i as f64 + 0.5;
            let z = 1.0 - 2.0 * k / n as f64;
            let a = k * 2.399963229728653;
            let r = (1.0 - z * z).sqrt();
            let d = DVec3::new(r * a.cos(), r * a.sin(), z);
            let h = t.height_m(d, 5_000.0);
            if h > 0.0 {
                land += 1.0;
                let w = t.biome_weights(d, h);
                for j in 0..7 { share[j] += w[j]; }
            }
        }
        let names = ["ice", "tundra", "taiga", "forest", "plains", "desert", "rain"];
        let s: Vec<String> = names.iter().zip(share).map(|(n, v)| format!("{n} {:.0}%", 100.0 * v / land)).collect();
        println!("seed {seed}: land {:.0}% | {}", 100.0 * land / n as f64, s.join(", "));
        let (mut tmin, mut tmax, mut mmin, mut mmax) = (9.0f64, -9.0f64, 9.0f64, -9.0f64);
        for i in 0..n {
            let k = i as f64 + 0.5;
            let z = 1.0 - 2.0 * k / n as f64;
            let a = k * 2.399963229728653;
            let r = (1.0 - z * z).sqrt();
            let (tt, mm) = t.climate(DVec3::new(r * a.cos(), r * a.sin(), z));
            tmin = tmin.min(tt); tmax = tmax.max(tt); mmin = mmin.min(mm); mmax = mmax.max(mm);
        }
        println!("   t {tmin:.2}..{tmax:.2}  m {mmin:.2}..{mmax:.2}");
    }
}

#[test]
fn cold_coasts_are_not_sandy() {
    use global_map_maker::planet::{beach, Biome, SAND};
    let ice = beach(Biome::Ice.cover(), Biome::Ice.sandy());
    let desert = beach(Biome::Desert.cover(), Biome::Desert.sandy());
    assert_eq!(desert, [SAND[0] * 1.0, SAND[1] * 1.0, SAND[2] * 1.0]);
    assert!(ice[2] > ice[0] - 0.02, "ice shore should be grey-blue, not sand: {ice:?}");
}
