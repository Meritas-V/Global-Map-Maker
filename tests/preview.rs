//! Software preview of the inside scene: draws the real ground mesh (same patches, same colors, same
//! haze) with a tiny CPU rasterizer into a PNG-able PPM. Not a test of anything: it exists so the look
//! can be judged without a GPU. Run: cargo test --release --test preview -- --ignored --nocapture
use global_map_maker::camera::{Camera, Fly};
use global_map_maker::ground::{build_ground, shade, visible_patches, GroundVertex, Lighting, SKY};
use global_map_maker::pixelview::PixelView;
use global_map_maker::terrain::{max_freq, HeightSource, Terrain, TerrainRef};
use global_map_maker::tiles::TileStore;
use glam::{DVec3, Vec4};

fn render(cam: &Camera, tiles: &TileStore, w: usize, h: usize) -> Vec<u8> {
    let tref = |cam: &Camera| {
        let t = TerrainRef::new(tiles, cam, h as f64);
        if cam.fly.is_none() { t.capped(256.0) } else { t }
    };
    let haze = cam.fly.is_some().then_some(SKY);
    let view = PixelView::from_camera(cam, w as u32, h as u32, Some(tref(cam)));
    let patches = visible_patches(&view);
    let origin = cam.eye64().normalize() * cam.floor;
    let light = cam.light_dir();
    let light = DVec3::new(light.x as f64, light.y as f64, light.z as f64).normalize();
    let lighting = Lighting::new(light, origin, cam.eye64(), haze);
    // Twice: the first pass makes the tiles, the second walks with exact heights.
    let _ = build_ground(&patches, origin, &view);
    let view = PixelView::from_camera(cam, w as u32, h as u32, Some(tref(cam)));
    let patches = visible_patches(&view);
    let verts = build_ground(&patches, origin, &view);
    let mvp = cam.matrix_rel(w as f64 / h as f64, origin);
    let bg = if cam.fly.is_some() { SKY } else { [0.02, 0.03, 0.08] };
    let mut color = vec![bg; w * h];
    let mut depth = vec![0.0f32; w * h];
    let proj: Vec<(Vec4, GroundVertex)> = verts.iter().map(|v| (mvp * Vec4::new(v.pos[0], v.pos[1], v.pos[2], 1.0), *v)).collect();
    for t in proj.chunks_exact(3) {
        if t.iter().any(|(c, _)| c.w <= 1e-30) {
            continue;
        }
        let s: Vec<(f32, f32, f32, GroundVertex)> = t
            .iter()
            .map(|(c, col)| ((c.x / c.w * 0.5 + 0.5) * w as f32, (1.0 - (c.y / c.w * 0.5 + 0.5)) * h as f32, c.z / c.w, *col))
            .collect();
        let (x0, x1) = (s.iter().map(|p| p.0).fold(f32::MAX, f32::min).floor().max(0.0), s.iter().map(|p| p.0).fold(f32::MIN, f32::max).ceil().min(w as f32 - 1.0));
        let (y0, y1) = (s.iter().map(|p| p.1).fold(f32::MAX, f32::min).floor().max(0.0), s.iter().map(|p| p.1).fold(f32::MIN, f32::max).ceil().min(h as f32 - 1.0));
        let area = (s[1].0 - s[0].0) * (s[2].1 - s[0].1) - (s[2].0 - s[0].0) * (s[1].1 - s[0].1);
        if area.abs() < 1e-9 {
            continue;
        }
        for y in y0 as usize..=y1 as usize {
            for x in x0 as usize..=x1 as usize {
                let (px, py) = (x as f32 + 0.5, y as f32 + 0.5);
                let w0 = ((s[1].0 - px) * (s[2].1 - py) - (s[2].0 - px) * (s[1].1 - py)) / area;
                let w1 = ((s[2].0 - px) * (s[0].1 - py) - (s[0].0 - px) * (s[2].1 - py)) / area;
                let w2 = 1.0 - w0 - w1;
                if w0 < 0.0 || w1 < 0.0 || w2 < 0.0 {
                    continue;
                }
                let z = w0 * s[0].2 + w1 * s[1].2 + w2 * s[2].2;
                let i = y * w + x;
                if z > depth[i] {
                    depth[i] = z;
                    // Per-pixel shading, like ground.wgsl: interpolate the attributes, then color the pixel.
                    let mix = |f: &dyn Fn(&GroundVertex) -> f32| w0 * f(&s[0].3) + w1 * f(&s[1].3) + w2 * f(&s[2].3);
                    let rel = DVec3::new(mix(&|v| v.pos[0]) as f64, mix(&|v| v.pos[1]) as f64, mix(&|v| v.pos[2]) as f64);
                    let nrm = [mix(&|v| v.normal()[0]), mix(&|v| v.normal()[1]), mix(&|v| v.normal()[2])];
                    color[i] = shade(mix(&|v| v.hr[0]) as f64, mix(&|v| v.hr[1]) as f64, nrm, mix(&|v| v.steepness()), rel, &lighting);
                }
            }
        }
    }
    let mut out = format!("P6\n{w} {h}\n255\n").into_bytes();
    for c in color {
        for k in 0..3 {
            // linear -> sRGB, like the GPU surface does
            let v = c[k].clamp(0.0, 1.0);
            let v = if v <= 0.0031308 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 };
            out.push((v * 255.0 + 0.5) as u8);
        }
    }
    out
}

/// A spot whose coarse height is close to `target` meters (the k-th such spot on a sweep, so sites differ).
fn find_site(t: &Terrain, target: f64, k: usize) -> DVec3 {
    let n = 80_000;
    let mut found = 0;
    for i in 0..n {
        let kk = i as f64 + 0.5;
        let z = 1.0 - 2.0 * kk / n as f64;
        let a = kk * 2.399963229728653;
        let r = (1.0 - z * z).sqrt();
        let d = DVec3::new(r * a.cos(), r * a.sin(), z);
        let h = t.height_m(d, 5_000.0);
        if (h - target).abs() < 250.0 {
            let fine = t.height_m(d, 200.0);
            if (fine - h).abs() > 50.0 {
                found += 1;
                if found > k {
                    println!("site {target} m: coarse {h:.0}, fine {fine:.0} at {d:?}");
                    return d;
                }
            }
        }
    }
    panic!("no site");
}

#[test]
#[ignore]
fn preview() {
    let tiles = TileStore::new(Terrain::new(2026));
    let (w, h) = (640usize, 360usize);
    let tag = std::env::var("PREVIEW_TAG").unwrap_or_else(|_| "x".into());
    let sites = [(2_300.0, 11usize), (4_300.0, 7usize)];
    // (name, km south of the site, altitude m, heading, tilt)
    let views: [(&str, f64, f64, f64, f64); 2] = [("far", 60.0, 3_000.0, 0.0, -0.12), ("near", 18.0, 1_500.0, 0.0, -0.08)];
    let mut idx = 0;
    for (target, k) in sites {
        let peak = find_site(tiles.terrain(), target, k);
        let (n, _e, _) = Fly { dir: peak, heading: 0.0, tilt: 0.0 }.frame();
        for (name, km, alt_m, heading, tilt) in views {
            let dir = (peak - n * (km * 1000.0 / 6_371_000.0)).normalize();
            let alt = alt_m / 6_371_000.0;
            let f = max_freq(alt, Camera::focal_px(h as f64));
            let mut cam = Camera { yaw: 0.0, pitch: 0.0, distance: 1.0, floor: 1.0, fly: Some(Fly { dir, heading, tilt }) };
            cam.floor = tiles.ground_radius(dir, f);
            cam.distance = cam.floor + alt;
            let img = render(&cam, &tiles, w, h);
            std::fs::write(format!("/tmp/preview_{tag}_{idx}_{target}_{name}.ppm"), img).unwrap();
            idx += 1;
        }
        // Outside: orbit over the site at three heights.
        for (name, height) in [("out_orbit", 2.5), ("out_mid", 0.35), ("out_close", 0.035)] {
            let mut cam = Camera { yaw: 0.0, pitch: 0.0, distance: 1.0 + height, floor: 1.0, fly: None };
            cam.orbit_over(peak);
            let f = global_map_maker::terrain::max_freq(height, Camera::focal_px(h as f64)).min(256.0);
            cam.floor = tiles.ground_radius(peak, f);
            cam.distance = cam.floor + height;
            let img = render(&cam, &tiles, w, h);
            std::fs::write(format!("/tmp/preview_{tag}_{idx}_{target}_{name}.ppm"), img).unwrap();
            idx += 1;
        }
    }
}
