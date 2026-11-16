//! Procedural planet terrain: one function from a direction on the sphere to a height in meters.
//!
//! 3D gradient noise (Perlin's improved noise) stacked into fractal layers, sampled straight on
//! the unit-sphere point, so there are no seams or pinched poles. Sea level is 0 m, the deepest
//! trench is about -11 km, the highest peaks about 8.8 km, at Earth's real scale (6,371 km radius).
//!
//! The height only knows how much detail to add through `max_freq`: octaves finer than what the
//! screen can show are left out (and fade in smoothly while zooming), so the same function works
//! from orbit down to centimeters.

use crate::camera::Camera;
use glam::DVec3;

pub const EARTH_RADIUS_M: f64 = 6_371_000.0;
/// Finest noise frequency ever used (feature size about 0.4 m).
pub const MAX_FREQ_CAP: f64 = 16_777_216.0;
/// The finest terrain feature drawn is about this many screen pixels wide.
pub const WAVE_PX: f64 = 40.0;
/// Tallest the land goes, meters.
pub const MAX_PEAK_M: f64 = 8_848.0;
/// Deepest the sea goes, meters.
pub const MAX_DEPTH_M: f64 = 10_994.0;

/// Ridge octaves that make up the mountain belts (frequencies 7, 14, 28, 56); finer ones are detail.
const MACRO_RIDGE_OCTAVES: u32 = 4;
/// Average of one ridged noise octave, `(1 - |n|)^2` with improved Perlin noise.
const RIDGE_MEAN: f64 = 0.64;
/// Height of the mountain belts themselves (the broad shape), before the crags are added on top.
const MACRO_PEAK_M: f64 = 4_800.0;
/// Ridged fractal on top of rugged ground: it makes the ridges and valleys (5-60 km scale and finer).
/// The signal (measured from its average) spreads about 0.12, so CRAG_AMP is roughly meters per 0.12.
const CRAG_AMP: f64 = 9_000.0;
const CRAG_PERSISTENCE: f64 = 0.64;
const HILLS_M: f64 = 600.0;
/// Ridge noise below this is plain, above it (over the span) it rises to peaks.
const RIDGE_LOW: f64 = 0.30;
const RIDGE_SPAN: f64 = 0.45;

/// Continent noise value above which there is land (tuned for about 45 % land: a fantasy world, a bit more land than Earth).
const SEA: f64 = 0.05;
/// How far below SEA the continent noise has to go to reach the abyssal plain.
const DEPTH_SPAN: f64 = 0.20;

/// Heights above 6 km bend over smoothly toward the highest peak, instead of being cut flat at it.
fn soft_peak(h: f64) -> f64 {
    const KNEE: f64 = 6_000.0;
    if h <= KNEE {
        return h;
    }
    let span = MAX_PEAK_M - KNEE;
    KNEE + span * ((h - KNEE) / span).tanh()
}

fn smooth(a: f64, b: f64, x: f64) -> f64 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

pub struct Terrain {
    perm: [u8; 512],
}

/// Anything that can say how high the ground is in a direction: the terrain function itself or the
/// tile layers built from it.
pub trait HeightSource: Sync {
    /// Height above (positive) or below (negative) sea level in meters, with detail up to `max_freq`.
    fn height_m(&self, d: DVec3, max_freq: f64) -> f64;

    /// Like `height_m`, but a source that makes its data lazily must not make any here: it answers from
    /// what exists (coarser data stands in). For quick tests that only need roughly the right height.
    fn height_m_peek(&self, d: DVec3, max_freq: f64) -> f64 {
        self.height_m(d, max_freq)
    }

    /// Radius of the visible surface in the direction `d` (a unit vector). The sea is a flat sphere.
    fn ground_radius(&self, d: DVec3, max_freq: f64) -> f64 {
        1.0 + self.height_m(d, max_freq).max(0.0) / EARTH_RADIUS_M
    }

    /// Get ready for lookups at these (direction, detail limit) points: sources that make their data
    /// lazily can build everything missing at once, in parallel. Does nothing by default.
    fn prefetch(&self, _points: &[(DVec3, f64)]) {}

    /// First point where a ray from `origin` along `dir` meets the surface (one detail limit everywhere).
    fn pick_ray(&self, origin: DVec3, dir: DVec3, max_freq: f64) -> Option<DVec3> {
        march(origin, dir, &|u| self.ground_radius(u, max_freq))
    }
}

/// Ray vs surface: first point where a ray from `origin` along `dir` goes below the ground.
/// `radius` gives the ground radius in a (unit) direction.
pub fn march(origin: DVec3, dir: DVec3, radius: &dyn Fn(DVec3) -> f64) -> Option<DVec3> {
    let rb = 1.0 + (MAX_PEAK_M + 50.0) / EARTH_RADIUS_M;
    let b = origin.dot(dir);
    let disc = b * b - (origin.dot(origin) - rb * rb);
    if disc < 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let (mut t, t_end) = ((-b - sq).max(0.0), -b + sq);
    let gap_at = |t: f64| {
        let p = origin + dir * t;
        let r = p.length();
        r - radius(p / r)
    };
    let mut prev = t;
    for _ in 0..600 {
        let gap = gap_at(t);
        if gap <= 1e-9 * t.max(1e-3) {
            // Hit (or overshot): bisect between the last point above ground and this one.
            let (mut lo, mut hi) = (prev, t);
            for _ in 0..40 {
                let mid = 0.5 * (lo + hi);
                if gap_at(mid) > 0.0 {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            return Some(origin + dir * hi);
        }
        prev = t;
        t += (gap * 0.6).max(1e-9 + t * 1e-6);
        if t > t_end {
            return None;
        }
    }
    None
}

/// A height source plus how much detail to ask for. Detail follows distance from the camera:
/// near ground gets the finest octaves the screen can show, far ground coarser ones (so far away
/// only a few cheap, coarse tiles are needed).
#[derive(Clone, Copy)]
pub struct TerrainRef<'a> {
    pub terrain: &'a dyn HeightSource,
    /// The finest detail limit (what the ground right under the camera gets).
    pub max_freq: f64,
    lod: Option<(DVec3, f64, f64)>, // eye, lens focal_px, radius of the ground under the camera
}

impl<'a> TerrainRef<'a> {
    /// Detail follows distance from this camera; `height_px` is the window height.
    pub fn new(terrain: &'a dyn HeightSource, camera: &Camera, height_px: f64) -> TerrainRef<'a> {
        let focal = Camera::focal_px(height_px);
        TerrainRef { terrain, max_freq: max_freq(camera.height(), focal), lod: Some((camera.eye64(), focal, camera.floor)) }
    }

    /// The same, but never finer than `cap` (the outside map view stays soft however close you get).
    pub fn capped(mut self, cap: f64) -> TerrainRef<'a> {
        self.max_freq = self.max_freq.min(cap).max(1.0);
        self
    }

    /// One detail limit everywhere.
    pub fn fixed(terrain: &'a dyn HeightSource, max_freq: f64) -> TerrainRef<'a> {
        TerrainRef { terrain, max_freq, lod: None }
    }

    /// Detail limit for the ground in direction `d`.
    pub fn freq_at(&self, d: DVec3) -> f64 {
        match self.lod {
            None => self.max_freq,
            Some((eye, focal, r)) => {
                let dist = (d * r - eye).length().max(1e-12);
                (focal / (WAVE_PX * dist)).clamp(1.0, self.max_freq.max(1.0))
            }
        }
    }

    pub fn height_m(&self, d: DVec3) -> f64 {
        self.terrain.height_m(d, self.freq_at(d))
    }

    /// Build ahead whatever lookups at these directions will need (see `HeightSource::prefetch`).
    pub fn prefetch(&self, dirs: &[DVec3]) {
        let pts: Vec<(DVec3, f64)> = dirs.iter().map(|d| (*d, self.freq_at(*d))).collect();
        self.terrain.prefetch(&pts);
    }

    /// Surface radius without making any tile (see `HeightSource::height_m_peek`): for culling tests.
    pub fn radius_peek(&self, d: DVec3) -> f64 {
        1.0 + self.terrain.height_m_peek(d, self.freq_at(d)).max(0.0) / EARTH_RADIUS_M
    }

    /// Distance from the globe's center to the visible surface (water is a flat sphere at sea level).
    pub fn radius(&self, d: DVec3) -> f64 {
        1.0 + self.height_m(d).max(0.0) / EARTH_RADIUS_M
    }

    /// First point where a ray meets the surface.
    pub fn pick(&self, origin: DVec3, dir: DVec3) -> Option<DVec3> {
        march(origin, dir, &|u| self.radius(u))
    }
}

impl HeightSource for Terrain {
    fn height_m(&self, d: DVec3, max_freq: f64) -> f64 {
        self.height_fn(d, max_freq)
    }
}

impl Terrain {
    pub fn new(seed: u64) -> Terrain {
        let mut p: Vec<u8> = (0..=255).collect();
        let mut s = seed;
        for i in (1..256).rev() {
            let j = (splitmix(&mut s) % (i as u64 + 1)) as usize;
            p.swap(i, j);
        }
        let mut perm = [0u8; 512];
        for i in 0..512 {
            perm[i] = p[i & 255];
        }
        Terrain { perm }
    }

    /// Perlin's improved gradient noise, roughly -1..1.
    fn noise(&self, x: f64, y: f64, z: f64) -> f64 {
        let (fx, fy, fz) = (x.floor(), y.floor(), z.floor());
        let xi = (fx as i64).rem_euclid(256) as usize;
        let yi = (fy as i64).rem_euclid(256) as usize;
        let zi = (fz as i64).rem_euclid(256) as usize;
        let (x, y, z) = (x - fx, y - fy, z - fz);
        let fade = |t: f64| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
        let (u, v, w) = (fade(x), fade(y), fade(z));
        let p = &self.perm;
        let a = p[xi] as usize + yi;
        let aa = p[a] as usize + zi;
        let ab = p[a + 1] as usize + zi;
        let b = p[xi + 1] as usize + yi;
        let ba = p[b] as usize + zi;
        let bb = p[b + 1] as usize + zi;
        let grad = |h: u8, x: f64, y: f64, z: f64| {
            let h = h & 15;
            let u = if h < 8 { x } else { y };
            let v = if h < 4 { y } else if h == 12 || h == 14 { x } else { z };
            (if h & 1 == 0 { u } else { -u }) + (if h & 2 == 0 { v } else { -v })
        };
        let lerp = |t: f64, a: f64, b: f64| a + t * (b - a);
        lerp(
            w,
            lerp(
                v,
                lerp(u, grad(p[aa], x, y, z), grad(p[ba], x - 1.0, y, z)),
                lerp(u, grad(p[ab], x, y - 1.0, z), grad(p[bb], x - 1.0, y - 1.0, z)),
            ),
            lerp(
                v,
                lerp(u, grad(p[aa + 1], x, y, z - 1.0), grad(p[ba + 1], x - 1.0, y, z - 1.0)),
                lerp(u, grad(p[ab + 1], x, y - 1.0, z - 1.0), grad(p[bb + 1], x - 1.0, y - 1.0, z - 1.0)),
            ),
        )
    }

    /// Octave weight: 1 well below the detail limit, fading to 0 at it.
    fn octave_weight(freq: f64, max_freq: f64) -> f64 {
        (max_freq / freq).log2().clamp(0.0, 1.0)
    }

    /// Fractal noise: each octave doubles the frequency and scales the strength by `persistence`.
    fn fbm(&self, d: DVec3, f0: f64, persistence: f64, octaves: u32, max_freq: f64, ridged: bool) -> f64 {
        let (mut sum, mut amp, mut norm, mut f) = (0.0, 1.0, 0.0, f0);
        for o in 0..octaves {
            norm += amp;
            let w = Self::octave_weight(f, max_freq);
            if w > 0.0 {
                // Each octave gets its own offset so the lattices don't line up.
                let off = 17.3 * o as f64;
                let n = self.noise(d.x * f + off, d.y * f - off, d.z * f + 0.5 * off);
                // Ridged octaves are measured from their average, so leaving octaves out (low detail) leaves the average
                // behind instead of a big hole: zooming out must not sink the mountains.
                let v = if ridged { (1.0 - n.abs()).powi(2) - RIDGE_MEAN } else { n };
                sum += v * amp * w;
            }
            amp *= persistence;
            f *= 2.0;
        }
        sum / norm
    }

    /// The terrain function itself: height in meters, with detail up to `max_freq`.
    pub fn height_fn(&self, d: DVec3, max_freq: f64) -> f64 {
        let c = self.fbm(d, 1.4, 0.5, 8, max_freq, false);
        let e = c - SEA;
        if e < 0.0 {
            let s = (-e / DEPTH_SPAN).clamp(0.0, 1.0);
            let mut h = -(250.0 * s + 3800.0 * s * s);
            if s > 0.35 {
                // Trenches: thin lines where a second noise crosses zero, only in the deep ocean.
                let n = self.fbm(d, 3.3, 0.5, 6, max_freq, false);
                let t = (1.0 - n.abs() / 0.06).max(0.0);
                h -= t * t * 7300.0 * smooth(0.35, 0.9, s);
            }
            return h.max(-MAX_DEPTH_M);
        }
        let r = smooth(0.0, 0.10, e);
        let low = 1100.0 * smooth(0.0, 0.4, e);
        let belt = smooth(0.02, 0.34, self.fbm(d, 2.6, 0.5, 4, max_freq, false) + 0.07);
        // Macro shape: the mountain belts themselves (ridge noise at frequencies 7..56).
        let ridge = RIDGE_MEAN + self.fbm(d, 7.0, 0.62, MACRO_RIDGE_OCTAVES, max_freq, true);
        let peaks = ((ridge - RIDGE_LOW) / RIDGE_SPAN).clamp(0.0, 1.0).powf(2.0);
        let macro_h = low + belt * smooth(0.02, 0.22, e) * MACRO_PEAK_M * peaks;
        // Detail: only on rugged high ground, so plains and valleys stay smooth and ridges stay sharp.
        let rugged = smooth(250.0, 2_500.0, macro_h);
        let crags = self.fbm(d, 7.0 * 16.0, CRAG_PERSISTENCE, 18, max_freq, true) * CRAG_AMP * rugged;
        let hills = self.fbm(d, 30.0, 0.58, 18, max_freq, false) * HILLS_M * (0.3 + 0.7 * rugged);
        soft_peak(macro_h + (hills + crags) * r)
    }
}

/// Detail limit for a camera `height` above the ground (globe radii) and lens `focal_px`.
pub fn max_freq(height: f64, focal_px: f64) -> f64 {
    (focal_px / (WAVE_PX * height.max(1e-12))).clamp(1.0, MAX_FREQ_CAP)
}

/// Terrain detail limit and ground radius under the camera for a window `height_px` tall.
pub fn camera_ground(terrain: &dyn HeightSource, camera: &Camera, height_px: f64) -> (f64, f64) {
    let d = camera.eye64().normalize();
    let f = max_freq(camera.height(), Camera::focal_px(height_px));
    (f, terrain.ground_radius(d, f))
}

fn ramp(stops: &[(f64, [f32; 3])], x: f64) -> [f32; 3] {
    if x <= stops[0].0 {
        return stops[0].1;
    }
    for w in stops.windows(2) {
        if x <= w[1].0 {
            let t = ((x - w[0].0) / (w[1].0 - w[0].0)) as f32;
            return [0, 1, 2].map(|i| w[0].1[i] + (w[1].1[i] - w[0].1[i]) * t);
        }
    }
    stops[stops.len() - 1].1
}

/// Surface color from height (m) and steepness (0 flat .. 1 vertical): water shades darker with
/// depth, land goes sand, grass, forest, rock, snow, and steep slopes show bare rock.
pub fn color(h: f64, steepness: f64) -> [f32; 3] {
    if h < 0.0 {
        return ramp(
            &[
                (0.0, [0.05, 0.40, 0.52]),
                (30.0, [0.03, 0.26, 0.48]),
                (200.0, [0.015, 0.14, 0.36]),
                (1_000.0, [0.008, 0.07, 0.24]),
                (4_000.0, [0.004, 0.03, 0.14]),
                (11_000.0, [0.002, 0.01, 0.07]),
            ],
            -h,
        );
    }
    let base = ramp(
        &[
            (0.0, [0.62, 0.50, 0.26]),
            (6.0, [0.58, 0.47, 0.24]),
            (40.0, [0.22, 0.42, 0.08]),
            (500.0, [0.15, 0.34, 0.06]),
            (1_200.0, [0.07, 0.22, 0.06]),
            (2_000.0, [0.30, 0.25, 0.17]),
            (3_000.0, [0.42, 0.38, 0.32]),
        ],
        h,
    );
    // `steepness` is 1 - cos(slope angle): 25 degrees = 0.09, 35 = 0.18, 45 = 0.29.
    let rock = [0.24f32, 0.20, 0.16];
    let t = (smooth(0.07, 0.34, steepness) * 0.85) as f32;
    let c = [0, 1, 2].map(|i| base[i] + (rock[i] - base[i]) * t);
    // Snow settles on high, gentler ground and slides off steep faces, so ridges show dark rock through it.
    let snow = (smooth(3_600.0, 4_900.0, h) * (1.0 - smooth(0.08, 0.36, steepness))) as f32;
    let white = [0.90f32, 0.93, 1.00];
    [0, 1, 2].map(|i| c[i] + (white[i] - c[i]) * snow)
}

/// Detail limit of the "range" height used for coloring: the mountain belts themselves, without crags.
pub const RANGE_FREQ: f64 = 54.33; // lands exactly on tile level 2: one tile lookup, no blending

/// Surface color with a zoom-independent hint of mountain ranges. `range_h` is the height from the
/// coarse (RANGE_FREQ) terrain at the same spot: it tints whole ranges grey at any zoom, so a range seen
/// from far away looks the same as when you get close (fine detail only adds slopes and crags on top).
pub fn color_in_range(h: f64, range_h: f64, steepness: f64) -> [f32; 3] {
    let base = color(h, steepness);
    if h < 0.0 {
        return base;
    }
    let grey = [0.36f32, 0.32, 0.28];
    let snowy = smooth(3_600.0, 4_700.0, h) as f32;
    let m = (smooth(1_300.0, 3_000.0, range_h) * 0.7) as f32 * (1.0 - snowy);
    [0, 1, 2].map(|i| base[i] + (grey[i] - base[i]) * m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mountain_ranges_get_a_grey_hint_but_plains_do_not() {
        let plain = color_in_range(150.0, 200.0, 0.0);
        let range = color_in_range(150.0, 3_000.0, 0.0); // a valley inside a high range
        assert_eq!(plain, color(150.0, 0.0));
        let spread = |c: [f32; 3]| c.iter().cloned().fold(0.0, f32::max) - c.iter().cloned().fold(1.0, f32::min);
        assert!(spread(range) < spread(plain), "range should be greyer: {range:?} vs {plain:?}");
        // The sea ignores it.
        assert_eq!(color_in_range(-500.0, 3_000.0, 0.0), color(-500.0, 0.0));
    }

    #[test]
    fn capping_detail_never_adds_detail() {
        let t = Terrain::new(3);
        let cam = Camera { yaw: 0.0, pitch: 0.0, distance: 1.001, floor: 1.0, fly: None };
        let r = TerrainRef::new(&t, &cam, 720.0);
        assert!(r.capped(200.0).max_freq <= 200.0);
        assert!(r.capped(1e12).max_freq == r.max_freq);
    }

    fn dir(i: u32) -> DVec3 {
        // Fibonacci sphere: evenly spread test directions.
        let n = 40_000.0;
        let k = i as f64 + 0.5;
        let z = 1.0 - 2.0 * k / n;
        let a = k * 2.399963229728653;
        let r = (1.0 - z * z).sqrt();
        DVec3::new(r * a.cos(), r * a.sin(), z)
    }

    /// Mountain character: slope and local relief over rugged high ground, at fine detail.
    #[test]
    #[ignore]
    fn print_relief() {
        let t = Terrain::new(2026);
        let f = 1e7;
        let r = 6_371_000.0;
        let mut slopes = Vec::new();
        let mut reliefs = Vec::new();
        let mut heights = Vec::new();
        let mut coarse_h: Vec<f64> = Vec::new();
        let n = 200_000;
        let (mut counted, mut high) = (0, 0);
        for i in 0..n {
            let k = i as f64 + 0.5;
            let z = 1.0 - 2.0 * k / n as f64;
            let a = k * 2.399963229728653;
            let rr = (1.0 - z * z).sqrt();
            let d = DVec3::new(rr * a.cos(), rr * a.sin(), z);
            counted += 1;
            // two tangent directions
            let e1 = d.cross(DVec3::Z).normalize();
            let e2 = d.cross(e1).normalize();
            let step = 500.0 / r;
            let h0 = t.height_m(d, f);
            if h0 < 1_500.0 {
                continue; // only mountain country
            }
            high += 1;
            coarse_h.push(t.height_m(d, RANGE_FREQ));
            heights.push(h0);
            let hx = t.height_m((d + e1 * step).normalize(), f);
            let hy = t.height_m((d + e2 * step).normalize(), f);
            let g = (((hx - h0) / 500.0).powi(2) + ((hy - h0) / 500.0).powi(2)).sqrt();
            slopes.push(g.atan().to_degrees());
            // relief in a 10 km radius, 16 samples
            let (mut lo, mut hi) = (h0, h0);
            for j in 0..16 {
                let ang = j as f64 * std::f64::consts::TAU / 16.0;
                let q = (d + (e1 * ang.cos() + e2 * ang.sin()) * (10_000.0 / r)).normalize();
                let h = t.height_m(q, f);
                lo = lo.min(h);
                hi = hi.max(h);
            }
            reliefs.push(hi - lo);
        }
        let q = |v: &mut Vec<f64>, p: f64| {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[((v.len() - 1) as f64 * p) as usize]
        };
        println!("rugged samples: {} (of {counted} points, {high} high)", slopes.len());
        for p in [0.1, 0.5, 0.9] {
            println!("q{p}: slope {:.1} deg, relief10km {:.0} m, height {:.0} m", q(&mut slopes, p), q(&mut reliefs, p), q(&mut heights, p));
        }
        println!("max height {:.0}", q(&mut heights, 1.0));
        for p in [0.05, 0.25, 0.5, 0.75, 0.95] {
            println!("range height (freq {RANGE_FREQ}) under mountain country, q{p}: {:.0} m", q(&mut coarse_h, p));
        }
    }

    #[test]
    #[ignore]
    fn print_ridged_distribution() {
        let t = Terrain::new(2026);
        for pers in [0.55, 0.62, 0.68] {
            let mut v: Vec<f64> = (0..20_000)
                .map(|i| {
                    let k = i as f64 + 0.5;
                    let z = 1.0 - 2.0 * k / 20_000.0;
                    let a = k * 2.399963229728653;
                    let rr = (1.0 - z * z).sqrt();
                    t.fbm(DVec3::new(rr * a.cos(), rr * a.sin(), z), 112.0, pers, 18, 1e7, true)
                })
                .collect();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let q = |p: f64| v[((v.len() - 1) as f64 * p) as usize];
            println!("persistence {pers}: q01 {:.3} q10 {:.3} q50 {:.3} q90 {:.3} q99 {:.3}", q(0.01), q(0.1), q(0.5), q(0.9), q(0.99));
        }
    }

    #[test]
    #[ignore]
    fn print_stats() {
        let t = Terrain::new(2026);
        let mut hs: Vec<f64> = (0..40_000).map(|i| t.height_m(dir(i), 1e9_f64.min(MAX_FREQ_CAP))).collect();
        hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = hs.len();
        let land: Vec<f64> = hs.iter().cloned().filter(|h| *h > 0.0).collect();
        println!("land fraction {:.3}", land.len() as f64 / n as f64);
        for q in [0.0, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0] {
            println!("q{:>4}: {:>8.0}", q, hs[((n - 1) as f64 * q) as usize]);
        }
        let l = land.len();
        for q in [0.1, 0.5, 0.9, 0.99, 0.999, 1.0] {
            println!("land q{:>5}: {:>8.0}", q, land[((l - 1) as f64 * q) as usize]);
        }
    }

    #[test]
    fn same_seed_same_world_different_seed_different_world() {
        let (a, b, c) = (Terrain::new(1), Terrain::new(1), Terrain::new(2));
        let mut diff = 0;
        for i in 0..500 {
            let d = dir(i * 7);
            assert_eq!(a.height_m(d, 1e6), b.height_m(d, 1e6));
            if (a.height_m(d, 1e6) - c.height_m(d, 1e6)).abs() > 1.0 {
                diff += 1;
            }
        }
        assert!(diff > 100, "{diff}");
    }

    #[test]
    fn realistic_ranges_and_land_share() {
        let t = Terrain::new(7);
        let hs: Vec<f64> = (0..20_000).map(|i| t.height_m(dir(i * 2), 1e7)).collect();
        let min = hs.iter().cloned().fold(f64::MAX, f64::min);
        let max = hs.iter().cloned().fold(f64::MIN, f64::max);
        let land = hs.iter().filter(|h| **h > 0.0).count() as f64 / hs.len() as f64;
        assert!(min >= -MAX_DEPTH_M - 1.0 && max <= MAX_PEAK_M + 1.0, "{min} {max}");
        assert!(min < -6_000.0, "deepest {min}: needs real trenches");
        assert!(max > 4_500.0, "highest {max}: needs real mountains");
        assert!(land > 0.35 && land < 0.6, "land share {land}");
    }

    #[test]
    fn the_surface_is_continuous() {
        let t = Terrain::new(3);
        for i in 0..300 {
            let d = dir(i * 13);
            let e = (d + DVec3::new(1.0, 0.5, -0.3).normalize() * 1e-7).normalize();
            let (a, b) = (t.height_m(d, 1e6), t.height_m(e, 1e6));
            assert!((a - b).abs() < 5.0, "jump {a} -> {b}");
        }
    }

    #[test]
    fn zooming_in_only_adds_small_detail() {
        let t = Terrain::new(5);
        for i in 0..200 {
            let d = dir(i * 31);
            let coarse = t.height_m(d, 1e4);
            let fine = t.height_m(d, 2e4);
            assert!((coarse - fine).abs() < 500.0, "{coarse} vs {fine}");
        }
    }

    #[test]
    fn water_is_flat_and_land_rises() {
        let t = Terrain::new(7);
        let mut saw_sea = false;
        let mut saw_land = false;
        for i in 0..2_000 {
            let d = dir(i * 5);
            let r = t.ground_radius(d, 1e6);
            if t.height_m(d, 1e6) <= 0.0 {
                assert_eq!(r, 1.0);
                saw_sea = true;
            } else {
                assert!(r > 1.0);
                saw_land = true;
            }
        }
        assert!(saw_sea && saw_land);
    }

    #[test]
    fn a_ray_from_space_lands_on_the_surface() {
        let t = Terrain::new(7);
        for i in 0..40 {
            let target = dir(i * 900);
            let origin = target * 3.0;
            let hit = t.pick_ray(origin, -target, 1e5).expect("hit");
            let r = hit.length();
            assert!((r - t.ground_radius(hit / r, 1e5)).abs() < 1e-7, "{r}");
            assert!(hit.normalize().dot(target) > 0.9999);
        }
    }

    #[test]
    fn colors_get_darker_with_depth_and_snow_is_high() {
        assert!(color(-10_000.0, 0.0)[2] < color(-100.0, 0.0)[2]);
        assert!(color(6_000.0, 0.0)[0] > 0.85);
        assert!(color(100.0, 0.0)[1] > color(100.0, 0.0)[0], "grass is green");
        // Steep slopes at the same height look more like rock.
        assert!(color(300.0, 0.6)[0] > color(300.0, 0.0)[0]);
    }

    #[test]
    fn detail_limit_follows_zoom() {
        let focal = 869.0;
        assert!(max_freq(3.0, focal) < max_freq(0.01, focal));
        assert_eq!(max_freq(1e-12, focal), MAX_FREQ_CAP);
    }
}
