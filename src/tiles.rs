//! Terrain tile layers: a pyramid of height tiles on the cube-sphere, generated lazily.
//!
//! A tile is a 64 x 64 cell grid of heights (meters) covering 1/2^level of a cube face, keyed by the
//! same ids the paint uses. Tiles are made the first time something looks at that spot at that
//! zoom: the terrain function is sampled on the tile's lattice (with a margin around it) and then
//! lightly eroded. The margin is as wide as the erosion reaches, so two neighbouring tiles agree
//! on their shared edge. Tiles are cached; the least recently used go first when the cache is full.
//!
//! Everything else (ground mesh, paint, picking) asks `TileStore` for heights through the
//! `HeightSource` trait, so tiles can later also carry things a pure function cannot: rivers,
//! lakes, biomes, hand edits.

use crate::pixelpaint::PixelId;
use crate::terrain::{HeightSource, Terrain, EARTH_RADIUS_M, MAX_FREQ_CAP};
use glam::DVec3;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread::JoinHandle;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Condvar, Mutex, RwLock};
use std::f64::consts::FRAC_PI_2;

/// Cells along a tile edge (the tile has (N+1)^2 height samples, edges shared with neighbours).
pub const TILE_N: usize = 64;
const SAMPLES: usize = TILE_N + 1;
/// Erosion passes per tile.
pub const ERODE_ITERS: usize = 6;
/// Extra samples around a tile while it is generated: as many as the erosion reaches.
const APRON: usize = ERODE_ITERS;
/// Deepest tile level (samples about 0.6 m apart).
pub const MAX_TILE_LEVEL: u8 = 18;
/// Tiles kept in the cache.
pub const CACHE_TILES: usize = 4096;

/// Steepest slope (rise over run) that loose material holds before it slides.
const TALUS: f64 = 0.85;
/// How much of the excess slope slides down per pass.
const SLIDE: f64 = 0.12;
/// How much of the dip in a concave spot gets filled per pass.
const FILL: f64 = 0.14;

/// Spacing between height samples of a tile at `level`, in radians.
pub fn spacing_rad(level: u8) -> f64 {
    FRAC_PI_2 / (TILE_N as f64 * (1u64 << level) as f64)
}

/// Detail limit used when generating tiles of `level` (about 3 samples per finest wavelength).
pub fn freq_for_level(level: u8) -> f64 {
    (1.0 / (3.0 * spacing_rad(level))).min(MAX_FREQ_CAP)
}

/// How fine a tile level a detail limit asks for (fractional: between two levels).
pub fn level_for(max_freq: f64) -> f64 {
    // spacing(level) = 1 / (3 f)  =>  2^level = (pi/2) * 3 f / N
    (FRAC_PI_2 * 3.0 * max_freq / TILE_N as f64).log2().clamp(0.0, MAX_TILE_LEVEL as f64)
}

pub struct Tile {
    h: Vec<f32>,
}

impl Tile {
    /// Height at fractions (0..1) across the tile, bilinear.
    pub fn bilinear(&self, fx: f64, fy: f64) -> f64 {
        let gx = (fx.clamp(0.0, 1.0) * TILE_N as f64).min(TILE_N as f64 - 1e-9);
        let gy = (fy.clamp(0.0, 1.0) * TILE_N as f64).min(TILE_N as f64 - 1e-9);
        let (i, j) = (gx.floor() as usize, gy.floor() as usize);
        let (tx, ty) = (gx - i as f64, gy - j as f64);
        let at = |i: usize, j: usize| self.h[j * SAMPLES + i] as f64;
        let a = at(i, j) * (1.0 - tx) + at(i + 1, j) * tx;
        let b = at(i, j + 1) * (1.0 - tx) + at(i + 1, j + 1) * tx;
        a * (1.0 - ty) + b * ty
    }

    /// Height at fractions (0..1) across the tile, smooth (Catmull-Rom through the samples): no creases
    /// along the sample grid, so slopes and shading do not show it. Passes through every sample like bilinear.
    pub fn cubic(&self, fx: f64, fy: f64) -> f64 {
        let gx = fx.clamp(0.0, 1.0) * TILE_N as f64;
        let gy = fy.clamp(0.0, 1.0) * TILE_N as f64;
        let (i, j) = ((gx.floor() as i64).min(TILE_N as i64 - 1), (gy.floor() as i64).min(TILE_N as i64 - 1));
        let (tx, ty) = (gx - i as f64, gy - j as f64);
        let at = |i: i64, j: i64| self.h[j.clamp(0, TILE_N as i64) as usize * SAMPLES + i.clamp(0, TILE_N as i64) as usize] as f64;
        let w = |t: f64| {
            [
                0.5 * (-t * t * t + 2.0 * t * t - t),
                0.5 * (3.0 * t * t * t - 5.0 * t * t + 2.0),
                0.5 * (-3.0 * t * t * t + 4.0 * t * t + t),
                0.5 * (t * t * t - t * t),
            ]
        };
        let (wx, wy) = (w(tx), w(ty));
        let mut sum = 0.0;
        for (b, wyb) in wy.iter().enumerate() {
            let row: f64 = (0..4).map(|a| wx[a] * at(i - 1 + a as i64, j - 1 + b as i64)).sum();
            sum += wyb * row;
        }
        sum
    }

    pub fn sample(&self, i: usize, j: usize) -> f32 {
        self.h[j * SAMPLES + i]
    }
}

/// Smooths the ground a little, in place on an m x m grid of heights (meters) with `spacing_m` between samples.
/// Land only (the sea is flat). Two things happen, in lockstep passes so the result does not depend on
/// the order cells are visited:
///  - slopes steeper than the talus slope slide material downhill (conserves material),
///  - concave spots (higher all around) fill in a little: valleys get smooth floors, ridges stay sharp.
/// Cells at the grid edge only see neighbours that exist, so the damage creeps in one cell per pass.
pub fn erode(g: &mut [f64], m: usize, spacing_m: f64, iters: usize) {
    let talus = TALUS * spacing_m;
    let mut delta = vec![0.0f64; m * m];
    for _ in 0..iters {
        delta.iter_mut().for_each(|d| *d = 0.0);
        for j in 0..m {
            for i in 0..m {
                let idx = j * m + i;
                let h = g[idx];
                let (mut lap, mut count) = (0.0f64, 0.0f64);
                for (di, dj) in [(-1i32, 0i32), (1, 0), (0, -1), (0, 1)] {
                    let (ni, nj) = (i as i32 + di, j as i32 + dj);
                    if ni < 0 || nj < 0 || ni >= m as i32 || nj >= m as i32 {
                        continue;
                    }
                    let nidx = nj as usize * m + ni as usize;
                    let hn = g[nidx];
                    lap += hn - h;
                    count += 1.0;
                    let drop = h - hn;
                    if h > 0.0 && drop > talus {
                        let amount = SLIDE * (drop - talus);
                        delta[idx] -= amount;
                        delta[nidx] += amount;
                    }
                }
                if h > 0.0 && lap > 0.0 {
                    delta[idx] += FILL * lap / count.max(1.0);
                }
            }
        }
        for (v, d) in g.iter_mut().zip(&delta) {
            *v += d;
        }
    }
}

/// Builds one tile: sample the terrain function on the lattice with an apron, erode, cut out the middle.
pub fn generate(terrain: &Terrain, id: PixelId) -> Tile {
    let m = SAMPLES + 2 * APRON;
    let f = freq_for_level(id.level);
    let mut g = vec![0.0f64; m * m];
    for gj in 0..m {
        for gi in 0..m {
            let fx = (gi as f64 - APRON as f64) / TILE_N as f64;
            let fy = (gj as f64 - APRON as f64) / TILE_N as f64;
            g[gj * m + gi] = terrain.height_fn(id.dir_at(fx, fy), f);
        }
    }
    erode(&mut g, m, spacing_rad(id.level) * EARTH_RADIUS_M, ERODE_ITERS);
    let mut h = Vec::with_capacity(SAMPLES * SAMPLES);
    for j in 0..SAMPLES {
        for i in 0..SAMPLES {
            h.push(g[(j + APRON) * m + i + APRON] as f32);
        }
    }
    Tile { h }
}

/// Bump when the tile file layout changes (a changed terrain function or erosion is noticed on its own).
const DISK_FORMAT: u64 = 1;

/// Tiles kept on disk between runs: the first run makes them, later runs just read them (about 0.05 ms
/// instead of 5 ms each). Files are written by a background thread, so making a tile never waits for the disk.
struct DiskCache {
    dir: PathBuf,
    tx: Mutex<Option<mpsc::Sender<(PathBuf, Vec<u8>)>>>,
    writer: Mutex<Option<JoinHandle<()>>>,
}

impl DiskCache {
    fn open(dir: PathBuf) -> DiskCache {
        let _ = std::fs::create_dir_all(&dir);
        let (tx, rx) = mpsc::channel::<(PathBuf, Vec<u8>)>();
        let writer = std::thread::Builder::new()
            .name("tile-writer".into())
            .spawn(move || {
                for (path, bytes) in rx {
                    if let Some(parent) = path.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    // Written under another name and renamed, so a file is either whole or not there.
                    let tmp = path.with_extension("tmp");
                    if std::fs::write(&tmp, &bytes).is_ok() {
                        let _ = std::fs::rename(&tmp, &path);
                    }
                }
            })
            .expect("tile writer thread");
        DiskCache { dir, tx: Mutex::new(Some(tx)), writer: Mutex::new(Some(writer)) }
    }

    fn path(&self, id: PixelId) -> PathBuf {
        self.dir.join(format!("L{}", id.level)).join(format!("{}_{}_{}.bin", id.face, id.x, id.y))
    }

    fn has(&self, id: PixelId) -> bool {
        self.path(id).exists()
    }

    fn load(&self, id: PixelId) -> Option<Tile> {
        let bytes = std::fs::read(self.path(id)).ok()?;
        if bytes.len() != SAMPLES * SAMPLES * 4 {
            return None;
        }
        Some(Tile { h: bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect() })
    }

    fn store(&self, id: PixelId, tile: &Tile) {
        let bytes: Vec<u8> = tile.h.iter().flat_map(|v| v.to_le_bytes()).collect();
        if let Some(tx) = self.tx.lock().unwrap().as_ref() {
            let _ = tx.send((self.path(id), bytes));
        }
    }
}

impl Drop for DiskCache {
    fn drop(&mut self) {
        self.tx.lock().unwrap().take(); // closes the queue; the writer finishes what is left and ends
        if let Some(w) = self.writer.lock().unwrap().take() {
            let _ = w.join();
        }
    }
}

/// Identifies what the tiles look like: the terrain function, the erosion and the layout. Tiles on disk made
/// by anything else are simply not used (they sit in another folder).
fn fingerprint(t: &Terrain) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut mix = |x: u64| {
        h ^= x;
        h = h.wrapping_mul(0x0100_0000_01b3);
    };
    for k in 0..12 {
        let d = DVec3::new((k as f64 * 1.3).sin(), (k as f64 * 0.7).cos(), (k as f64 * 2.1).sin() + 0.1).normalize();
        for f in [1.0, 40.0, 3000.0] {
            mix(t.height_fn(d, f).to_bits());
        }
    }
    let m = 24;
    let mut g: Vec<f64> = (0..m * m).map(|i| ((i * 7919) % 101) as f64 * 3.0).collect();
    erode(&mut g, m, 30.0, ERODE_ITERS);
    for x in g {
        mix(x.to_bits());
    }
    mix(TILE_N as u64);
    mix(DISK_FORMAT);
    h
}

struct Cached {
    tile: Tile,
    used: AtomicU64,
}

/// Safe to share between threads: lookups only take a read lock (the "last used" stamp is atomic),
/// so the ground mesh can be built on all cores at once.
pub struct TileStore {
    terrain: Terrain,
    cache: RwLock<HashMap<PixelId, Cached>>,
    tick: AtomicU64,
    generated: AtomicUsize,
    capacity: usize,
    /// Tiles being made right now: a second thread that needs one waits for it instead of making it again.
    making: Mutex<HashSet<PixelId>>,
    made: Condvar,
    disk: Option<DiskCache>,
    from_disk: AtomicUsize,
}

impl TileStore {
    pub fn new(terrain: Terrain) -> TileStore {
        TileStore::with_capacity(terrain, CACHE_TILES)
    }

    pub fn with_capacity(terrain: Terrain, capacity: usize) -> TileStore {
        TileStore {
            terrain,
            cache: RwLock::new(HashMap::new()),
            tick: AtomicU64::new(0),
            generated: AtomicUsize::new(0),
            capacity: capacity.max(8),
            making: Mutex::new(HashSet::new()),
            made: Condvar::new(),
            disk: None,
            from_disk: AtomicUsize::new(0),
        }
    }

    /// A store that keeps its tiles in `root` between runs (in a folder named after what the terrain looks like).
    pub fn with_disk(terrain: Terrain, root: &Path) -> TileStore {
        let dir = root.join(format!("{:016x}", fingerprint(&terrain)));
        let mut s = TileStore::with_capacity(terrain, CACHE_TILES);
        s.disk = Some(DiskCache::open(dir));
        s
    }

    /// Tiles read from disk instead of generated.
    pub fn from_disk(&self) -> usize {
        self.from_disk.load(Ordering::Relaxed)
    }

    /// Makes sure every tile up to `max_level` is on disk (coarse levels first), with `threads` workers. Tiles
    /// already there are skipped. Does nothing without a disk cache.
    pub fn pregenerate(&self, max_level: u8, threads: usize) {
        let Some(disk) = self.disk.as_ref() else { return };
        let mut ids = Vec::new();
        for level in 0..=max_level {
            let n = 1u32 << level;
            for face in 0..6u8 {
                for y in 0..n {
                    for x in 0..n {
                        ids.push(PixelId::new(face, level, x, y));
                    }
                }
            }
        }
        let next = AtomicUsize::new(0);
        std::thread::scope(|sc| {
            for _ in 0..threads.max(1) {
                sc.spawn(|| loop {
                    let q = next.fetch_add(1, Ordering::Relaxed);
                    let Some(id) = ids.get(q) else { break };
                    if !disk.has(*id) {
                        let t = generate(&self.terrain, *id);
                        self.generated.fetch_add(1, Ordering::Relaxed);
                        disk.store(*id, &t);
                    }
                });
            }
        });
    }

    pub fn terrain(&self) -> &Terrain {
        &self.terrain
    }

    /// How many tiles have been generated so far (including ones evicted since).
    pub fn generated(&self) -> usize {
        self.generated.load(Ordering::Relaxed)
    }

    pub fn cached(&self) -> usize {
        self.cache.read().unwrap().len()
    }

    fn next_tick(&self) -> u64 {
        self.tick.fetch_add(1, Ordering::Relaxed) + 1
    }

    /// Runs `f` on the tile if it is already made (never makes one).
    fn with_cached<R>(&self, id: PixelId, f: impl FnOnce(&Tile) -> R) -> Option<R> {
        let cache = self.cache.read().unwrap();
        let c = cache.get(&id)?;
        c.used.store(self.next_tick(), Ordering::Relaxed);
        Some(f(&c.tile))
    }

    /// Runs `f` on the tile, making it first if it does not exist yet.
    pub fn with_tile<R>(&self, id: PixelId, f: impl FnOnce(&Tile) -> R) -> R {
        let mut f = Some(f);
        if let Some(r) = self.with_cached(id, |t| (f.take().unwrap())(t)) {
            return r;
        }
        // One thread makes a tile; others needing it meanwhile wait and then read it.
        loop {
            let mut making = self.making.lock().unwrap();
            if making.contains(&id) {
                while making.contains(&id) {
                    making = self.made.wait(making).unwrap();
                }
                drop(making);
                if let Some(r) = self.with_cached(id, |t| (f.take().unwrap())(t)) {
                    return r;
                }
                continue; // evicted again already: try to make it ourselves
            }
            making.insert(id);
            break;
        }
        let tile = match self.disk.as_ref().and_then(|d| d.load(id)) {
            Some(t) => {
                self.from_disk.fetch_add(1, Ordering::Relaxed);
                t
            }
            None => {
                let t = generate(&self.terrain, id);
                self.generated.fetch_add(1, Ordering::Relaxed);
                if let Some(d) = &self.disk {
                    d.store(id, &t);
                }
                t
            }
        };
        let r = (f.take().unwrap())(&tile);
        self.insert(id, tile);
        self.making.lock().unwrap().remove(&id);
        self.made.notify_all();
        r
    }

    fn insert(&self, id: PixelId, tile: Tile) {
        let tick = self.next_tick();
        let mut cache = self.cache.write().unwrap();
        if cache.len() >= self.capacity {
            // Drop the quarter that was used longest ago.
            let mut stamps: Vec<u64> = cache.values().map(|c| c.used.load(Ordering::Relaxed)).collect();
            stamps.sort_unstable();
            let cut = stamps[stamps.len() / 4];
            cache.retain(|_, c| c.used.load(Ordering::Relaxed) > cut);
        }
        cache.insert(id, Cached { tile, used: AtomicU64::new(tick) });
    }

    /// Height (m) at a direction from the tiles of exactly `level`.
    pub fn sample(&self, d: DVec3, level: u8) -> f64 {
        let id = PixelId::from_dir(d, level);
        let (fx, fy) = id.frac_in(d);
        self.with_tile(id, |t| t.cubic(fx, fy))
    }

    /// Height from `level`, or from the nearest coarser level that is already made. Never makes a tile,
    /// except level 0 (a handful of tiles for the whole planet) if nothing at all exists yet.
    pub fn sample_peek(&self, d: DVec3, level: u8) -> f64 {
        let mut lv = level;
        loop {
            let id = PixelId::from_dir(d, lv);
            let (fx, fy) = id.frac_in(d);
            if let Some(h) = self.with_cached(id, |t| t.cubic(fx, fy)) {
                return h;
            }
            if lv == 0 {
                return self.sample(d, 0);
            }
            lv -= 1;
        }
    }
}

impl HeightSource for TileStore {
    /// Makes every tile these lookups will touch that does not exist yet, on all CPU cores at once.
    fn prefetch(&self, points: &[(DVec3, f64)]) {
        let mut need: Vec<PixelId> = Vec::new();
        {
            let cache = self.cache.read().unwrap();
            let mut seen = HashSet::new();
            for (d, f) in points {
                let l = level_for(*f);
                let lo = l.floor() as u8;
                let mut levels = vec![lo];
                if l - lo as f64 >= 1e-3 && lo < MAX_TILE_LEVEL {
                    levels.push(lo + 1);
                }
                for lv in levels {
                    let id = PixelId::from_dir(*d, lv);
                    if !cache.contains_key(&id) && seen.insert(id) {
                        need.push(id);
                    }
                }
            }
        }
        need.truncate(self.capacity / 2);
        if need.len() < 4 {
            return; // few enough to make on demand
        }
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
        let chunk = need.len().div_ceil(threads);
        let terrain = &self.terrain;
        let made: Vec<(PixelId, Tile)> = std::thread::scope(|scope| {
            let handles: Vec<_> = need
                .chunks(chunk)
                .map(|ids| scope.spawn(move || ids.iter().map(|id| (*id, generate(terrain, *id))).collect::<Vec<_>>()))
                .collect();
            handles.into_iter().flat_map(|h| h.join().expect("tile thread")).collect()
        });
        self.generated.fetch_add(made.len(), Ordering::Relaxed);
        for (id, tile) in made {
            self.insert(id, tile);
        }
    }

    /// Blends the two tile levels around the one the detail limit asks for, so detail fades in smoothly while zooming.
    fn height_m(&self, d: DVec3, max_freq: f64) -> f64 {
        let l = level_for(max_freq);
        let lo = l.floor() as u8;
        let t = l - lo as f64;
        let a = self.sample(d, lo);
        if t < 1e-3 || lo >= MAX_TILE_LEVEL {
            return a;
        }
        let b = self.sample(d, lo + 1);
        a + (b - a) * t
    }

    /// The same without ever making a tile (coarser data stands in for what is missing).
    fn height_m_peek(&self, d: DVec3, max_freq: f64) -> f64 {
        let l = level_for(max_freq);
        let lo = l.floor() as u8;
        let t = l - lo as f64;
        let a = self.sample_peek(d, lo);
        if t < 1e-3 || lo >= MAX_TILE_LEVEL {
            return a;
        }
        let b = self.sample_peek(d, lo + 1);
        a + (b - a) * t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(i: u32) -> DVec3 {
        let n = 40_000.0;
        let k = i as f64 + 0.5;
        let z = 1.0 - 2.0 * k / n;
        let a = k * 2.399963229728653;
        let r = (1.0 - z * z).sqrt();
        DVec3::new(r * a.cos(), r * a.sin(), z)
    }

    fn store() -> TileStore {
        TileStore::new(Terrain::new(7))
    }

    #[test]
    fn tiles_are_made_once_and_then_reused() {
        let s = store();
        let d = dir(123);
        let a = s.sample(d, 5);
        let n = s.generated();
        assert_eq!(n, 1);
        assert_eq!(s.sample(d, 5), a);
        assert_eq!(s.generated(), n);
        s.sample(d, 6);
        assert_eq!(s.generated(), n + 1);
    }

    #[test]
    fn neighbouring_tiles_agree_on_their_shared_edge() {
        // The reason for the apron: no cracks between tiles of one face.
        let s = store();
        let level = 7;
        let mut checked = 0;
        for k in 0..30u32 {
            let d = dir(k * 1300 + 17);
            let a = PixelId::from_dir(d, level);
            let max = (1u32 << level) - 1;
            if a.x >= max || a.y >= max {
                continue;
            }
            let right = PixelId::new(a.face, level, a.x + 1, a.y);
            let below = PixelId::new(a.face, level, a.x, a.y + 1);
            s.with_tile(a, |ta| {
                s.with_tile(right, |tr| {
                    for j in 0..SAMPLES {
                        let (x, y) = (ta.sample(TILE_N, j) as f64, tr.sample(0, j) as f64);
                        assert!((x - y).abs() < 0.05, "right edge row {j}: {x} vs {y}");
                    }
                });
                s.with_tile(below, |tb| {
                    for i in 0..SAMPLES {
                        let (x, y) = (ta.sample(i, TILE_N) as f64, tb.sample(i, 0) as f64);
                        assert!((x - y).abs() < 0.05, "bottom edge col {i}: {x} vs {y}");
                    }
                });
            });
            checked += 1;
        }
        assert!(checked >= 15, "{checked}");
    }

    #[test]
    fn tile_heights_follow_the_terrain_function() {
        // Erosion only touches the surface a little: most samples stay close to the raw function.
        let s = store();
        let level = 8;
        let f = freq_for_level(level);
        let mut diffs = Vec::new();
        for k in 0..40u32 {
            let d = dir(k * 977 + 5);
            let want = s.terrain().height_fn(d, f);
            diffs.push((s.sample(d, level) - want).abs());
        }
        diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!(diffs[diffs.len() / 2] < 30.0, "median difference {}", diffs[diffs.len() / 2]);
    }

    #[test]
    fn erosion_slides_spikes_and_fills_pits_but_keeps_ridges() {
        let m = 21;
        let sp = 100.0;
        // A needle on a plateau: its flanks are far steeper than the talus slope.
        let mut g = vec![500.0f64; m * m];
        g[10 * m + 10] = 5_000.0;
        let before: f64 = g.iter().sum();
        erode(&mut g, m, sp, 40);
        let after: f64 = g.iter().sum();
        assert!(g[10 * m + 10] < 5_000.0 - 100.0, "needle did not slide: {}", g[10 * m + 10]);
        assert!(g.iter().all(|v| v.is_finite()));
        assert!(after >= before - 1e-6, "slides never lose material: {before} -> {after}");
        // A pit gets filled, a ridge line keeps its crest.
        let mut pit = vec![500.0f64; m * m];
        pit[10 * m + 10] = 450.0;
        erode(&mut pit, m, sp, 6);
        assert!(pit[10 * m + 10] > 450.0 + 10.0);
        let mut ridge = vec![500.0f64; m * m];
        for j in 0..m {
            ridge[j * m + 10] = 520.0;
        }
        erode(&mut ridge, m, sp, 6);
        assert!((ridge[10 * m + 10] - 520.0).abs() < 1.0, "ridge crest changed: {}", ridge[10 * m + 10]);
        // Water stays put.
        let mut sea = vec![-200.0f64; m * m];
        sea[10 * m + 10] = -100.0;
        erode(&mut sea, m, sp, 6);
        assert_eq!(sea[10 * m + 10], -100.0);
    }

    #[test]
    fn levels_blend_smoothly() {
        let s = store();
        for k in 0..20u32 {
            let d = dir(k * 2003 + 9);
            let mut last = s.height_m(d, 1e3);
            let mut f = 1e3;
            for _ in 0..40 {
                f *= 1.04;
                let h = s.height_m(d, f);
                assert!((h - last).abs() < 150.0, "jump {last} -> {h} at f={f}");
                last = h;
            }
        }
    }

    #[test]
    fn a_point_moving_a_hair_changes_height_a_hair() {
        let s = store();
        for k in 0..60u32 {
            let d = dir(k * 641 + 3);
            let e = (d + DVec3::new(0.3, -0.2, 0.9).normalize() * 3e-8).normalize();
            let (a, b) = (s.height_m(d, 1e5), s.height_m(e, 1e5));
            assert!((a - b).abs() < 5.0, "{a} vs {b}");
        }
    }

    #[test]
    fn the_cache_never_grows_past_its_capacity() {
        let s = TileStore::with_capacity(Terrain::new(7), 16);
        for k in 0..200u32 {
            s.sample(dir(k * 131), 6);
            assert!(s.cached() <= 16, "{}", s.cached());
        }
        assert!(s.generated() > 16);
    }

    #[test]
    fn level_choice_follows_the_detail_limit() {
        assert_eq!(level_for(1.0), 0.0);
        assert!(level_for(1e3) < level_for(1e6));
        assert_eq!(level_for(1e12), MAX_TILE_LEVEL as f64);
        assert!(freq_for_level(10) > freq_for_level(5));
        // The tile level asked for always has at least the detail limit's frequency.
        for f in [10.0, 1e3, 1e5] {
            let l = level_for(f).ceil() as u8;
            assert!(freq_for_level(l) >= f * 0.99);
        }
    }

    #[test]
    fn making_a_tile_is_fast_enough() {
        let s = store();
        let start = std::time::Instant::now();
        for k in 0..8u32 {
            s.sample(dir(k * 4001), 12);
        }
        let per = start.elapsed().as_secs_f64() * 1000.0 / 8.0;
        println!("{per:.1} ms per tile");
        assert!(per < 120.0, "{per} ms per tile");
    }

    #[test]
    fn cubic_sampling_passes_through_the_samples_and_is_smooth() {
        let st = store();
        let id = PixelId::from_dir(dir(3), 6);
        st.with_tile(id, |t| {
            for (i, j) in [(0usize, 0usize), (10, 20), (64, 64), (33, 7)] {
                let (fx, fy) = (i as f64 / TILE_N as f64, j as f64 / TILE_N as f64);
                assert!((t.cubic(fx, fy) - t.sample(i, j) as f64).abs() < 1e-3);
            }
            // Between samples it stays close to the bilinear value (no wild overshoot).
            for k in 0..200 {
                let (fx, fy) = ((k * 37 % 200) as f64 / 200.0, (k * 91 % 200) as f64 / 200.0);
                assert!((t.cubic(fx, fy) - t.bilinear(fx, fy)).abs() < 600.0);
            }
        });
    }

    #[test]
    fn tiles_saved_to_disk_are_read_back_instead_of_generated() {
        let dir = std::env::temp_dir().join(format!("gmm-tiles-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let id = PixelId::new(1, 6, 9, 20);
        let d = dir_for(&id);
        let first = {
            let s = TileStore::with_disk(Terrain::new(7), &dir);
            let v = s.sample(d, 6);
            assert_eq!(s.generated(), 1);
            v
        }; // dropping the store waits for the writer, so the file is there
        let s = TileStore::with_disk(Terrain::new(7), &dir);
        let again = s.sample(d, 6);
        assert_eq!((s.generated(), s.from_disk()), (0, 1), "second run should read the tile, not make it");
        assert_eq!(first, again);
        // Another terrain does not pick these tiles up.
        let other = TileStore::with_disk(Terrain::new(8), &dir);
        other.sample(d, 6);
        assert_eq!(other.from_disk(), 0);
        drop((s, other));
        // Pregeneration fills in the coarse levels, and only what is missing.
        let p = TileStore::with_disk(Terrain::new(7), &dir);
        p.pregenerate(1, 2);
        assert_eq!(p.generated(), 6 + 24);
        p.pregenerate(1, 2);
        assert_eq!(p.generated(), 6 + 24);
        drop(p);
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn dir_for(id: &PixelId) -> DVec3 {
        id.center_dir()
    }
}
