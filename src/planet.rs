//! What makes one planet different from another: the settings, and the biomes they produce.
//!
//! A biome comes from climate: temperature (cold at the poles and on mountains) and moisture (wet near the
//! equator and the coast, dry inland and in the desert belts), laid out like Whittaker's chart. Every biome has a
//! ground colour and a land shape (how rugged, dunes, mesas, rock domes). Memberships are soft, so neighbouring
//! biomes blend instead of meeting at a line.

/// The land biomes.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Biome {
    Ice,
    Tundra,
    Taiga,
    Forest,
    Plains,
    Desert,
    Rainforest,
}

pub const BIOME_COUNT: usize = 7;

impl Biome {
    pub const ALL: [Biome; BIOME_COUNT] =
        [Biome::Ice, Biome::Tundra, Biome::Taiga, Biome::Forest, Biome::Plains, Biome::Desert, Biome::Rainforest];

    pub fn name(self) -> &'static str {
        match self {
            Biome::Ice => "Ice",
            Biome::Tundra => "Tundra",
            Biome::Taiga => "Taiga",
            Biome::Forest => "Forest",
            Biome::Plains => "Plains",
            Biome::Desert => "Desert",
            Biome::Rainforest => "Rainforest",
        }
    }

    pub fn index(self) -> usize {
        Biome::ALL.iter().position(|b| *b == self).unwrap()
    }

    /// Ground cover colour at low and middle heights (the height ramp darkens it toward the forest line and
    /// turns it to rock and snow higher up).
    pub fn cover(self) -> [f32; 3] {
        match self {
            Biome::Ice => [0.68, 0.73, 0.80],
            Biome::Tundra => [0.45, 0.47, 0.36],
            Biome::Taiga => [0.12, 0.28, 0.15],
            Biome::Forest => [0.22, 0.42, 0.08],
            Biome::Plains => [0.38, 0.52, 0.16],
            Biome::Desert => [0.80, 0.66, 0.40],
            Biome::Rainforest => [0.06, 0.30, 0.08],
        }
    }

    /// How sandy its beaches are (0 = pebbles and rock, 1 = sand): cold coasts have none, warm ones all.
    pub fn sandy(self) -> f32 {
        match self {
            Biome::Ice => 0.0,
            Biome::Tundra => 0.1,
            Biome::Taiga => 0.3,
            Biome::Forest => 0.75,
            Biome::Plains => 0.85,
            Biome::Desert => 1.0,
            Biome::Rainforest => 1.0,
        }
    }

    /// How this biome shapes the land.
    pub fn shape(self) -> Shape {
        let s = |hills, mountains, dunes, mesas, domes| Shape { hills, mountains, dunes, mesas, domes };
        match self {
            // Ice sheets are flat; the rock peaks that poke out of them are the mountains, kept.
            Biome::Ice => s(0.25, 0.8, 0.0, 0.0, 0.0),
            Biome::Tundra => s(0.6, 0.9, 0.0, 0.0, 0.0),
            Biome::Taiga => s(0.8, 1.0, 0.0, 0.0, 0.0),
            Biome::Forest => s(1.0, 1.0, 0.0, 0.0, 0.0),
            // Savanna and steppe: wide flat country with the odd rock dome (inselberg).
            Biome::Plains => s(0.55, 0.7, 0.0, 0.0, 1.0),
            // Desert: dune fields, flat-topped mesas.
            Biome::Desert => s(0.9, 0.9, 1.0, 1.0, 0.0),
            // Rainforest: deeply cut, rounded hills.
            Biome::Rainforest => s(1.25, 1.0, 0.0, 0.0, 0.0),
        }
    }
}

/// Land shape parameters of a biome (blended by membership). All 1.0 / 0.0 = the plain terrain.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shape {
    /// Multiplies hills and rolling relief.
    pub hills: f64,
    /// Multiplies how high the mountain belts rise.
    pub mountains: f64,
    /// 0..1: how much of the land is dune field (low, dry ground only).
    pub dunes: f64,
    /// 0..1: how much of the land is cut into flat-topped mesas.
    pub mesas: f64,
    /// 0..1: scattered rock domes on the plains.
    pub domes: f64,
}

/// How the biomes are laid out over the planet.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BiomeMode {
    /// From climate: poles cold, equator warm, wet and dry belts, mountains cold.
    Earthlike,
    /// Climate ignored: random blobs of every biome.
    Patchwork,
    /// The whole planet is one biome.
    Single(Biome),
}

/// Everything the user chooses for a planet.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanetSettings {
    pub seed: u64,
    /// Where the sea surface is, meters (0 = the default sea level; high = an ocean planet; negative = drained).
    pub sea_level_m: f64,
    /// 0..1, how much of the planet is land (0.5 = the default).
    pub land: f64,
    /// 0..2, how high and rugged the mountains are (1 = default).
    pub mountains: f64,
    /// 0..1, cold to hot (0.5 = Earth-like).
    pub climate: f64,
    /// 0..1, dry to wet (0.5 = Earth-like).
    pub wetness: f64,
    pub biomes: BiomeMode,
    /// How many natural cave sites there are (used once caves exist).
    pub cave_amount: f64,
}

impl Default for PlanetSettings {
    fn default() -> Self {
        PlanetSettings {
            seed: 2026,
            sea_level_m: 0.0,
            land: 0.5,
            mountains: 1.0,
            climate: 0.5,
            wetness: 0.5,
            biomes: BiomeMode::Earthlike,
            cave_amount: 10.0,
        }
    }
}

fn smooth(a: f64, b: f64, x: f64) -> f64 {
    let t = ((x - a) / (b - a)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Soft biome memberships (sum to 1) for a temperature and a moisture, both 0..1.
pub fn weights(t: f64, m: f64) -> [f64; BIOME_COUNT] {
    let ice = 1.0 - smooth(0.10, 0.20, t);
    let tundra_t = smooth(0.10, 0.20, t) * (1.0 - smooth(0.24, 0.34, t));
    let cold_t = smooth(0.24, 0.34, t) * (1.0 - smooth(0.42, 0.52, t));
    let temp_t = smooth(0.42, 0.52, t) * (1.0 - smooth(0.68, 0.78, t));
    let hot_t = smooth(0.68, 0.78, t);
    let dry = 1.0 - smooth(0.18, 0.32, m);
    let wet = smooth(0.55, 0.70, m);
    let forest_m = smooth(0.26, 0.40, m);
    let mut w = [0.0; BIOME_COUNT];
    w[Biome::Ice.index()] = ice;
    w[Biome::Tundra.index()] = tundra_t + cold_t * dry;
    w[Biome::Taiga.index()] = cold_t * (1.0 - dry);
    w[Biome::Forest.index()] = temp_t * (1.0 - dry) * forest_m;
    w[Biome::Plains.index()] = temp_t * (1.0 - dry) * (1.0 - forest_m) + hot_t * (1.0 - dry) * (1.0 - wet);
    w[Biome::Desert.index()] = (temp_t + hot_t) * dry;
    w[Biome::Rainforest.index()] = hot_t * (1.0 - dry) * wet;
    let sum: f64 = w.iter().sum();
    if sum > 1e-9 {
        w.map(|x| x / sum)
    } else {
        let mut f = [0.0; BIOME_COUNT];
        f[Biome::Forest.index()] = 1.0;
        f
    }
}

/// Blends per-biome values by membership.
pub fn blend_shape(w: &[f64; BIOME_COUNT]) -> Shape {
    let mut s = Shape { hills: 0.0, mountains: 0.0, dunes: 0.0, mesas: 0.0, domes: 0.0 };
    for (b, k) in Biome::ALL.iter().zip(w) {
        let sh = b.shape();
        s.hills += sh.hills * k;
        s.mountains += sh.mountains * k;
        s.dunes += sh.dunes * k;
        s.mesas += sh.mesas * k;
        s.domes += sh.domes * k;
    }
    s
}

pub fn blend_sandy(w: &[f64; BIOME_COUNT]) -> f32 {
    Biome::ALL.iter().zip(w).map(|(b, k)| b.sandy() * *k as f32).sum()
}

/// The beach colours: sand, and the stone that replaces it on cold coasts (grey, a little of the biome's colour).
pub const SAND: [f32; 3] = [0.62, 0.50, 0.26];
pub fn beach(cover: [f32; 3], sandy: f32) -> [f32; 3] {
    let stone = [0.40f32, 0.40, 0.41];
    [0, 1, 2].map(|i| {
        let st = stone[i] * 0.9 + cover[i] * 0.1;
        st + (SAND[i] - st) * sandy
    })
}

pub fn blend_cover(w: &[f64; BIOME_COUNT]) -> [f32; 3] {
    let mut c = [0.0f32; 3];
    for (b, k) in Biome::ALL.iter().zip(w) {
        let bc = b.cover();
        for i in 0..3 {
            c[i] += bc[i] * *k as f32;
        }
    }
    c
}

/// The cover colour the ground has with no biome at all (the temperate-forest green the terrain always had).
pub const DEFAULT_COVER: [f32; 3] = [0.22, 0.42, 0.08];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weights_sum_to_one_everywhere() {
        for ti in 0..=20 {
            for mi in 0..=20 {
                let w = weights(ti as f64 / 20.0, mi as f64 / 20.0);
                assert!((w.iter().sum::<f64>() - 1.0).abs() < 1e-9);
                assert!(w.iter().all(|x| *x >= -1e-12));
            }
        }
    }

    #[test]
    fn climate_picks_the_biome_it_should() {
        let top = |t, m| {
            let w = weights(t, m);
            Biome::ALL[w.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0]
        };
        assert_eq!(top(0.02, 0.5), Biome::Ice);
        assert_eq!(top(0.22, 0.6), Biome::Tundra);
        assert_eq!(top(0.38, 0.6), Biome::Taiga);
        assert_eq!(top(0.60, 0.7), Biome::Forest);
        assert_eq!(top(0.90, 0.10), Biome::Desert);
        assert_eq!(top(0.90, 0.90), Biome::Rainforest);
        assert_eq!(top(0.90, 0.45), Biome::Plains);
    }

    #[test]
    fn the_default_settings_are_the_old_planet() {
        let s = PlanetSettings::default();
        assert_eq!((s.sea_level_m, s.land, s.mountains), (0.0, 0.5, 1.0));
        assert_eq!(DEFAULT_COVER, Biome::Forest.cover());
    }
}
