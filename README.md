# Global Map Maker

A 3D globe editor for making fantasy worlds, with a chunky-pixel look that stays crisp at every zoom level.

> Status: early. The sphere core (HEALPix cells <-> directions, with tests) works. No window or renderer yet. This file is also the parking lot for the plan.

Run the tests with `cargo test`.

## The core idea

- A planet you can spin and zoom, rendered as pixels ("cells").
- **Resolution follows zoom.** From orbit a cell is big. Zoom in and a cell splits into more cells (e.g. 2x2 -> 20x20 -> 200x200), so you always see crisp pixels, never blur.
- **Brushes stick to the world, not the screen.** A brush paints at the cell size of the level you're working at, and what you painted stays at that size when you zoom further in.
- Editable terrain: paint terrain types, raise mountains, dig into the planet.
- Custom file format for saving worlds.

## Feature wishlist

### v0.1: prove the look
- [ ] Window + camera that orbits a sphere
- [ ] Sphere made of cells, with a zoom-dependent subdivision level
- [ ] Per-cell color / terrain type
- [ ] One basic paint brush

### v0.2: editing
- [ ] Brush size and shape options
- [ ] Terrain type palette (ocean, grass, forest, desert, snow, rock, ...)
- [ ] Height tools: raise / lower / smooth (mountains, valleys)
- [ ] Undo / redo
- [ ] Save / load

### Later
- [ ] Digging into the planet (cross-section view, caves, layers)
- [ ] Custom brushes and terrain templates the user can define
- [ ] Custom file format with a documented spec
- [ ] Export (flat map projections, PNG, heightmaps)
- [ ] Labels / markers / borders for fantasy map stuff

## Technical ideas (open, not decided)

**Language:** Rust. Reasons: performance headroom if the data gets heavy, memory control, good GPU ecosystem.
Candidates: `wgpu` + `winit` (more control, more work) or `bevy` (faster start, more opinionated). Decide after a tiny prototype.

**Sphere layout:** a plain lat/long grid pinches at the poles, so cells get distorted. Better options:
- Cube-sphere: 6 faces, each a quadtree. Subdivision maps naturally onto the "2x2 splits into more" idea.
- Icosphere / hex-style grids: more even cells, harder to subdivide cleanly.
**Decided: HEALPix (nested scheme)**: 12 base faces, equal-area cells, each face a quadtree. Implemented in `src/sphere.rs` (cell <-> direction on the sphere, with tests).

**Detail on demand:** only store and draw detail where the user has actually painted or is looking. Coarse cells everywhere else. A cell with no finer data simply renders as one big pixel.

**Digging / volume:** the hard part. Options:
- Heightmap plus a few stacked layers (cheap, covers mountains and shallow digging).
- Sparse voxels / octree under the surface (real caves, much heavier).
Start with heightmap + layers and see if it's enough.

**File format:** sparse and chunked, since most of the planet stays coarse.
Rough shape: header (magic, version, planet settings) -> per-face quadtree chunks -> terrain palette -> optional layers. Compress chunks. Keep it versioned from day one.

## Controls (hex mode, the default)
| What | How |
|---|---|
| Turn the globe / zoom | left-drag / scroll |
| Use the current tool | right mouse (drag for pencil, eraser, line, circle) |
| Color | 1-9 (deep red, grass, forest, sand, rock, snow, dirt, shallow water, deep water) |
| Tools | B pencil, E or 0 eraser, L line, C circle (Shift = outline), F fill, I eyedropper |
| Brush size | [ and ] |
| Undo | Ctrl+Z |
| Zoom in on the globe past ~190 km: enter the Inside scene (fly over the land) | scroll |
| Inside: look around | left-drag |
| Inside: fly (Shift = faster) | W A S D |
| Inside: height (scroll out past the top = back to space) | scroll |
| Inside: back to space | Esc |
| Inside: render distance (shorter = faster) | PageUp / PageDown (title shows km) |
| Ground: flat sphere with terrain (default) or hex grid | G |
| Switch hex / pixel layer | P |
| Pixel size (pixel layer) | [ and ] |
| Pixel brush radius (pixel layer) | - and = |
| Switch globe type | G (smooth / faceted / hex) |

The grid detail follows the zoom (H3 hexagons, down to about 1 m on an Earth-sized globe). Painting is per level: painting a big hex wipes the finer paint inside it, and zooming out shows the average. The bottom-left corner shows a distance scale bar (Earth-sized globe) and a ladder of the 16 zoom levels.

## Open questions

- How many zoom levels, and how big is one cell in real-world terms at each?
- Should neighbors at different detail levels blend, or show hard edges? (Hard edges fit the pixel look.)
- Is terrain one value per cell, or several (type, height, moisture, ...)?
- Do brushes work in cell space (always N cells wide) or world space (always N km wide)?
- Name for the project.

## License

TBD.

## Tile cache
Height tiles are saved to `cache/tiles/` (ignored by git) and the coarse levels are made in the background on the first run (~140 MB). Changing the terrain function starts a new cache folder automatically; delete old folders in `cache/tiles/` to reclaim space.
