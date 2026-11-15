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

## Open questions

- How many zoom levels, and how big is one cell in real-world terms at each?
- Should neighbors at different detail levels blend, or show hard edges? (Hard edges fit the pixel look.)
- Is terrain one value per cell, or several (type, height, moisture, ...)?
- Do brushes work in cell space (always N cells wide) or world space (always N km wide)?
- Name for the project.

## License

TBD.
