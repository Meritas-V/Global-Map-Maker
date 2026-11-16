//! Window + GPU + drawing the globe. Mouse drag rotates, scroll zooms.

use global_map_maker::camera::{Camera, Fly};
use global_map_maker::hexmesh::build_hex_cells;
use global_map_maker::hexpaint::{self, Paint, PaintStore};
use global_map_maker::tools::{self, Tool};
use global_map_maker::hexview::{cell_width, choose_res, pick_cell, visible_cells as visible_hexes, MAX_PAINT_RES};
use global_map_maker::pixelpaint::{PixelId, PixelStore};
use global_map_maker::pixelview::{
    self, build_vertices, disk, highlight_vertices, pick_pixel, split_for_relief, visible_pixels, PixelVertex, PixelView,
    Surface, LIFT_PX,
};
use global_map_maker::overlay::{build_overlay, meters_per_pixel, OverlayVertex};
use h3o::{CellIndex, Resolution};
use glam::DVec3;
use std::collections::HashSet;
use global_map_maker::ground::{build_patches, crowd, patch_key, visible_patches, GroundVertex, Lighting, PatchKey, MAX_PATCHES, SKY};
use std::collections::HashMap;
use global_map_maker::terrain::{Terrain, TerrainRef};
use global_map_maker::tiles::TileStore;
use global_map_maker::mesh::Vertex;
use std::sync::Arc;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalPosition,
    event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    keyboard::{Key, NamedKey},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};

/// Room for the ground / hex triangles (hexes: 18 vertices each).
const MAX_VERTS: usize = 600_000;
/// Ground patch meshes kept between frames.
const PATCH_CACHE_MAX: usize = 9_000;
/// Room for the pixel layer's triangles.
const PIXEL_MAX_VERTS: usize = 900_000;
/// Brush radius in pixels for the pixel layer ( - and = change it).
const PIXEL_RADII: [u32; 4] = [0, 1, 2, 4];

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Layer {
    Hex,
    Pixel,
}

/// Outside (the globe): terrain never gets finer than this, so the map stays a clean map.
const OUTSIDE_CAP: f64 = 256.0;
/// Zooming in on the globe past this height (globe radii above the ground, ~190 km) enters the inside scene.
const OUTSIDE_MIN_H: f64 = 0.03;
/// Where the camera is put when leaving the inside scene.
const OUTSIDE_EXIT_H: f64 = 0.045;
/// Inside: lowest and highest the camera may be above the ground (about 1.6 m and 127 km); zooming out past the top leaves.
const INSIDE_MIN_ALT: f64 = 2.5e-7;
const INSIDE_MAX_ALT: f64 = 0.02;
/// Height and downward tilt when entering the inside scene.
const INSIDE_ENTER_ALT: f64 = 0.012;
const INSIDE_ENTER_TILT: f64 = -0.8;
/// Fly speed: this many heights above the ground per second (shift = faster).
/// One cached ground patch mesh on the GPU.
struct GpuPatch {
    buf: wgpu::Buffer,
    count: u32,
    center: DVec3,
    used: u64,
}

/// How quickly the flying camera's ground level follows the land (seconds): it glides over ridges instead of bobbing.
const GLIDE_TAU: f64 = 0.45;
/// Inside, the camera never gets closer to the real ground than this fraction of its height.
const CLEARANCE: f64 = 0.25;
const FLY_SPEED: f64 = 0.8;
const FLY_BOOST: f64 = 5.0;

/// Room for the corner display (scale bar, level ladder, text).
const OVERLAY_MAX_VERTS: usize = 12_000;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

struct Gpu {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    pipeline: wgpu::RenderPipeline,
    overlay_pipeline: wgpu::RenderPipeline,
    /// Pixels over the hexes: no depth test (see pixelview.rs).
    pixel_pipeline: wgpu::RenderPipeline,
    /// Pixels on the terrain: depth-tested against the ground.
    pixel_pipeline_depth: wgpu::RenderPipeline,
    pixel_buf: wgpu::Buffer,
    pixel_count: u32,
    /// The planet's terrain (flat-sphere ground only) and the detail limit for the current frame.
    terrain: TileStore,
    max_freq: f64,
    overlay_buf: wgpu::Buffer,
    overlay_count: u32,
    vertex_buf: wgpu::Buffer,
    vertex_count: u32,
    ground_pipeline: wgpu::RenderPipeline,
    /// Per-patch offsets (one vec4 each), written every frame; patch meshes themselves are kept between frames.
    inst_buf: wgpu::Buffer,
    patch_cache: HashMap<PatchKey, GpuPatch>,
    patch_tick: u64,
    draw_list: Vec<(wgpu::Buffer, u32)>,
    ground_light: Lighting,
    ground_uniform: wgpu::Buffer,
    ground_bind: wgpu::BindGroup,
    /// Set when the camera or window changed, so the set of cells must be recomputed.
    dirty: bool,
    /// Only the pixel layer / highlight changed (painting, hover): the ground mesh is kept.
    pixels_dirty: bool,
    /// Extra rebuilds left to settle after new height tiles appeared mid-frame.
    settle: u8,
    hex_count: usize,
    /// Inside: height above the (smoothed) ground the user asked for, in globe radii.
    fly_alt: f64,
    /// Inside: ground level the camera glides on (a smoothed version of the land under it).
    floor_smooth: f64,
    last_glide: Instant,
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    depth_view: wgpu::TextureView,
    camera: Camera,
    /// true = hex grid ground (G switches), false = plain flat sphere. Pixels work on both.
    hex: bool,
    paint: PaintStore,
    /// Mouse position in window pixels.
    cursor: Option<(f64, f64)>,
    /// The hex under the mouse, at the current display resolution.
    hover: Option<CellIndex>,
    disp_res: u8,
    /// Hex vertices are stored relative to this point (under the camera), see camera.rs.
    origin: DVec3,
    /// Brush size index ([ and ]), see BRUSH_RINGS.
    brush: usize,
    tool: Tool,
    /// Current paint color (palette index, keys 1-9).
    color: Paint,
    shift: bool,
    /// Where a line / circle drag started.
    shape_start: Option<CellIndex>,
    /// Which layer the tools work on (P switches).
    layer: Layer,
    pixels: PixelStore,
    /// Pixel size: 2^-pixel_k of a hex wide ([ and ] in the pixel layer).
    pixel_k: u8,
    /// Index into PIXEL_RADII.
    pixel_rad: usize,
    pixel_hover: Option<PixelId>,
    pixel_start: Option<PixelId>,
    last_pixel: Option<PixelId>,
    /// Which layer each undoable stroke belongs to, oldest first (Ctrl+Z pops the last).
    undo_order: Vec<Layer>,

    last_paint_cell: Option<CellIndex>,
    /// Fly keys held right now: W, A, S, D.
    keys: [bool; 4],
    last_frame: Instant,
}

/// Rings around the middle hex for brush sizes 1..5 (0 rings = just the one hex).
const BRUSH_RINGS: [u32; 5] = [0, 1, 2, 4, 8];
/// Most hexes one bucket fill may paint.
const FILL_CAP: usize = 30_000;

fn make_depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some("depth"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: DEPTH_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

impl Gpu {
    fn new(window: Arc<Window>) -> Gpu {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance.create_surface(window.clone()).unwrap();

        // Try every GPU the system offers, discrete ones first, until one works.
        let mut adapters = pollster::block_on(instance.enumerate_adapters(wgpu::Backends::all()));
        adapters.sort_by_key(|a| match a.get_info().device_type {
            wgpu::DeviceType::DiscreteGpu => 0,
            wgpu::DeviceType::IntegratedGpu => 1,
            _ => 2,
        });
        let mut found = None;
        for adapter in adapters {
            if !adapter.is_surface_supported(&surface) {
                continue;
            }
            if let Ok((device, queue)) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
            {
                let info = adapter.get_info();
                println!("Using GPU: {} ({:?})", info.name, info.backend);
                found = Some((adapter, device, queue));
                break;
            }
        }
        let (adapter, device, queue) = found.expect("no GPU could be opened");

        let size = window.inner_size();
        let config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .expect("surface not supported by this GPU");
        surface.configure(&device, &config);

        // Room for the most cells we'll ever draw. Refilled whenever the camera moves.
        let vertex_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globe vertices"),
            size: (MAX_VERTS * std::mem::size_of::<Vertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // One 4x4 matrix the shader reads, rewritten every frame.
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("camera matrix"),
            size: 80, // 4x4 matrix (64 bytes) + light (16 bytes)
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: uniform_buf.as_entire_binding() }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("globe shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bind_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("globe pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<Vertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(config.format.into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Greater), // reversed depth: nearer = larger
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        // The ground: per-pixel shaded (ground.wgsl), its own vertex format and uniforms.
        let inst_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ground patch offsets"),
            size: (MAX_PATCHES * 16) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ground_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ground uniforms"),
            size: 144, // mvp + sun + eye + origin + sky + refm
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ground_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: ground_uniform.as_entire_binding() }],
        });
        let ground_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ground shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("ground.wgsl").into()),
        });
        let ground_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ground pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &ground_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[
                    Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<GroundVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Snorm8x4, 2 => Float32x2],
                    }),
                    Some(wgpu::VertexBufferLayout {
                        array_stride: 16,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![3 => Float32x4],
                    }),
                ],
            },
            fragment: Some(wgpu::FragmentState {
                module: &ground_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(config.format.into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });

        // Flat 2D shapes on top (scale bar etc.): no camera matrix, no depth test.
        let overlay_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("overlay shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("overlay.wgsl").into()),
        });
        let overlay_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            immediate_size: 0,
        });
        let overlay_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("overlay pipeline"),
            layout: Some(&overlay_layout),
            vertex: wgpu::VertexState {
                module: &overlay_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<OverlayVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x3],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &overlay_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(config.format.into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Always),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        // Pixel layer: quads on the globe, blended over the hexes, hidden behind the horizon by the depth test.
        let pixel_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pixel shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("pixel.wgsl").into()),
        });
        let make_pixel_pipeline = |compare: wgpu::CompareFunction| device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("pixel pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &pixel_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<PixelVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x4, 2 => Float32],
                })],
            },
            fragment: Some(wgpu::FragmentState {
                module: &pixel_shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(compare),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        let pixel_pipeline = make_pixel_pipeline(wgpu::CompareFunction::Always);
        let pixel_pipeline_depth = make_pixel_pipeline(wgpu::CompareFunction::GreaterEqual);
        let pixel_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pixel vertices"),
            size: (PIXEL_MAX_VERTS * std::mem::size_of::<PixelVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let overlay_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("overlay vertices"),
            size: (OVERLAY_MAX_VERTS * std::mem::size_of::<OverlayVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let depth_view = make_depth_view(&device, config.width, config.height);
        let camera = Camera { yaw: 0.0, pitch: 0.3, distance: 3.2, floor: 1.0, fly: None };

        let mut gpu = Gpu {
            window,
            surface,
            device,
            queue,
            config,
            pipeline,
            overlay_pipeline,
            pixel_pipeline,
            pixel_pipeline_depth,
            terrain: TileStore::new(Terrain::new(2026)),
            max_freq: 1.0,
            pixel_buf,
            pixel_count: 0,
            overlay_buf,
            overlay_count: 0,
            vertex_buf,
            vertex_count: 0,
            ground_pipeline,
            inst_buf,
            patch_cache: HashMap::new(),
            patch_tick: 0,
            draw_list: Vec::new(),
            ground_light: Lighting::new(DVec3::Z, DVec3::X, DVec3::X * 2.0, None),
            ground_uniform,
            ground_bind,
            dirty: true,
            pixels_dirty: false,
            settle: 0,
            hex_count: 0,
            fly_alt: INSIDE_ENTER_ALT,
            floor_smooth: 1.0,
            last_glide: Instant::now(),
            uniform_buf,
            bind_group,
            depth_view,
            camera,
            hex: false,
            paint: PaintStore::default(),
            cursor: None,
            hover: None,
            disp_res: 0,
            origin: DVec3::ZERO,
            brush: 1,
            tool: Tool::Pencil,
            color: hexpaint::RED,
            shift: false,
            shape_start: None,
            layer: Layer::Pixel,
            pixels: PixelStore::default(),
            pixel_k: 4,
            pixel_rad: 1,
            pixel_hover: None,
            pixel_start: None,
            last_pixel: None,
            undo_order: Vec::new(),
            last_paint_cell: None,
            keys: [false; 4],
            last_frame: Instant::now(),
        };
        gpu.rebuild();
        gpu
    }

    /// Inside = the free-flying scene over the land; otherwise the globe seen from outside.
    fn inside(&self) -> bool {
        self.camera.fly.is_some()
    }

    /// How the terrain is read for the current scene: inside gets all the detail the screen can show,
    /// outside is capped so the globe stays a clean map.
    fn terrain_ref(&self) -> TerrainRef<'_> {
        let t = TerrainRef::new(&self.terrain, &self.camera, self.config.height as f64);
        if self.inside() { t } else { t.capped(OUTSIDE_CAP) }
    }

    /// (detail limit, ground radius) under the camera for the current scene.
    fn ground_under_camera(&self) -> (f64, f64) {
        let f = self.terrain_ref().max_freq;
        (f, global_map_maker::terrain::HeightSource::ground_radius(&self.terrain, self.camera.eye64().normalize(), f))
    }

    /// Ground level for the flying camera: the land averaged over a disk about as wide as the camera is high
    /// (center + 8 points around), so a ridge raises the camera gradually, well before it is overhead.
    fn smoothed_ground(&self, max_freq: f64) -> f64 {
        use global_map_maker::terrain::HeightSource;
        let Some(f) = self.camera.fly else { return self.camera.floor };
        let (n, e, _) = f.frame();
        let rad = 1.5 * self.fly_alt;
        let mut sum = self.terrain.ground_radius(f.dir.normalize(), max_freq);
        for k in 0..8 {
            let a = k as f64 * std::f64::consts::FRAC_PI_4;
            let d = (f.dir.normalize() + (n * a.cos() + e * a.sin()) * rad).normalize();
            sum += self.terrain.ground_radius(d, max_freq);
        }
        sum / 9.0
    }

    /// Zoomed in far enough on the globe: drop into the scene over the spot under the camera.
    fn enter_inside(&mut self) {
        let dir = self.camera.eye64().normalize();
        self.camera.fly = Some(Fly { dir, heading: 0.0, tilt: INSIDE_ENTER_TILT });
        self.fly_alt = INSIDE_ENTER_ALT;
        self.floor_smooth = self.camera.floor;
        self.last_glide = Instant::now();
        self.camera.distance = self.camera.floor + INSIDE_ENTER_ALT;
        self.shape_start = None;
        self.keys = [false; 4];
        self.dirty = true;
    }

    /// Back out to space, over whatever you were flying above.
    fn leave_inside(&mut self) {
        if let Some(f) = self.camera.fly.take() {
            self.camera.orbit_over(f.dir);
            self.camera.distance = self.camera.floor + OUTSIDE_EXIT_H;
            self.keys = [false; 4];
            self.dirty = true;
        }
    }

    fn moving(&self) -> bool {
        self.inside() && self.keys.iter().any(|k| *k)
    }

    /// Move the flying camera for the time since the last frame (W A S D, shift = faster).
    fn step(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f64().min(0.1);
        self.last_frame = now;
        if !self.moving() {
            return;
        }
        let alt = self.fly_alt.max(INSIDE_MIN_ALT);
        let speed = alt * FLY_SPEED * if self.shift { FLY_BOOST } else { 1.0 };
        let [w, a, s, d] = self.keys.map(|k| k as i32 as f64);
        let (ahead, right) = (w - s, d - a);
        if let Some(f) = self.camera.fly.as_mut() {
            let (n, e, _) = f.frame();
            let (sh, ch) = f.heading.sin_cos();
            let mv = (n * ch + e * sh) * ahead + (e * ch - n * sh) * right;
            let len = mv.length();
            if len > 0.0 {
                f.dir = (f.dir.normalize() + mv / len * speed * dt).normalize();
                self.dirty = true;
            }
        }
    }

    /// Work out what the camera needs (ground + pixels) and upload the triangles.
    fn rebuild(&mut self) {
        let tiles_before = self.terrain.generated();
        let (w, h) = (self.config.width, self.config.height);
        // Height above the ground is what zoom and detail follow: keep it while the ground under the
        // camera rises and falls.
        let above = self.camera.height();
        let mut gliding = false;
        if self.inside() && !self.hex {
            // Flying: ride on a smoothed ground level (like a ghost, not a walker) but never closer to the
            // real ground than CLEARANCE x height.
            let (f, exact) = self.ground_under_camera();
            let target = self.smoothed_ground(f);
            let now = Instant::now();
            let dt = now.duration_since(self.last_glide).as_secs_f64().min(0.25);
            self.last_glide = now;
            self.floor_smooth += (target - self.floor_smooth) * (1.0 - (-dt / GLIDE_TAU).exp());
            self.camera.floor = self.floor_smooth;
            self.camera.distance = (self.floor_smooth + self.fly_alt).max(exact + CLEARANCE * self.fly_alt);
            self.max_freq = self.ground_under_camera().0;
            gliding = (target - self.floor_smooth).abs() > 0.01 * self.fly_alt;
        } else {
            if self.hex {
                self.camera.floor = 1.0;
            } else {
                let (f, floor) = self.ground_under_camera();
                self.max_freq = f;
                self.camera.floor = floor;
            }
            self.camera.distance = self.camera.floor + above;
            if !self.hex {
                // The floor moved, so the detail limit for this height changes a little too.
                self.max_freq = self.ground_under_camera().0;
            }
        }
        let res = choose_res(&self.camera, w, h);
        self.disp_res = res;
        // Vertices are stored relative to the point on the ground under the camera (keeps f32 precise).
        self.origin = self.camera.eye64().normalize() * if self.hex { 1.0 } else { self.camera.floor };
        let mut vertices = Vec::new();
        let mut ground: Option<(Vec<PatchKey>, Vec<(PatchKey, global_map_maker::ground::PatchMesh)>)> = None;
        let mut light_now = None;
        if self.hex {
            let r = Resolution::try_from(res).unwrap();
            let cells = visible_hexes(res, &self.camera, w, h);
            self.hex_count = cells.len();
            self.hover = self.cell_under_cursor();
            let highlighted = self.footprint();
            let coverage = self.paint.coverage(r);
            vertices = build_hex_cells(&cells, r, &self.paint, &coverage, &highlighted, self.origin);
        } else {
            self.hover = None;
            let tref = self.terrain_ref();
            let view = PixelView::from_camera(&self.camera, w, h, Some(tref));
            let light = self.camera.light_dir();
            let light = DVec3::new(light.x as f64, light.y as f64, light.z as f64).normalize();
            let haze = self.inside().then_some(SKY);
            let lighting = Lighting::new(light, self.origin, self.camera.eye64(), haze);
            let patches = visible_patches(&view);
            let c = crowd(patches.len());
            let keys: Vec<PatchKey> = patches.iter().map(|id| patch_key(*id, &view, c)).collect();
            // Only patches not seen before are built (on all cores); the rest are reused as they are.
            let missing: Vec<PatchKey> = keys.iter().copied().filter(|k| !self.patch_cache.contains_key(k)).collect();
            let meshes = build_patches(&missing, &view);
            ground = Some((keys, missing.into_iter().zip(meshes).collect()));
            light_now = Some(lighting);
        }
        if let Some(l) = light_now {
            self.ground_light = l;
        }
        self.draw_list.clear();
        if let Some((keys, built)) = ground {
            self.patch_tick += 1;
            for (k, mesh) in built {
                let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("ground patch"),
                    size: (mesh.verts.len() * std::mem::size_of::<GroundVertex>()) as u64,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.queue.write_buffer(&buf, 0, bytemuck::cast_slice(&mesh.verts));
                self.patch_cache.insert(k, GpuPatch { buf, count: mesh.verts.len() as u32, center: mesh.center, used: 0 });
            }
            let mut inst: Vec<[f32; 4]> = Vec::with_capacity(keys.len());
            for k in &keys {
                if let Some(p) = self.patch_cache.get_mut(k) {
                    p.used = self.patch_tick;
                    let o = p.center - self.origin;
                    inst.push([o.x as f32, o.y as f32, o.z as f32, 0.0]);
                    self.draw_list.push((p.buf.clone(), p.count));
                }
            }
            inst.truncate(MAX_PATCHES);
            self.draw_list.truncate(MAX_PATCHES);
            self.queue.write_buffer(&self.inst_buf, 0, bytemuck::cast_slice(&inst));
            // Keep the meshes of recently seen patches (flying back and forth is free), drop the oldest beyond that.
            if self.patch_cache.len() > PATCH_CACHE_MAX {
                let mut ages: Vec<(u64, PatchKey)> = self.patch_cache.iter().map(|(k, p)| (p.used, *k)).collect();
                ages.sort_unstable_by_key(|a| a.0);
                for (_, k) in ages.into_iter().take(self.patch_cache.len() - PATCH_CACHE_MAX * 2 / 3) {
                    self.patch_cache.remove(&k);
                }
            }
        }

        vertices.truncate(MAX_VERTS / 3 * 3);
        self.queue.write_buffer(&self.vertex_buf, 0, bytemuck::cast_slice(&vertices));
        self.vertex_count = vertices.len() as u32;

        self.rebuild_pixels();
        self.dirty = gliding; // keep going until the camera has settled on the ground level
        // Tiles made while building change what the walk should have picked: look again (a few times at most).
        if self.terrain.generated() != tiles_before && self.settle < 3 {
            self.settle += 1;
            self.dirty = true;
        } else if !self.dirty {
            self.settle = 0;
        }
        self.update_title();
    }

    /// The pixel layer (paint + highlight) on top of the ground; the ground mesh stays as it is.
    fn rebuild_pixels(&mut self) {
        let (w, h) = (self.config.width, self.config.height);
        self.pixel_hover = self.pixel_under_cursor();
        let eye = self.camera.eye64();
        let tref = (!self.hex).then(|| self.terrain_ref());
        let view = PixelView::from_camera(&self.camera, w, h, tref);
        let surface = Surface { terrain: tref, lift_px: if tref.is_some() { LIFT_PX } else { 0.0 }, focal_px: view.focal_px };
        let mut quads = visible_pixels(&self.pixels, &view);
        if tref.is_some() {
            quads = split_for_relief(quads, &view);
        }
        let mut pv = build_vertices(&quads, self.origin, eye, &surface);
        if self.layer == Layer::Pixel {
            let fp: Vec<PixelId> = self.pixel_footprint().into_iter().collect();
            pv.extend(highlight_vertices(&fp, self.origin, eye, &surface, &view));
        }
        pv.truncate(PIXEL_MAX_VERTS / 3 * 3);
        self.queue.write_buffer(&self.pixel_buf, 0, bytemuck::cast_slice(&pv));
        self.pixel_count = pv.len() as u32;
        self.pixels_dirty = false;
    }

    fn update_title(&mut self) {
        let res = self.disp_res;
        let hexes = self.hex_count;
        let what = if self.tool == Tool::Eraser { 0 } else { self.color };
        let title = match self.layer {
            Layer::Hex => format!(
                "Global Map Maker | HEX ground, hex layer | {} | {} | brush {} | hex level {} | {} hexes | {} painted | right-drag = use tool, P = pixel layer, G = flat sphere",
                self.tool.name(),
                hexpaint::name(what),
                BRUSH_RINGS[self.brush] * 2 + 1,
                res,
                hexes,
                self.paint.painted_cells()
            ),
            Layer::Pixel => format!(
                "Global Map Maker | {} | PIXEL layer | {}{} | {} | pixels 1/{} hex | brush {} | {} pixels painted | [ ] pixel size, - = brush, G = {}",
                if self.hex { "HEX ground" } else if self.inside() { "INSIDE (WASD fly, drag look, scroll height, Esc = space)" } else { "OUTSIDE (zoom in to enter)" },
                self.tool.name(),
                if matches!(self.tool, Tool::Circle | Tool::Fill) { " (hex layer only)" } else { "" },
                hexpaint::name(what),
                1u32 << self.pixel_k,
                PIXEL_RADII[self.pixel_rad] * 2 + 1,
                self.pixels.painted_pixels(),
                if self.hex { "flat sphere, P = hex layer" } else { "hex ground" }
            ),
        };
        self.window.set_title(&title);
    }

    fn cell_under_cursor(&self) -> Option<CellIndex> {
        let (x, y) = self.cursor?;
        pick_cell(self.disp_res, &self.camera, self.config.width, self.config.height, x, y)
    }

    /// The paint value the current tool lays down.
    fn value(&self) -> Paint {
        if self.tool == Tool::Eraser {
            0
        } else {
            self.color
        }
    }

    /// The hexes the tool would paint right now (also what lights up under the mouse).
    fn footprint(&self) -> HashSet<CellIndex> {
        let rings = BRUSH_RINGS[self.brush];
        if self.layer == Layer::Pixel {
            return HashSet::new();
        }
        let Some(h) = self.hover else { return HashSet::new() };
        match (self.tool, self.shape_start) {
            (Tool::Pencil | Tool::Eraser, _) => tools::expand([h], rings),
            (Tool::Line, Some(s)) => tools::expand(tools::line_cells(s, h), rings),
            (Tool::Circle, Some(s)) => {
                let r = tools::distance(s, h).unwrap_or(0);
                // Thickness from the brush is added to the radius, not hex by hex (much cheaper).
                if self.shift {
                    let outer: HashSet<CellIndex> = tools::disk_cells(s, r + rings).into_iter().collect();
                    if r > rings {
                        let inner: HashSet<CellIndex> = tools::disk_cells(s, r - rings - 1).into_iter().collect();
                        outer.difference(&inner).copied().collect()
                    } else {
                        outer
                    }
                } else {
                    tools::disk_cells(s, r + rings).into_iter().collect()
                }
            }
            (Tool::Line | Tool::Circle, None) => tools::expand([h], rings),
            _ => [h].into_iter().collect(),
        }
    }

    fn paint_set(&mut self, cells: impl IntoIterator<Item = CellIndex>, value: Paint) {
        for c in cells {
            self.paint.paint(c, value);
        }
        self.dirty = true;
        self.window.request_redraw();
    }

    /// Pencil / eraser: paint under the cursor, plus the hexes between here and the last spot.
    fn paint_at_cursor(&mut self) {
        let Some(center) = self.cell_under_cursor() else { return };
        let path = match self.last_paint_cell {
            Some(prev) if prev != center => tools::line_cells(prev, center),
            _ => vec![center],
        };
        let targets = tools::expand(path, BRUSH_RINGS[self.brush]);
        let value = self.value();
        self.paint_set(targets, value);
        self.last_paint_cell = Some(center);
    }

    /// Right mouse button went down.
    fn tool_press(&mut self) {
        if self.layer == Layer::Pixel {
            return self.pixel_press();
        }
        self.hover = self.cell_under_cursor();
        let Some(h) = self.hover else { return };
        match self.tool {
            Tool::Pencil | Tool::Eraser => {
                self.paint.begin_stroke();
                self.last_paint_cell = None;
                self.paint_at_cursor();
            }
            Tool::Line | Tool::Circle => self.shape_start = Some(h),
            Tool::Fill => {
                let (w, hh) = (self.config.width, self.config.height);
                let visible: HashSet<CellIndex> = visible_hexes(self.disp_res, &self.camera, w, hh).into_iter().collect();
                let cells = tools::flood_fill(h, &visible, &self.paint, FILL_CAP);
                let value = self.color;
                self.paint.begin_stroke();
                self.paint_set(cells, value);
                self.end_hex_stroke();
            }
            Tool::Eyedropper => {
                let picked = self.paint.effective(h);
                if picked != 0 {
                    self.color = picked;
                }
                self.tool = Tool::Pencil;
                self.dirty = true;
                self.window.request_redraw();
            }
        }
    }

    /// Right mouse button came up.
    fn tool_release(&mut self) {
        if self.layer == Layer::Pixel {
            return self.pixel_release();
        }
        match self.tool {
            Tool::Pencil | Tool::Eraser => {
                self.end_hex_stroke();
                self.last_paint_cell = None;
            }
            Tool::Line | Tool::Circle => {
                if self.shape_start.is_some() {
                    let cells = self.footprint();
                    let value = self.value();
                    self.paint.begin_stroke();
                    self.paint_set(cells, value);
                    self.end_hex_stroke();
                }
                self.shape_start = None;
                self.dirty = true;
                self.window.request_redraw();
            }
            _ => {}
        }
    }

    fn pixel_level(&self) -> u8 {
        pixelview::pixel_level(cell_width(self.disp_res), self.pixel_k)
    }

    fn pixel_under_cursor(&self) -> Option<PixelId> {
        let (x, y) = self.cursor?;
        let tref = (!self.hex).then(|| self.terrain_ref());
        pick_pixel(&self.camera, self.config.width, self.config.height, x, y, self.pixel_level(), tref)
    }

    /// The pixels the tool would paint right now (also what lights up under the mouse).
    fn pixel_footprint(&self) -> HashSet<PixelId> {
        let Some(h) = self.pixel_hover else { return HashSet::new() };
        let rad = PIXEL_RADII[self.pixel_rad];
        let centers = match (self.tool, self.pixel_start) {
            (Tool::Line, Some(s)) => pixelview::line(s, h),
            _ => vec![h],
        };
        match self.tool {
            Tool::Pencil | Tool::Eraser | Tool::Line => centers.into_iter().flat_map(|c| disk(c, rad)).collect(),
            _ => [h].into_iter().collect(),
        }
    }

    fn pixel_paint_set(&mut self, ids: impl IntoIterator<Item = PixelId>, value: Paint) {
        for id in ids {
            self.pixels.paint(id, value);
        }
        self.pixels_dirty = true;
        self.window.request_redraw();
    }

    fn pixel_paint_at_cursor(&mut self) {
        let Some(c) = self.pixel_under_cursor() else { return };
        let path = match self.last_pixel {
            Some(p) if p != c => pixelview::line(p, c),
            _ => vec![c],
        };
        let rad = PIXEL_RADII[self.pixel_rad];
        let targets: HashSet<PixelId> = path.into_iter().flat_map(|p| disk(p, rad)).collect();
        let value = self.value();
        self.pixel_paint_set(targets, value);
        self.last_pixel = Some(c);
    }

    fn pixel_press(&mut self) {
        self.pixel_hover = self.pixel_under_cursor();
        let Some(h) = self.pixel_hover else { return };
        match self.tool {
            Tool::Pencil | Tool::Eraser => {
                self.pixels.begin_stroke();
                self.last_pixel = None;
                self.pixel_paint_at_cursor();
            }
            Tool::Line => self.pixel_start = Some(h),
            Tool::Eyedropper => {
                let picked = self.pixels.effective(h);
                if picked != 0 {
                    self.color = picked;
                }
                self.tool = Tool::Pencil;
                self.pixels_dirty = true;
                self.window.request_redraw();
            }
            Tool::Circle | Tool::Fill => {} // hex layer only for now
        }
    }

    fn pixel_release(&mut self) {
        match self.tool {
            Tool::Pencil | Tool::Eraser => {
                self.end_pixel_stroke();
                self.last_pixel = None;
            }
            Tool::Line => {
                if self.pixel_start.is_some() {
                    let ids = self.pixel_footprint();
                    let value = self.value();
                    self.pixels.begin_stroke();
                    self.pixel_paint_set(ids, value);
                    self.end_pixel_stroke();
                }
                self.pixel_start = None;
                self.pixels_dirty = true;
                self.window.request_redraw();
            }
            _ => {}
        }
    }

    fn end_hex_stroke(&mut self) {
        let before = self.paint.undo_depth();
        self.paint.end_stroke();
        if self.paint.undo_depth() > before {
            self.undo_order.push(Layer::Hex);
        }
    }

    fn end_pixel_stroke(&mut self) {
        let before = self.pixels.undo_depth();
        self.pixels.end_stroke();
        if self.pixels.undo_depth() > before {
            self.undo_order.push(Layer::Pixel);
        }
    }

    /// Ctrl+Z: undo the latest stroke on whichever layer it was.
    fn undo(&mut self) {
        match self.undo_order.pop() {
            Some(Layer::Hex) => {
                self.paint.undo();
                self.dirty = true;
            }
            Some(Layer::Pixel) => {
                self.pixels.undo();
                self.pixels_dirty = true;
            }
            None => {}
        }
    }

    fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return; // minimized
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.depth_view = make_depth_view(&self.device, width, height);
        self.dirty = true;
    }

    fn draw(&mut self) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            _ => return, // skip this frame (resizing, occluded, ...)
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        if self.dirty {
            self.rebuild();
        } else if self.pixels_dirty {
            self.rebuild_pixels();
            self.update_title();
        }

        // Rebuild the camera matrix from the current angles and send it to the GPU.
        let mut uniforms = [0f32; 20];
        let m = self.camera.matrix_rel(self.config.width as f64 / self.config.height as f64, self.origin);
        let light = self.camera.light_dir();
        uniforms[..16].copy_from_slice(&m.to_cols_array());
        uniforms[16..].copy_from_slice(&[light.x, light.y, light.z, 0.0]); // colors are shaded when the mesh is built
        self.queue.write_buffer(&self.uniform_buf, 0, bytemuck::cast_slice(&uniforms));

        {
            let l = &self.ground_light;
            let f = |v: DVec3, w: f32| [v.x as f32, v.y as f32, v.z as f32, w];
            let sky = l.haze.unwrap_or([0.0; 3]);
            let mut g = [0f32; 36];
            g[..16].copy_from_slice(&m.to_cols_array());
            g[16..20].copy_from_slice(&f(l.sun, 0.0));
            g[20..24].copy_from_slice(&f(l.eye_rel, l.horizon as f32));
            g[24..28].copy_from_slice(&f(l.origin, 0.0));
            g[28..32].copy_from_slice(&[sky[0], sky[1], sky[2], if l.haze.is_some() { 1.0 } else { 0.0 }]);
            let r = global_map_maker::terrain::EARTH_RADIUS_M;
            let wrap = |x: f64| (x * r).rem_euclid(4096.0) as f32;
            g[32..].copy_from_slice(&[wrap(l.origin.x), wrap(l.origin.y), wrap(l.origin.z), r as f32]);
            self.queue.write_buffer(&self.ground_uniform, 0, bytemuck::cast_slice(&g));
        }

        // Scale bar + zoom-depth ladder.
        self.overlay_count = 0;
        {
            let mpp = meters_per_pixel(self.camera.height(), Camera::focal_px(self.config.height as f64));
            let mut hud = build_overlay(self.config.width, self.config.height, mpp, self.disp_res, MAX_PAINT_RES);
            hud.truncate(OVERLAY_MAX_VERTS);
            self.queue.write_buffer(&self.overlay_buf, 0, bytemuck::cast_slice(&hud));
            self.overlay_count = hud.len() as u32;
        }

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(if self.inside() {
                            wgpu::Color { r: SKY[0] as f64, g: SKY[1] as f64, b: SKY[2] as f64, a: 1.0 }
                        } else {
                            wgpu::Color { r: 0.02, g: 0.03, b: 0.08, a: 1.0 }
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0), // reversed depth: far = 0
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            if self.hex {
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
                pass.draw(0..self.vertex_count, 0..1);
            } else {
                pass.set_pipeline(&self.ground_pipeline);
                pass.set_bind_group(0, &self.ground_bind, &[]);
                pass.set_vertex_buffer(1, self.inst_buf.slice(..));
                for (k, (buf, count)) in self.draw_list.iter().enumerate() {
                    pass.set_vertex_buffer(0, buf.slice(..));
                    pass.draw(0..*count, k as u32..k as u32 + 1);
                }
            }
            if self.pixel_count > 0 {
                pass.set_pipeline(if self.hex { &self.pixel_pipeline } else { &self.pixel_pipeline_depth });
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.set_vertex_buffer(0, self.pixel_buf.slice(..));
                pass.draw(0..self.pixel_count, 0..1);
            }
            if self.overlay_count > 0 {
                pass.set_pipeline(&self.overlay_pipeline);
                pass.set_vertex_buffer(0, self.overlay_buf.slice(..));
                pass.draw(0..self.overlay_count, 0..1);
            }
        }
        self.queue.submit(Some(encoder.finish()));
        self.window.pre_present_notify();
        self.queue.present(frame);
    }
}

#[derive(Default)]
struct App {
    gpu: Option<Gpu>,
    dragging: bool,
    painting: bool,
    ctrl: bool,
    last_cursor: Option<PhysicalPosition<f64>>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let attrs = Window::default_attributes().with_title("Global Map Maker");
        let window = Arc::new(event_loop.create_window(attrs).unwrap());
        self.gpu = Some(Gpu::new(window));
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(gpu) = self.gpu.as_mut() else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                gpu.resize(size.width, size.height);
                gpu.window.request_redraw();
            }
            WindowEvent::ModifiersChanged(m) => {
                self.ctrl = m.state().control_key();
                gpu.shift = m.state().shift_key();
                gpu.dirty = true; // shift changes the circle preview
                gpu.window.request_redraw();
            }
            WindowEvent::MouseInput { state, button: MouseButton::Right, .. } => {
                if state == ElementState::Pressed {
                    self.painting = true;
                    gpu.tool_press();
                } else if self.painting {
                    self.painting = false;
                    gpu.tool_release();
                }
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                self.dragging = state == ElementState::Pressed;
                if !self.dragging {
                    self.last_cursor = None;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                gpu.cursor = Some((position.x, position.y));
                {
                    let moved = match gpu.layer {
                        Layer::Hex => gpu.cell_under_cursor() != gpu.hover,
                        Layer::Pixel => gpu.pixel_under_cursor() != gpu.pixel_hover,
                    };
                    if self.painting && matches!(gpu.tool, Tool::Pencil | Tool::Eraser) {
                        match gpu.layer {
                            Layer::Hex => gpu.paint_at_cursor(),
                            Layer::Pixel => gpu.pixel_paint_at_cursor(),
                        }
                    } else if moved {
                        // The highlight moved: only the pixel layer needs rebuilding (the hex ground shows it, so it needs all).
                        match gpu.layer {
                            Layer::Hex => gpu.dirty = true,
                            Layer::Pixel => gpu.pixels_dirty = true,
                        }
                        gpu.window.request_redraw();
                    }
                }
                if self.dragging {
                    if let (Some(last), true) = (self.last_cursor, gpu.inside()) {
                        // Inside: dragging turns your head.
                        if let Some(f) = gpu.camera.fly.as_mut() {
                            f.heading += (position.x - last.x) * 0.004;
                            f.tilt = (f.tilt - (position.y - last.y) * 0.004).clamp(-1.5, 1.2);
                        }
                        gpu.dirty = true;
                        gpu.window.request_redraw();
                    } else if let Some(last) = self.last_cursor {
                        // Slower spin when zoomed in, so the surface doesn't race past.
                        let speed = 0.0025 * gpu.camera.height().clamp(1e-6, 3.0);
                        gpu.camera.yaw += (position.x - last.x) * speed;
                        let limit = 89f64.to_radians();
                        gpu.camera.pitch =
                            (gpu.camera.pitch + (position.y - last.y) * speed).clamp(-limit, limit);
                        gpu.dirty = true;
                        gpu.window.request_redraw();
                    }
                    self.last_cursor = Some(position);
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
                };
                // Zoom by a percentage of the distance to the surface, so it feels even at any height.
                let height = gpu.camera.height() * (-(scroll as f64) * 0.2).exp();
                if gpu.inside() {
                    let alt = gpu.fly_alt * (-(scroll as f64) * 0.2).exp();
                    if alt > INSIDE_MAX_ALT && scroll < 0.0 {
                        gpu.leave_inside(); // zoomed out past the top of the scene: back to space
                    } else {
                        gpu.fly_alt = alt.clamp(INSIDE_MIN_ALT, INSIDE_MAX_ALT);
                        gpu.camera.distance = gpu.camera.floor + gpu.fly_alt;
                    }
                } else if !gpu.hex && height < OUTSIDE_MIN_H && scroll > 0.0 {
                    gpu.enter_inside(); // zoomed in far enough: down into the land
                } else {
                    let min = if gpu.hex { 5e-6 } else { OUTSIDE_MIN_H };
                    gpu.camera.distance = gpu.camera.floor + height.clamp(min, 10.0);
                }
                gpu.dirty = true;
                gpu.window.request_redraw();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                // Fly keys: remember which are held; the redraw loop moves the camera while they are.
                if let Key::Character(c) = &event.logical_key {
                    let slot = match c.as_str() {
                        "w" | "W" => Some(0),
                        "a" | "A" => Some(1),
                        "s" | "S" => Some(2),
                        "d" | "D" => Some(3),
                        _ => None,
                    };
                    if let (Some(i), true) = (slot, gpu.inside() && !self.ctrl) {
                        let down = event.state == ElementState::Pressed;
                        if down && !gpu.moving() {
                            gpu.last_frame = Instant::now();
                        }
                        gpu.keys[i] = down;
                        gpu.window.request_redraw();
                    } else if slot.is_some() && event.state == ElementState::Released {
                        gpu.keys[slot.unwrap()] = false;
                    }
                }
                if event.state == ElementState::Pressed && !event.repeat && event.logical_key == Key::Named(NamedKey::Escape) {
                    gpu.leave_inside();
                    gpu.window.request_redraw();
                }
                if event.state == ElementState::Pressed && !event.repeat {
                    if let Key::Character(c) = &event.logical_key {
                        match c.as_str() {
                            // G switches the ground: hex grid <-> plain flat sphere (pixels only there).
                            "g" | "G" => {
                                gpu.hex = !gpu.hex;
                                if gpu.hex {
                                    gpu.leave_inside();
                                }
                                if !gpu.hex {
                                    gpu.layer = Layer::Pixel;
                                    gpu.shape_start = None;
                                }
                            }
                            "z" | "Z" if self.ctrl && !self.painting => {
                                gpu.undo();
                            }
                            "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" if !self.ctrl => {
                                gpu.color = c.as_str().parse().unwrap();
                                if matches!(gpu.tool, Tool::Eraser | Tool::Eyedropper) {
                                    gpu.tool = Tool::Pencil;
                                }
                            }
                            "0" if !self.ctrl => gpu.tool = Tool::Eraser,
                            "b" | "B" if !self.ctrl => gpu.tool = Tool::Pencil,
                            "e" | "E" if !self.ctrl => gpu.tool = Tool::Eraser,
                            "l" | "L" if !self.ctrl => gpu.tool = Tool::Line,
                            "c" | "C" if !self.ctrl => gpu.tool = Tool::Circle,
                            "f" | "F" if !self.ctrl => gpu.tool = Tool::Fill,
                            "i" | "I" if !self.ctrl => gpu.tool = Tool::Eyedropper,
                            "p" | "P" if !self.ctrl && gpu.hex => {
                                gpu.layer = if gpu.layer == Layer::Hex { Layer::Pixel } else { Layer::Hex };
                                gpu.shape_start = None;
                                gpu.pixel_start = None;
                            }
                            "[" if gpu.layer == Layer::Pixel => gpu.pixel_k = (gpu.pixel_k - 1).max(1),
                            "]" if gpu.layer == Layer::Pixel => gpu.pixel_k = (gpu.pixel_k + 1).min(10),
                            "-" if gpu.layer == Layer::Pixel => gpu.pixel_rad = gpu.pixel_rad.saturating_sub(1),
                            "=" | "+" if gpu.layer == Layer::Pixel => {
                                gpu.pixel_rad = (gpu.pixel_rad + 1).min(PIXEL_RADII.len() - 1)
                            }
                            "[" => gpu.brush = gpu.brush.saturating_sub(1),
                            "]" => gpu.brush = (gpu.brush + 1).min(BRUSH_RINGS.len() - 1),
                            _ => return,
                        }
                        gpu.dirty = true;
                        gpu.window.request_redraw();
                    }
                }
            }
            WindowEvent::RedrawRequested => {
                gpu.step();
                gpu.draw();
                if gpu.moving() || gpu.dirty {
                    gpu.window.request_redraw(); // keep flying while keys are held; settle after new tiles
                }
            }
            _ => {}
        }
    }
}

fn main() {
    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Wait);
    event_loop.run_app(&mut App::default()).unwrap();
}
