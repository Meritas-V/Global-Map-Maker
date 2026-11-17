//! Window + GPU + drawing the globe. Mouse drag rotates, scroll zooms.

use global_map_maker::camera::{Camera, Fly};
use global_map_maker::hexmesh::build_hex_cells;
use global_map_maker::hexpaint::{self, Paint, PaintStore};
use global_map_maker::ribbon::Ribbon;
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
use global_map_maker::ground::{build_patches, patch_key, visible_patches, GroundVertex, Lighting, PatchKey, PatchMesh, MAX_PATCHES, SKY};
use global_map_maker::pixelpaint::PixelId as PatchId;
use global_map_maker::stream::{plan_frame, Streamer, BASE_LEVEL};
use std::sync::Arc;
use std::collections::HashMap;
use global_map_maker::planet::PlanetSettings;
use global_map_maker::terrain::{Terrain, TerrainRef};
use global_map_maker::tiles::TileStore;
use global_map_maker::mesh::Vertex;
use std::time::Instant;
use winit::{
    application::ApplicationHandler,
    dpi::PhysicalPosition,
    event::{DeviceEvent, DeviceId, ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    keyboard::{Key, NamedKey},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{CursorGrabMode, Window, WindowId},
};

/// Room for the ground / hex triangles (hexes: 18 vertices each).
const MAX_VERTS: usize = 600_000;
/// Ground patch meshes kept between frames.
/// The old paint tools (pencil, line, fill, ...) are put aside for a later model-making mode: the ribbon is for
/// terraforming. Flip this to bring painting back (its code is untouched).
const PAINT_ENABLED: bool = false;
const PATCH_CACHE_MAX: usize = 24_000;
/// Height tiles up to this level are made ahead in the background (about 140 MB on disk, once).
const PREGEN_LEVEL: u8 = 5;
/// Numbers behind the loading readout in the window title.
struct LoadStats {
    t: Instant,
    gen0: usize,
    disk0: usize,
    meshes: usize,
    /// Meshes made again that had been thrown out of the cache earlier (should stay near 0).
    reloads: usize,
    missing: usize,
    drawn: usize,
    rebuild_ms: f64,
    line: String,
}

impl LoadStats {
    fn new() -> LoadStats {
        LoadStats { t: Instant::now(), gen0: 0, disk0: 0, meshes: 0, reloads: 0, missing: 0, drawn: 0, rebuild_ms: 0.0, line: String::new() }
    }
}

/// Finished patches picked up from the loader per frame.
const UPLOADS_PER_FRAME: usize = 200;
/// Most time one frame spends putting finished meshes on the GPU.
const UPLOAD_BUDGET_S: f64 = 0.004;
/// How far ahead (seconds of flying) the loader looks.
const AHEAD_SECONDS: f64 = 1.5;
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
    /// Vertices first, then (from `index_at`) the u16 indices.
    buf: wgpu::Buffer,
    index_at: u64,
    /// Number of indices to draw.
    count: u32,
    center: DVec3,
    used: u64,
    /// Which hand-edit version the mesh was made at (see PatchMesh::edit_version).
    edit_version: u64,
}

/// How quickly the flying camera's ground level follows the land (seconds): it glides over ridges instead of bobbing.
const GLIDE_TAU: f64 = 0.45;
/// Inside, the camera never gets closer to the real ground than this fraction of its height.
const CLEARANCE: f64 = 0.25;
/// Render distance inside: default and limits (km).
const RENDER_KM_DEFAULT: f64 = 300.0;
const RENDER_KM_MIN: f64 = 1.5;
const RENDER_KM_MAX: f64 = 1500.0;
/// Mouse look while the cursor is locked: radians per raw mouse count.
const LOOK_SPEED: f64 = 0.003;
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
    terrain: Arc<TileStore>,
    /// Background loader for ground patches (see stream.rs).
    streamer: Streamer,
    /// Newest ready mesh of each patch, and what was drawn last frame (stand-ins while the real ones load).
    by_id: HashMap<PatchId, PatchKey>,
    ahead_wanted: Vec<(f64, PatchKey)>,
    frame_no: u64,
    last_title: String,
    /// The camera numbers the last ground plan was made for.
    last_sig: Option<[f64; 12]>,
    /// Loading readout for the window title (refreshed about once a second) and what it is made from.
    stats: LoadStats,
    max_freq: f64,
    overlay_buf: wgpu::Buffer,
    overlay_count: u32,
    vertex_buf: wgpu::Buffer,
    vertex_count: u32,
    ground_pipeline: wgpu::RenderPipeline,
    /// Per-patch offsets (one vec4 each), written every frame; patch meshes themselves are kept between frames.
    inst_buf: wgpu::Buffer,
    patch_cache: HashMap<PatchKey, GpuPatch>,
    /// Keys thrown out of the cache recently, to notice when one has to be made again.
    evicted: HashSet<PatchKey>,
    /// The tool ribbon (egui) along the top of the window.
    ribbon: Ribbon,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    patch_tick: u64,
    draw_list: Vec<(wgpu::Buffer, u64, u32)>,
    ground_light: Lighting,
    ground_uniform: wgpu::Buffer,
    ground_bind: wgpu::BindGroup,
    /// Set when the camera or window changed, so the set of cells must be recomputed.
    dirty: bool,
    /// Only the pixel layer / highlight changed (painting, hover): the ground mesh is kept.
    pixels_dirty: bool,
    hex_count: usize,
    /// Inside: height above the (smoothed) ground the user asked for, in globe radii.
    fly_alt: f64,
    /// Inside: render distance in km (PageUp / PageDown).
    render_km: f64,
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
    /// Inside: climb above (+) or sink below (-) the zoom height as a fraction of it (Space / Ctrl). Zoom stays.
    lift: f64,
    space: bool,
    ctrl: bool,
    alt: bool,
    /// Tab: the cursor is free by default (else it is locked to the view while flying). Holding Alt flips it.
    free_default: bool,
    /// Middle click: the scroll wheel no longer zooms.
    zoom_locked: bool,
    /// What the OS was last told about the cursor (None = apply again).
    cursor_locked: Option<bool>,
    focused: bool,
    /// Left mouse is down inside: the selected tool works on the ground under the crosshair / cursor.
    sculpting: bool,
    /// Flatten: the height the stroke started at.
    flat_target: f64,
    /// Where the last brush dab went, and the time since it (see `sculpt`).
    last_dab: Option<DVec3>,
    dab_timer: f64,
    /// The meshes drawn last frame: after an edit these are remade first, the rest when the stroke ends.
    drawn_keys: HashSet<PatchKey>,
    /// The strokes being shown (see sculpt_overlay.rs): two windows, the texture they live in, which one the brush
    /// is writing to, the meshes each is waiting for (with the ones finished so far), and meshes set aside.
    ov_tex: wgpu::Texture,
    ov_win: [Option<global_map_maker::sculpt_overlay::Window>; 2],
    ov_live: Option<usize>,
    ov_commit: [Option<HashSet<PatchKey>>; 2],
    /// Debug: layers switched off (F1..F6), bit set = off. See the `dbg` uniform in ground.wgsl.
    dbg: u32,
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

/// While the pointer rests, the brush dabs this often (seconds).
const DAB_REST_S: f64 = 1.0 / 12.0;

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

        // Tiles are kept on disk between runs (cache/tiles, ignored by git); the coarse levels are made in the
        // background once, so the next start finds them ready.
        let terrain = Arc::new(TileStore::with_disk(Terrain::with_settings(&PlanetSettings::default()), std::path::Path::new("cache/tiles")));
        {
            let t = terrain.clone();
            let workers = (std::thread::available_parallelism().map_or(2, |n| n.get()) / 4).max(1);
            std::thread::Builder::new().name("tile-pregen".into()).spawn(move || t.pregenerate(PREGEN_LEVEL, workers)).ok();
        }

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
            size: (MAX_PATCHES * 32) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let ground_uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ground uniforms"),
            size: 256, // mvp + sun + eye + origin + sky + refm + dbg + 2 stroke windows
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // The stroke being made: two windows of heights, one layer each (see sculpt_overlay.rs).
        let ov_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("stroke windows"),
            size: wgpu::Extent3d { width: global_map_maker::sculpt_overlay::N as u32, height: global_map_maker::sculpt_overlay::N as u32, depth_or_array_layers: 2 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let ov_view = ov_tex.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() });
        let ground_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: None,
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension: wgpu::TextureViewDimension::D2Array, multisampled: false },
                    count: None,
                },
            ],
        });
        let ground_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&ground_bgl)],
            immediate_size: 0,
        });
        let ground_bind = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &ground_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: ground_uniform.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&ov_view) },
            ],
        });
        let ground_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ground shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("ground.wgsl").into()),
        });
        let ground_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ground pipeline"),
            layout: Some(&ground_layout),
            vertex: wgpu::VertexState {
                module: &ground_shader,
                entry_point: Some("vs_main"),
                compilation_options: Default::default(),
                buffers: &[
                    Some(wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<GroundVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Snorm8x4, 2 => Float32x2, 4 => Float32, 5 => Snorm8x4, 6 => Unorm8x4, 7 => Float32],
                    }),
                    Some(wgpu::VertexBufferLayout {
                        array_stride: 32,
                        step_mode: wgpu::VertexStepMode::Instance,
                        attributes: &wgpu::vertex_attr_array![3 => Float32x4, 8 => Float32x4],
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
        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &*window,
            Some(window.scale_factor() as f32),
            window.theme(),
            Some(device.limits().max_texture_dimension_2d as usize),
        );
        let egui_renderer = egui_wgpu::Renderer::new(&device, config.format, egui_wgpu::RendererOptions::default());
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
            terrain: terrain.clone(),
            streamer: Streamer::new(terrain, std::thread::available_parallelism().map_or(2, |n| n.get().saturating_sub(1)).clamp(1, 8)),
            by_id: HashMap::new(),
            ahead_wanted: Vec::new(),
            frame_no: 0,
            last_title: String::new(),
            last_sig: None,
            stats: LoadStats::new(),
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
            evicted: HashSet::new(),
            ribbon: Ribbon::default(),
            egui_ctx,
            egui_state,
            egui_renderer,
            patch_tick: 0,
            draw_list: Vec::new(),
            ground_light: Lighting::new(DVec3::Z, DVec3::X, DVec3::X * 2.0, None),
            ground_uniform,
            ground_bind,
            dirty: true,
            pixels_dirty: false,
            hex_count: 0,
            fly_alt: INSIDE_ENTER_ALT,
            render_km: RENDER_KM_DEFAULT,
            floor_smooth: 1.0,
            last_glide: Instant::now(),
            uniform_buf,
            bind_group,
            depth_view,
            camera,
            hex: false,
            paint: PaintStore::default(),
            cursor: None,
            lift: 0.0,
            space: false,
            ctrl: false,
            alt: false,
            free_default: false,
            zoom_locked: false,
            cursor_locked: None,
            focused: true,
            sculpting: false,
            flat_target: 0.0,
            last_dab: None,
            dab_timer: 0.0,
            drawn_keys: HashSet::new(),
            ov_tex,
            ov_win: [None, None],
            ov_live: None,
            ov_commit: [None, None],
            dbg: 0,
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
    /// Render distance in globe radii (infinite outside, where the whole globe is the view).
    fn render_dist(&self) -> f64 {
        if self.inside() && !self.hex { self.render_km * 1000.0 / global_map_maker::terrain::EARTH_RADIUS_M } else { f64::INFINITY }
    }

    fn terrain_ref(&self) -> TerrainRef<'_> {
        let t = TerrainRef::new(&*self.terrain, &self.camera, self.config.height as f64);
        if self.inside() { t } else { t.capped(OUTSIDE_CAP) }
    }

    /// (detail limit, ground radius) under the camera for the current scene.
    fn ground_under_camera(&self) -> (f64, f64) {
        let f = self.terrain_ref().max_freq;
        (f, global_map_maker::terrain::HeightSource::ground_radius_peek(&*self.terrain, self.camera.eye64().normalize(), f))
    }

    /// Ground level for the flying camera: the land averaged over a disk about as wide as the camera is high
    /// (center + 8 points around), so a ridge raises the camera gradually, well before it is overhead.
    fn smoothed_ground(&self, max_freq: f64) -> f64 {
        use global_map_maker::terrain::HeightSource;
        let Some(f) = self.camera.fly else { return self.camera.floor };
        let (n, e, _) = f.frame();
        let rad = 1.5 * self.fly_alt;
        let mut sum = self.terrain.ground_radius_peek(f.dir.normalize(), max_freq);
        let mut top = sum;
        for k in 0..8 {
            let a = k as f64 * std::f64::consts::FRAC_PI_4;
            let d = (f.dir.normalize() + (n * a.cos() + e * a.sin()) * rad).normalize();
            let r = self.terrain.ground_radius_peek(d, max_freq);
            sum += r;
            top = top.max(r);
        }
        // Half average, half highest point around: the camera still glides over ridges, but a rock or crest
        // taller than the average can no longer end up around (or in front of) the lens.
        0.5 * (sum / 9.0 + top)
    }

    /// Zoomed in far enough on the globe: drop into the scene over the spot under the camera.
    fn enter_inside(&mut self) {
        let dir = self.camera.eye64().normalize();
        self.camera.fly = Some(Fly { dir, heading: 0.0, tilt: INSIDE_ENTER_TILT });
        self.fly_alt = INSIDE_ENTER_ALT;
        self.lift = 0.0;
        self.floor_smooth = self.camera.floor;
        self.last_glide = Instant::now();
        self.camera.distance = self.camera.floor + INSIDE_ENTER_ALT;
        self.shape_start = None;
        self.keys = [false; 4];
        self.dirty = true;
    }

    /// Back out to space, over whatever you were flying above.
    fn leave_inside(&mut self) {
        self.stroke_end();
        if let Some(f) = self.camera.fly.take() {
            self.camera.orbit_over(f.dir);
            self.camera.distance = self.camera.floor + OUTSIDE_EXIT_H;
            self.keys = [false; 4];
            self.dirty = true;
        }
    }

    /// The ground point under the crosshair (cursor locked) or the mouse (cursor free), as a unit direction.
    fn brush_point(&self) -> Option<DVec3> {
        self.brush_hit().map(|p| p.normalize())
    }

    /// The same as a point on the ground (not only its direction).
    fn brush_hit(&self) -> Option<DVec3> {
        let (x, y) = self.cursor?;
        let (origin, dir) = global_map_maker::hexview::pick_ray(&self.camera, self.config.width, self.config.height, x, y);
        self.terrain_ref().pick_peek(origin, dir)
    }

    /// The stroke is over: remake the meshes that were not on screen while it went on.
    fn stroke_end(&mut self) {
        self.sculpting = false;
        if let Some(i) = self.ov_live.take() {
            self.begin_commit(i);
            self.window.request_redraw();
        }
    }

    /// The brush outline on the ground, in screen points (None where it is behind the camera).
    fn brush_ring(&self) -> Vec<Option<egui::Pos2>> {
        use global_map_maker::terrain::EARTH_RADIUS_M;
        if !self.inside() || self.hex || !self.ribbon.visible_tool_works() {
            return Vec::new();
        }
        let Some(hit) = self.brush_hit() else { return Vec::new() };
        let c = hit.normalize();
        let radius = self.ribbon.radius_m();
        let helper = if c.z.abs() < 0.9 { DVec3::Z } else { DVec3::X };
        let t1 = c.cross(helper).normalize();
        let t2 = c.cross(t1);
        let (w, h) = (self.config.width as f64, self.config.height as f64);
        let ppp = self.egui_ctx.pixels_per_point() as f64;
        let view = self.camera.view64();
        let tan_y = (global_map_maker::camera::Camera::fov_y() * 0.5).tan();
        let tref = self.terrain_ref();
        // The ring follows the ground around the point, but its middle is the very point the pointer hits: whatever
        // the ground height said here when the pointer was picked and when the ring is drawn, they cannot drift apart.
        let lift = hit.length() - tref.radius_peek(c);
        (0..72)
            .map(|i| {
                let a = i as f64 / 72.0 * std::f64::consts::TAU;
                let d = (c + (t1 * a.cos() + t2 * a.sin()) * (radius / EARTH_RADIUS_M)).normalize();
                let p = view.transform_point3(d * (tref.radius_peek(d) + lift));
                if p.z >= -1e-12 {
                    return None;
                }
                let (nx, ny) = (p.x / -p.z / (tan_y * w / h), p.y / -p.z / tan_y);
                Some(egui::pos2(((nx + 1.0) * 0.5 * w / ppp) as f32, ((1.0 - ny) * 0.5 * h / ppp) as f32))
            })
            .collect()
    }

    /// Throws the current planet away and builds the one the Planet view describes: new tile store (its cache
    /// folder is named after what the terrain looks like, so settings never mix), new loader, empty mesh caches.
    fn new_planet(&mut self) {
        let settings = self.ribbon.planet.clone();
        self.stroke_end();
        let terrain = Arc::new(TileStore::with_disk(Terrain::with_settings(&settings), std::path::Path::new("cache/tiles")));
        {
            let t = terrain.clone();
            let workers = (std::thread::available_parallelism().map_or(2, |n| n.get()) / 4).max(1);
            std::thread::Builder::new().name("tile-pregen".into()).spawn(move || t.pregenerate(PREGEN_LEVEL, workers)).ok();
        }
        self.streamer = Streamer::new(terrain.clone(), std::thread::available_parallelism().map_or(2, |n| n.get().saturating_sub(1)).clamp(1, 8));
        self.terrain = terrain;
        self.patch_cache.clear();
        self.by_id.clear();
        self.evicted.clear();
        self.draw_list.clear();
        self.ahead_wanted.clear();
        self.drawn_keys.clear();
        self.ov_win = [None, None];
        self.ov_live = None;
        self.ov_commit = [None, None];
        self.last_sig = None;
        self.dirty = true;
        self.window.request_redraw();
    }

    fn stroke_start(&mut self) {
        self.last_dab = None;
        self.dab_timer = 0.0;
        // Flatten starts at the average height under the brush.
        if let Some(d) = self.brush_point() {
            let level = global_map_maker::edits::level_for_radius(self.ribbon.radius_m());
            self.flat_target = Self::brush_mean(&self.terrain, d, self.ribbon.radius_m(), level);
        }
    }

    /// One frame of the selected tool: change the ground under the brush. The meshes are not remade while the
    /// stroke goes on; a window of heights around the brush shows it instead (see sculpt_overlay.rs), and the real
    /// meshes come when the stroke ends.
    fn sculpt(&mut self, dt: f64) {
        use global_map_maker::edits::{falloff_weight, level_for_radius};
        use global_map_maker::ribbon::TerraTool;
        use global_map_maker::sculpt_overlay::Window;
        let tool = self.ribbon.tool;
        if !self.sculpting || !self.inside() || self.hex || !tool.works() || !matches!(tool, TerraTool::Raise | TerraTool::Lower | TerraTool::Smooth | TerraTool::Flatten) {
            return;
        }
        let Some(center) = self.brush_point() else { return };
        let radius = self.ribbon.radius_m();
        let level = level_for_radius(radius);
        // The window the stroke is drawn into: a new one when the brush nears its edge (the old one is made real).
        let fits = self.ov_live.and_then(|i| self.ov_win[i].as_ref()).is_some_and(|w| w.fits(center, radius));
        if !fits {
            if let Some(i) = self.ov_live.take() {
                self.begin_commit(i);
            }
            // Both windows busy (one still being made real): the brush waits a moment.
            let Some(i) = (0..2).find(|&i| self.ov_win[i].is_none()) else { return };
            let mut win = Window::new(center, radius);
            win.v_first = self.terrain.edits().version();
            win.tint = matches!(tool, TerraTool::Raise | TerraTool::Lower);
            self.last_sig = None; // the drawn meshes must be flagged for this window
            self.ov_win[i] = Some(win);
            self.ov_live = Some(i);
        }
        let i = self.ov_live.unwrap();
        // Dabs, like Blender's spaced strokes: one every 20% of the brush radius along the drag, plus one every
        // twelfth of a second while the pointer stays put (so holding builds up). Each dab applies the tool once, so
        // the result does not depend on the frame rate.
        let spacing = 0.2 * radius;
        self.dab_timer += dt;
        let mut dabs: Vec<DVec3> = match self.last_dab {
            None => vec![center],
            Some(last) => {
                let (d, newest) = global_map_maker::sculpt_overlay::dabs_along(last, center, spacing, 4);
                if !d.is_empty() {
                    self.last_dab = Some(newest);
                }
                d
            }
        };
        if dabs.is_empty() && self.dab_timer >= DAB_REST_S {
            dabs.push(center);
        }
        if dabs.is_empty() {
            return;
        }
        self.dab_timer = 0.0;
        self.last_dab = Some(*dabs.last().unwrap());
        let (strength, kind) = (self.ribbon.brush.strength as f64, self.ribbon.brush.falloff);
        let store = self.terrain.clone();
        let w = |dist: f64| falloff_weight(kind, dist / radius);
        // Smooth works one level finer than the brush's usual grid: it can only even out what its grid can hold, and the
        // ridges you see are finer than a fourteenth of the brush.
        let level = if tool == TerraTool::Smooth { (level + 1).min(global_map_maker::tiles::MAX_TILE_LEVEL) } else { level };
        let cell = global_map_maker::tiles::spacing_rad(level);
        let sign = if tool == TerraTool::Raise { 1.0 } else { -1.0 };
        // Per dab: raise/lower add a share of the radius; smooth pulls toward the neighbours' average, flatten toward
        // a target (the average height under the brush, followed along so a mountain's height is not dragged around).
        let raise = sign * 0.06 * strength * radius;
        let k_relax = 1.0 - (-strength * if tool == TerraTool::Smooth { 2.5 } else { 1.2 }).exp();
        for dab in dabs {
            if tool == TerraTool::Flatten {
                let mean = Self::brush_mean(&store, dab, radius, level);
                self.flat_target += (mean - self.flat_target) * 0.25;
            }
            let target = self.flat_target;
            let amount = |d: DVec3, dist: f64| -> f64 {
                match tool {
                    TerraTool::Raise | TerraTool::Lower => raise * w(dist),
                    TerraTool::Smooth => {
                        let helper = if d.z.abs() < 0.9 { DVec3::Z } else { DVec3::X };
                        let t1 = d.cross(helper).normalize();
                        let t2 = d.cross(t1);
                        let (a, b) = ((t1 + t2).normalize(), (t1 - t2).normalize());
                        let avg = [t1, -t1, t2, -t2, a, -a, b, -b].iter().map(|t| store.height_for_edit((d + *t * cell).normalize(), level)).sum::<f64>() / 8.0;
                        (avg - store.height_for_edit(d, level)) * k_relax * w(dist)
                    }
                    TerraTool::Flatten => (target - store.height_for_edit(d, level)) * k_relax * w(dist),
                    _ => 0.0,
                }
            };
            if let Some(win) = self.ov_win[i].as_mut() {
                win.add(dab, radius, &amount);
            }
            store.edits().stroke(level, dab, radius, &amount);
        }
        self.dirty = true;
    }

    /// Average ground height on a ring at 85% of the brush radius around `center`. The ring is where the brush has
    /// hardly any effect, so the reference is not pulled down by the flattening itself (reading the middle made
    /// Flatten sink further with every dab, most visibly on a beach).
    fn brush_mean(store: &TileStore, center: DVec3, radius: f64, level: u8) -> f64 {
        use global_map_maker::terrain::EARTH_RADIUS_M;
        let helper = if center.z.abs() < 0.9 { DVec3::Z } else { DVec3::X };
        let t1 = center.cross(helper).normalize();
        let t2 = center.cross(t1);
        let mut sum = 0.0;
        for k in 0..12 {
            let a = k as f64 * std::f64::consts::TAU / 12.0;
            let off = (t1 * a.cos() + t2 * a.sin()) * (0.85 * radius / EARTH_RADIUS_M);
            sum += store.height_for_edit((center + off).normalize(), level);
        }
        sum / 12.0
    }

    /// The stroke in window `i` is done: remake every mesh it reached (in the background). The window keeps
    /// showing it until all of them are ready, then they are swapped in together.
    fn begin_commit(&mut self, i: usize) {
        use global_map_maker::edits::level_for_radius;
        use global_map_maker::terrain::EARTH_RADIUS_M;
        let Some(win) = self.ov_win[i].as_ref() else { return };
        let Some((c, reach_m)) = win.touched_disc() else {
            self.ov_win[i] = None;
            return;
        };
        let radius = win.radius_m;
        let level = level_for_radius(radius);
        let cell = global_map_maker::tiles::spacing_rad(level);
        let reach = (reach_m + 3.0 * cell * EARTH_RADIUS_M) / EARTH_RADIUS_M;
        let keys: HashSet<PatchKey> = self
            .patch_cache
            .keys()
            .filter(|k| {
                // Fine enough to show the edit directly, or coarse patches whose level of the edit pyramid has it
                // (cells no bigger than about the brush): that is how the globe learns of what was made inside.
                (k.lvl as i32 + 2 >= level as i32 || global_map_maker::tiles::spacing_rad(k.lvl) * EARTH_RADIUS_M <= 1.5 * radius) && {
                    let ang = k.id.center_dir().dot(c).clamp(-1.0, 1.0).acos();
                    ang <= global_map_maker::pixelview::width_rad(k.id.level) * 0.8 + reach
                }
            })
            .copied()
            .collect();
        // Only what is on screen (and the always-kept coarse base) is remade and waited for. The rest of what the
        // edit reached is dropped from the cache: it is made again, with the edit in it, when it is next needed.
        let (keys, dropped): (HashSet<PatchKey>, HashSet<PatchKey>) = keys.into_iter().partition(|k| self.drawn_keys.contains(k) || k.id.level <= BASE_LEVEL);
        for k in dropped {
            self.patch_cache.remove(&k);
            if self.by_id.get(&k.id) == Some(&k) {
                self.by_id.remove(&k.id);
            }
        }
        if keys.is_empty() {
            self.ov_win[i] = None;
            return;
        }
        // The loader takes the last first: the ones under the brush go last in the list.
        let mut list: Vec<PatchKey> = keys.iter().copied().collect();
        list.sort_by(|a, b| a.id.center_dir().dot(c).partial_cmp(&b.id.center_dir().dot(c)).unwrap_or(std::cmp::Ordering::Equal));
        self.streamer.rebuild(&list);
        self.ov_commit[i] = Some(keys);
        self.dirty = true;
    }

    /// Every mesh window `i` was waiting for is in: the window can go.
    fn finish_commit(&mut self, i: usize) {
        self.ov_commit[i] = None;
        self.ov_win[i] = None;
        self.last_sig = None;
        self.dirty = true;
    }

    /// A mesh the loader finished goes into the cache (a few per frame, see `ingest_patches`). Each mesh knows which
    /// strokes it already shows (its edit version), so one made while a stroke was going is not drawn twice over,
    /// and a window goes as soon as the meshes it asked for are in.
    fn take_patch(&mut self, k: PatchKey, mesh: PatchMesh) {
        self.insert_patch(k, mesh);
        for i in 0..2 {
            if let Some(pending) = self.ov_commit[i].as_mut() {
                pending.remove(&k);
                if pending.is_empty() {
                    self.finish_commit(i);
                }
            }
        }
    }

    /// Sends the changed rows of the stroke windows to the GPU, and says which windows are on.
    fn upload_windows(&mut self) -> [f32; 24] {
        let mut out = [0f32; 24];
        for i in 0..2 {
            if let Some(w) = self.ov_win[i].as_mut() {
                if let Some((x, y, ww, hh, v)) = w.take_dirty() {
                    self.queue.write_texture(
                        wgpu::TexelCopyTextureInfo { texture: &self.ov_tex, mip_level: 0, origin: wgpu::Origin3d { x: x as u32, y: y as u32, z: i as u32 }, aspect: wgpu::TextureAspect::All },
                        bytemuck::cast_slice(&v),
                        wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(ww as u32 * 4), rows_per_image: Some(hh as u32) },
                        wgpu::Extent3d { width: ww as u32, height: hh as u32, depth_or_array_layers: 1 },
                    );
                }
                out[i * 12..i * 12 + 12].copy_from_slice(&w.uniform(self.ground_light.origin));
            }
        }
        out
    }

    fn moving(&self) -> bool {
        self.inside() && (self.keys.iter().any(|k| *k) || self.space != self.ctrl)
    }

    /// Camera height above the ground: the zoom height, plus whatever Space / Ctrl lifted it by.
    fn fly_height(&self) -> f64 {
        self.fly_alt * (1.0 + self.lift)
    }

    fn clamp_lift(&mut self) {
        let top = (INSIDE_MAX_ALT / self.fly_alt.max(INSIDE_MIN_ALT) - 1.0).clamp(0.0, 20.0);
        self.lift = self.lift.clamp(-(1.0 - CLEARANCE), top);
    }

    /// Locked to the camera: flying inside, unless Tab / Alt say otherwise.
    fn cursor_wanted_locked(&self) -> bool {
        self.focused && self.inside() && self.free_default == self.alt
    }

    fn centre_cursor(&self) {
        let s = self.window.inner_size();
        let _ = self.window.set_cursor_position(PhysicalPosition::new(s.width as f64 / 2.0, s.height as f64 / 2.0));
    }

    /// Hides and locks the cursor while flying (mouse look), frees it otherwise.
    fn sync_cursor(&mut self) {
        let want = self.cursor_wanted_locked();
        if self.cursor_locked == Some(want) {
            return;
        }
        self.cursor_locked = Some(want);
        if want {
            let _ = self
                .window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| self.window.set_cursor_grab(CursorGrabMode::Confined));
            self.window.set_cursor_visible(false);
            let s = self.window.inner_size();
            self.cursor = Some((s.width as f64 / 2.0, s.height as f64 / 2.0));
            self.centre_cursor();
        } else {
            let _ = self.window.set_cursor_grab(CursorGrabMode::None);
            self.window.set_cursor_visible(true);
        }
    }

    /// Move the flying camera for the time since the last frame (W A S D, shift = faster).
    fn step(&mut self) {
        let now = Instant::now();
        let dt = now.duration_since(self.last_frame).as_secs_f64().min(0.3);
        self.last_frame = now;
        self.sculpt(dt);
        if !self.moving() {
            return;
        }
        let alt = self.fly_height().max(INSIDE_MIN_ALT);
        let boost = if self.shift { FLY_BOOST } else { 1.0 };
        let speed = alt * FLY_SPEED * boost;
        let [w, a, s, d] = self.keys.map(|k| k as i32 as f64);
        let (ahead, right) = (w - s, d - a);
        if let Some(f) = self.camera.fly.as_mut() {
            let (n, e, _) = f.frame();
            let (sh, ch) = f.heading.sin_cos();
            let mv = (n * ch + e * sh) * ahead + (e * ch - n * sh) * right;
            let len = mv.length();
            if len > 0.0 {
                f.move_to((f.dir.normalize() + mv / len * speed * dt).normalize());
                self.dirty = true;
            }
        }
        // Space / Ctrl: up and down at the current zoom's speed. The zoom (fly_alt) itself does not change.
        let climb = self.space as i32 as f64 - self.ctrl as i32 as f64;
        if climb != 0.0 {
            self.lift += climb * FLY_SPEED * boost * dt * (1.0 + self.lift).max(0.25);
            self.clamp_lift();
            self.dirty = true;
        }
    }

    /// Work out what the camera needs (ground + pixels) and upload the triangles.
    fn rebuild(&mut self) {
        let t_rebuild = Instant::now();
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
            self.camera.distance = (self.floor_smooth + self.fly_height()).max(exact + CLEARANCE * self.fly_alt);
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
        let mut ground: Option<(Vec<PatchKey>, Vec<PatchKey>, Vec<(PatchKey, PatchMesh)>)> = None;
        let mut light_now = None;
        let mut ahead_out: Option<Vec<(f64, PatchKey)>> = None;
        if self.hex {
            let r = Resolution::try_from(res).unwrap();
            let cells = visible_hexes(res, &self.camera, w, h);
            self.hex_count = cells.len();
            self.hover = self.cell_under_cursor();
            let highlighted = self.footprint();
            let coverage = self.paint.coverage(r);
            vertices = build_hex_cells(&cells, r, &self.paint, &coverage, &highlighted, self.origin);
        } else {
            let n_new = self.ingest_patches(); // whatever the loader finished since last frame
            self.hover = None;
            // Nothing new (same camera, same render distance, no mesh arrived): the plan would come out exactly as
            // last frame, so it is not worked out again (the loader keeps going with its wish list meanwhile).
            let fly = self.camera.fly.map_or([0.0; 5], |f| [f.dir.x, f.dir.y, f.dir.z, f.heading, f.tilt]);
            let sig = [self.camera.yaw, self.camera.pitch, self.camera.distance, self.camera.floor, self.render_km, w as f64, h as f64, fly[0], fly[1], fly[2], fly[3], fly[4]];
            let unchanged = n_new == 0 && self.last_sig == Some(sig);
            self.last_sig = Some(sig);
            if !unchanged {
            let tref = self.terrain_ref();
            let view = PixelView::from_camera(&self.camera, w, h, Some(tref)).with_max_dist(self.render_dist());
            let light = self.camera.light_dir();
            let light = DVec3::new(light.x as f64, light.y as f64, light.z as f64).normalize();
            let haze = self.inside().then_some(SKY);
            let mut lighting = Lighting::new(light, self.origin, self.camera.eye64(), haze).with_alt_shift(self.terrain.terrain().settings().sea_level_m);
            if self.inside() {
                lighting = lighting.with_render_distance(self.render_dist());
            }
            let patches = visible_patches(&view);
            let keys: Vec<PatchKey> = patches.iter().map(|id| patch_key(*id, &view)).collect();
            let ready = |k: &PatchKey| self.patch_cache.contains_key(k);
            let key_of = |id: PatchId| patch_key(id, &view);
            let eye = self.camera.eye64();
            let dist = |k: &PatchKey| (k.id.center_dir() * self.camera.floor - eye).length();
            let mut plan = plan_frame(&keys, &ready, &self.by_id, &key_of);
            // Coarse base patches nothing stands in for yet (cold start, long jump) are made right now, so the
            // ground is never empty; everything finer than that streams in without blocking.
            let mut built: Vec<(PatchKey, PatchMesh)> = Vec::new();
            if !plan.holes.is_empty() {
                let now_keys = plan.holes.clone();
                built = now_keys.iter().copied().zip(build_patches(&now_keys, &view)).collect();
                let mut by_id = self.by_id.clone();
                for (k, _) in &built {
                    by_id.insert(k.id, *k);
                }
                let have: std::collections::HashSet<PatchKey> = built.iter().map(|(k, _)| *k).collect();
                plan = plan_frame(&keys, &|k| have.contains(k) || ready(k), &by_id, &key_of);
            }
            // Missing meshes go to the background loader: first the coarse ancestors of what is wanted (coarse to
            // fine: they are the stand-ins, so they must exist before the fine ones), then the wanted ones, nearest first.
            let mut wanted: Vec<(f64, PatchKey)> = plan.chain.iter().filter(|k| !ready(k)).map(|k| (k.id.level as f64 + dist(k) - 100.0, *k)).collect();
            wanted.extend(plan.missing.iter().map(|k| (dist(k), *k)));
            // Look ahead while flying: what the view will need a moment from now is loaded in advance.
            let mut ahead_new: Option<Vec<(f64, PatchKey)>> = None;
            if self.moving() {
                if self.frame_no % 4 == 0 {
                    let mut ahead_list = Vec::new();
                    let mut cam = self.camera.clone();
                    if let Some(f) = cam.fly.as_mut() {
                        let (n, e, _) = f.frame();
                        let (sh, ch) = f.heading.sin_cos();
                        let [wk, ak, sk, dk] = self.keys.map(|k| k as i32 as f64);
                        let mv = (n * ch + e * sh) * (wk - sk) + (e * ch - n * sh) * (dk - ak);
                        if mv.length() > 0.0 {
                            let speed = self.fly_height().max(INSIDE_MIN_ALT) * FLY_SPEED * if self.shift { FLY_BOOST } else { 1.0 };
                            f.dir = (f.dir.normalize() + mv / mv.length() * speed * AHEAD_SECONDS).normalize();
                        }
                    }
                    let vt = TerrainRef::new(&*self.terrain, &cam, self.config.height as f64);
                    let va = PixelView::from_camera(&cam, w, h, Some(vt)).with_max_dist(self.render_dist());
                    let pa = visible_patches(&va);
                    let ea = cam.eye64();
                    let ahead_keys: Vec<PatchKey> = pa.iter().map(|id| patch_key(*id, &va)).collect();
                    for k in &ahead_keys {
                        if !ready(k) {
                            ahead_list.push(((k.id.center_dir() * self.camera.floor - ea).length() + 5.0, *k));
                        }
                    }
                    let akey = |id: PatchId| patch_key(id, &va);
                    for k in global_map_maker::stream::chain_of(&ahead_keys, &akey) {
                        if !ready(&k) {
                            ahead_list.push((k.id.level as f64 + (k.id.center_dir() * self.camera.floor - ea).length() + 5.0 - 100.0, k));
                        }
                    }
                    ahead_new = Some(ahead_list);
                }
                let src = ahead_new.as_ref().unwrap_or(&self.ahead_wanted);
                wanted.extend(src.iter().filter(|(_, k)| !ready(k)).copied());
            } else {
                ahead_new = Some(Vec::new());
            }
            self.streamer.request(wanted);
            // Wanted meshes and their ancestors count as used, so the cache never throws away the chain a
            // stand-in is made of (or a fine mesh whose coarse ones are missing).
            let mut touch = std::mem::take(&mut plan.chain);
            touch.extend(keys.iter().copied());
            let draw = plan.draw;
            self.drawn_keys = draw.iter().copied().collect();
            ahead_out = ahead_new;
            self.stats.missing = plan.missing.len();
            self.stats.drawn = draw.len();
            ground = Some((draw, touch, built));
            light_now = Some(lighting);
            }
        }
        if let Some(l) = light_now {
            self.ground_light = l;
        }
        if let Some(a) = ahead_out {
            self.ahead_wanted = a;
        }
        if let Some((draw, touch, built)) = ground {
            self.draw_list.clear();
            self.patch_tick += 1;
            for (k, mesh) in built {
                self.insert_patch(k, mesh);
            }
            let mut inst: Vec<[f32; 8]> = Vec::with_capacity(draw.len());
            let shown: [Option<u64>; 2] = [0, 1].map(|i| self.ov_win[i].as_ref().map(|w| w.v_first));
            for k in touch.iter().chain(self.ahead_wanted.iter().map(|(_, k)| k)) {
                if let Some(p) = self.patch_cache.get_mut(k) {
                    p.used = self.patch_tick;
                }
            }
            for k in &draw {
                if let Some(p) = self.patch_cache.get_mut(k) {
                    p.used = self.patch_tick;
                    let o = p.center - self.origin;
                    // A stroke window is added to the meshes that were made before it started; the others already show
                    // what they show (see PatchMesh::edit_version).
                    let f = |i: usize| if shown[i].is_some_and(|v| p.edit_version <= v) { 1.0 } else { 0.0 };
                    inst.push([o.x as f32, o.y as f32, o.z as f32, global_map_maker::pixelview::width_rad(k.id.level) as f32, f(0), f(1), 0.0, 0.0]);
                    self.draw_list.push((p.buf.clone(), p.index_at, p.count));
                }
            }
            inst.truncate(MAX_PATCHES);
            self.draw_list.truncate(MAX_PATCHES);
            self.queue.write_buffer(&self.inst_buf, 0, bytemuck::cast_slice(&inst));
            // Keep the meshes of recently seen patches (flying back and forth is free), drop the oldest beyond that.
            if self.patch_cache.len() > PATCH_CACHE_MAX {
                let mut ages: Vec<(u64, PatchKey)> = self.patch_cache.iter().filter(|(k, _)| k.id.level > BASE_LEVEL).map(|(k, p)| (p.used, *k)).collect();
                ages.sort_unstable_by_key(|a| a.0);
                for (_, k) in ages.into_iter().take(self.patch_cache.len().saturating_sub(PATCH_CACHE_MAX * 85 / 100)) {
                    self.patch_cache.remove(&k);
                    if self.evicted.len() > 100_000 {
                        self.evicted.clear();
                    }
                    self.evicted.insert(k);
                    if self.by_id.get(&k.id) == Some(&k) {
                        self.by_id.remove(&k.id);
                    }
                }
            }
        }
        self.frame_no += 1;

        vertices.truncate(MAX_VERTS / 3 * 3);
        self.queue.write_buffer(&self.vertex_buf, 0, bytemuck::cast_slice(&vertices));
        self.vertex_count = vertices.len() as u32;

        self.rebuild_pixels();
        // Keep going until the camera has settled on the ground level and the loader has caught up.
        self.dirty = gliding || self.streamer.busy() > 0;
        self.stats.rebuild_ms += (t_rebuild.elapsed().as_secs_f64() * 1e3 - self.stats.rebuild_ms) * 0.1;
        self.update_title();
    }

    /// Puts a finished patch mesh on the GPU and into the cache.
    fn insert_patch(&mut self, k: PatchKey, mesh: PatchMesh) {
        // One buffer per patch: the points, then the triangle indices (padded to a multiple of 4 bytes).
        let index_at = (mesh.ivert.len() * std::mem::size_of::<GroundVertex>()) as u64;
        let mut index = mesh.index;
        let count = index.len() as u32;
        if index.len() % 2 == 1 {
            index.push(0);
        }
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ground patch"),
            size: index_at + (index.len() * 2) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buf, 0, bytemuck::cast_slice(&mesh.ivert));
        self.queue.write_buffer(&buf, index_at, bytemuck::cast_slice(&index));
        if self.evicted.remove(&k) {
            self.stats.reloads += 1;
        }
        self.patch_cache.insert(k, GpuPatch { buf, index_at, count, center: mesh.center, used: self.patch_tick, edit_version: mesh.edit_version });
        self.by_id.insert(k.id, k);
    }

    /// Takes in what the background loader finished (a limited number per frame, so uploads never stall a frame).
    /// Returns how many were taken in. Uploads stop after a few milliseconds, so a burst of finished meshes
    /// is spread over a few frames instead of stalling one.
    fn ingest_patches(&mut self) -> usize {
        let t0 = Instant::now();
        let mut n = 0;
        // While sculpting, remade meshes get more of the frame, so the ground follows the brush closely.
        let budget = if self.sculpting { UPLOAD_BUDGET_S * 3.0 } else { UPLOAD_BUDGET_S };
        while n < UPLOADS_PER_FRAME && t0.elapsed().as_secs_f64() < budget {
            let batch = self.streamer.take_done(16);
            if batch.is_empty() {
                break;
            }
            for (k, mesh) in batch {
                n += 1;
                self.stats.meshes += 1;
                self.take_patch(k, mesh);
            }
        }
        n
    }

    /// The pixel layer (paint + highlight) on top of the ground; the ground mesh stays as it is.
    fn rebuild_pixels(&mut self) {
        let (w, h) = (self.config.width, self.config.height);
        self.pixel_hover = self.pixel_under_cursor();
        let eye = self.camera.eye64();
        let tref = (!self.hex).then(|| self.terrain_ref());
        let view = PixelView::from_camera(&self.camera, w, h, tref).with_max_dist(self.render_dist());
        let surface = Surface { terrain: tref, lift_px: if tref.is_some() { LIFT_PX } else { 0.0 }, focal_px: view.focal_px };
        let mut quads = visible_pixels(&self.pixels, &view);
        if tref.is_some() {
            quads = split_for_relief(quads, &view);
        }
        let mut pv = build_vertices(&quads, self.origin, eye, &surface);
        if PAINT_ENABLED && self.layer == Layer::Pixel {
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
                if self.hex { "HEX ground".to_string() } else if self.inside() { format!("INSIDE (WASD+Space/Ctrl fly, mouse look, Tab cursor, scroll zoom{}, view {:.0} km: PgUp/PgDn, Esc = space)", if self.zoom_locked { " LOCKED" } else { "" }, self.render_km) } else { "OUTSIDE (zoom in to enter)".to_string() },
                self.tool.name(),
                if matches!(self.tool, Tool::Circle | Tool::Fill) { " (hex layer only)" } else { "" },
                hexpaint::name(what),
                1u32 << self.pixel_k,
                PIXEL_RADII[self.pixel_rad] * 2 + 1,
                self.pixels.painted_pixels(),
                if self.hex { "flat sphere, P = hex layer" } else { "hex ground" }
            ),
        };
        // Loading readout (inside scene): what is still wanted but not made, and how fast the loader works.
        if self.inside() && !self.hex {
            let now = Instant::now();
            let dt = now.duration_since(self.stats.t).as_secs_f64();
            if dt >= 1.0 {
                let (made, disk) = (self.terrain.generated(), self.terrain.from_disk());
                self.stats.line = format!(
                    "LOAD missing {} | queue {} | tiles {:.0}/s made {:.0}/s disk | meshes {:.0}/s | reloads {} | drawn {} | rebuild {:.1} ms",
                    self.stats.missing,
                    self.streamer.busy(),
                    (made - self.stats.gen0) as f64 / dt,
                    (disk - self.stats.disk0) as f64 / dt,
                    self.stats.meshes as f64 / dt,
                    self.stats.reloads,
                    self.stats.drawn,
                    self.stats.rebuild_ms
                );
                self.stats.t = now;
                self.stats.gen0 = made;
                self.stats.disk0 = disk;
                self.stats.meshes = 0;
                self.stats.reloads = 0;
            }
        }
        let title = if self.stats.line.is_empty() || !self.inside() || self.hex {
            title
        } else {
            title.replacen("Global Map Maker | ", &format!("Global Map Maker | {} | ", self.stats.line), 1)
        };
        // Setting the title is slow on some systems (it repaints the window frame): only when it changed.
        if title != self.last_title {
            let names = ["noise", "dry+wet", "beach rule", "detail fades", "biome colour", "bump", "stone/sand variation"];
            let off: Vec<&str> = (0..7).filter(|i| self.dbg >> i & 1 == 1).map(|i| names[i]).collect();
            let title = if off.is_empty() { title } else { format!("{title} | DEBUG OFF (F1-F7): {}", off.join(", ")) };
            self.window.set_title(&title);
            self.last_title = title;
        }
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

        let ov = self.upload_windows();
        {
            let l = &self.ground_light;
            let f = |v: DVec3, w: f32| [v.x as f32, v.y as f32, v.z as f32, w];
            let sky = l.haze.unwrap_or([0.0; 3]);
            let mut g = [0f32; 64];
            g[..16].copy_from_slice(&m.to_cols_array());
            g[16..20].copy_from_slice(&f(l.sun, Camera::focal_px(self.config.height as f64) as f32));
            g[20..24].copy_from_slice(&f(l.eye_rel, l.horizon as f32));
            g[24..28].copy_from_slice(&f(l.origin, l.alt_shift as f32)); // w: colour height shift (sea level)
            g[28..32].copy_from_slice(&[sky[0], sky[1], sky[2], if l.haze.is_some() { 1.0 } else { 0.0 }]);
            let r = global_map_maker::terrain::EARTH_RADIUS_M;
            let wrap = |x: f64| (x * r).rem_euclid(4096.0) as f32;
            g[32..36].copy_from_slice(&[wrap(l.origin.x), wrap(l.origin.y), wrap(l.origin.z), r as f32]);
            g[36..40].copy_from_slice(&[self.dbg as f32, 0.0, 0.0, 0.0]);
            g[40..64].copy_from_slice(&ov);
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

        // The ribbon: one egui frame per redraw.
        let raw_input = self.egui_state.take_egui_input(&self.window);
        self.ribbon.ring = self.brush_ring();
        self.ribbon.inside = self.inside();
        self.ribbon.height_m = if self.inside() { self.fly_height() } else { self.camera.height() } * global_map_maker::terrain::EARTH_RADIUS_M;
        self.ribbon.cursor_free = self.cursor_locked != Some(true);
        self.ribbon.zoom_locked = self.zoom_locked;
        let ui_out = self.egui_ctx.run_ui(raw_input, |ui| {
            self.ribbon.show(ui);
            self.ribbon.crosshair(ui);
        });
        self.egui_state.handle_platform_output(&self.window, ui_out.platform_output);
        let ui_jobs = self.egui_ctx.tessellate(ui_out.shapes, ui_out.pixels_per_point);
        for (id, deltas) in &ui_out.textures_delta.set {
            for delta in deltas {
                self.egui_renderer.update_texture(&self.device, &self.queue, *id, delta);
            }
        }
        let screen = egui_wgpu::ScreenDescriptor { size_in_pixels: [self.config.width, self.config.height], pixels_per_point: ui_out.pixels_per_point };
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        let ui_extra = self.egui_renderer.update_buffers(&self.device, &self.queue, &mut encoder, &ui_jobs, &screen);
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
                for (k, (buf, index_at, count)) in self.draw_list.iter().enumerate() {
                    pass.set_vertex_buffer(0, buf.slice(..*index_at));
                    pass.set_index_buffer(buf.slice(*index_at..), wgpu::IndexFormat::Uint16);
                    pass.draw_indexed(0..*count, 0, k as u32..k as u32 + 1);
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
        {
            // The ribbon goes on top of everything, without depth.
            let mut pass = encoder
                .begin_render_pass(&wgpu::RenderPassDescriptor {
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                    })],
                    ..Default::default()
                })
                .forget_lifetime();
            self.egui_renderer.render(&mut pass, &ui_jobs, &screen);
        }
        self.queue.submit(ui_extra.into_iter().chain(Some(encoder.finish())));
        for id in &ui_out.textures_delta.free {
            self.egui_renderer.free_texture(id);
        }
        // Hover effects and animations ask for another frame.
        if ui_out.viewport_output.get(&egui::ViewportId::ROOT).is_some_and(|v| v.repaint_delay.is_zero()) {
            self.window.request_redraw();
        }
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

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        // Locked cursor: raw mouse motion turns the head (no button needed).
        let DeviceEvent::MouseMotion { delta } = event else { return };
        let Some(gpu) = self.gpu.as_mut() else { return };
        if gpu.cursor_locked == Some(true) && gpu.inside() {
            if let Some(f) = gpu.camera.fly.as_mut() {
                f.heading += delta.0 * LOOK_SPEED;
                f.tilt = (f.tilt - delta.1 * LOOK_SPEED).clamp(-1.5, 1.2);
            }
            gpu.dirty = true;
            gpu.window.request_redraw();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(gpu) = self.gpu.as_mut() else { return };
        // The ribbon gets every event first; what it uses (clicks on it, typing in a field) does not reach the map.
        // Keys belong to the map (Tab and Space must not move egui's focus or press its buttons) unless a text field
        // has focus; while the cursor is locked the panel sees no mouse either.
        let skip_ui = match event {
            WindowEvent::KeyboardInput { .. } => !gpu.egui_ctx.egui_wants_keyboard_input(),
            WindowEvent::CursorMoved { .. } | WindowEvent::MouseInput { .. } | WindowEvent::MouseWheel { .. } => {
                gpu.cursor_locked == Some(true)
            }
            _ => false,
        };
        let ui = if skip_ui {
            egui_winit::EventResponse { repaint: false, consumed: false }
        } else {
            gpu.egui_state.on_window_event(&gpu.window, &event)
        };
        if ui.repaint {
            gpu.window.request_redraw();
        }
        let always = matches!(
            event,
            WindowEvent::RedrawRequested | WindowEvent::Resized(_) | WindowEvent::CloseRequested | WindowEvent::ModifiersChanged(_)
        ) || matches!(event, WindowEvent::MouseInput { state: ElementState::Released, .. })
            || matches!(event, WindowEvent::KeyboardInput { event: ref k, .. } if k.state == ElementState::Released);
        if ui.consumed && !always {
            return;
        }
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                gpu.resize(size.width, size.height);
                gpu.window.request_redraw();
            }
            WindowEvent::ModifiersChanged(m) => {
                self.ctrl = m.state().control_key();
                if self.ctrl && !gpu.ctrl && !gpu.moving() {
                    gpu.last_frame = Instant::now();
                }
                gpu.ctrl = self.ctrl;
                gpu.alt = m.state().alt_key();
                gpu.shift = m.state().shift_key();
                gpu.dirty = true; // shift changes the circle preview
                gpu.window.request_redraw();
            }
            WindowEvent::MouseInput { state, button: MouseButton::Right, .. } if PAINT_ENABLED => {
                if state == ElementState::Pressed {
                    self.painting = true;
                    gpu.tool_press();
                } else if self.painting {
                    self.painting = false;
                    gpu.tool_release();
                }
            }
            WindowEvent::Focused(on) => {
                gpu.focused = on;
                if !on {
                    gpu.stroke_end();
                    gpu.keys = [false; 4];
                    gpu.space = false;
                    gpu.ctrl = false;
                    gpu.alt = false;
                    self.ctrl = false;
                }
                gpu.cursor_locked = None;
                gpu.window.request_redraw();
            }
            WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Middle, .. } => {
                gpu.zoom_locked = !gpu.zoom_locked;
                gpu.window.request_redraw();
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } if gpu.inside() && !gpu.hex => {
                // Inside, the left button works the selected tool (looking around is the locked cursor).
                let down = state == ElementState::Pressed;
                if down != gpu.sculpting {
                    if down {
                        gpu.sculpting = true;
                        gpu.stroke_start();
                    } else {
                        gpu.stroke_end();
                    }
                    gpu.window.request_redraw();
                }
            }
            WindowEvent::MouseInput { state, button: MouseButton::Left, .. } => {
                self.dragging = state == ElementState::Pressed;
                if !self.dragging {
                    self.last_cursor = None;
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if gpu.cursor_locked == Some(true) {
                    // Locked: the view turns with raw mouse motion (device_event); keep the cursor in the middle.
                    let s = gpu.window.inner_size();
                    let (cx, cy) = (s.width as f64 / 2.0, s.height as f64 / 2.0);
                    if (position.x - cx).abs() > 2.0 || (position.y - cy).abs() > 2.0 {
                        gpu.centre_cursor();
                    }
                    gpu.cursor = Some((cx, cy));
                    return;
                }
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
                        // No stop at the poles: the pitch just keeps going over the top (the globe turns upside
                        // down), and while it is upside down a sideways drag turns the other way round the axis,
                        // so the surface still follows the mouse.
                        let tau = std::f64::consts::TAU;
                        let flipped = gpu.camera.pitch.cos() < 0.0;
                        let side = if flipped { -1.0 } else { 1.0 };
                        gpu.camera.yaw = (gpu.camera.yaw + side * (position.x - last.x) * speed).rem_euclid(tau);
                        gpu.camera.pitch = (gpu.camera.pitch + (position.y - last.y) * speed + std::f64::consts::PI).rem_euclid(tau) - std::f64::consts::PI;
                        gpu.dirty = true;
                        gpu.window.request_redraw();
                    }
                    self.last_cursor = Some(position);
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if gpu.zoom_locked {
                    return;
                }
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
                        gpu.clamp_lift();
                        gpu.camera.distance = gpu.camera.floor + gpu.fly_height();
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
                    if let (Some(i), true) = (slot, gpu.inside()) {
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
                match &event.logical_key {
                    Key::Named(NamedKey::Space) => {
                        let down = event.state == ElementState::Pressed;
                        if down && !gpu.moving() {
                            gpu.last_frame = Instant::now();
                        }
                        gpu.space = down;
                        gpu.window.request_redraw();
                    }
                    Key::Named(NamedKey::Tab) if event.state == ElementState::Pressed && !event.repeat => {
                        gpu.free_default = !gpu.free_default;
                        gpu.window.request_redraw();
                    }
                    Key::Named(n @ (NamedKey::F1 | NamedKey::F2 | NamedKey::F3 | NamedKey::F4 | NamedKey::F5 | NamedKey::F6 | NamedKey::F7)) if event.state == ElementState::Pressed && !event.repeat => {
                        let i = match n { NamedKey::F1 => 0, NamedKey::F2 => 1, NamedKey::F3 => 2, NamedKey::F4 => 3, NamedKey::F5 => 4, NamedKey::F6 => 5, _ => 6 };
                        gpu.dbg ^= 1 << i;
                        if i == 2 {
                            // F3 changes how the meshes are made, so the made ones are thrown away.
                            global_map_maker::ground::BEACH_NEIGHBOUR_RULE.store(gpu.dbg & 4 == 0, std::sync::atomic::Ordering::Relaxed);
                            let (mut now, later): (Vec<PatchKey>, Vec<PatchKey>) = gpu.patch_cache.keys().copied().partition(|k| gpu.drawn_keys.contains(k));
                            now.extend(later);
                            gpu.streamer.rebuild(&now);
                            gpu.last_sig = None;
                        }
                        gpu.update_title();
                        gpu.dirty = true;
                        gpu.window.request_redraw();
                    }
                    Key::Character(c) if event.state == ElementState::Pressed && !event.repeat && matches!(c.as_str(), "t" | "T") => {
                        gpu.ribbon.visible = !gpu.ribbon.visible;
                        gpu.window.request_redraw();
                    }
                    _ => {}
                }
                if event.state == ElementState::Pressed && gpu.inside() {
                    let k = match event.logical_key {
                        Key::Named(NamedKey::PageUp) => Some(1.25),
                        Key::Named(NamedKey::PageDown) => Some(0.8),
                        _ => None,
                    };
                    if let Some(k) = k {
                        gpu.render_km = (gpu.render_km * k).clamp(RENDER_KM_MIN, RENDER_KM_MAX);
                        gpu.dirty = true;
                        gpu.window.request_redraw();
                    }
                }
                if event.state == ElementState::Pressed && !event.repeat && event.logical_key == Key::Named(NamedKey::Escape) {
                    gpu.leave_inside();
                    gpu.window.request_redraw();
                }
                if event.state == ElementState::Pressed && !event.repeat {
                    if let Key::Character(c) = &event.logical_key {
                        // Only G (ground switch) stays while the paint tools are put aside.
                        if !PAINT_ENABLED && !matches!(c.as_str(), "g" | "G") {
                            return;
                        }
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
                gpu.sync_cursor();
                if gpu.ribbon.apply_planet {
                    gpu.ribbon.apply_planet = false;
                    gpu.new_planet();
                }
                gpu.step();
                gpu.draw();
                if gpu.moving() || gpu.dirty || gpu.sculpting {
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
