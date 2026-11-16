//! The one point format the globe shader takes.

/// One point sent to the GPU: where it is (relative to the camera-side origin) and what color it is.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pub pos: [f32; 3],
    pub color: [f32; 3],
}
