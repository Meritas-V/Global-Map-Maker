//! Orbit camera: dragging changes yaw/pitch, scrolling changes distance.

use crate::lod::View;
use glam::{Mat4, Vec3};

const FOV_Y_DEG: f32 = 45.0;

pub struct Camera {
    pub yaw: f32,
    pub pitch: f32,
    /// From the globe's center. The globe has radius 1, so height above the surface = distance - 1.
    pub distance: f32,
}

impl Camera {
    /// Globe coordinates -> where the globe sits in the scene.
    pub fn model(&self) -> Mat4 {
        // Our sphere has north along +z, the screen has up along +y: tip it upright first.
        let upright = Mat4::from_rotation_x(-std::f32::consts::FRAC_PI_2);
        let spin = Mat4::from_rotation_x(self.pitch) * Mat4::from_rotation_y(self.yaw);
        spin * upright
    }

    /// Globe coordinates -> screen clip space, in one matrix.
    pub fn matrix(&self, aspect: f32) -> Mat4 {
        let view = Mat4::from_translation(Vec3::new(0.0, 0.0, -self.distance));
        // Near plane has to be closer than the smallest height we allow zooming to.
        let proj = glam::camera::rh::proj::directx::perspective(FOV_Y_DEG.to_radians(), aspect, 0.001, 20.0);
        proj * view * self.model()
    }

    /// Where the camera is, in the globe's own coordinates.
    pub fn eye(&self) -> Vec3 {
        self.model().inverse().transform_point3(Vec3::new(0.0, 0.0, self.distance))
    }

    /// What the level-of-detail code needs to know.
    pub fn view(&self, width_px: u32, height_px: u32) -> View {
        let (w, h) = (width_px.max(1) as f32, height_px.max(1) as f32);
        View {
            eye: self.eye(),
            mvp: self.matrix(w / h),
            focal_px: (h * 0.5) / (FOV_Y_DEG.to_radians() * 0.5).tan(),
            viewport: (w, h),
        }
    }
}
