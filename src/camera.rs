//! The camera, in two modes.
//!
//! Orbit (outside): dragging changes yaw/pitch, scrolling changes distance, and the camera always
//! looks at the globe's center. Fly (inside): the camera sits at a point above the ground and
//! looks wherever `Fly` says, like a person flying over the land.
//!
//! All the math is f64. Everything is drawn "camera-relative": every vertex is stored as an
//! offset from a point right under the camera (computed in f64), so even a 1-meter feature on a
//! planet-sized globe keeps full precision when it reaches the f32 GPU.

use glam::{DMat4, DVec3, DVec4, Mat4, Vec3};

const FOV_Y_DEG: f64 = 45.0;

/// Looking around from a point above the ground (inside scene).
#[derive(Clone, Copy, Debug)]
pub struct Fly {
    /// Direction from the globe's center to the camera (unit vector): where you are.
    pub dir: DVec3,
    /// Compass heading in radians: 0 = north, pi/2 = east.
    pub heading: f64,
    /// Looking up (+) or down (-), radians.
    pub tilt: f64,
}

impl Fly {
    /// Local directions at this spot: (north, east, up).
    pub fn frame(&self) -> (DVec3, DVec3, DVec3) {
        let up = self.dir.normalize();
        let mut east = DVec3::Z.cross(up);
        if east.length() < 1e-9 {
            east = DVec3::X.cross(up);
        }
        let east = east.normalize();
        (up.cross(east), east, up)
    }

    /// Where the camera looks, as a unit vector in globe coordinates.
    pub fn forward(&self) -> DVec3 {
        let (n, e, up) = self.frame();
        let (sh, ch) = self.heading.sin_cos();
        let (st, ct) = self.tilt.sin_cos();
        ((n * ch + e * sh) * ct + up * st).normalize()
    }

    /// The sun, as a direction toward it in globe coordinates: fixed relative to the local compass, so shading
    /// stays the same while you turn and the land keeps its look as you fly.
    pub fn sun(&self) -> DVec3 {
        let (n, e, up) = self.frame();
        (up * 0.6 - e * 0.6 - n * 0.5).normalize()
    }
}

pub struct Camera {
    pub yaw: f64,
    pub pitch: f64,
    /// From the globe's center. The globe has radius 1.
    pub distance: f64,
    /// Radius of the ground right under the camera (1.0 on the bare sphere, higher over mountains),
    /// so height above the ground = distance - floor.
    pub floor: f64,
    /// Some = the inside scene (free flying); None = orbiting the globe from outside.
    pub fly: Option<Fly>,
}

/// Right-handed perspective with depth 0..1 (what wgpu expects), "reversed": depth is 1 at the near
/// plane and falls toward 0 far away, with no far plane at all. A float depth buffer is far more
/// precise this way, which matters when the near plane is millimeters and the horizon kilometers.
pub fn perspective(fov_y: f64, aspect: f64, near: f64) -> DMat4 {
    let f = 1.0 / (fov_y * 0.5).tan();
    DMat4::from_cols(
        DVec4::new(f / aspect, 0.0, 0.0, 0.0),
        DVec4::new(0.0, f, 0.0, 0.0),
        DVec4::new(0.0, 0.0, 0.0, -1.0),
        DVec4::new(0.0, 0.0, near, 0.0),
    )
}

/// View matrix for a camera at `eye` looking along `forward`, with `up` roughly upward (right-handed, looks down -z).
fn look_to(eye: DVec3, forward: DVec3, up: DVec3) -> DMat4 {
    let f = forward.normalize();
    let r = f.cross(up).normalize();
    let u = r.cross(f);
    DMat4::from_cols(
        DVec4::new(r.x, u.x, -f.x, 0.0),
        DVec4::new(r.y, u.y, -f.y, 0.0),
        DVec4::new(r.z, u.z, -f.z, 0.0),
        DVec4::new(-r.dot(eye), -u.dot(eye), f.dot(eye), 1.0),
    )
}

impl Camera {
    /// Height above the ground under the camera, in globe radii.
    pub fn height(&self) -> f64 {
        self.distance - self.floor
    }

    /// Orbit camera: globe coordinates -> where the globe sits in the scene.
    fn model64(&self) -> DMat4 {
        // Our sphere has north along +z, the screen has up along +y: tip it upright first.
        let upright = DMat4::from_rotation_x(-std::f64::consts::FRAC_PI_2);
        let spin = DMat4::from_rotation_x(self.pitch) * DMat4::from_rotation_y(self.yaw);
        spin * upright
    }

    /// Globe coordinates -> camera space (camera at the origin looking down -z), in f64.
    pub fn view64(&self) -> DMat4 {
        match &self.fly {
            None => DMat4::from_translation(DVec3::new(0.0, 0.0, -self.distance)) * self.model64(),
            Some(f) => look_to(self.eye64(), f.forward(), f.frame().2),
        }
    }

    /// Vertical field of view, radians.
    pub fn fov_y() -> f64 {
        FOV_Y_DEG.to_radians()
    }

    /// Globe coordinates -> screen clip space, in f64.
    pub fn matrix64(&self, aspect: f64) -> DMat4 {
        // Nothing is ever nearer than the ground (ground just ahead can be a bit higher than right below, hence the small factor), so the near plane follows the zoom.
        let near = (self.height() * 0.004).max(1e-10);
        perspective(FOV_Y_DEG.to_radians(), aspect, near) * self.view64()
    }

    /// The same, for f32 vertices in plain globe coordinates.
    pub fn matrix(&self, aspect: f32) -> Mat4 {
        self.matrix64(aspect as f64).as_mat4()
    }

    /// For camera-relative vertices: input is (globe position - origin), done in f64, handed over as f32.
    pub fn matrix_rel(&self, aspect: f64, origin: DVec3) -> Mat4 {
        (self.matrix64(aspect) * DMat4::from_translation(origin)).as_mat4()
    }

    /// Where the camera is, in the globe's own coordinates.
    pub fn eye64(&self) -> DVec3 {
        match &self.fly {
            None => self.model64().inverse().transform_point3(DVec3::new(0.0, 0.0, self.distance)),
            Some(f) => f.dir.normalize() * self.distance,
        }
    }

    pub fn eye(&self) -> Vec3 {
        self.eye64().as_vec3()
    }

    /// Direction the light comes from, in the globe's own coordinates.
    /// Orbit: fixed relative to the camera (up-left, over the viewer's shoulder), so facets stay readable as you spin.
    /// Fly: the sun, fixed relative to the local compass.
    pub fn light_dir(&self) -> Vec3 {
        match &self.fly {
            Some(f) => f.sun().as_vec3(),
            None => self.model64().inverse().transform_vector3(DVec3::new(-0.6, 0.5, 0.5).normalize()).as_vec3(),
        }
    }

    /// Orbit the globe so the camera sits over `dir` (what you were flying over).
    pub fn orbit_over(&mut self, dir: DVec3) {
        // Undo the model: rotate the point into the "upright" frame, then pick yaw and pitch that bring it to +z.
        let d = dir.normalize();
        let v = DVec3::new(d.x, d.z, -d.y);
        self.yaw = (-v.x).atan2(v.z);
        let rho = (v.x * v.x + v.z * v.z).sqrt();
        let limit = 89f64.to_radians();
        self.pitch = v.y.atan2(rho).clamp(-limit, limit);
    }

    /// Pixels per unit of size at distance 1 (how "zoomed" the lens is).
    pub fn focal_px(height_px: f64) -> f64 {
        (height_px * 0.5) / (FOV_Y_DEG.to_radians() * 0.5).tan()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn orbit(yaw: f64, pitch: f64) -> Camera {
        Camera { yaw, pitch, distance: 1.5, floor: 1.0, fly: None }
    }

    #[test]
    fn orbit_over_puts_the_camera_where_it_was() {
        for (yaw, pitch) in [(0.0, 0.0), (0.7, 0.3), (-2.0, -1.0), (3.0, 1.2), (-0.4, 0.9)] {
            let c = orbit(yaw, pitch);
            let dir = c.eye64().normalize();
            let mut back = orbit(0.1, 0.1);
            back.orbit_over(dir);
            let d2 = back.eye64().normalize();
            assert!((d2 - dir).length() < 1e-9, "yaw {yaw} pitch {pitch}: {dir:?} vs {d2:?}");
        }
    }

    #[test]
    fn fly_frame_is_a_compass() {
        let f = Fly { dir: DVec3::X, heading: 0.0, tilt: 0.0 };
        let (n, e, up) = f.frame();
        assert!((n - DVec3::Z).length() < 1e-12 && (e - DVec3::Y).length() < 1e-12 && (up - DVec3::X).length() < 1e-12);
        // Heading east, level: looking along +y.
        let f = Fly { heading: std::f64::consts::FRAC_PI_2, ..f };
        assert!((f.forward() - DVec3::Y).length() < 1e-12);
        // Straight down looks at the ground.
        let f = Fly { tilt: -std::f64::consts::FRAC_PI_2, ..f };
        assert!((f.forward() + DVec3::X).length() < 1e-12);
    }

    #[test]
    fn fly_camera_sees_what_is_ahead_and_not_what_is_behind() {
        let dir = DVec3::new(0.3, 0.5, 0.8).normalize();
        let c = Camera { yaw: 0.0, pitch: 0.0, distance: 1.001, floor: 1.0, fly: Some(Fly { dir, heading: 0.4, tilt: -0.2 }) };
        let f = c.fly.unwrap();
        let eye = c.eye64();
        let ahead = eye + f.forward() * 0.0005;
        let behind = eye - f.forward() * 0.0005;
        let a = c.view64().transform_point3(ahead);
        let b = c.view64().transform_point3(behind);
        assert!(a.z < 0.0 && a.x.abs() < 1e-9 && a.y.abs() < 1e-9, "{a:?}");
        assert!(b.z > 0.0, "{b:?}");
        // Camera sits at the origin of its own space.
        assert!(c.view64().transform_point3(eye).length() < 1e-9);
    }

    #[test]
    fn reversed_depth_is_one_at_the_near_plane_and_falls_with_distance() {
        let p = perspective(0.8, 1.5, 0.01);
        let depth = |z: f64| {
            let c = p * DVec4::new(0.0, 0.0, -z, 1.0);
            c.z / c.w
        };
        assert!((depth(0.01) - 1.0).abs() < 1e-12);
        assert!(depth(0.1) < depth(0.05) && depth(1e6) > 0.0 && depth(1e6) < 1e-6);
    }
}
