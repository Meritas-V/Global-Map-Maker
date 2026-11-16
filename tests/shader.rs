//! Checks the shaders on the CPU, so a typo shows up in `cargo test` instead of at window-open time.
use naga::valid::{Capabilities, ValidationFlags, Validator};

fn check(src: &str) {
    let module = naga::front::wgsl::parse_str(src).expect("shader failed to parse");
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .expect("shader failed validation");
}

#[test]
fn globe_shader_is_valid() {
    check(include_str!("../src/shader.wgsl"));
}

#[test]
fn overlay_shader_is_valid() {
    check(include_str!("../src/overlay.wgsl"));
}

#[test]
fn pixel_shader_is_valid() {
    check(include_str!("../src/pixel.wgsl"));
}

#[test]
fn ground_shader_is_valid() {
    check(include_str!("../src/ground.wgsl"));
}
