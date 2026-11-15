//! Checks the shader on the CPU, so a typo shows up in `cargo test` instead of at window-open time.
use naga::valid::{Capabilities, ValidationFlags, Validator};

#[test]
fn shader_is_valid() {
    let src = include_str!("../src/shader.wgsl");
    let module = naga::front::wgsl::parse_str(src).expect("shader failed to parse");
    Validator::new(ValidationFlags::all(), Capabilities::all())
        .validate(&module)
        .expect("shader failed validation");
}
