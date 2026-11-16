//! The heads-up display in the corner: a distance scale bar and a ladder showing how deep
//! the zoom is. Drawn in screen pixels with a tiny built-in pixel font (no font files needed).
//! Pure geometry, so it can be tested without a window.

use bytemuck::{Pod, Zeroable};

/// One corner of an on-screen rectangle: position in clip space (-1..1) and color.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
pub struct OverlayVertex {
    pub pos: [f32; 2],
    pub color: [f32; 3],
}

/// The globe has radius 1; for the scale bar we pretend it is Earth-sized.
pub const EARTH_RADIUS_M: f64 = 6_371_000.0;

/// Ground distance covered by one screen pixel, under the middle of the screen.
pub fn meters_per_pixel(height_above_surface: f64, focal_px: f64) -> f64 {
    height_above_surface * EARTH_RADIUS_M / focal_px
}

/// A round length (1, 2 or 5 times a power of ten, in meters) whose bar is at most `target_px`
/// wide. Returns (meters, bar width in pixels).
pub fn nice_length(m_per_px: f64, target_px: f64) -> (f64, f64) {
    let raw = (m_per_px * target_px).max(1e-9);
    let base = 10f64.powf(raw.log10().floor());
    let meters = [5.0, 2.0, 1.0].iter().map(|k| k * base).find(|m| *m <= raw * 1.0000001).unwrap_or(base);
    (meters, meters / m_per_px)
}

/// "5 km", "200 m", "20 cm", "2 mm".
pub fn format_length(meters: f64) -> String {
    let (value, unit) = if meters >= 1000.0 {
        (meters / 1000.0, "km")
    } else if meters >= 1.0 {
        (meters, "m")
    } else if meters >= 0.01 {
        (meters * 100.0, "cm")
    } else {
        (meters * 1000.0, "mm")
    };
    format!("{} {}", value.round() as u64, unit)
}

/// 5 x 7 pixel glyphs; each row is 5 bits, leftmost pixel = highest bit.
fn glyph(c: char) -> Option<[u8; 7]> {
    Some(match c {
        '0' => [0b01110, 0b10001, 0b10011, 0b10101, 0b11001, 0b10001, 0b01110],
        '1' => [0b00100, 0b01100, 0b00100, 0b00100, 0b00100, 0b00100, 0b01110],
        '2' => [0b01110, 0b10001, 0b00001, 0b00010, 0b00100, 0b01000, 0b11111],
        '3' => [0b11110, 0b00001, 0b00001, 0b01110, 0b00001, 0b00001, 0b11110],
        '4' => [0b00010, 0b00110, 0b01010, 0b10010, 0b11111, 0b00010, 0b00010],
        '5' => [0b11111, 0b10000, 0b11110, 0b00001, 0b00001, 0b10001, 0b01110],
        '6' => [0b00110, 0b01000, 0b10000, 0b11110, 0b10001, 0b10001, 0b01110],
        '7' => [0b11111, 0b00001, 0b00010, 0b00100, 0b01000, 0b01000, 0b01000],
        '8' => [0b01110, 0b10001, 0b10001, 0b01110, 0b10001, 0b10001, 0b01110],
        '9' => [0b01110, 0b10001, 0b10001, 0b01111, 0b00001, 0b00010, 0b01100],
        'k' => [0b10000, 0b10000, 0b10010, 0b10100, 0b11000, 0b10100, 0b10010],
        'm' => [0b00000, 0b00000, 0b11010, 0b10101, 0b10101, 0b10101, 0b10101],
        'c' => [0b00000, 0b00000, 0b01110, 0b10000, 0b10000, 0b10000, 0b01110],
        'L' => [0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b10000, 0b11111],
        'V' => [0b10001, 0b10001, 0b10001, 0b10001, 0b01010, 0b01010, 0b00100],
        '/' => [0b00001, 0b00001, 0b00010, 0b00100, 0b01000, 0b10000, 0b10000],
        ' ' => [0; 7],
        _ => return None,
    })
}

struct Canvas {
    w: f32,
    h: f32,
    out: Vec<OverlayVertex>,
}

impl Canvas {
    /// A filled rectangle; x, y from the top-left corner, in pixels.
    fn rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: [f32; 3]) {
        let to_clip = |px: f32, py: f32| [px / self.w * 2.0 - 1.0, 1.0 - py / self.h * 2.0];
        let (a, b, c, d) = (to_clip(x, y), to_clip(x + w, y), to_clip(x + w, y + h), to_clip(x, y + h));
        for pos in [a, b, c, a, c, d] {
            self.out.push(OverlayVertex { pos, color });
        }
    }

    /// Text with each font pixel drawn `scale` pixels big. Returns the width used.
    fn text(&mut self, s: &str, x: f32, y: f32, scale: f32, color: [f32; 3]) -> f32 {
        let mut cx = x;
        for ch in s.chars() {
            if let Some(rows) = glyph(ch) {
                for (ry, row) in rows.iter().enumerate() {
                    for rx in 0..5 {
                        if row & (0b10000 >> rx) != 0 {
                            self.rect(cx + rx as f32 * scale, y + ry as f32 * scale, scale, scale, color);
                        }
                    }
                }
            }
            cx += 6.0 * scale;
        }
        cx - x
    }
}

pub fn text_width(s: &str, scale: f32) -> f32 {
    s.chars().count() as f32 * 6.0 * scale
}

const PANEL: [f32; 3] = [0.04, 0.05, 0.08];
const WHITE: [f32; 3] = [0.95, 0.96, 1.0];
const ON: [f32; 3] = [0.30, 0.78, 0.70];
const OFF: [f32; 3] = [0.17, 0.21, 0.27];

/// Everything for the bottom-left corner: scale bar with its length, and the zoom-depth ladder.
/// `res` is the current hex level, `max_res` the finest one.
pub fn build_overlay(width: u32, height: u32, m_per_px: f64, res: u8, max_res: u8) -> Vec<OverlayVertex> {
    let mut c = Canvas { w: width.max(1) as f32, h: height.max(1) as f32, out: Vec::new() };
    let margin = 12.0;
    let pad = 10.0;
    let step = 14.0; // ladder square pitch
    let ladder_w = (max_res as f32 + 1.0) * step - 4.0;
    let (meters, bar_px) = nice_length(m_per_px, 160.0);
    let bar_px = bar_px as f32;
    let inner_w = ladder_w.max(bar_px).max(100.0);
    let panel_w = inner_w + 2.0 * pad;
    let panel_h = 92.0;
    let (px0, py0) = (margin, c.h - margin - panel_h);
    c.rect(px0, py0, panel_w, panel_h, PANEL);

    let x0 = px0 + pad;
    // Row 1: the length of the bar (left) and the level (right).
    let label = format_length(meters);
    c.text(&label, x0, py0 + pad, 2.0, WHITE);
    let level = format!("LV {}/{}", res, max_res);
    let lw = text_width(&level, 2.0) - 2.0 * 1.0;
    c.text(&level, x0 + inner_w - lw, py0 + pad, 2.0, ON);
    // Row 2: the bar, with end ticks.
    let by = py0 + pad + 14.0 + 12.0;
    c.rect(x0, by, bar_px, 4.0, WHITE);
    c.rect(x0, by - 5.0, 3.0, 14.0, WHITE);
    c.rect(x0 + bar_px - 3.0, by - 5.0, 3.0, 14.0, WHITE);
    // Row 3: the ladder, one square per level; filled up to where we are.
    let ly = by + 20.0;
    for i in 0..=max_res {
        let color = if i == res { WHITE } else if i < res { ON } else { OFF };
        let h = if i == res { 14.0 } else { 10.0 };
        c.rect(x0 + i as f32 * step, ly + (14.0 - h), 10.0, h, color);
    }
    c.out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nice_lengths_are_1_2_or_5_and_fit() {
        for mpp in [0.00001, 0.02, 0.3, 1.0, 7.7, 120.0, 10_700.0, 25_000.0] {
            let (m, px) = nice_length(mpp, 160.0);
            assert!(px <= 160.0 + 1e-6 && px >= 160.0 / 2.6, "{mpp}: {px}");
            let lead = m / 10f64.powf(m.log10().floor());
            assert!([1.0, 2.0, 5.0].iter().any(|k| (lead - k).abs() < 1e-6), "{m} is not 1/2/5 x 10^n");
        }
    }

    #[test]
    fn lengths_read_naturally() {
        assert_eq!(format_length(5000.0), "5 km");
        assert_eq!(format_length(200.0), "200 m");
        assert_eq!(format_length(1.0), "1 m");
        assert_eq!(format_length(0.2), "20 cm");
        assert_eq!(format_length(0.002), "2 mm");
        assert_eq!(format_length(1_000_000.0), "1000 km");
    }

    #[test]
    fn every_character_we_print_has_a_glyph() {
        for ch in "0123456789kmcLV/ ".chars() {
            assert!(glyph(ch).is_some(), "{ch}");
        }
        for m in [0.002, 0.2, 1.0, 5.0, 200.0, 5000.0, 2_000_000.0] {
            for ch in format_length(m).chars() {
                assert!(glyph(ch).is_some(), "{ch} in {}", format_length(m));
            }
        }
        for res in 0..=15 {
            for ch in format!("LV {res}/15").chars() {
                assert!(glyph(ch).is_some(), "{ch}");
            }
        }
    }

    #[test]
    fn the_hud_fits_on_a_normal_window_and_stays_in_the_bottom_left() {
        for (w, h) in [(1280u32, 720u32), (1920, 1080), (800, 600)] {
            let v = build_overlay(w, h, 12.0, 9, 15);
            assert!(!v.is_empty() && v.len() % 3 == 0);
            for p in &v {
                assert!(p.pos[0] >= -1.0 && p.pos[0] <= 1.0 && p.pos[1] >= -1.0 && p.pos[1] <= 1.0, "{p:?}");
                assert!(p.pos[0] < 0.0 || w < 800, "should stay on the left side of the window");
                assert!(p.pos[1] < 0.0 || h < 400, "should stay in the lower half of the window");
            }
        }
    }

    #[test]
    fn deeper_zoom_lights_more_of_the_ladder() {
        let count = |res: u8| {
            build_overlay(1280, 720, 12.0, res, 15).iter().filter(|v| v.color == ON).count()
        };
        // Text also uses ON for the level, so compare the growth, not the exact value.
        assert!(count(12) > count(3));
    }

    #[test]
    fn scale_follows_height_and_focal_length() {
        // 1 pixel at 1 unit height with focal 1000 px covers 1/1000 of the radius.
        let m = meters_per_pixel(1.0, 1000.0);
        assert!((m - 6371.0).abs() < 1e-6);
        assert!(meters_per_pixel(1e-5, 1303.0) < 0.1, "1 m hexes should read as centimeters per pixel");
    }
}
