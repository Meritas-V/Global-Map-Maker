//! The tool panel: a narrow strip of terraforming tools down the left of the window; ">>" opens a bigger window
//! beside it with the brush, a guide to every tool, and the keys.
//!
//! Only the interface lives here (egui); it knows nothing about the GPU or the terrain, so the tool list can be
//! tested without a window. The tools are not wired to the terrain yet: choosing one only selects it.

use crate::planet::{Biome, BiomeMode, PlanetSettings};
use egui::{Button, RichText, Ui};

/// Everything the ribbon can switch on. One tab holds a handful of related tools.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Tab {
    Sculpt,
    Landforms,
    Water,
    Dig,
}

impl Tab {
    pub const ALL: [Tab; 4] = [Tab::Sculpt, Tab::Landforms, Tab::Water, Tab::Dig];

    pub fn name(self) -> &'static str {
        match self {
            Tab::Sculpt => "Sculpt",
            Tab::Landforms => "Landforms",
            Tab::Water => "Water",
            Tab::Dig => "Dig",
        }
    }
}

/// One terraforming tool.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum TerraTool {
    Raise,
    Lower,
    Smooth,
    Flatten,
    Roughen,
    Plateau,
    Hill,
    Mountain,
    Crater,
    Valley,
    River,
    Lake,
    SeaLevel,
    Pit,
    Trench,
    Tunnel,
}

impl TerraTool {
    pub const ALL: [TerraTool; 16] = [
        TerraTool::Raise,
        TerraTool::Lower,
        TerraTool::Smooth,
        TerraTool::Flatten,
        TerraTool::Roughen,
        TerraTool::Plateau,
        TerraTool::Hill,
        TerraTool::Mountain,
        TerraTool::Crater,
        TerraTool::Valley,
        TerraTool::River,
        TerraTool::Lake,
        TerraTool::SeaLevel,
        TerraTool::Pit,
        TerraTool::Trench,
        TerraTool::Tunnel,
    ];

    pub fn name(self) -> &'static str {
        match self {
            TerraTool::Raise => "Raise",
            TerraTool::Lower => "Lower",
            TerraTool::Smooth => "Smooth",
            TerraTool::Flatten => "Flatten",
            TerraTool::Roughen => "Roughen",
            TerraTool::Plateau => "Plateau",
            TerraTool::Hill => "Hill",
            TerraTool::Mountain => "Mountain",
            TerraTool::Crater => "Crater",
            TerraTool::Valley => "Valley",
            TerraTool::River => "River",
            TerraTool::Lake => "Lake",
            TerraTool::SeaLevel => "Sea level",
            TerraTool::Pit => "Pit",
            TerraTool::Trench => "Trench",
            TerraTool::Tunnel => "Tunnel",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            TerraTool::Raise => "Push the ground up under the brush.",
            TerraTool::Lower => "Press the ground down under the brush.",
            TerraTool::Smooth => "Even out bumps and sharp edges.",
            TerraTool::Flatten => "Level the ground to the height where you started.",
            TerraTool::Roughen => "Add small bumps and crags.",
            TerraTool::Plateau => "Raise a flat-topped block with steep sides.",
            TerraTool::Hill => "Stamp a soft rounded hill.",
            TerraTool::Mountain => "Stamp a mountain with ridges and a peak.",
            TerraTool::Crater => "Stamp a bowl with a raised rim.",
            TerraTool::Valley => "Carve a V-shaped valley along a line.",
            TerraTool::River => "Cut a riverbed along a line and fill it with water.",
            TerraTool::Lake => "Dig a basin and fill it to a flat level.",
            TerraTool::SeaLevel => "Choose the height where land turns into sea.",
            TerraTool::Pit => "Dig straight down.",
            TerraTool::Trench => "Dig a long, narrow cut along a line.",
            TerraTool::Tunnel => "Bore a cave or tunnel into the ground.",
        }
    }

    pub fn tab(self) -> Tab {
        match self {
            TerraTool::Raise | TerraTool::Lower | TerraTool::Smooth | TerraTool::Flatten | TerraTool::Roughen => Tab::Sculpt,
            TerraTool::Plateau | TerraTool::Hill | TerraTool::Mountain | TerraTool::Crater | TerraTool::Valley => Tab::Landforms,
            TerraTool::River | TerraTool::Lake | TerraTool::SeaLevel => Tab::Water,
            TerraTool::Pit | TerraTool::Trench | TerraTool::Tunnel => Tab::Dig,
        }
    }

    /// Whether the tool actually changes the ground yet.
    pub fn works(self) -> bool {
        matches!(self, TerraTool::Raise | TerraTool::Lower | TerraTool::Smooth | TerraTool::Flatten)
    }
}

/// How strongly the brush acts toward its edge.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Falloff {
    Soft,
    Linear,
    Hard,
}

impl Falloff {
    pub const ALL: [Falloff; 3] = [Falloff::Soft, Falloff::Linear, Falloff::Hard];

    pub fn name(self) -> &'static str {
        match self {
            Falloff::Soft => "Soft",
            Falloff::Linear => "Linear",
            Falloff::Hard => "Hard",
        }
    }
}

/// Brush settings shared by all tools.
#[derive(Clone, Copy, Debug)]
pub struct Brush {
    /// Radius as a fraction of the camera's height above the ground: the brush covers the same part of the
    /// screen at any zoom (and so a smaller patch of ground the closer you are).
    pub rel: f64,
    /// 0..1, how hard the tool pushes.
    pub strength: f32,
    pub falloff: Falloff,
}

impl Default for Brush {
    fn default() -> Self {
        Brush { rel: 0.2, strength: 0.5, falloff: Falloff::Soft }
    }
}

pub const RADIUS_MIN_M: f64 = 1.0;
pub const RADIUS_MAX_M: f64 = 200_000.0;

/// What the big window is showing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum View {
    /// Brush settings and the selected tool.
    Brush,
    /// Every tool with what it does.
    Guide,
    /// Keys and state.
    Keys,
    /// The planet's settings (seed, sea level, biomes ...).
    Planet,
}

impl View {
    pub const ALL: [View; 4] = [View::Brush, View::Guide, View::Keys, View::Planet];

    pub fn name(self) -> &'static str {
        match self {
            View::Brush => "Brush",
            View::Guide => "Guide",
            View::Keys => "Keys",
            View::Planet => "Planet",
        }
    }
}

/// Width of the tool strip's buttons, points.
const STRIP_W: f32 = 58.0;
/// Width of the big window's content.
const BIG_W: f32 = 250.0;

/// What the panel remembers between frames.
pub struct Ribbon {
    /// The tab of the selected tool (the strip groups tools by tab).
    pub tab: Tab,
    pub tool: TerraTool,
    pub brush: Brush,
    pub view: View,
    /// Camera height above the ground, meters (set by the app every frame): the brush radius follows it.
    pub height_m: f64,
    /// The brush's outline on the ground, projected to the screen (points; None = behind the camera). Set by the app.
    pub ring: Vec<Option<egui::Pos2>>,
    /// The planet settings being edited (made real by "Make this planet").
    pub planet: PlanetSettings,
    /// Set by the Planet view's button; the app takes it and builds the planet.
    pub apply_planet: bool,
    /// The strip is shown (T toggles it).
    pub visible: bool,
    /// The big window next to the strip is open (the ">>" button).
    pub open: bool,
    /// Shown in the Keys view; set by the app every frame.
    pub inside: bool,
    pub cursor_free: bool,
    pub zoom_locked: bool,
}

impl Default for Ribbon {
    fn default() -> Self {
        Ribbon {
            tab: Tab::Sculpt,
            tool: TerraTool::Raise,
            brush: Brush::default(),
            view: View::Brush,
            height_m: 1000.0,
            planet: PlanetSettings::default(),
            apply_planet: false,
            ring: Vec::new(),
            visible: true,
            open: false,
            inside: false,
            cursor_free: true,
            zoom_locked: false,
        }
    }
}

/// Short text for a strip button.
fn label(tool: TerraTool) -> &'static str {
    match tool {
        TerraTool::SeaLevel => "Sea lvl",
        t => t.name(),
    }
}

impl Ribbon {
    /// Whether the selected tool changes the ground (so the brush outline is shown).
    pub fn visible_tool_works(&self) -> bool {
        self.tool.works()
    }

    /// Brush radius on the ground right now, meters.
    pub fn radius_m(&self) -> f64 {
        (self.brush.rel * self.height_m).clamp(RADIUS_MIN_M, RADIUS_MAX_M)
    }

    fn pick(&mut self, tool: TerraTool) {
        self.tool = tool;
        self.tab = tool.tab();
    }

    /// Draws the narrow tool strip down the left side, and the big window beside it when it is open.
    pub fn show(&mut self, ui: &mut Ui) {
        if !self.visible {
            return;
        }
        let ctx = ui.ctx().clone();
        egui::Window::new("tool strip")
            .title_bar(false)
            .anchor(egui::Align2::LEFT_CENTER, [8.0, 0.0])
            .resizable(false)
            .show(&ctx, |ui| {
                ui.set_width(STRIP_W);
                ui.spacing_mut().item_spacing.y = 3.0;
                let mut last_tab = None;
                for tool in TerraTool::ALL {
                    if last_tab.is_some() && last_tab != Some(tool.tab()) {
                        ui.add_space(4.0);
                        ui.separator();
                        ui.add_space(1.0);
                    }
                    last_tab = Some(tool.tab());
                    let b = Button::new(RichText::new(label(tool)).size(11.0)).selected(self.tool == tool);
                    if ui.add_sized([STRIP_W, 24.0], b).on_hover_text(format!("{}: {}", tool.name(), tool.hint())).clicked() {
                        self.pick(tool);
                    }
                }
                ui.add_space(4.0);
                ui.separator();
                let more = Button::new(RichText::new(if self.open { "<<" } else { ">>" }).size(15.0).strong());
                if ui.add_sized([STRIP_W, 26.0], more).on_hover_text("Brush, guide and keys").clicked() {
                    self.open = !self.open;
                }
            });
        if self.open {
            egui::Window::new("Terraform")
                .anchor(egui::Align2::LEFT_CENTER, [8.0 + STRIP_W + 30.0, 0.0])
                .resizable(false)
                .collapsible(false)
                .show(&ctx, |ui| {
                    ui.set_width(BIG_W);
                    ui.horizontal(|ui| {
                        for v in View::ALL {
                            ui.selectable_value(&mut self.view, v, v.name());
                        }
                    });
                    ui.separator();
                    match self.view {
                        View::Brush => self.brush_view(ui),
                        View::Guide => self.guide_view(ui),
                        View::Keys => self.keys_view(ui),
                        View::Planet => self.planet_view(ui),
                    }
                });
        }
    }

    /// The brush outline on the ground: a dark line under a white one, broken where it goes behind the camera.
    fn draw_ring(&self, ctx: &egui::Context) {
        if self.ring.len() < 3 {
            return;
        }
        let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::NULL));
        let mut run: Vec<egui::Pos2> = Vec::new();
        let mut runs: Vec<Vec<egui::Pos2>> = Vec::new();
        for p in self.ring.iter().chain(self.ring.first()) {
            match p {
                Some(p) => run.push(*p),
                None => runs.push(std::mem::take(&mut run)),
            }
        }
        runs.push(run);
        for run in runs.into_iter().filter(|r| r.len() > 1) {
            painter.add(egui::Shape::line(run.clone(), egui::Stroke::new(3.5, egui::Color32::from_black_alpha(150))));
            painter.add(egui::Shape::line(run, egui::Stroke::new(1.5, egui::Color32::from_white_alpha(230))));
        }
    }

    /// A small + in the middle of the screen while the cursor is locked to the camera (that is where tools aim).
    pub fn crosshair(&self, ui: &Ui) {
        let ctx = ui.ctx();
        self.draw_ring(ctx);
        if !self.inside || self.cursor_free {
            return;
        }
        let c = ctx.content_rect().center();
        let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::NULL));
        let r = 9.0;
        // A dark outline under a white line, so it shows on snow and on sea alike.
        for (w, col) in [(3.5, egui::Color32::from_black_alpha(170)), (1.5, egui::Color32::WHITE)] {
            let stroke = egui::Stroke::new(w, col);
            painter.line_segment([c - egui::vec2(r, 0.0), c + egui::vec2(r, 0.0)], stroke);
            painter.line_segment([c - egui::vec2(0.0, r), c + egui::vec2(0.0, r)], stroke);
        }
    }

    fn brush_view(&mut self, ui: &mut Ui) {
        ui.label(RichText::new(self.tool.name()).strong());
        ui.label(self.tool.hint());
        if !self.tool.works() {
            ui.label(RichText::new("Not wired to the terrain yet.").weak().italics());
        }
        ui.separator();
        let mut r = self.radius_m();
        if ui.add(egui::Slider::new(&mut r, RADIUS_MIN_M..=RADIUS_MAX_M).logarithmic(true).text("radius (m)")).changed() {
            self.brush.rel = r / self.height_m.max(1e-3);
        }
        ui.label(RichText::new("The radius follows your height.").small().weak());
        ui.add(egui::Slider::new(&mut self.brush.strength, 0.0..=1.0).text("strength"));
        ui.horizontal(|ui| {
            ui.label("falloff");
            for f in Falloff::ALL {
                ui.selectable_value(&mut self.brush.falloff, f, f.name());
            }
        });
    }

    /// Every tool, grouped by tab, with its one-line description. Clicking a name selects it.
    fn guide_view(&mut self, ui: &mut Ui) {
        egui::ScrollArea::vertical().max_height(380.0).show(ui, |ui| {
            for tab in Tab::ALL {
                ui.label(RichText::new(tab.name()).strong());
                for tool in TerraTool::ALL.iter().copied().filter(|t| t.tab() == tab) {
                    ui.horizontal_wrapped(|ui| {
                        if ui.selectable_label(self.tool == tool, tool.name()).clicked() {
                            self.pick(tool);
                        }
                        ui.label(RichText::new(tool.hint()).small().weak());
                    });
                }
                ui.add_space(4.0);
            }
        });
    }

    fn planet_view(&mut self, ui: &mut Ui) {
        let p = &mut self.planet;
        ui.horizontal(|ui| {
            ui.label("seed");
            ui.add(egui::DragValue::new(&mut p.seed).speed(1.0));
            if ui.small_button("random").clicked() {
                // Not random in the cryptographic sense: the clock is enough for a new world.
                p.seed = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(1, |d| d.as_nanos() as u64) % 1_000_000;
            }
        });
        ui.add(egui::Slider::new(&mut p.sea_level_m, -2000.0..=9000.0).text("sea level (m)"));
        ui.add(egui::Slider::new(&mut p.land, 0.0..=1.0).text("land"));
        ui.add(egui::Slider::new(&mut p.mountains, 0.0..=2.0).text("mountains"));
        ui.add(egui::Slider::new(&mut p.climate, 0.0..=1.0).text("climate (cold - hot)"));
        ui.add(egui::Slider::new(&mut p.wetness, 0.0..=1.0).text("wetness"));
        ui.separator();
        ui.label(RichText::new("Biomes").small().weak());
        let mode_name = match p.biomes {
            BiomeMode::Earthlike => "Earth-like".to_string(),
            BiomeMode::Patchwork => "Patchwork".to_string(),
            BiomeMode::Single(b) => format!("Only {}", b.name().to_lowercase()),
        };
        egui::ComboBox::from_id_salt("biome mode").selected_text(mode_name).show_ui(ui, |ui| {
            ui.selectable_value(&mut p.biomes, BiomeMode::Earthlike, "Earth-like");
            ui.selectable_value(&mut p.biomes, BiomeMode::Patchwork, "Patchwork");
            for b in Biome::ALL {
                ui.selectable_value(&mut p.biomes, BiomeMode::Single(b), format!("Only {}", b.name().to_lowercase()));
            }
        });
        ui.separator();
        ui.add_enabled(false, egui::Slider::new(&mut p.cave_amount, 0.0..=100.0).text("caves (not yet)"));
        ui.separator();
        if ui.button("Make this planet").clicked() {
            self.apply_planet = true;
        }
        ui.label(RichText::new("Builds a new planet. Edits to the current one are lost.").small().weak());
    }

    fn keys_view(&mut self, ui: &mut Ui) {
        let cursor = if !self.inside {
            "free (globe)"
        } else if self.cursor_free {
            "free"
        } else {
            "locked to the camera"
        };
        ui.label(format!("Cursor: {cursor}"));
        ui.label(format!("Zoom: {}", if self.zoom_locked { "LOCKED" } else { "free" }));
        ui.separator();
        for line in HELP {
            ui.label(RichText::new(*line).weak());
        }
    }
}

/// The key help shown in the Keys view.
const HELP: &[&str] = &[
    "Left mouse: use the tool",
    "Tab: free / lock cursor",
    "Alt (hold): flip it",
    "WASD: move, Shift: sprint",
    "Space: up, Ctrl: down",
    "Scroll: zoom",
    "Middle click: lock zoom",
    "T: hide the tool strip",
    "Esc: leave the land",
    "F1-F7: debug: switch a ground layer off",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tool_is_in_exactly_one_tab_and_every_tab_has_tools() {
        for tab in Tab::ALL {
            assert!(TerraTool::ALL.iter().any(|t| t.tab() == tab), "{} is empty", tab.name());
        }
        let mut seen = std::collections::HashSet::new();
        for t in TerraTool::ALL {
            assert!(seen.insert(t), "{t:?} listed twice");
        }
        assert_eq!(seen.len(), TerraTool::ALL.len());
    }

    #[test]
    fn tool_names_are_unique_and_have_hints() {
        let mut names = std::collections::HashSet::new();
        for t in TerraTool::ALL {
            assert!(names.insert(t.name()), "duplicate name {}", t.name());
            assert!(!t.hint().is_empty());
        }
    }

    #[test]
    fn default_ribbon_starts_on_a_tool_of_its_own_tab() {
        let r = Ribbon::default();
        assert_eq!(r.tool.tab(), r.tab);
        assert!(r.radius_m() >= RADIUS_MIN_M && r.radius_m() <= RADIUS_MAX_M);
    }

    #[test]
    fn the_ribbon_draws_without_a_window() {
        // Run one egui frame headless: the ribbon must lay out and produce shapes.
        let ctx = egui::Context::default();
        let mut ribbon = Ribbon::default();
        let out = ctx.run_ui(egui::RawInput::default(), |ui| ribbon.show(ui));
        assert!(!out.shapes.is_empty());
    }
}
