use clap::ValueEnum;

const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
const GRAY: [f32; 4] = [0.5, 0.5, 0.5, 1.0];
const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
const HALF_RED: [f32; 4] = [0.25, 0.0, 0.0, 1.0];

#[derive(Debug, Clone, Copy, Default, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum ColorScheme {
    #[default]
    BlackAndWhite,
    Inverted,
    Newspaper,
}

impl ColorScheme {
    /// The basic color of a live cell.
    pub fn live_color(&self) -> [f32; 4] {
        match self {
            ColorScheme::Newspaper | ColorScheme::BlackAndWhite => BLACK,
            ColorScheme::Inverted => WHITE,
        }
    }

    /// The basic color of a dead cell.
    pub fn fallow_color(&self) -> [f32; 4] {
        match self {
            ColorScheme::Newspaper | ColorScheme::BlackAndWhite => WHITE,
            ColorScheme::Inverted => BLACK,
        }
    }

    /// The color of a cell that just became live.
    pub fn spawn_color(&self) -> [f32; 4] {
        self.live_color()
    }

    /// The color of a cell that just died.
    pub fn died_color(&self) -> [f32; 4] {
        match self {
            ColorScheme::Newspaper => HALF_RED,
            _ => self.fallow_color(),
        }
    }

    /// The color of a cell that has remained live.
    pub fn survivor_color(&self) -> [f32; 4] {
        GRAY
    }
}
