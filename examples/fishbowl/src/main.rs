use burn::backend::Wgpu;
use burn::prelude::{Backend, s};
use clap::{Parser, ValueEnum};
use conway::{Conway, ConwayConfig};
use glutin_window::GlutinWindow as Window;
use opengl_graphics::{GlGraphics, OpenGL};
use piston::event_loop::{EventSettings, Events};
use piston::input::{RenderArgs, RenderEvent, UpdateArgs, UpdateEvent};
use piston::window::WindowSettings;
use piston::{EventLoop, OpenGLWindow};

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

pub struct App<B: Backend> {
    gl: GlGraphics, // OpenGL drawing backend.
    conway: Conway<B>,
    update_noise: f64,
    color_scheme: ColorScheme,
}

impl<B: Backend> App<B> {
    fn render(
        &mut self,
        args: &RenderArgs,
    ) {
        use graphics::*;

        let fallow_color = self.color_scheme.fallow_color();
        let spawn_color = self.color_scheme.spawn_color();
        let died_color = self.color_scheme.died_color();
        let survivor_color = self.color_scheme.survivor_color();

        let state_data = self.conway.read_slice(s![.., ..]);
        let previous_data = self.conway.read_previous_slice(s![.., ..]);

        let [h, w] = self.conway.shape();
        let [win_w, win_h] = args.viewport().draw_size;
        let draw_scale = [(win_w as f64) / (w as f64), (win_h as f64) / (h as f64)];

        // TODD: this should all be a one-step Image::draw().

        self.gl.draw(args.viewport(), |c, gl| {
            for (w_idx, col) in state_data.iter().enumerate() {
                for (h_idx, is_live) in col.iter().enumerate() {
                    let is_live = *is_live;

                    let was_live = previous_data
                        .as_ref()
                        .map(|prev| prev[w_idx][h_idx])
                        .unwrap_or(false);

                    let color = match (was_live, is_live) {
                        (false, false) => fallow_color,
                        (false, true) => spawn_color,
                        (true, false) => died_color,
                        (true, true) => survivor_color,
                    };

                    let pos = [0., 0., draw_scale[0], draw_scale[1]];

                    let transform = c
                        .transform
                        .trans(w_idx as f64 * draw_scale[0], h_idx as f64 * draw_scale[1]);

                    Rectangle::new(color).draw(pos, &c.draw_state, transform, gl);
                }
            }
        });
    }

    fn update(
        &mut self,
        _args: &UpdateArgs,
    ) {
        self.conway.fuzz(self.update_noise);
        self.conway.wrap();
        self.conway.step()
    }
}

/// Conway's Game of Life demo for Burn.
#[derive(Parser, Debug)]
#[command(long_about = None)]
pub struct Args {
    /// The width and height of the grid.
    #[arg(long, default_value_t = 200)]
    pub grid_size: usize,

    /// The initial density of the grid.
    #[arg(long, default_value_t = 0.2)]
    pub initial_density: f64,

    /// The noise to apply to the grid.
    #[arg(long, default_value_t = 0.001)]
    pub update_noise: f64,

    /// The number of steps to target per second.
    #[arg(long, default_value_t = 30)]
    pub fps: u64,

    /// The initial window zoom.
    #[arg(long, default_value_t = 2.5)]
    pub zoom: f64,

    /// The color scheme to use.
    #[arg(long, default_value = "inverted")]
    pub color_scheme: ColorScheme,
}

fn main() {
    let args = Args::parse();

    run::<Wgpu>(&args);
}

fn run<B: Backend>(args: &Args) {
    println!("Args: {:?}", args);

    let device = Default::default();

    let mut conway: Conway<B> = ConwayConfig::new([args.grid_size, args.grid_size]).init(&device);
    conway.fuzz(args.initial_density);
    conway.wrap();

    // Change this to OpenGL::V2_1 if not working.
    let opengl = OpenGL::V3_2;

    // Create a Glutin window.
    let gs = (args.grid_size as f64 * args.zoom) as u32;
    let mut window: Window = WindowSettings::new("life", [gs, gs])
        .graphics_api(opengl)
        .exit_on_esc(true)
        .build()
        .unwrap();

    // Load the OpenGL function pointers
    gl::load_with(|s| window.get_proc_address(s) as *const _);

    // Create a new game and run it.
    let mut app = App {
        gl: GlGraphics::new(opengl),
        conway,
        update_noise: args.update_noise,
        color_scheme: args.color_scheme,
    };

    let mut events = Events::new(EventSettings::new());
    events.set_ups(args.fps);

    while let Some(e) = events.next(&mut window) {
        if let Some(args) = e.render_args() {
            app.render(&args);
        }

        if let Some(args) = e.update_args() {
            app.update(&args);
        }
    }
}
