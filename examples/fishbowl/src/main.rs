use app::FishbowlApp;
use burn::backend::Wgpu;
use burn::prelude::Backend;
use clap::Parser;
use color::ColorScheme;
use conway::{Conway, ConwayConfig};
use glutin_window::GlutinWindow as Window;
use opengl_graphics::{GlGraphics, OpenGL};
use piston::event_loop::{EventSettings, Events};
use piston::input::{RenderEvent, UpdateEvent};
use piston::window::WindowSettings;
use piston::{EventLoop, OpenGLWindow};

mod app;
mod color;

/// Conway's Game of Life demo for Burn.
#[derive(Parser, Debug)]
#[command(long_about = None)]
pub struct Args {
    /// The width and height of the grid.
    #[arg(long, default_value_t = 400)]
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
    println!("{:#?}", args);

    run::<Wgpu>(&args);
}

fn run<B: Backend>(args: &Args) {
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
    let mut app = FishbowlApp {
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
