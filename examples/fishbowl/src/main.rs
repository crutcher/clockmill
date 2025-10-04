use crate::sim::Simulation;
use app::FishbowlApp;
use burn::prelude::Backend;
use clap::Parser;
use clockmill::simulations::surface::conway::{Conway, ConwayConfig};
use color::ColorScheme;
use glutin_window::GlutinWindow as Window;
use opengl_graphics::{GlGraphics, OpenGL};
use piston::event_loop::{EventSettings, Events};
use piston::input::RenderEvent;
use piston::window::WindowSettings;
use piston::{EventLoop, OpenGLWindow};

mod app;
mod color;
mod sim;

fn parse_shape(s: &str) -> Result<[usize; 2], String> {
    if s.contains(",") {
        let parts: Vec<&str> = s.split(',').collect();
        if parts.len() != 2 {
            return Err("Shape must be in the format WIDTH,HEIGHT".to_string());
        }
        let width = parts[0]
            .parse::<usize>()
            .map_err(|_| "Invalid width".to_string())?;
        let height = parts[1]
            .parse::<usize>()
            .map_err(|_| "Invalid height".to_string())?;
        Ok([width, height])
    } else {
        let size = s.parse::<usize>().map_err(|_| "Invalid size".to_string())?;
        Ok([size, size])
    }
}

/// Conway's Game of Life demo for Burn.
#[derive(Parser, Debug)]
#[command(long_about = None)]
pub struct Args {
    /// The grid shape as `HEIGHT,WIDTH`, or `SIZE`.
    #[arg(long, value_parser=parse_shape, default_value="400,600")]
    pub grid_shape: [usize; 2],

    /// The number of steps to skip on init.
    #[arg(long, default_value_t = 10)]
    pub init_skip_steps: usize,

    /// The initial density of the grid.
    #[arg(long, default_value_t = 0.1)]
    pub initial_density: f64,

    /// The noise to apply to the grid on each step.
    #[arg(long, default_value_t = 0.0001)]
    pub update_noise: f64,

    /// The frames per second.
    #[arg(long, default_value_t = 60)]
    pub fps: u64,

    /// The tics per second.
    #[arg(long, default_value_t = 4.)]
    pub tps: f32,

    /// The initial window zoom.
    #[arg(long, default_value_t = 1.5)]
    pub zoom: f64,

    /// The opacity between frames.
    #[arg(long, default_value_t = 0.8)]
    pub opacity: f32,

    /// The color scheme to use.
    #[arg(long, default_value = "inverted")]
    pub color_scheme: ColorScheme,
}

fn main() {
    let args = Args::parse();
    println!("{:#?}", args);

    #[cfg(feature = "cuda")]
    run::<burn::backend::Cuda>(&args);

    #[cfg(feature = "wgpu")]
    run::<burn::backend::Wgpu>(&args);

    #[cfg(feature = "metal")]
    run::<burn::backend::Metal>(&args);
}

fn run<B: Backend>(args: &Args) {
    let device = Default::default();

    let mut conway: Conway<B> = ConwayConfig::new(args.grid_shape).init(&device);
    conway.fuzz(args.initial_density);
    conway.wrap();
    conway.step_no_wrap();
    for _ in 0..args.init_skip_steps {
        conway.fuzz(args.update_noise);
        conway.wrap();
        conway.step_no_wrap();
    }

    let sim = Simulation::new(
        conway,
        args.update_noise,
        std::time::Duration::from_secs_f32(1.0 / args.tps),
    );

    // Change this to OpenGL::V2_1 if not working.
    let opengl = OpenGL::V3_2;

    // Create a Glutin window.
    let mut window: Window = WindowSettings::new(
        "fishbowl",
        [
            args.grid_shape[1] as f64 * args.zoom,
            args.grid_shape[0] as f64 * args.zoom,
        ],
    )
    .graphics_api(opengl)
    .exit_on_esc(true)
    .build()
    .unwrap();

    // Load the OpenGL function pointers
    gl::load_with(|s| window.get_proc_address(s) as *const _);

    // Create a new game and run it.
    let mut app = FishbowlApp {
        gl: GlGraphics::new(opengl),
        state_handle: sim.state.clone(),
        opacity: args.opacity,
    };

    let mut events = Events::new(EventSettings::new());
    events.set_ups(args.fps);

    while let Some(e) = events.next(&mut window) {
        if let Some(args) = e.render_args() {
            app.render(&args);
        }
    }

    sim.shutdown();
}
