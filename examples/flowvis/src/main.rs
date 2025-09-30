use app::FlowVisApp;
use burn::prelude::{s, Backend};
use clap::Parser;
use clockmill::simulations::surface::fluids::lattice_boltzmann::{LBM, LBMConfig, LBMOperations};
use color::ColorScheme;
use glutin_window::GlutinWindow as Window;
use opengl_graphics::{GlGraphics, OpenGL};
use piston::event_loop::{EventSettings, Events};
use piston::input::{RenderEvent, UpdateEvent};
use piston::window::WindowSettings;
use piston::{EventLoop, OpenGLWindow};

mod app;
mod color;

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

    /// The number of steps to take per frame.
    #[arg(long, default_value_t = 2)]
    pub step_rate: usize,

    /// The number of steps to skip on init.
    #[arg(long, default_value_t = 0)]
    pub init_skip_steps: usize,

    /// The initial density of the grid.
    #[arg(long, default_value_t = 0.03)]
    pub initial_density: f64,

    /// The noise to apply to the grid on each step.
    #[arg(long, default_value_t = 0.0001)]
    pub update_noise: f64,

    /// The max frames per second.
    #[arg(long, default_value_t = 1)]
    pub fps: u64,

    /// The initial window zoom.
    #[arg(long, default_value_t = 1.5)]
    pub zoom: f64,

    /// The opacity between frames.
    #[arg(long, default_value_t = 0.95)]
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

    let mut world_state: LBM<B> = LBMConfig::new(args.grid_shape).init(&device);
    world_state.state =
        world_state.state.clone()
        .slice_fill(s![0..20, 0, 1, 2], 1.0);

    // Change this to OpenGL::V2_1 if not working.
    let opengl = OpenGL::V3_2;

    // Create a Glutin window.
    let mut window: Window = WindowSettings::new(
        "flowvis",
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
    let mut app = FlowVisApp {
        gl: GlGraphics::new(opengl),
        world_state,
        ops: LBMOperations::init(&device),
        _color_scheme: args.color_scheme,
        step_rate: args.step_rate,
        opacity: args.opacity,
    };
    for _ in 0..args.init_skip_steps {
        app.advance_frame();
    }

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
