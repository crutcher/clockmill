use app::FlowVisApp;
use burn::prelude::{Backend, s};
use clap::Parser;
use clockmill::simulations::surface::fluids::lattice_boltzmann::{LBMD2Q9Config, LBMD2Q9State};
use glutin_window::GlutinWindow as Window;
use opengl_graphics::{GlGraphics, OpenGL};
use piston::event_loop::{EventSettings, Events};
use piston::input::{RenderEvent, UpdateEvent};
use piston::window::WindowSettings;
use piston::{EventLoop, OpenGLWindow};

mod app;

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
    #[arg(long, value_parser=parse_shape, default_value="100")]
    pub grid_shape: [usize; 2],

    /// The number of steps to take per frame.
    #[arg(long, default_value_t = 1)]
    pub step_rate: usize,

    /// The number of steps to skip on init.
    #[arg(long, default_value_t = 0)]
    pub init_skip_steps: usize,

    /// The max frames per second.
    #[arg(long, default_value_t = 1)]
    pub fps: u64,

    /// The initial window zoom.
    #[arg(long, default_value_t = 2.5)]
    pub zoom: f64,
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

    let mut world_state: LBMD2Q9State<B> = LBMD2Q9Config::new(args.grid_shape).init(&device);
    world_state.dist = world_state
        .dist
        .slice_fill(s![.., .., 1, 1], 1.0)
        .slice_fill(s![40, 50, 1, 1], 10.0)
        .slice_fill(s![50, 50, 1, 1], 10.0);

    // Create a new game and run it.
    let mut app = FlowVisApp {
        gl: GlGraphics::new(opengl),
        world_state,
        step_rate: args.step_rate,
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
