use burn::Tensor;
use burn::prelude::{Backend, Bool, ElementConversion, s};
use burn::tensor::DType::F32;
use clap::Parser;
use clockmill::simulations::surface::fluids::lbm::d2q9::operations::{
    RelaxationParam, density, direction_vectors, macroscopic_momentum,
};
use clockmill::simulations::surface::fluids::lbm::d2q9::world::{
    LBMD2Q9Config, LBMD2Q9State, LBMMeta,
};
use glutin_window::GlutinWindow as Window;
use opengl_graphics::{GlGraphics, OpenGL};
use piston::event_loop::{EventSettings, Events};
use piston::input::RenderEvent;
use piston::window::WindowSettings;
use piston::{EventLoop, OpenGLWindow, RenderArgs};
use rand::Rng;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::thread::JoinHandle;
use std::time::Duration;

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
    #[arg(long, value_parser=parse_shape, default_value="250")]
    pub grid_shape: [usize; 2],

    /// The max frames per second.
    #[arg(long, default_value_t = 40)]
    pub fps: u64,

    /// The tics per second.
    #[arg(long, default_value_t = 40.)]
    pub tps: f32,

    /// The initial window zoom.
    #[arg(long, default_value_t = 2.5)]
    pub zoom: f64,

    /// The display opacity of updates.
    #[arg(long, default_value_t = 0.5)]
    pub opacity: f32,

    /// The number of steps to skip on init.
    #[arg(long, default_value_t = 1)]
    pub init_skip_steps: usize,

    /// The collision relaxation tau.
    #[arg(long, default_value_t = 0.9)]
    pub tau: f64,
}

fn main() {
    let args = Args::parse();
    println!("{:#?}", args);

    #[cfg(feature = "cuda")]
    run::<burn::backend::Cuda<f32, i32>>(&args);

    #[cfg(feature = "wgpu")]
    run::<burn::backend::Wgpu>(&args);

    #[cfg(feature = "metal")]
    run::<burn::backend::Metal>(&args);
}

fn run<B: Backend>(args: &Args) {
    let device = Default::default();

    // Change this to OpenGL::V2_1 if not working.
    let opengl = OpenGL::V3_2;

    let dtype = F32;

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

    let mut world_state: LBMD2Q9State<B> = LBMD2Q9Config::new(args.grid_shape)
        .with_relaxation(RelaxationParam::Tau(args.tau))
        .init(&device);
    world_state.dist = world_state
        .dist
        .slice_fill(s![50, 20, 1, 1], 50.0)
        .slice_fill(s![90, 100, 1, 1], 50.0);

    world_state.solid_mask = world_state
        .solid_mask
        .slice_fill(s![30, 40..60], true)
        .slice_fill(s![125, 75..150], true)
        .slice_fill(s![150, 50..75], true)
        .slice_fill(s![150, 100..125], true);

    world_state.omega = world_state
        .omega
        .slice_fill(s![-150..-50, -150..-50], 0.05)
        .slice_fill(
            s![-125..-75, -125..-75],
            RelaxationParam::Tau(args.tau).as_omega_value(),
        );

    let mut world_state = world_state.to_dtype(dtype);
    world_state.save_correct_total_mass();

    for _ in 0..args.init_skip_steps {
        world_state.advance_step();
    }

    let solid_mask: Tensor<B, 2, Bool> = world_state.solid_mask.clone();

    let sim = Simulation::new(
        world_state,
        std::time::Duration::from_secs_f32(1.0 / args.tps),
    );

    // Create a new game and run it.
    let mut app = FlowVisApp {
        gl: GlGraphics::new(opengl),
        state_handle: sim.state.clone(),
        solid_mask,
        opacity: args.opacity,
    };

    let mut events = Events::new(EventSettings::new());
    events.set_ups(args.fps);

    while let Some(e) = events.next(&mut window) {
        if let Some(render_args) = e.render_args() {
            app.render(&render_args);
        }
    }

    sim.shutdown();
}

pub struct Simulation<B: Backend> {
    handle: Option<JoinHandle<()>>,
    shutdown: Arc<AtomicBool>,
    pub state: Arc<Mutex<Tensor<B, 4>>>,
}

impl<B: Backend> Simulation<B> {
    pub fn new(
        world: LBMD2Q9State<B>,
        step_duration: Duration,
    ) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let state = Arc::new(Mutex::new(world.dist.clone()));

        let shutdown_clone = shutdown.clone();
        let state_clone = state.clone();

        let handle = thread::spawn(move || {
            let mut world = world;

            while !shutdown_clone.load(Ordering::Relaxed) {
                if false && world.step_count.is_multiple_of(250) {
                    let [width, height] = world.shape();
                    let mut dist = world.dist.clone();

                    let extra: f64 = dist.clone().max().into_scalar().elem();

                    let k = rand::rng().random_range(2..=4);
                    for _ in 0..k {
                        let ry = rand::rng().random_range(1..=height - 1);
                        let rx = rand::rng().random_range(1..=width - 1);

                        let existing: f64 =
                            dist.clone().slice(s![ry, rx, 1, 1]).into_scalar().elem();
                        dist = dist.slice_fill(s![ry, rx, 1, 1], existing + extra);

                        world.correct_total_mass += extra;
                    }

                    world.dist = dist;
                }

                // Export
                world.advance_step();
                *state_clone.lock().unwrap() = world.dist.clone();

                thread::sleep(step_duration);
            }
        });

        Simulation {
            handle: Some(handle),
            shutdown,
            state,
        }
    }
    pub fn shutdown(mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            handle.join().unwrap();
        }
    }
}

pub struct FlowVisApp<B: Backend> {
    pub gl: GlGraphics, // OpenGL drawing backend.
    pub state_handle: Arc<Mutex<Tensor<B, 4>>>,
    pub solid_mask: Tensor<B, 2, Bool>,
    pub opacity: f32,
}

impl<B: Backend> FlowVisApp<B> {
    pub fn get_world_shape(&self) -> [usize; 2] {
        self.solid_mask.shape().dims()
    }

    pub fn get_state(&self) -> Tensor<B, 4> {
        let lock = self.state_handle.lock();
        lock.unwrap().clone()
    }

    pub fn vis_cells(&self) -> Vec<Vec<(f32, f32)>> {
        let [height, width] = self.get_world_shape();
        let state = self.get_state();

        let e = direction_vectors(&state.device()).cast(state.dtype());

        // let (_rho, u) = moments(dist.clone(), e.clone());
        // let cells = fast_powi_2(u);
        let cells = macroscopic_momentum(state.clone(), e.clone());

        let scale = cells.clone().max().into_scalar();
        let cells = ((cells / scale) + 1.0) / 2.0;
        // let cells = cells.mul_scalar(std::f64::consts::PI / 2.0).sin();

        let cells = cells.cast(F32).to_data().into_vec::<f32>().unwrap();
        assert_eq!(cells.len(), height * width * 2);

        let mut result = vec![vec![(0.0, 0.0); width]; height];
        for h in 0..height {
            for w in 0..width {
                let vy: f32 = cells[h * width * 2 + w * 2];
                let vx: f32 = cells[h * width * 2 + w * 2 + 1];

                result[h][w] = (vy, vx);
            }
        }
        result
    }

    pub fn solid_cells(&self) -> Vec<Vec<bool>> {
        let [height, width] = self.get_world_shape();

        let cells = self.solid_mask.clone().to_data().into_vec::<u8>().unwrap();
        let mut result = vec![vec![false; width]; height];
        for y in 0..height {
            for x in 0..width {
                result[y][x] = cells[(y * width) + x] == 1;
            }
        }
        result
    }

    pub fn render(
        &mut self,
        args: &RenderArgs,
    ) {
        use graphics::*;

        let rho = density(self.get_state());
        let rho = rho.powi_scalar(2.0);
        let max_rho = rho.clone().max().into_scalar();
        let rho = rho.div_scalar(max_rho);
        let rho = rho.cast(F32).to_data().into_vec::<f32>().unwrap();

        let vis_cells = self.vis_cells();
        let solid_cells = self.solid_cells();

        let [height, width] = self.get_world_shape();
        let [view_width, view_height] = args.viewport().draw_size;

        let [x_step, y_step] = [
            (view_width as f64) / (width as f64),
            (view_height as f64) / (height as f64),
        ];

        self.gl.draw(args.viewport(), |c, gl| {
            for y in 0..height {
                for x in 0..width {
                    let (uy, ux) = vis_cells[y][x];
                    let is_solid = solid_cells[y][x];

                    let d = rho[y * width + x];

                    let color = if is_solid {
                        [1., 1., 1., 1.]
                    } else if uy.is_finite() && ux.is_finite() {
                        [d, uy, ux, self.opacity]
                    } else {
                        [0., 0., 0., 1.]
                    };

                    let pos = [0., 0., x_step, y_step];

                    let transform = c.transform.trans(x as f64 * x_step, y as f64 * y_step);

                    Rectangle::new(color).draw(pos, &c.draw_state, transform, gl);
                }
            }
        });
    }
}
