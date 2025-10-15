use burn::Tensor;
use burn::prelude::{Backend, Bool, ElementConversion, s};
use burn::tensor::DType::F32;
use clap::Parser;
use clockmill::framework::config_parsers::parse_shape;
use clockmill::simulations::surface::fluids::lbm::d2q9::SPEED_OF_SOUND;
use clockmill::simulations::surface::fluids::lbm::d2q9::relaxation::RelaxationParam;
use clockmill::simulations::surface::fluids::lbm::d2q9::simulation::{
    LBMD2Q9Config, LBMD2Q9State, LBMMeta,
};
use clockmill::simulations::surface::fluids::lbm::d2q9::space::LbmTables;
use clockmill::simulations::surface::fluids::lbm::d2q9::space::{density, macroscopic_momentum};
use glutin_window::GlutinWindow as Window;
use indicatif::ProgressBar;
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

/// Conway's Game of Life demo for Burn.
#[derive(Parser, Debug)]
#[command(long_about = None)]
pub struct Args {
    /// The grid shape as `HEIGHT,WIDTH`, or `SIZE`.
    #[arg(long, value_parser=parse_shape, default_value="250")]
    pub grid_shape: [usize; 2],

    /// The max frames per second.
    #[arg(long, default_value_t = 60)]
    pub fps: u64,

    /// The tics per second.
    #[arg(long, default_value_t = 150.0)]
    pub tps: f32,

    /// The initial window zoom.
    #[arg(long, default_value_t = 2.5)]
    pub zoom: f64,

    /// The display opacity of updates.
    #[arg(long, default_value_t = 1.0)]
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

    let [height, width] = args.grid_shape;

    let background_density = SPEED_OF_SOUND / 100.0;

    let mut world_state: LBMD2Q9State<B> = LBMD2Q9Config::new(args.grid_shape)
        .with_relaxation(RelaxationParam::Tau(args.tau))
        .init(&device, background_density);
    world_state.dist = world_state
        .dist
        .slice_fill(s![50, 20, 1, 1], 5.0 * background_density)
        .slice_fill(s![20, 100, 1, 1], 5.0 * background_density);

    world_state.solid_mask = world_state
        .solid_mask
        .slice_fill(s![30, 40..60], true)
        .slice_fill(s![125, 75..150], true)
        .slice_fill(s![150, 50..75], true)
        .slice_fill(s![150, 100..125], true);

    let mut world_state = world_state.to_dtype(dtype);
    world_state.save_correct_total_mass();

    for _ in 0..args.init_skip_steps {
        world_state.advance_step();
    }

    let solid_mask: Tensor<B, 2, Bool> = world_state.solid_mask.clone();

    let sim_delay = if args.tps > 0.0 {
        Some(Duration::from_secs_f32(1.0 / args.tps))
    } else {
        None
    };
    let sim = Simulation::new(world_state, sim_delay);

    // Create a Glutin window.
    let mut window: Window = WindowSettings::new(
        "flowvis",
        [width as f64 * args.zoom, height as f64 * args.zoom],
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
        step_duration: Option<Duration>,
    ) -> Self {
        let shutdown = Arc::new(AtomicBool::new(false));
        let state = Arc::new(Mutex::new(world.dist.clone()));
        let [height, width] = world.shape();

        let shutdown_clone = shutdown.clone();
        let state_clone = state.clone();

        let handle = thread::spawn(move || {
            let progress = ProgressBar::new_spinner();

            let mut world = world;

            let mut stash = 0.0;

            let delay_smoothing = 20;
            let mut avg_delay = std::time::Duration::from_secs_f32(0.0);
            let mut last_time = std::time::Instant::now();

            while !shutdown_clone.load(Ordering::Relaxed) {
                {
                    let now = std::time::Instant::now();
                    let dt = now - last_time;
                    avg_delay = (avg_delay * delay_smoothing + dt) / (delay_smoothing + 1);
                    last_time = now;
                }
                let avg_tps = 1.0 / avg_delay.as_secs_f32();
                progress.set_message(format!("sim:{:.0}tps", avg_tps));
                progress.tick();

                let dist = world.dist.clone();

                let drift = width / 4;
                let period = 400.0;

                let offset = ((world.step_count as f32 * std::f32::consts::PI / period).sin()
                    * drift as f32) as isize;

                let start = offset + (width as isize / 2);

                let r = 0.2;
                let outflow_slice = s![-1, start..start + 10, 0, ..];
                let outflow = dist.clone().slice(outflow_slice);

                stash += r * outflow.clone().sum().into_scalar().elem::<f32>();

                let mut dist = dist
                    .clone()
                    .slice_assign(outflow_slice, (1.0 - r) * outflow.clone());

                if world.step_count.is_multiple_of(60) {
                    let (ry, rx) = loop {
                        let ry = rand::rng().random_range(10..height - 10);
                        let rx = rand::rng().random_range(10..width - 10);

                        if world
                            .solid_mask
                            .clone()
                            .slice(s![ry, rx])
                            .into_scalar()
                            .elem::<bool>()
                        {
                            continue;
                        }

                        break (ry, rx);
                    };

                    let existing: f32 = dist.clone().slice(s![ry, rx, 1, 1]).into_scalar().elem();

                    dist = dist.slice_fill(s![ry, rx, 1, 1], existing + stash);
                    stash = 0.0;
                }

                world.dist = dist;

                // Export
                world.advance_step();
                *state_clone.lock().unwrap() = world.dist.clone();

                if let Some(step_duration) = step_duration {
                    thread::sleep(step_duration);
                }
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

        let constants: LbmTables<B> = LbmTables::for_dist(&state);

        // let (_rho, u) = moments(dist.clone(), e.clone());
        // let cells = fast_powi_2(u);
        let cells = macroscopic_momentum(state.clone(), constants.e_vec());

        // let scale = cells.clone().max().into_scalar();
        let scale = SPEED_OF_SOUND / 1000.0;

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

        let cells = self
            .solid_mask
            .clone()
            .int()
            .to_data()
            .into_vec::<i32>()
            .unwrap();
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
        let [view_width, view_height] = args.viewport().window_size;

        let [x_step, y_step] = [view_width / (width as f64), view_height / (height as f64)];

        self.gl.draw(args.viewport(), |c, gl| {
            for y in 0..height {
                for x in 0..width {
                    let (uy, ux) = vis_cells[y][x];
                    let is_solid = solid_cells[y][x];

                    let _d = rho[y * width + x];

                    let color = if is_solid {
                        [1., 1., 1., 1.]
                    } else if uy.is_finite() && ux.is_finite() {
                        [0.0, uy, ux, self.opacity]
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
