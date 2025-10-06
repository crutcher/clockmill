use burn::prelude::{Backend, s};
use burn::tensor::DType::F32;
use clap::Parser;
use clockmill::simulations::surface::fluids::lbm::d2q9::operations::{RelaxationParam, moments};
use clockmill::simulations::surface::fluids::lbm::d2q9::world::{
    LBMD2Q9Config, LBMD2Q9State, LBMMeta,
};
use glutin_window::GlutinWindow as Window;
use opengl_graphics::{GlGraphics, OpenGL};
use piston::event_loop::{EventSettings, Events};
use piston::input::{RenderEvent, UpdateEvent};
use piston::window::WindowSettings;
use piston::{EventLoop, OpenGLWindow, RenderArgs, UpdateArgs};

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
    #[arg(long, value_parser=parse_shape, default_value="500")]
    pub grid_shape: [usize; 2],

    /// The max frames per second.
    #[arg(long, default_value_t = 10)]
    pub fps: u64,

    /// The initial window zoom.
    #[arg(long, default_value_t = 1.0)]
    pub zoom: f64,

    /// The number of steps to take per frame.
    #[arg(long, default_value_t = 1)]
    pub step_rate: usize,

    /// The number of steps to skip on init.
    #[arg(long, default_value_t = 0)]
    pub init_skip_steps: usize,

    /// The collision relaxation tau.
    #[arg(long, default_value_t = 1.0)]
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
        .slice_fill(s![50, 20, 1, 1], 100.0)
        .slice_fill(s![90, 100, 1, 1], 200.0);

    world_state.solid_mask = world_state
        .solid_mask
        .slice_fill(s![30, 40..60], true)
        .slice_fill(s![.., ..2], true)
        .slice_fill(s![.., -2..], true)
        .slice_fill(s![..2, ..], true)
        .slice_fill(s![-2.., ..], true);

    let mut world_state = world_state.to_dtype(F32);
    world_state.save_correct_total_mass();

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
        if let Some(render_args) = e.render_args() {
            app.render(&render_args);
        }

        if let Some(update_args) = e.update_args() {
            app.update(&update_args);
        }
    }
}

/*
// arctan(x) = ln(
//   (x/sqrt(1+x²)) + sqrt(1 + (x²/(1+x²)))
// )
fn arctan<B: Backend, const D: usize>(x: &Tensor<B, D>) -> Tensor<B, D> {
    let x2: Tensor<B, D> = fast_powi_2(x.clone());
    let t: Tensor<B, D> = 1.0 + x2.clone();

    (
        (x.clone() / t.clone().sqrt())
            + ((x2 / t) + 1).sqrt()
    ).log()
}

fn atan2<B: Backend, const D: usize>(y: &Tensor<B, D>, x: &Tensor<B, D>) -> Tensor<B, D> {
    if x
    let at = arctan(y/x)

}

 */

pub struct FlowVisApp<B: Backend> {
    pub gl: GlGraphics, // OpenGL drawing backend.
    pub world_state: LBMD2Q9State<B>,
    pub step_rate: usize,
}

impl<B: Backend> FlowVisApp<B> {
    pub fn vis_cells(&self) -> Vec<Vec<(f32, f32)>> {
        let [height, width] = self.world_state.shape();
        let dist = &self.world_state.dist;

        let e = self.world_state.e.clone();

        let (_rho, u) = moments(dist.clone(), e.clone());

        let scale = u.clone().max().into_scalar();

        let cells = (u / scale).clamp(0.0, 1.0);
        let cells = cells.mul_scalar(std::f64::consts::PI / 2.0).sin();

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
        let [h, w] = self.world_state.shape();
        let mask = &self.world_state.solid_mask;

        let cells = mask.to_data().into_vec::<u8>().unwrap();
        let mut result = vec![vec![false; w]; h];
        for y in 0..h {
            for x in 0..w {
                result[y][x] = cells[(y * w) + x] == 1;
            }
        }
        result
    }

    pub fn render(
        &mut self,
        args: &RenderArgs,
    ) {
        use graphics::*;

        let vis_cells = self.vis_cells();
        let solid_cells = self.solid_cells();

        let [h, w] = self.world_state.shape();
        let [view_width, view_height] = args.viewport().draw_size;

        let [x_step, y_step] = [
            (view_width as f64) / (w as f64),
            (view_height as f64) / (h as f64),
        ];

        self.gl.draw(args.viewport(), |c, gl| {
            for y in 0..h {
                for x in 0..w {
                    let (uy, ux) = vis_cells[y][x];
                    let is_solid = solid_cells[y][x];

                    let color = if is_solid {
                        [1., 1., 1., 1.]
                    } else if uy.is_finite() && ux.is_finite() {
                        [0.0, uy, ux, 1.]
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

    pub fn update(
        &mut self,
        _args: &UpdateArgs,
    ) {
        self.advance_frame();
    }

    pub fn advance_frame(&mut self) {
        for _ in 0..self.step_rate {
            self.world_state.advance_step();
        }
    }
}
