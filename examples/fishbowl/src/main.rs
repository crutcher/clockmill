#![allow(unused)]

use bimm_contracts::{assert_shape_contract_periodically, unpack_shape_contract};
use burn::backend::Cuda;
use burn::prelude::{Backend, Bool, Int, Tensor, s};
use burn::tensor::DType::F16;
use burn::tensor::Distribution;
use burn::tensor::module::unfold4d;
use burn::tensor::ops::UnfoldOptions;
use clap::Parser;
use glutin_window::GlutinWindow as Window;
use graphics::Graphics;
use opengl_graphics::{GlGraphics, OpenGL};
use piston::Key::{B, G};
use piston::event_loop::{EventSettings, Events};
use piston::input::{RenderArgs, RenderEvent, UpdateArgs, UpdateEvent};
use piston::window::WindowSettings;
use piston::{EventLoop, OpenGLWindow};
use std::env::args;
use std::time::Instant;

const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

pub struct App<B: Backend> {
    gl: GlGraphics, // OpenGL drawing backend.
    state: Tensor<B, 2, Bool>,
    update_noise: f64,
}

impl<B: Backend> App<B> {
    fn render(
        &mut self,
        args: &RenderArgs,
    ) {
        use graphics::*;

        let [h, w] = self.state.shape().dims();
        let data = self.state.clone();

        let data = data.to_data();
        let data: &[u8] = data.as_slice().unwrap();

        let [win_w, win_h] = args.viewport().draw_size;

        let draw_scale = (win_w as f64)/(w as f64);

        self.gl.draw(args.viewport(), |c, gl| {
            for w_idx in 0..w {
                for h_idx in 0..h {
                    let cell = data[h_idx * w + w_idx];

                    let color = if cell == 1 { BLACK } else { WHITE };

                    let pos = [0., 0., draw_scale, draw_scale];

                    let transform = c.transform.trans(
                        w_idx as f64 * draw_scale,
                        h_idx as f64 * draw_scale);

                    Rectangle::new(color).draw(
                        pos,
                        &c.draw_state,
                        transform,
                        gl,
                    );
                }
            }
        });
    }

    fn update(
        &mut self,
        args: &UpdateArgs,
    ) {
        self.state = conway(self.state.clone());

        if self.update_noise > 0.0 {
            let noise = Tensor::<B, 2>::random(
                self.state.shape(),
                Distribution::Bernoulli(self.update_noise),
                &self.state.device(),
            )
            .equal_elem(1.0);

            self.state = self.state.clone().bool_or(noise);
        }
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
    #[arg(long, default_value_t = 1e-4)]
    pub update_noise: f64,

    /// The number of steps to target per second.
    #[arg(long, default_value_t = 20)]
    pub fps: u64,

    /// The initial window zoom.
    #[arg(long, default_value_t = 2.5)]
    pub zoom: f64,
}

fn main() {
    let args = Args::parse();

    run::<Cuda>(&args);
}

fn run<B: Backend>(args: &Args) {
    println!("Args: {:?}", args);

    let device = Default::default();

    let mut state: Tensor<B, 2, Bool> = Tensor::<B, 2>::random(
        [args.grid_size, args.grid_size],
        Distribution::Bernoulli(args.initial_density),
        &device,
    )
    .equal_elem(1.0);

    // Change this to OpenGL::V2_1 if not working.
    let opengl = OpenGL::V3_2;

    // Create a Glutin window.
    let gs =  (args.grid_size as f64 * args.zoom) as u32;
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
        state: state,
        update_noise: args.update_noise,
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

fn conway<B: Backend>(state: Tensor<B, 2, Bool>) -> Tensor<B, 2, Bool> {
    let [h, w] = state.shape().dims();

    let h_blocks: Tensor<B, 3, Bool> = state.clone().unfold(0, 3, 1);
    assert_shape_contract_periodically!(
        ["h_wins" = "height" - "pad", "width", "kernel"],
        &h_blocks.shape().dims,
        &[("height", h), ("width", w), ("kernel", 3), ("pad", 2)]
    );

    let blocks: Tensor<B, 4, Bool> = h_blocks.unfold(1, 3, 1);
    assert_shape_contract_periodically!(
        [
            "h_wins" = "height" - "pad",
            "w_wins" = "width" - "pad",
            "kernel",
            "kernel"
        ],
        &blocks.shape().dims,
        &[("height", h), ("width", w), ("kernel", 3), ("pad", 2)]
    );

    let blocks: Tensor<B, 3, Int> = blocks
        .reshape([h - 2, w - 2, 3 * 3])
        .permute([2, 0, 1])
        .int();

    let block_sum = blocks.clone().sum_dim(0);
    let neighbor_count = block_sum - blocks.slice(s![5, .., ..]);

    let neighbor_count = neighbor_count.reshape([h - 2, w - 2]);

    conway_transition(state, neighbor_count)
}

fn conway_transition<B: Backend>(
    state: Tensor<B, 2, Bool>,
    neighbor_count: Tensor<B, 2, Int>,
) -> Tensor<B, 2, Bool> {
    let inner = state.clone().slice(s![1..-1, 1..-1]);

    let survivors = inner.bool_and(neighbor_count.clone().equal_elem(2));
    let spawns = neighbor_count.equal_elem(3);

    let update = survivors.bool_or(spawns);

    state.slice_assign(s![1..-1, 1..-1], update)
}
