#![allow(unused)]

use bimm_contracts::{assert_shape_contract_periodically, unpack_shape_contract};
use burn::backend::{Wgpu};
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
use conway::{Conway, ConwayConfig};

const WHITE: [f32; 4] = [1.0, 1.0, 1.0, 1.0];
const BLACK: [f32; 4] = [0.0, 0.0, 0.0, 1.0];
const GREEN: [f32; 4] = [0.0, 1.0, 0.0, 1.0];
const RED: [f32; 4] = [1.0, 0.0, 0.0, 1.0];

pub struct App<B: Backend> {
    gl: GlGraphics, // OpenGL drawing backend.
    conway: Conway<B>,
    update_noise: f64,
}

impl<B: Backend> App<B> {
    fn render(
        &mut self,
        args: &RenderArgs,
    ) {
        use graphics::*;


        let [h, w] = self.conway.shape();
        let data = self.conway.read_slice(s![.., ..]);

        let [win_w, win_h] = args.viewport().draw_size;

        let draw_scale = (win_w as f64)/(w as f64);

        self.gl.draw(args.viewport(), |c, gl| {
            for w_idx in 0..w {
                for h_idx in 0..h {
                    let cell = data[h_idx][w_idx];

                    let color = if cell { BLACK } else { WHITE };

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
        self.conway.fuzz(self.update_noise);
        self.conway.step()
    }
}

/// Conway's Game of Life demo for Burn.
#[derive(Parser, Debug)]
#[command(long_about = None)]
pub struct Args {
    /// The width and height of the grid.
    #[arg(long, default_value_t = 600)]
    pub grid_size: usize,

    /// The initial density of the grid.
    #[arg(long, default_value_t = 0.2)]
    pub initial_density: f64,

    /// The noise to apply to the grid.
    #[arg(long, default_value_t = 1e-5)]
    pub update_noise: f64,

    /// The number of steps to target per second.
    #[arg(long, default_value_t = 60)]
    pub fps: u64,

    /// The initial window zoom.
    #[arg(long, default_value_t = 1.5)]
    pub zoom: f64,
}

fn main() {
    let args = Args::parse();

    run::<Wgpu>(&args);
}

fn run<B: Backend>(args: &Args) {
    println!("Args: {:?}", args);

    let device = Default::default();

    let mut conway: Conway<B> = ConwayConfig {
        shape: [args.grid_size, args.grid_size]
    }.init(&device);
    conway.fuzz(args.initial_density);

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
        conway,
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

