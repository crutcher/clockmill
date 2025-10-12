use burn::Tensor;
use burn::prelude::{Backend, Bool};
use opengl_graphics::GlGraphics;
use piston::RenderArgs;
use std::sync::{Arc, Mutex};

pub struct FishbowlApp<B: Backend> {
    pub gl: GlGraphics, // OpenGL drawing backend.
    pub state_handle: Arc<Mutex<Tensor<B, 2, Bool>>>,
    pub opacity: f32,
}

impl<B: Backend> FishbowlApp<B> {
    pub fn get_state(&self) -> Tensor<B, 2, Bool> {
        let lock = self.state_handle.lock();
        lock.unwrap().clone()
    }

    pub fn render(
        &mut self,
        args: &RenderArgs,
    ) {
        use graphics::*;

        let state_data = self.get_state();
        let [h, w] = state_data.shape().dims();
        let state_vec = state_data.int().to_data().to_vec::<i32>().unwrap();

        let [win_w, win_h] = args.viewport().window_size;
        let draw_scale = [win_w / (w as f64), win_h / (h as f64)];

        // TODD: this should all be a one-step Image::draw().

        self.gl.draw(args.viewport(), |c, gl| {
            for h_idx in 0..h {
                for w_idx in 0..w {
                    let offset = h_idx * w + w_idx;
                    let is_live = state_vec[offset] == 1;

                    let mut color = if is_live {
                        [1.0, 1.0, 1.0, 1.0]
                    } else {
                        [0.0, 0.0, 0.0, 1.0]
                    };

                    /*
                    let was_live = previous_data
                        .as_ref()
                        .map(|prev| prev[h_idx][w_idx])
                        .unwrap_or(false);

                    let mut color = match (was_live, is_live) {
                        (false, false) => fallow_color,
                        (false, true) => spawn_color,
                        (true, false) => died_color,
                        (true, true) => survivor_color,
                    };
                     */

                    color[3] *= self.opacity;

                    let pos = [0., 0., draw_scale[0], draw_scale[1]];

                    let transform = c
                        .transform
                        .trans(w_idx as f64 * draw_scale[0], h_idx as f64 * draw_scale[1]);

                    Rectangle::new(color).draw(pos, &c.draw_state, transform, gl);
                }
            }
        });
    }
}
