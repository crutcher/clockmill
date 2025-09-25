use crate::color::ColorScheme;
use burn::prelude::Backend;
use clockmill::simulations::surface::conway::Conway;
use opengl_graphics::GlGraphics;
use piston::{RenderArgs, UpdateArgs};

pub struct FishbowlApp<B: Backend> {
    pub gl: GlGraphics, // OpenGL drawing backend.
    pub conway: Conway<B>,
    pub update_noise: f64,
    pub color_scheme: ColorScheme,
}

impl<B: Backend> FishbowlApp<B> {
    pub fn render(
        &mut self,
        args: &RenderArgs,
    ) {
        use burn::prelude::s;
        use graphics::*;

        let fallow_color = self.color_scheme.fallow_color();
        let spawn_color = self.color_scheme.spawn_color();
        let died_color = self.color_scheme.died_color();
        let survivor_color = self.color_scheme.survivor_color();

        let state_data = self.conway.read_slice(s![.., ..]);
        let previous_data = self.conway.read_previous_slice(s![.., ..]);

        let [h, w] = self.conway.shape();
        let [win_w, win_h] = args.viewport().draw_size;
        let draw_scale = [(win_w as f64) / (w as f64), (win_h as f64) / (h as f64)];

        // TODD: this should all be a one-step Image::draw().

        self.gl.draw(args.viewport(), |c, gl| {
            for (w_idx, col) in state_data.iter().enumerate() {
                for (h_idx, is_live) in col.iter().enumerate() {
                    let is_live = *is_live;

                    let was_live = previous_data
                        .as_ref()
                        .map(|prev| prev[w_idx][h_idx])
                        .unwrap_or(false);

                    let color = match (was_live, is_live) {
                        (false, false) => fallow_color,
                        (false, true) => spawn_color,
                        (true, false) => died_color,
                        (true, true) => survivor_color,
                    };

                    let pos = [0., 0., draw_scale[0], draw_scale[1]];

                    let transform = c
                        .transform
                        .trans(w_idx as f64 * draw_scale[0], h_idx as f64 * draw_scale[1]);

                    Rectangle::new(color).draw(pos, &c.draw_state, transform, gl);
                }
            }
        });
    }

    pub fn update(
        &mut self,
        _args: &UpdateArgs,
    ) {
        self.conway.fuzz(self.update_noise);
        self.conway.wrap();
        self.conway.step_no_wrap()
    }
}
