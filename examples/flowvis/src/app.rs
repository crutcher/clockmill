use crate::color::ColorScheme;
use burn::prelude::{Backend, s};
use burn::tensor::DType::F32;
use clockmill::simulations::surface::fluids::lattice_boltzmann::{
    LBMD2Q9Operations, LBMD2Q9State, LBMMeta, UCellTerms,
};
use opengl_graphics::GlGraphics;
use piston::{RenderArgs, UpdateArgs};

pub struct FlowVisApp<B: Backend> {
    pub gl: GlGraphics, // OpenGL drawing backend.
    pub world_state: LBMD2Q9State<B>,
    pub ops: LBMD2Q9Operations<B>,
    pub _color_scheme: ColorScheme,
    pub step_rate: usize,
    pub opacity: f32,
}

impl<B: Backend> FlowVisApp<B> {
    pub fn render(
        &mut self,
        args: &RenderArgs,
    ) {
        use graphics::*;

        let world_state = &self.world_state;
        let UCellTerms { u_sq, .. } = self
            .ops
            .ucell_partials(world_state.state.clone())
            .equi_terms();

        let u = u_sq.sqrt().squeeze_dims::<2>(&[2, 3]);
        let shape = u.shape();
        let u_max = u.clone().max().expand(shape);
        let u_norm = u.div(u_max);

        let u_norm = u_norm.cast(F32);

        let [h, w] = self.world_state.shape();
        let [win_w, win_h] = args.viewport().draw_size;
        let draw_scale = [(win_w as f64) / (w as f64), (win_h as f64) / (h as f64)];

        let u_norm = u_norm.into_data();
        let u_norm_slice = u_norm.as_slice::<f32>().unwrap();

        self.gl.draw(args.viewport(), |c, gl| {
            for h_idx in 0..h {
                for w_idx in 0..w {
                    let v = u_norm_slice[h_idx * w + w_idx];

                    let mut color = [v, v, v, 1.0];
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

    pub fn update(
        &mut self,
        _args: &UpdateArgs,
    ) {
        self.advance_frame();
    }

    pub fn advance_frame(&mut self) {
        let tau = 0.55;
        for _ in 0..self.step_rate {
            let state = self
                .world_state
                .state
                .clone()
                .slice_fill(s![0..20, 0, .., 2], 0.5);
            let state = self.ops.collision(state, tau);
            let interior = self
                .ops
                .interior_streaming(state.clone(), self.world_state.solid_mask.clone());
            let state = state.slice_assign(s![1..-1, 1..-1, .., ..], interior);

            self.world_state.state = state;
        }
    }
}
