use burn::prelude::{Backend, s};
use clockmill::simulations::surface::fluids::lattice_boltzmann::{LBMD2Q9State, LBMMeta};
use clockmill::simulations::surface::fluids::lbm::d2q9::operations::{
    RelaxationParam, bgk_collision, direction_vectors, equilibrium, macroscopic_velocity,
    population_density, stream_distribution_interior, weight_matrix,
};
use opengl_graphics::GlGraphics;
use piston::{RenderArgs, UpdateArgs};

pub struct FlowVisApp<B: Backend> {
    pub gl: GlGraphics, // OpenGL drawing backend.
    pub world_state: LBMD2Q9State<B>,
    pub step_rate: usize,
}

impl<B: Backend> FlowVisApp<B> {
    pub fn render(
        &mut self,
        args: &RenderArgs,
    ) {
        use graphics::*;

        let world_state = &self.world_state;
        let [h, w] = world_state.shape();
        let dist = &world_state.dist;

        let cells = population_density(dist.clone()).add_scalar(1.0).log();
        let max_rho = cells.clone().max_dim(1).max_dim(1);
        let cells = cells / max_rho;
        let cells = cells.to_data().into_vec::<f32>().unwrap();
        assert_eq!(cells.len(), h * w);

        let [win_w, win_h] = args.viewport().draw_size;
        let draw_scale = [(win_w as f64) / (w as f64), (win_h as f64) / (h as f64)];

        self.gl.draw(args.viewport(), |c, gl| {
            for h_idx in 0..h {
                for w_idx in 0..w {
                    let v = cells[h_idx * w + w_idx];

                    let color = [v, v, v, 1.0];

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
        let relaxation = RelaxationParam::Omega(0.1);

        for _ in 0..self.step_rate {
            let device = self.world_state.device();
            let dist = self.world_state.dist.clone();

            let rho = population_density(dist.clone());
            let e = direction_vectors(&device);
            let w = weight_matrix(&device);
            let u = macroscopic_velocity(dist.clone(), rho.clone(), e.clone());

            let eq = equilibrium(rho.clone(), u.clone(), e.clone(), w.clone());

            let dist = bgk_collision(dist.clone(), eq.clone(), relaxation);

            let interior =
                stream_distribution_interior(dist.clone(), self.world_state.solid_mask.clone());

            let dist = dist.slice_assign(s![1..-1, 1..-1], interior);

            self.world_state.dist = dist;
        }
    }
}
