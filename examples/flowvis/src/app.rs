use burn::prelude::{Backend, s};
use burn::tensor::f16;
use clockmill::simulations::surface::fluids::lbm::d2q9::operations::{
    RelaxationParam, bgk_collision, direction_vectors, equilibrium, macroscopic_velocity,
    population_density, stream_distribution_interior, weight_matrix,
};
use clockmill::simulations::surface::fluids::lbm::d2q9::world::{LBMD2Q9State, LBMMeta};
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

        let cells = population_density(dist.clone());
        let max_rho = cells.clone().max_dim(0).max_dim(1).into_scalar();
        let cells = cells / max_rho;
        let cells = cells.to_data().into_vec::<f16>().unwrap();
        assert_eq!(cells.len(), h * w);

        let [view_width, view_height] = args.viewport().draw_size;
        let [x_step, y_step] = [
            (view_width as f64) / (w as f64),
            (view_height as f64) / (h as f64),
        ];

        self.gl.draw(args.viewport(), |c, gl| {
            for y in 0..h {
                for x in 0..w {
                    let v: f32 = cells[x + y * w].into();

                    let color = if v.is_finite() {
                        [0., 1. - v, v, 1.0]
                    } else {
                        [1., 0., 0., 1.]
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
        let relaxation = RelaxationParam::Omega(1.5);

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
