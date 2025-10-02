use burn::prelude::{Backend, s};
use clockmill::simulations::surface::fluids::lbm::d2q9::operations::{
    RelaxationParam, bgk_collision, direction_vectors, macroscopic_momentum,
    stream_interior_cells, weight_matrix,
};
use clockmill::simulations::surface::fluids::lbm::d2q9::world::{LBMD2Q9State, LBMMeta};
use opengl_graphics::GlGraphics;
use piston::{RenderArgs, UpdateArgs};

pub struct FlowVisApp<B: Backend> {
    pub gl: GlGraphics, // OpenGL drawing backend.
    pub world_state: LBMD2Q9State<B>,
    pub step_rate: usize,
    pub relaxation: RelaxationParam,
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

        let device = world_state.device();
        let e = direction_vectors(&device);
        let momentum = macroscopic_momentum(dist.clone(), e.clone());
        let max_mom = momentum.clone().max().into_scalar();
        let norm_momentum = momentum / max_mom;

        let cells = norm_momentum.to_data().into_vec::<f32>().unwrap();
        assert_eq!(cells.len(), h * w * 2);

        let [view_width, view_height] = args.viewport().draw_size;
        let [x_step, y_step] = [
            (view_width as f64) / (w as f64),
            (view_height as f64) / (h as f64),
        ];

        self.gl.draw(args.viewport(), |c, gl| {
            for y in 0..h {
                for x in 0..w {
                    let vy: f32 = cells[x * 2 + y * 2 * w];
                    let vx: f32 = cells[1 + x * 2 + y * 2 * w];

                    let color = if vy.is_finite() && vx.is_finite() {
                        [0., vy, vx, 1.0]
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
        let device = self.world_state.device();
        let mut dist = self.world_state.dist.clone();
        let e = direction_vectors(&device);
        let w = weight_matrix(&device);

        for _ in 0..self.step_rate {
            dist = bgk_collision(dist.clone(), e.clone(), w.clone(), self.relaxation);

            let interior =
                stream_interior_cells(dist.clone(), self.world_state.solid_mask.clone());

            dist = dist.slice_assign(s![1..-1, 1..-1], interior);
        }

        self.world_state.dist = dist;
    }
}
