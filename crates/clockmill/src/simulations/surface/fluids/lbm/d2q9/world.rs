#![allow(dead_code)]
//! # LBM D2Q9 World Module

use crate::simulations::surface::fluids::lbm::d2q9::operations::{
    RelaxationParam, bgk_collision, direction_vectors, half_stream_x, half_stream_y,
    stream_interior_windows, weight_matrix, with_spherical_reflection,
};
use burn::Tensor;
use burn::config::Config;
use burn::module::Module;
use burn::prelude::{Backend, Bool, ElementConversion, s};
use burn::tensor::DType;

/// Introspection trait for [`LBMD2Q9State`]
pub trait LBMMeta {
    /// Get the shape of the simulation: `[HEIGHT, WIDTH]`
    fn shape(&self) -> [usize; 2];

    /// Get the height of the simulation.
    fn height(&self) -> usize {
        self.shape()[0]
    }

    /// Get the width of the simulation.
    fn width(&self) -> usize {
        self.shape()[1]
    }
}

/// Config for [`LBMD2Q9State`]
///
/// Implements [`LBMMeta`].
#[derive(Config, Debug)]
pub struct LBMD2Q9Config {
    /// The shape of the simulation: `[HEIGHT, WIDTH]`
    pub shape: [usize; 2],

    /// Relaxation Param
    #[config(default = "RelaxationParam::Tau(0.5)")]
    pub relaxation: RelaxationParam,
}

impl LBMMeta for LBMD2Q9Config {
    fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl LBMD2Q9Config {
    /// Initialize a [`LBMD2Q9State`] module.
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> LBMD2Q9State<B> {
        let [height, width] = self.shape;

        let solid_mask = Tensor::<B, 2>::zeros([height, width], device).bool();

        let initial_rho = 1.0;
        let e = direction_vectors(device);
        let w = weight_matrix(device);

        // Start off in a relaxed state.
        let state = Tensor::<B, 4>::empty([height, width, 3, 3], device).slice_assign(
            s![.., ..],
            w.clone().unsqueeze::<4>().expand([height, width, 3, 3]) * initial_rho,
        );
        let total_mass = state.clone().sum().into_scalar().elem();

        self.relaxation.validate();

        let omega =
            Tensor::<B, 2>::ones([height, width], device) * self.relaxation.as_omega_value();

        LBMD2Q9State {
            step_count: 0,
            dist: state,
            correct_total_mass: total_mass,
            solid_mask,
            e,
            w,
            omega,
        }
    }
}

/// Lattice-Boltzmann Fluid Simulation State Module
#[derive(Module, Debug)]
pub struct LBMD2Q9State<B: Backend> {
    /// The current simulation step.
    pub step_count: u64,

    /// Total Mass.
    pub correct_total_mass: f64,

    /// The grid velocity: ``[H, W, UY=3, UX=3]``
    /// Here the 0-9 velocity terms are unfolded
    /// into the ``UY`` and ``UX`` dims.
    pub dist: Tensor<B, 4>,

    /// The solid mask: ``[H, W]``
    pub solid_mask: Tensor<B, 2, Bool>,

    /// The relaxation field.
    pub omega: Tensor<B, 2>,

    /// Direction Vectors
    pub e: Tensor<B, 3>,

    /// Weight Matrix
    pub w: Tensor<B, 2>,
}

impl<B: Backend> LBMMeta for LBMD2Q9State<B> {
    fn shape(&self) -> [usize; 2] {
        let dims = &self.dist.shape().dims;
        [dims[0], dims[1]]
    }
}

impl<B: Backend> LBMD2Q9State<B> {
    /// Get the device the module is on.
    pub fn device(&self) -> B::Device {
        self.dist.device()
    }

    /// Get the datatype of the state.
    pub fn dtype(&self) -> DType {
        self.dist.dtype()
    }

    /// Recast the datatype of the state.
    pub fn to_dtype(
        self,
        dtype: DType,
    ) -> Self {
        Self {
            dist: self.dist.cast(dtype),
            e: self.e.cast(dtype),
            w: self.w.cast(dtype),
            ..self
        }
    }

    /// Get the current simulation step count.
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Set the current simulation step count.
    pub fn set_step_count(
        &mut self,
        step: u64,
    ) {
        self.step_count = step;
    }

    /// Reset the simulation step count to zero.
    pub fn reset_step_count(&mut self) {
        self.set_step_count(0)
    }

    /// Get the mass correction term.
    pub fn correction_term(&self) -> f64 {
        self.correct_total_mass / self.current_total_mass()
    }

    fn stream_left_edge(
        &self,
        thermal_dist: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        Tensor::cat(
            vec![
                half_stream_y(thermal_dist.clone().slice(s![.., 1, .., 0])),
                thermal_dist.clone().slice(s![1..-1, 0, .., -2..]),
            ],
            3,
        )
    }
    fn stream_right_edge(
        &self,
        thermal_dist: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        Tensor::cat(
            vec![
                thermal_dist.clone().slice(s![1..-1, -1, .., ..2]),
                half_stream_y(thermal_dist.clone().slice(s![.., -2, .., -1])),
            ],
            3,
        )
    }
    fn stream_top_edge(
        &self,
        thermal_dist: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        Tensor::cat(
            vec![
                half_stream_x(thermal_dist.clone().slice(s![1, .., 0, ..])),
                thermal_dist.clone().slice(s![0, 1..-1, -2.., ..]),
            ],
            2,
        )
    }
    fn stream_bottom_edge(
        &self,
        thermal_dist: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        Tensor::cat(
            vec![
                thermal_dist.clone().slice(s![-1, 1..-1, ..2, ..]),
                half_stream_x(thermal_dist.clone().slice(s![-2, .., -1, ..])),
            ],
            2,
        )
    }

    /// Advance the world simulation by one step.
    pub fn advance_step(&mut self) {
        let dist = self.dist.clone();

        let solid_mask = self
            .solid_mask
            .clone()
            .slice_fill(s![0, ..], true)
            .slice_fill(s![-1, ..], true)
            .slice_fill(s![.., 0], true)
            .slice_fill(s![.., -1], true);

        // Local Updates:
        // 1. Internal cell collisions.
        let col_dist = bgk_collision(
            dist.clone(),
            self.e.clone(),
            self.w.clone(),
            self.omega.clone(),
            None, // Some(self.correction_term()),
        );
        let thermal_dist = with_spherical_reflection(dist.clone(), col_dist, solid_mask);

        let mut streaming_dist = thermal_dist.zeros_like();
        streaming_dist = streaming_dist.slice_assign(
            s![1..-1, 1..-1],
            stream_interior_windows(thermal_dist.clone()),
        );

        streaming_dist = streaming_dist.slice_assign(
            s![0, .., 1, 1],
            thermal_dist.clone().slice(s![0, .., 1, 1]),
        );
        streaming_dist = streaming_dist.slice_assign(
            s![.., 0, 1, 1],
            thermal_dist.clone().slice(s![.., 0, 1, 1]),
        );
        streaming_dist = streaming_dist.slice_assign(
            s![-1, .., 1, 1],
            thermal_dist.clone().slice(s![-1, .., 1, 1]),
        );
        streaming_dist = streaming_dist.slice_assign(
            s![.., -1, 1, 1],
            thermal_dist.clone().slice(s![.., -1, 1, 1]),
        );

        // stream top edges.
        // e[-1, -1]; v[0, 0]
        streaming_dist = streaming_dist.slice_assign(
            s![0, ..-1, 0, 0],
            thermal_dist.clone().slice(s![1, 1.., 0, 0]),
        );
        // e[-1, 0]; v[0, 1]
        streaming_dist = streaming_dist.slice_assign(
            s![0, .., 0, 1],
            thermal_dist.clone().slice(s![1, .., 0, 1]),
        );
        // e[-1, 1]; v[0, 2]
        streaming_dist = streaming_dist.slice_assign(
            s![0, 1.., 0, 2],
            thermal_dist.clone().slice(s![1, ..-1, 0, 2]),
        );

        // stream bottom edges.
        // e[1, -1]; v[2, 0]
        streaming_dist = streaming_dist.slice_assign(
            s![-1, ..-1, 2, 0],
            thermal_dist.clone().slice(s![-2, 1.., 2, 0]),
        );
        // e[1, 0]; v[2, 1]
        streaming_dist = streaming_dist.slice_assign(
            s![-1, .., 2, 1],
            thermal_dist.clone().slice(s![1, .., 2, 1]),
        );
        // e[1, 1]; v[2, 2]
        streaming_dist = streaming_dist.slice_assign(
            s![-1, 1.., 2, 2],
            thermal_dist.clone().slice(s![-2, ..-1, 2, 2]),
        );

        // stream left edges.
        // e[-1, -1]; v[0, 0]
        streaming_dist = streaming_dist.slice_assign(
            s![..-1, 0, 0, 0],
            thermal_dist.clone().slice(s![1.., 1, 0, 0]),
        );
        // e[0, -1]; v[1, 0]
        streaming_dist = streaming_dist.slice_assign(
            s![.., 0, 1, 0],
            thermal_dist.clone().slice(s![.., 1, 1, 0]),
        );
        // e[1, -1]; v[2, 0]
        streaming_dist = streaming_dist.slice_assign(
            s![1.., 0, 2, 0],
            thermal_dist.clone().slice(s![..-1, 1, 2, 0]),
        );

        // stream right edges.
        // e[-1, 1]; v[0, 2]
        streaming_dist = streaming_dist.slice_assign(
            s![..-1, -1, 0, 2],
            thermal_dist.clone().slice(s![1.., -2, 0, 2]),
        );
        // e[0, 1]; v[1, 2]
        streaming_dist = streaming_dist.slice_assign(
            s![.., -1, 1, 2],
            thermal_dist.clone().slice(s![.., -2, 1, 2]),
        );
        // e[1, 1]; v[2, 2]
        streaming_dist = streaming_dist.slice_assign(
            s![1.., -1, 2, 2],
            thermal_dist.clone().slice(s![..-1, -2, 2, 2]),
        );

        // TODO: better handle of numerical instability.
        // let dist = dist.clone().mask_fill(dist.is_finite().bool_not(), 0.0);

        self.dist = streaming_dist;
        self.step_count += 1;
    }

    /// Get the current mass of the simm.
    pub fn current_total_mass(&self) -> f64 {
        self.dist.clone().sum().into_scalar().elem()
    }

    /// Save the total energy of the system.
    pub fn save_correct_total_mass(&mut self) {
        self.correct_total_mass = self.current_total_mass();
    }
}
