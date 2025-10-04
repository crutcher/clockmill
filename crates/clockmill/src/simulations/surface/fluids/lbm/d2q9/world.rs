//! # LBM D2Q9 World Module

use crate::simulations::surface::fluids::lbm::d2q9::operations::{
    RelaxationParam, combined_isotropic_collision, direction_vectors, stream_interior_cells,
    weight_matrix,
};
use burn::Tensor;
use burn::config::Config;
use burn::module::{Ignored, Module};
use burn::prelude::{Backend, Bool, s};
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

        self.relaxation.validate();

        LBMD2Q9State {
            step_count: 0,
            dist: state,
            solid_mask,
            e,
            w,
            relaxation: Ignored(self.relaxation),
        }
    }
}

/// Lattice-Boltzmann Fluid Simulation State Module
#[derive(Module, Debug)]
pub struct LBMD2Q9State<B: Backend> {
    /// The current simulation step.
    pub step_count: u64,

    /// The grid velocity: ``[H, W, UY=3, UX=3]``
    /// Here the 0-9 velocity terms are unfolded
    /// into the ``UY`` and ``UX`` dims.
    pub dist: Tensor<B, 4>,

    /// The solid mask: ``[H, W]``
    pub solid_mask: Tensor<B, 2, Bool>,

    /// Direction Vectors
    pub e: Tensor<B, 3>,

    /// Weight Matrix
    pub w: Tensor<B, 2>,

    /// Relaxation Param
    pub relaxation: Ignored<RelaxationParam>,
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

    /// Get the current simulation step count.
    pub fn step_count(&self) -> u64 {
        self.step_count
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

    /// Advance the world simulation by one step.
    pub fn advance_step(&mut self) {
        let dist = combined_isotropic_collision(
            self.dist.clone(),
            self.e.clone(),
            self.w.clone(),
            self.solid_mask.clone(),
            *self.relaxation,
        );

        let interior_updates = stream_interior_cells(dist.clone());

        // TODO: handle boundary cells.
        let dist = dist.slice_assign(s![1..-1, 1..-1], interior_updates);

        // TODO: better handle of numerical instability.
        // let dist = dist.clone().mask_fill(dist.is_finite().bool_not(), 0.0);

        self.dist = dist;
        self.step_count += 1;
    }
}
