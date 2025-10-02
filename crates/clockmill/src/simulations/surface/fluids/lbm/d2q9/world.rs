//! # LBM D2Q9 World Module

use burn::Tensor;
use burn::config::Config;
use burn::module::Module;
use burn::prelude::{Backend, Bool};
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
        fill_value: f32,
        device: &B::Device,
    ) -> LBMD2Q9State<B> {
        let [h, w] = self.shape;
        let state = Tensor::<B, 4>::full([h, w, 3, 3], fill_value, device);
        let solid_mask = Tensor::<B, 2>::zeros([h, w], device).bool();

        LBMD2Q9State {
            step_count: 0,
            dist: state,
            solid_mask,
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
}
