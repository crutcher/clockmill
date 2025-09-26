//! # Lattice-Boltzmann Fluid Simulation

use burn::Tensor;
use burn::config::Config;
use burn::module::Module;
use burn::prelude::Backend;

/// Introspection trait for [`LBM`]
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

/// Config for [`LBM`]
///
/// Implements [`LBMMeta`].
#[derive(Config, Debug)]
pub struct LBMConfig {
    /// The shape of the simulation: `[HEIGHT, WIDTH]`
    pub shape: [usize; 2],
}

impl LBMMeta for LBMConfig {
    fn shape(&self) -> [usize; 2] {
        self.shape
    }
}

impl LBMConfig {
    /// Initialize a [`LBM`] module.
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> LBM<B> {
        let state = Tensor::<B, 4>::zeros([self.shape[0], self.shape[1], 3, 3], device);

        LBM {
            step_count: 0,
            state,
        }
    }
}

/// Lattice-Boltzmann Fluid Simulation State Module
#[derive(Module, Debug)]
pub struct LBM<B: Backend> {
    /// The current simulation step.
    pub step_count: u64,

    /// The world state: ``[H, W, X, Y]``
    pub state: Tensor<B, 4>,
}

impl<B: Backend> LBMMeta for LBM<B> {
    fn shape(&self) -> [usize; 2] {
        let dims = &self.state.shape().dims;
        [dims[0], dims[1]]
    }
}

impl<B: Backend> LBM<B> {
    /// Get the device the module is on.
    pub fn device(&self) -> B::Device {
        self.state.device()
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;

    #[test]
    fn test_lbm_init() {
        let device = Default::default();

        let config = LBMConfig::new([10, 12]);
        assert_eq!(config.shape(), [10, 12]);

        let lbm: LBM<Wgpu> = config.init(&device);
        assert_eq!(lbm.shape(), [10, 12]);
        assert_eq!(lbm.step_count(), 0);
        assert_eq!(lbm.device(), device);
        assert_eq!(lbm.state.shape().dims(), [10, 12, 3, 3]);
    }
}
