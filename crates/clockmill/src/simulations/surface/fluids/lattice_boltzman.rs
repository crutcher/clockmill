//! # Lattice-Boltzmann Fluid Simulation

use burn::Tensor;
use burn::config::Config;
use burn::module::Module;
use burn::prelude::Backend;
use burn::tensor::Slice;

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

    /// The world state: ``[H, W, V, U]``
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

/// LBM Operations
#[derive(Module, Debug)]
pub struct LBMOperations<B: Backend> {
    /// Vertical direction vectors in 3x3 layout.
    pub ev: Tensor<B, 2>,

    /// Horizontal direction vectors in 3x3 layout.
    pub eu: Tensor<B, 2>,

    /// Weights in 3x3 layout.
    pub w: Tensor<B, 2>,
}

impl<B: Backend> LBMOperations<B> {
    /// Initialize LBM operations.
    pub fn init(device: &B::Device) -> Self {
        let eu = Tensor::<B, 2>::from_data(
            [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]],
            device,
        );
        let ev = Tensor::<B, 2>::from_data(
            [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]],
            device,
        );
        let w = Tensor::<B, 2>::from_data(
            [
                [1.0 / 36.0, 1.0 / 9.0, 1.0 / 36.0],
                [1.0 / 9.0, 4.0 / 9.0, 1.0 / 9.0],
                [1.0 / 36.0, 1.0 / 9.0, 1.0 / 36.0],
            ],
            device,
        );

        Self { ev, eu, w }
    }

    /// LBM Equilibrium.
    ///
    /// # Arguments
    ///
    /// - `state`: LBM state tensor, with shape ``[..., V, U]`` over cell flow state.
    ///
    /// # Returns
    ///
    /// The equilibrium tensor, with the same shape as `state`.
    pub fn equilibrium<const D: usize>(
        &self,
        state: Tensor<B, D>,
    ) -> Tensor<B, D> {
        let shape = state.shape();
        let rank = shape.num_dims();
        assert!(rank >= 2, "Rank must be at least 2: got {}", rank);
        assert_eq!(rank, D);

        let y_dim = rank - 2;
        let x_dim = rank - 1;

        let eu: Tensor<B, D> = self.eu.clone().expand::<D, _>(shape.clone());
        let ev: Tensor<B, D> = self.ev.clone().expand::<D, _>(shape.clone());
        let w: Tensor<B, D> = self.w.clone().expand::<D, _>(shape.clone());

        // Compute density (sum over V, U dimensions)
        let rho = state.clone().sum_dim(y_dim).sum_dim(x_dim);

        let dv: Tensor<B, D> = (state.clone() * ev.clone())
            .sum_dim(y_dim)
            .sum_dim(x_dim)
            .div(rho.clone())
            .expand(shape.clone());

        let du: Tensor<B, D> = (state.clone() * eu.clone())
            .sum_dim(y_dim)
            .sum_dim(x_dim)
            .div(rho.clone())
            .expand(shape.clone());

        let u_dot_e: Tensor<B, D> = ev * dv.clone() + eu * du.clone();
        let u_sq: Tensor<B, D> = dv.powi_scalar(2) + du.powi_scalar(2);

        w * rho * (1.0 + 3.0 * u_dot_e.clone() + 4.5 * u_dot_e.powi_scalar(2) - 1.5 * u_sq)
    }

    /// LBM Collision step.
    ///
    /// # Arguments
    ///
    /// - `state`: LBM state tensor, with shape ``[..., V, U]`` over cell flow state.
    ///
    /// # Returns
    ///
    /// The collision tensor, with the same shape as `state`.
    pub fn collision<const D: usize>(
        &self,
        state: Tensor<B, D>,
        tau: f32,
    ) -> Tensor<B, D> {
        let equi = self.equilibrium(state.clone());
        let delta = equi - state.clone();
        state + delta / tau
    }
}

/// Lattice-Boltzmann Method Streaming Operation.
///
/// This operation applies the LBM streaming operation,
/// swapping complementary direction pairs from neighboring cells.
///
/// ```text
/// # From (HxW grid of VxU cells)
/// | _, _, _ |; | _, _, _ |; | _, _, _ |
/// | _, _, _ |; | _, _, _ |; | _, _, _ |
/// | _, _, a |; | _, b, _ |; | c, _, _ |
///
/// | _, _, _ |; | _, _, _ |; | _, _, _ |
/// | _, _, d |; | _, e, _ |; | f, _, _ |
/// | _, _, _ |; | _, _, _ |; | _, _, _ |
///
/// | _, _, h |; | _, i, _ |; | j, _, _ |
/// | _, _, _ |; | _, _, _ |; | _, _, _ |
/// | _, _, _ |; | _, _, _ |; | _, _, _ |
///
/// # To
/// | a, b, c |
/// | d, e, f |
/// | h, i, j |
/// ```
pub fn streaming_window_op<B: Backend, const D: usize, const D2: usize>(
    state: Tensor<B, D>
) -> Tensor<B, D2> {
    // state: [..., V=3, U=3, H_KERN=3, W_KERN=3]
    // output: [..., V=3, U=3]

    assert_eq!(D - 2, D2, "D ({D}) - 2 must equal D2 ({D2})");

    #[cfg(debug_assertions)]
    bimm_contracts::assert_shape_contract_periodically!(
        [..., "V", "U", "H_KERN", "W_KERN"],
        &state.shape().dims,
        &[("V", 3), ("U", 3), ("H_KERN", 3), ("W_KERN", 3)]
    );

    let mut ranges: [Slice; D] = (0..D)
        .map(|_| Slice::new(0, None, 1))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    let mut rows = Vec::with_capacity(3);
    for h_idx in 0isize..3 {
        let target_v = h_idx;
        let source_v = 2 - h_idx;

        ranges[D - 4] = Slice::new(source_v, Some(source_v + 1), 1);
        ranges[D - 2] = Slice::new(target_v, Some(target_v + 1), 1);

        let mut columns = Vec::with_capacity(3);
        for w_idx in 0isize..3 {
            let target_u = w_idx;
            let source_u = 2 - w_idx;

            ranges[D - 3] = Slice::new(source_u, Some(source_u + 1), 1);
            ranges[D - 1] = Slice::new(target_u, Some(target_u + 1), 1);

            let column = state
                .clone()
                .slice(ranges.clone())
                .squeeze_dims::<D2>(&[-2, -1]);

            columns.push(column);
        }
        // Concatenate along U dimension
        rows.push(Tensor::cat(columns, D - 3));
    }

    // Concatenate along V dimension
    Tensor::cat(rows, D - 4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;
    use burn::tensor::Distribution;

    #[test]
    #[rustfmt::skip]
    fn test_streaming_window_op() {
        type B = Wgpu;
        let device = Default::default();

        let window: Tensor<B, 4> = Tensor::from_data([
            [
                [
                    [0., 1., 2.],
                    [3., 4., 5.],
                    [6., 7., 8.]
                ],
                [
                    [9., 10., 11.],
                    [12., 13., 14.],
                    [15., 16., 17.]
                ],
                [
                    [18., 19., 20.],
                    [21., 22., 23.],
                    [24., 25., 26.]
                ],
            ],
            [
                [
                    [27., 28., 29.],
                    [30., 31., 32.],
                    [33., 34., 35.]
                ],
                [
                    [36., 37., 38.],
                    [39., 40., 41.],
                    [42., 43., 44.]
                ],
                [
                    [45., 46., 47.],
                    [48., 49., 50.],
                    [51., 52., 53.]
                ],
            ],
            [
                [
                    [54., 55., 56.],
                    [57., 58., 59.],
                    [60., 61., 62.]
                ],
                [
                    [63., 64., 65.],
                    [66., 67., 68.],
                    [69., 70., 71.]
                ],
                [
                    [72., 73., 74.],
                    [75., 76., 77.],
                    [78., 79., 80.]
                ],
            ],
        ], &device);
        let window = window.permute([2, 3, 0, 1]).unsqueeze();

        // println!("{:?}", window.shape());

        let result: Tensor<B, 3> = streaming_window_op::<B, 5, 3>(window.clone());
        assert_eq!(result.shape().dims, vec![1, 3, 3]);

        let expected: Tensor<B, 3> = Tensor::from_data([[
            [8., 16., 24.],
            [32., 40., 48.],
            [56., 64., 72.],
        ]], &device);

        // println!("result: {:#?}", result.to_data().as_slice::<f32>());
        // println!("expected: {:#?}", expected.to_data().as_slice::<f32>());

        result.to_data().assert_eq(&expected.to_data(), false);
    }

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

    #[test]
    fn test_equilibrium() {
        let device = Default::default();

        let ops = LBMOperations::<Wgpu>::init(&device);

        let state = Tensor::<Wgpu, 3>::random([1, 3, 3], Distribution::Normal(0., 1.), &device);
        let _eq = ops.equilibrium(state.clone());
        let _col = ops.collision(state.clone(), 0.5);
    }
}
