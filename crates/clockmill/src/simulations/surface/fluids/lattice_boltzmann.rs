//! # Lattice-Boltzmann Fluid Simulation

use burn::Tensor;
use burn::config::Config;
use burn::module::Module;
use burn::prelude::{Backend, Bool, Shape};
use burn::tensor::{DType, Slice};

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
        let [h, w] = self.shape;
        let state = Tensor::<B, 4>::zeros([h, w, 3, 3], device);
        let solid_mask = Tensor::<B, 2>::zeros([h, w], device).bool();

        LBM {
            step_count: 0,
            state,
            solid_mask,
        }
    }
}

/// Lattice-Boltzmann Fluid Simulation State Module
#[derive(Module, Debug)]
pub struct LBM<B: Backend> {
    /// The current simulation step.
    pub step_count: u64,

    /// The world state: ``[H, W, UY, UX]``
    pub state: Tensor<B, 4>,

    /// The solid mask: ``[H, W]``
    pub solid_mask: Tensor<B, 2, Bool>,
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

    /// Recast the datatype of the state.
    pub fn cast(
        self,
        dtype: DType,
    ) -> Self {
        Self {
            state: self.state.cast(dtype),
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

/// LBM Operations
#[derive(Module, Debug)]
pub struct LBMOperations<B: Backend> {
    /// Vertical direction vectors in 3x3 layout.
    pub ey: Tensor<B, 2>,

    /// Horizontal direction vectors in 3x3 layout.
    pub ex: Tensor<B, 2>,

    /// Weights in 3x3 layout.
    pub w: Tensor<B, 2>,
}

/// Partials of the VU Equilibrium Operation.
#[derive(Clone, Debug)]
pub struct UCellPartials<B: Backend, const D: usize> {
    /// `UY` unit vector bias; shape expanded to ``[..., UY, UX; D]``
    pub ey: Tensor<B, D>,

    /// `UX` unit vector bias; shape expanded to ``[..., UY, UX; D]``
    pub ex: Tensor<B, D>,

    /// `UYxUX` sum; shape expanded to ``[..., 1, 1; D]``
    pub rho: Tensor<B, D>,

    /// `UY` partial; shape expanded to ``[..., 1, 1; D]``
    pub duy: Tensor<B, D>,

    /// `UX` partial; shape expanded to ``[..., 1, 1; D]``
    pub dux: Tensor<B, D>,
}

/// Final Terms of the VU Equilibrium Operation.
#[derive(Clone, Debug)]
pub struct UCellTerms<B: Backend, const D: usize> {
    /// `UXU` sum; shape expanded to ``[..., 1, 1; D]``
    pub rho: Tensor<B, D>,

    /// TODO
    pub u_dot_e: Tensor<B, D>,

    /// TODO
    pub u_sq: Tensor<B, D>,
}

impl<B: Backend, const D: usize> UCellPartials<B, D> {
    /// Get the shape of the state.
    pub fn shape(&self) -> Shape {
        self.ey.shape()
    }

    /// Get the [`UCellTerms`] for these partials.
    pub fn equi_terms(self) -> UCellTerms<B, D> {
        let shape = self.shape();

        let u_dot_e: Tensor<B, D> = (self.ey * self.duy.clone().expand(shape.clone()))
            + self.ex * self.dux.clone().expand(shape.clone());

        let u_sq: Tensor<B, D> = self.duy.powi_scalar(2) + self.dux.powi_scalar(2);

        UCellTerms {
            rho: self.rho,
            u_dot_e,
            u_sq,
        }
    }
}

impl<B: Backend> LBMOperations<B> {
    /// Initialize LBM operations.
    pub fn init(device: &B::Device) -> Self {
        let ey = Tensor::<B, 2>::from_data(
            [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]],
            device,
        );
        let ex = Tensor::<B, 2>::from_data(
            [[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]],
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

        Self { ey, ex, w }
    }

    /// Cast the operations to the given dtype.
    pub fn cast(
        self,
        dtype: DType,
    ) -> Self {
        Self {
            ey: self.ey.cast(dtype),
            ex: self.ex.cast(dtype),
            w: self.w.cast(dtype),
        }
    }

    /// LBM Cellular Equilibrium.
    ///
    /// # Arguments
    ///
    /// - `state`: LBM state tensor, with shape ``[..., UY, UX]`` over cell flow state.
    ///
    /// # Returns
    ///
    /// The equilibrium tensor, with the same shape as `state`.
    pub fn ucell_equilibrium(
        &self,
        state: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let shape = state.shape();

        let UCellTerms { rho, u_dot_e, u_sq } = self.ucell_partials(state).equi_terms();

        let w: Tensor<B, 4> = self.w.clone().expand(shape.clone());

        w * rho
            * (1.0 + 3.0 * u_dot_e.clone() + 4.5 * u_dot_e.powi_scalar(2)
                - 1.5 * u_sq.expand(shape))
    }

    /// Compute the directional VU cell partials.
    ///
    /// These are intermediate sums used in computing equilibrium flow;
    /// primarily exposed as partial values for testing.
    ///
    /// # Argument
    ///
    /// - `state`: a rank ``D`` state with shape ``[..., UY, UX]``.
    ///
    /// # Returns
    ///
    /// A rank ``D`` [`UCellPartials`].
    pub fn ucell_partials<const D: usize>(
        &self,
        state: Tensor<B, D>,
    ) -> UCellPartials<B, D> {
        let shape = state.shape();
        assert!(D >= 2, "Rank must be at least 2: got {}", D);

        let ex: Tensor<B, D> = self.ex.clone().expand(shape.clone());
        let ey: Tensor<B, D> = self.ey.clone().expand(shape.clone());

        let rho = sum_cell_2d(state.clone());
        let duy = sum_cell_2d(state.clone() * ey.clone()).div(rho.clone());
        let dux = sum_cell_2d(state.clone() * ex.clone()).div(rho.clone());

        UCellPartials {
            ey,
            ex,
            rho,
            duy,
            dux,
        }
    }

    /// LBM Collision step.
    ///
    /// # Arguments
    ///
    /// - `state`: LBM state tensor, with shape ``[..., UY, UX]`` over cell flow state.
    ///
    /// # Returns
    ///
    /// The collision tensor, with the same shape as `state`.
    pub fn collision(
        &self,
        state: Tensor<B, 4>,
        tau: f32,
    ) -> Tensor<B, 4> {
        let equi = self.ucell_equilibrium(state.clone());
        let delta = equi - state.clone();
        state + delta / tau
    }

    /// Compute the streaming updates for the non-border cells of a state.
    ///
    /// # Arguments
    ///
    /// - `state`: a ``[H, W, V=3, U=3]`` input.
    /// - `solid_mask`: a ``[H, W]`` solid mask.
    ///
    /// # Returns
    ///
    /// The stream updates for the ``[1:-1, 1:-1, V=3, U=3]`` interior.
    pub fn interior_streaming_updates(
        &self,
        state: Tensor<B, 4>,
        solid_mask: Tensor<B, 2, Bool>,
    ) -> Tensor<B, 4> {
        streaming_window_op(
            state
                .unfold::<5, usize>(0, 3, 1)
                .unfold::<6, usize>(1, 3, 1),
            solid_mask
                .unfold::<3, usize>(0, 3, 1)
                .unfold::<4, usize>(1, 3, 1),
        )
    }
}

/// Sums a ``[..., A, B]`` cell.
///
/// # Arguments
///
/// - `state`: a ``[..., A, B]`` input.
///
/// # Returns
///
/// A ``[..., 1, 1]`` result.
pub fn sum_cell_2d<B: Backend, const D: usize>(state: Tensor<B, D>) -> Tensor<B, D> {
    state.sum_dim(D - 2).sum_dim(D - 1)
}

/// Lattice-Boltzmann Method Streaming Operation.
///
/// This operation applies the LBM streaming operation,
/// swapping complementary direction pairs from neighboring cells.
///
/// # Example
///
/// Note: This is the ``[..., H=3, W=3, UY=3, UX=3]`` view;
/// though the input is ``[..., UY=3, UX=3, H=3, W=3]``.
///
/// ### From
///
/// ```text
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
/// ```
///
/// ### To
///
/// ```text
/// | a, b, c |
/// | d, e, f |
/// | h, i, j |
/// ```
pub fn streaming_window_op<B: Backend, const D: usize, const D2: usize>(
    state: Tensor<B, D>,
    _solid_mask: Tensor<B, D2, Bool>,
) -> Tensor<B, D2> {
    // state: [..., UY=3, UX=3, H_KERN=3, W_KERN=3]
    // output: [..., UY=3, UX=3]

    assert_eq!(D - 2, D2, "D ({D}) - 2 must equal D2 ({D2})");

    #[cfg(debug_assertions)]
    bimm_contracts::assert_shape_contract_periodically!(
        [..., "UY", "UX", "HK", "WK"],
        &state.shape().dims,
        &[("UY", 3), ("UX", 3), ("HK", 3), ("WK", 3)]
    );

    let mut ranges: [Slice; D] = (0..D)
        .map(|_| Slice::new(0, None, 1))
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    let mut rows = Vec::with_capacity(3);
    for h_idx in 0isize..3 {
        let target_vy = h_idx;
        let source_vy = 2 - h_idx;

        ranges[D - 4] = Slice::new(source_vy, Some(source_vy + 1), 1);
        ranges[D - 2] = Slice::new(target_vy, Some(target_vy + 1), 1);

        let mut columns = Vec::with_capacity(3);
        for w_idx in 0isize..3 {
            let target_vx = w_idx;
            let source_vx = 2 - w_idx;

            ranges[D - 3] = Slice::new(source_vx, Some(source_vx + 1), 1);
            ranges[D - 1] = Slice::new(target_vx, Some(target_vx + 1), 1);

            let column = state
                .clone()
                .slice(ranges.clone())
                .squeeze_dims::<D2>(&[-2, -1]);

            columns.push(column);
        }
        // Concatenate along U dimension
        rows.push(Tensor::cat(columns, D2 - 1));
    }

    // Concatenate along V dimension
    Tensor::cat(rows, D2 - 2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;
    use burn::tensor::{Distribution, Tolerance};

    #[test]
    fn test_expand_vu_cell_sum() {
        type B = Wgpu;
        let device = Default::default();

        let input: Tensor<B, 3> = Tensor::from_data(
            [
                [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                [[10., 20., 30.], [40., 50., 60.], [70., 80., 90.]],
            ],
            &device,
        );

        let result = sum_cell_2d::<B, 3>(input.clone());

        let expected: Tensor<B, 3> = Tensor::from_data([[[45.]], [[450.]]], &device);
        result.to_data().assert_eq(&expected.to_data(), true);
    }

    #[test]
    fn test_vu_partials() {
        type B = Wgpu;
        let device = Default::default();

        let ops: LBMOperations<B> = LBMOperations::init(&device);

        let input: Tensor<B, 3> = Tensor::from_data(
            [
                [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                [[10., 20., 30.], [40., 50., 60.], [70., 80., 90.]],
            ],
            &device,
        );
        let shape = input.shape();

        let partials = ops.ucell_partials::<3>(input.clone());

        partials.ey.clone().to_data().assert_eq(
            &Tensor::<B, 3>::from_data(
                [
                    [[1., 1., 1.], [0., 0., 0.], [-1., -1., -1.]],
                    [[1., 1., 1.], [0., 0., 0.], [-1., -1., -1.]],
                ],
                &device,
            )
            .to_data(),
            true,
        );
        partials.ex.clone().to_data().assert_eq(
            &Tensor::<B, 3>::from_data(
                [
                    [[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]],
                    [[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]],
                ],
                &device,
            )
            .to_data(),
            true,
        );

        partials.rho.clone().to_data().assert_eq(
            &Tensor::<B, 3>::from_data([[[45.]], [[450.]]], &device).to_data(),
            true,
        );

        partials.duy.clone().to_data().assert_eq(
            &Tensor::<B, 3>::from_data(
                [
                    [[((1. + 2. + 3.) - (7. + 8. + 9.)) / 45.]],
                    [[((10. + 20. + 30.) - (70. + 80. + 90.)) / 450.]],
                ],
                &device,
            )
            .to_data(),
            true,
        );

        partials.dux.clone().to_data().assert_eq(
            &Tensor::<B, 3>::from_data(
                [
                    [[((3. + 6. + 9.) - (1. + 4. + 7.)) / 45.]],
                    [[((30. + 60. + 90.) - (10. + 40. + 70.)) / 450.]],
                ],
                &device,
            )
            .to_data(),
            true,
        );

        let terms = partials.clone().equi_terms();
        terms.rho.clone().to_data().assert_eq(
            &Tensor::<B, 3>::from_data([[[45.]], [[450.]]], &device).to_data(),
            true,
        );

        let vs_a = (((1. + 2. + 3.) - (7. + 8. + 9.)) / 45f32).powi(2);
        let vs_b = (((10. + 20. + 30.) - (70. + 80. + 90.)) / 450f32).powi(2);

        let us_a = (((3. + 6. + 9.) - (1. + 4. + 7.)) / 45f32).powi(2);
        let us_b = (((30. + 60. + 90.) - (10. + 40. + 70.)) / 450f32).powi(2);

        terms.u_sq.clone().to_data().assert_approx_eq::<f32>(
            &Tensor::<B, 3>::from_data([[[vs_a + us_a]], [[vs_b + us_b]]], &device).to_data(),
            Tolerance::default(),
        );

        terms.u_dot_e.clone().to_data().assert_approx_eq::<f32>(
            &(partials.ey.clone() * partials.duy.clone().expand(shape.clone())
                + partials.ex.clone() * partials.dux.clone().expand(shape))
            .to_data(),
            Tolerance::default(),
        );
    }

    #[test]
    #[rustfmt::skip]
    fn test_interior_streaming_updates() {
        type B = Wgpu;
        let device = Default::default();

        let state: Tensor<B, 4> = Tensor::from_data([
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
        let solid_mask = Tensor::<B, 2>::zeros([3, 3], &device).bool();

        let ops = LBMOperations::<B>::init(&device);

        let result = ops.interior_streaming_updates(state.clone(), solid_mask.clone());

        assert_eq!(result.shape().dims, vec![1, 1, 3, 3]);

        let expected: Tensor<B, 4> = Tensor::from_data([[[
            [8., 16., 24.],
            [32., 40., 48.],
            [56., 64., 72.],
        ]]], &device);

        result.to_data().assert_eq(&expected.to_data(), false);
    }

    #[test]
    fn test_lbm_init() {
        type B = Wgpu;
        let device = Default::default();

        let config = LBMConfig::new([10, 12]);
        assert_eq!(config.shape(), [10, 12]);

        let lbm: LBM<B> = config.init(&device);
        assert_eq!(lbm.shape(), [10, 12]);
        assert_eq!(lbm.step_count(), 0);
        assert_eq!(lbm.device(), device);
        assert_eq!(lbm.state.shape().dims(), [10, 12, 3, 3]);
    }

    #[test]
    fn test_equilibrium() {
        type B = Wgpu;
        let device = Default::default();

        let ops = LBMOperations::<B>::init(&device);

        let state = Tensor::<B, 4>::random([1, 1, 3, 3], Distribution::Normal(0., 1.), &device);
        let _eq = ops.ucell_equilibrium(state.clone());
        let _col = ops.collision(state.clone(), 0.5);
    }

    #[test]
    fn test_streaming_updates() {
        type B = Wgpu;
        let device = Default::default();

        let ops = LBMOperations::<B>::init(&device);

        let state = Tensor::<B, 4>::random([10, 10, 3, 3], Distribution::Normal(0., 1.), &device);
        let solid_mask = Tensor::<B, 2>::zeros([10, 10], &device).bool();

        let updates = ops.interior_streaming_updates(state.clone(), solid_mask.clone());
        assert_eq!(updates.shape().dims(), [10 - 2, 10 - 2, 3, 3]);
    }
}
