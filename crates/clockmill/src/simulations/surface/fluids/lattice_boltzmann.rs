//! # Lattice-Boltzmann Fluid Simulation

use bimm_contracts::{assert_shape_contract_periodically, unpack_shape_contract};
use burn::Tensor;
use burn::config::Config;
use burn::module::Module;
use burn::prelude::{Backend, Bool, Shape};
use burn::tensor::{DType, Slice};

use crate::compat::operations::sum_dims;

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
        device: &B::Device,
    ) -> LBMD2Q9State<B> {
        let [h, w] = self.shape;
        let state = Tensor::<B, 4>::zeros([h, w, 3, 3], device);
        let solid_mask = Tensor::<B, 2>::zeros([h, w], device).bool();

        LBMD2Q9State {
            step_count: 0,
            velocity: state,
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
    pub velocity: Tensor<B, 4>,

    /// The solid mask: ``[H, W]``
    pub solid_mask: Tensor<B, 2, Bool>,
}

impl<B: Backend> LBMMeta for LBMD2Q9State<B> {
    fn shape(&self) -> [usize; 2] {
        let dims = &self.velocity.shape().dims;
        [dims[0], dims[1]]
    }
}

impl<B: Backend> LBMD2Q9State<B> {
    /// Get the device the module is on.
    pub fn device(&self) -> B::Device {
        self.velocity.device()
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
            velocity: self.velocity.cast(dtype),
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
pub struct LBMD2Q9Operations<B: Backend> {
    /// Vertical direction vectors in 3x3 layout.
    pub ey: Tensor<B, 2>,

    /// Horizontal direction vectors in 3x3 layout.
    pub ex: Tensor<B, 2>,

    /// Weights in 3x3 layout.
    pub w: Tensor<B, 2>,
}

/// Partials of the VU Equilibrium Operation.
#[derive(Clone, Debug)]
pub struct EquilibriumPartials<B: Backend> {
    /// `UY` unit vector bias; broadcast to ``[H, W, UY=3, UX=3]``
    pub ey: Tensor<B, 4>,

    /// `UX` unit vector bias; broadcast to ``[H, W, UY=3, UX=3]``
    pub ex: Tensor<B, 4>,

    /// The grid density, broadcast to ``[H, W, 1, 1]``
    pub rho: Tensor<B, 4>,

    /// `UY` partial; broadcast to ``[H, W, 1, 1]``
    pub duy: Tensor<B, 4>,

    /// `UX` partial; broadcast to ``[H, W, 1, 1]``
    pub dux: Tensor<B, 4>,
}

impl<B: Backend> EquilibriumPartials<B> {
    /// Get the shape of the state.
    pub fn shape(&self) -> Shape {
        self.ey.shape()
    }

    /// Get the [`EquilibriumTerms`] for these partials.
    pub fn equi_terms(self) -> EquilibriumTerms<B> {
        let shape = self.shape();

        let u_dot_e: Tensor<B, 4> = (self.ey * self.duy.clone().expand(shape.clone()))
            + self.ex * self.dux.clone().expand(shape.clone());

        let u_sq: Tensor<B, 4> = self.duy.powi_scalar(2) + self.dux.powi_scalar(2);

        EquilibriumTerms {
            rho: self.rho,
            u_dot_e,
            u_sq,
        }
    }
}

/// Final Terms of the VU Equilibrium Operation.
#[derive(Clone, Debug)]
pub struct EquilibriumTerms<B: Backend> {
    /// The grid density, broadcast to ``[H, W, 1, 1]``
    pub rho: Tensor<B, 4>,

    /// TODO
    pub u_dot_e: Tensor<B, 4>,

    /// TODO
    pub u_sq: Tensor<B, 4>,
}


impl<B: Backend> LBMD2Q9Operations<B> {
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

    /// Re-cast to the given dtype.
    pub fn to_dtype(
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
    /// - `velocity`: with shape ``[H, W, UY=3, UX=3]``.
    ///
    /// # Returns
    ///
    /// The velocity equilibrium, with the same shape.
    pub fn equilibrium(
        &self,
        velocity: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let shape = velocity.shape();

        let EquilibriumTerms { rho, u_dot_e, u_sq } = self.equilibrium_partials(velocity).equi_terms();

        let w: Tensor<B, 4> = self.w.clone().expand(shape.clone());

        w * rho
            * (1.0 + 3.0 * u_dot_e.clone() + 4.5 * u_dot_e.powi_scalar(2)
                - 1.5 * u_sq.expand(shape))
    }

    /// placeholder.
    pub fn density(
        &self,
        velocity: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        sum_dims(velocity, &[2, 3])
    }

    /// Compute the directional VU cell partials.
    ///
    /// These are intermediate sums used in computing equilibrium flow;
    /// primarily exposed as partial values for testing.
    ///
    /// # Argument
    ///
    /// - `velocity`: with shape ``[H, W, UY=3, UX=3]``.
    ///
    /// # Returns
    ///
    /// A rank ``D`` [`EquilibriumPartials`].
    pub fn equilibrium_partials(
        &self,
        velocity: Tensor<B, 4>,
    ) -> EquilibriumPartials<B> {
        let shape = velocity.shape();

        let ex: Tensor<B, 4> = self.ex.clone().expand(shape.clone());
        let ey: Tensor<B, 4> = self.ey.clone().expand(shape.clone());

        // The grid density, broadcast to ``[H, W, 1, 1]``
        let rho = self.density(velocity.clone()).expand(shape.clone());

        let duy = sum_dims(velocity.clone() * ey.clone(), &[- 2, - 1]).div(rho.clone());
        let dux = sum_dims(velocity * ex.clone(), &[- 2, - 1]).div(rho.clone());

        EquilibriumPartials {
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
    /// - `velocity`: with shape ``[H, W, UY, UX]``
    ///
    /// # Returns
    ///
    /// The updated velocity tensor.
    pub fn collision(
        &self,
        velocity: Tensor<B, 4>,
        tau: f32,
    ) -> Tensor<B, 4> {
        let equi = self.equilibrium(velocity.clone());
        // s + (e - s) / t
        // ( s t + e - s ) / t
        // ( s t - s + e ) / t
        // ( s (t - 1) + e ) / t
        (velocity * (tau - 1.0) + equi) / tau
    }

    /// Compute the streaming updates for the non-border cells of a state.
    ///
    /// # Arguments
    ///
    /// - `velocity`: a ``[H, W, V=3, U=3]`` input.
    /// - `solid_mask`: a ``[H, W]`` solid mask.
    ///
    /// # Returns
    ///
    /// The updated velocity for the ``[1:-1, 1:-1, V=3, U=3]`` interior.
    pub fn interior_streaming(
        &self,
        velocity: Tensor<B, 4>,
        _solid_mask: Tensor<B, 2, Bool>,
    ) -> Tensor<B, 4> {
        let [h, w] = unpack_shape_contract!(
            ["H", "W", "UY", "UX"],
            &velocity.shape().dims,
            &["H", "W"],
            &[("UY", 3), ("UX", 3)]
        );

        // Map the state into no-copy 3x3 neighborhood windows.
        let velocity_windows = velocity
            .unfold::<5, usize>(0, 3, 1)
            .unfold::<6, usize>(1, 3, 1);

        // TODO: implement bounce.
        // This requires computing 3x3x2 columns;
        // and using a where operation on the solid_mask:
        // cell = where(mask_cell, bounce_cell, stream_cell)

        assert_shape_contract_periodically!(
            ["H" - "PAD", "W" - "PAD", "UY", "UX", "HK", "WK"],
            &velocity_windows.shape().dims,
            &[
                ("H", h),
                ("W", w),
                ("PAD", 2),
                ("UY", 3),
                ("UX", 3),
                ("HK", 3),
                ("WK", 3)
            ]
        );

        // Allocate a mutable selector over the windows.
        let mut ranges: [Slice; 6] = (0..6)
            .map(|_| Slice::new(0, None, 1))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let mut rows = Vec::with_capacity(3);
        for hk_idx in 0..3 {
            // For each hk index, we have a row of 3 columns.

            // Select the hk neighbor from the windows:
            ranges[4] = Slice::new(hk_idx, Some(hk_idx + 1), 1);

            // Select the complimentary vy flow in that neighbor.
            let vy_source_idx = 2 - hk_idx;
            ranges[2] = Slice::new(vy_source_idx, Some(vy_source_idx + 1), 1);

            let mut columns = Vec::with_capacity(3);
            for wk_idx in 0..3 {
                // For each wk index, we have a column of 3 cells.

                // Select the wk neighbor from the windows:
                ranges[5] = Slice::new(wk_idx, Some(wk_idx + 1), 1);

                // Select the complimentary vx flow in that neighbor.
                let vx_source_idx = 2 - wk_idx;
                ranges[3] = Slice::new(vx_source_idx, Some(vx_source_idx + 1), 1);

                // `ranges` now contains a slice selector for one of the 9 cells:
                // ranges = [.., .., vy_source_idx, vx_source_idx, h_win_idx, w_win_idx]
                let cell = velocity_windows
                    .clone()
                    .slice(ranges.clone())
                    .squeeze_dims::<4>(&[-2, -1]);

                assert_shape_contract_periodically!(
                    ["H" - "PAD", "W" - "PAD", "C", "C"],
                    &cell.shape().dims,
                    &[("H", h), ("W", w), ("PAD", 2), ("C", 1)]
                );

                // Collect the cell into the column vector.
                columns.push(cell);
            }
            // Concatenate the 3 column cells into a row.
            let row = Tensor::cat(columns, 3);

            assert_shape_contract_periodically!(
                ["H" - "PAD", "W" - "PAD", "C", "UX"],
                &row.shape().dims,
                &[("H", h), ("W", w), ("PAD", 2), ("C", 1), ("UX", 3)]
            );

            // Collect the row into the rows vector.
            rows.push(row);
        }
        // Concatenate the 3 rows into a result.
        let result = Tensor::cat(rows, 2);

        assert_shape_contract_periodically!(
            ["H" - "PAD", "W" - "PAD", "UY", "UX"],
            &result.shape().dims,
            &[("H", h), ("W", w), ("PAD", 2), ("UY", 3), ("UX", 3)]
        );

        result
    }
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

        let state = input.clone();
        let result = sum_dims(state, &[- 2, - 1]);

        let expected: Tensor<B, 3> = Tensor::from_data([[[45.]], [[450.]]], &device);
        result.to_data().assert_eq(&expected.to_data(), true);
    }

    #[test]
    fn test_vu_partials() {
        type B = Wgpu;
        let device = Default::default();

        let ops: LBMD2Q9Operations<B> = LBMD2Q9Operations::init(&device);

        let input: Tensor<B, 4> = Tensor::from_data(
            [[
                [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                [[10., 20., 30.], [40., 50., 60.], [70., 80., 90.]],
            ]],
            &device,
        );
        let shape = input.shape();

        let partials = ops.equilibrium_partials(input.clone());

        partials.ey.clone().to_data().assert_eq(
            &Tensor::<B, 4>::from_data(
                [[
                    [[1., 1., 1.], [0., 0., 0.], [-1., -1., -1.]],
                    [[1., 1., 1.], [0., 0., 0.], [-1., -1., -1.]],
                ]],
                &device,
            )
            .to_data(),
            true,
        );
        partials.ex.clone().to_data().assert_eq(
            &Tensor::<B, 4>::from_data(
                [[
                    [[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]],
                    [[-1., 0., 1.], [-1., 0., 1.], [-1., 0., 1.]],
                ]],
                &device,
            )
            .to_data(),
            true,
        );

        partials.rho.clone().to_data().assert_eq(
            &Tensor::<B, 4>::from_data([[[[45.]], [[450.]]]], &device).to_data(),
            true,
        );

        partials.duy.clone().to_data().assert_eq(
            &Tensor::<B, 4>::from_data(
                [[
                    [[((1. + 2. + 3.) - (7. + 8. + 9.)) / 45.]],
                    [[((10. + 20. + 30.) - (70. + 80. + 90.)) / 450.]],
                ]],
                &device,
            )
            .to_data(),
            true,
        );

        partials.dux.clone().to_data().assert_eq(
            &Tensor::<B, 4>::from_data(
                [[
                    [[((3. + 6. + 9.) - (1. + 4. + 7.)) / 45.]],
                    [[((30. + 60. + 90.) - (10. + 40. + 70.)) / 450.]],
                ]],
                &device,
            )
            .to_data(),
            true,
        );

        let terms = partials.clone().equi_terms();
        terms.rho.clone().to_data().assert_eq(
            &Tensor::<B, 4>::from_data([[[[45.]], [[450.]]]], &device).to_data(),
            true,
        );

        let vs_a = (((1. + 2. + 3.) - (7. + 8. + 9.)) / 45f32).powi(2);
        let vs_b = (((10. + 20. + 30.) - (70. + 80. + 90.)) / 450f32).powi(2);

        let us_a = (((3. + 6. + 9.) - (1. + 4. + 7.)) / 45f32).powi(2);
        let us_b = (((30. + 60. + 90.) - (10. + 40. + 70.)) / 450f32).powi(2);

        terms.u_sq.clone().to_data().assert_approx_eq::<f32>(
            &Tensor::<B, 4>::from_data([[[[vs_a + us_a]], [[vs_b + us_b]]]], &device).to_data(),
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

        let ops = LBMD2Q9Operations::<B>::init(&device);

        let result = ops.interior_streaming(state.clone(), solid_mask.clone());

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

        let config = LBMD2Q9Config::new([10, 12]);
        assert_eq!(config.shape(), [10, 12]);

        let lbm: LBMD2Q9State<B> = config.init(&device);
        assert_eq!(lbm.shape(), [10, 12]);
        assert_eq!(lbm.step_count(), 0);
        assert_eq!(lbm.device(), device);
        assert_eq!(lbm.velocity.shape().dims(), [10, 12, 3, 3]);
    }

    #[test]
    fn test_equilibrium() {
        type B = Wgpu;
        let device = Default::default();

        let ops = LBMD2Q9Operations::<B>::init(&device);

        let state = Tensor::<B, 4>::random([1, 1, 3, 3], Distribution::Normal(0., 1.), &device);
        let _eq = ops.equilibrium(state.clone());
        let _col = ops.collision(state.clone(), 0.5);
    }

    #[test]
    fn test_streaming_updates() {
        type B = Wgpu;
        let device = Default::default();

        let ops = LBMD2Q9Operations::<B>::init(&device);

        let state = Tensor::<B, 4>::random([10, 10, 3, 3], Distribution::Normal(0., 1.), &device);
        let solid_mask = Tensor::<B, 2>::zeros([10, 10], &device).bool();

        let updates = ops.interior_streaming(state.clone(), solid_mask.clone());
        assert_eq!(updates.shape().dims(), [10 - 2, 10 - 2, 3, 3]);
    }
}
