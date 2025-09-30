//! # Lattice-Boltzmann Fluid Simulation

use burn::Tensor;
use burn::config::Config;
use burn::module::Module;
use burn::prelude::{Backend, Shape};
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

/// Partials of the VU Equilibrium Operation.
#[derive(Clone, Debug)]
pub struct VuEquiPartials<B: Backend, const D: usize> {
    /// `Vy` unit vector bias; shape expanded to ``[..., Vy, Vx; D]``
    pub ev: Tensor<B, D>,

    /// `Vx` unit vector bias; shape expanded to ``[..., Vy, Vx; D]``
    pub eu: Tensor<B, D>,

    /// `VyxVx` sum; shape expanded to ``[..., 1, 1; D]``
    pub rho: Tensor<B, D>,

    /// `Vy` partial; shape expanded to ``[..., 1, 1; D]``
    pub dvy: Tensor<B, D>,

    /// `Vx` partial; shape expanded to ``[..., 1, 1; D]``
    pub dvx: Tensor<B, D>,
}

/// Final Terms of the VU Equilibrium Operation.
#[derive(Clone, Debug)]
pub struct VuEquiTerms<B: Backend, const D: usize> {
    /// `VxU` sum; shape expanded to ``[..., 1, 1; D]``
    pub rho: Tensor<B, D>,

    /// TODO
    pub u_dot_e: Tensor<B, D>,

    /// TODO
    pub u_sq: Tensor<B, D>,
}

impl<B: Backend, const D: usize> VuEquiPartials<B, D> {
    /// Get the shape of the state.
    pub fn shape(&self) -> Shape {
        self.ev.shape()
    }

    /// Get the [`VuEquiTerms`] for these partials.
    pub fn equi_terms(self) -> VuEquiTerms<B, D> {
        let shape = self.shape();

        let u_dot_e: Tensor<B, D> = (self.ev * self.dvy.clone().expand(shape.clone()))
            + self.eu * self.dvx.clone().expand(shape.clone());

        let u_sq: Tensor<B, D> = self.dvy.powi_scalar(2) + self.dvx.powi_scalar(2);

        VuEquiTerms {
            rho: self.rho,
            u_dot_e,
            u_sq,
        }
    }
}

impl<B: Backend> LBMOperations<B> {
    /// Initialize LBM operations.
    pub fn init(device: &B::Device) -> Self {
        let ev = Tensor::<B, 2>::from_data(
            [[1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]],
            device,
        );
        let eu = Tensor::<B, 2>::from_data(
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

        Self { ev, eu, w }
    }

    /// Cast the operations to the given dtype.
    pub fn cast(
        self,
        dtype: DType,
    ) -> Self {
        Self {
            ev: self.ev.cast(dtype),
            eu: self.eu.cast(dtype),
            w: self.w.cast(dtype),
        }
    }

    /// LBM Cellular Equilibrium.
    ///
    /// # Arguments
    ///
    /// - `state`: LBM state tensor, with shape ``[..., Vy, Vx]`` over cell flow state.
    ///
    /// # Returns
    ///
    /// The equilibrium tensor, with the same shape as `state`.
    pub fn vu_cell_equilibrium<const D: usize>(
        &self,
        state: Tensor<B, D>,
    ) -> Tensor<B, D> {
        let shape = state.shape();
        assert!(D >= 2, "Rank must be at least 2: got {}", D);

        let VuEquiTerms { rho, u_dot_e, u_sq } = self.vu_equi_partials(state).equi_terms();

        let w: Tensor<B, D> = self.w.clone().expand::<D, _>(shape.clone());

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
    /// - `state`: a rank ``D`` state with shape ``[..., Vy, Vx]``.
    ///
    /// # Returns
    ///
    /// A rank ``D`` [`VuEquiPartials`].
    pub fn vu_equi_partials<const D: usize>(
        &self,
        state: Tensor<B, D>,
    ) -> VuEquiPartials<B, D> {
        let shape = state.shape();
        assert!(D >= 2, "Rank must be at least 2: got {}", D);

        let eu: Tensor<B, D> = self.eu.clone().expand(shape.clone());
        let ev: Tensor<B, D> = self.ev.clone().expand(shape.clone());

        let rho = sum_cell_2d(state.clone());
        let dv = sum_cell_2d(state.clone() * ev.clone()).div(rho.clone());
        let du = sum_cell_2d(state.clone() * eu.clone()).div(rho.clone());

        VuEquiPartials {
            ev,
            eu,
            rho,
            dvy: dv,
            dvx: du,
        }
    }

    /// LBM Collision step.
    ///
    /// # Arguments
    ///
    /// - `state`: LBM state tensor, with shape ``[..., Vy, Vx]`` over cell flow state.
    ///
    /// # Returns
    ///
    /// The collision tensor, with the same shape as `state`.
    pub fn collision<const D: usize>(
        &self,
        state: Tensor<B, D>,
        tau: f32,
    ) -> Tensor<B, D> {
        let equi = self.vu_cell_equilibrium(state.clone());
        let delta = equi - state.clone();
        state + delta / tau
    }

    /// Compute the streaming updates for the non-border cells of a state.
    ///
    /// # Arguments
    ///
    /// - `state`: a ``[H, W, V=3, U=3]`` input.
    ///
    /// # Returns
    ///
    /// The stream updates for the ``[1:-1, 1:-1, V=3, U=3]`` interior.
    pub fn interior_streaming_updates(
        &self,
        state: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        streaming_window_op(
            state
                .unfold::<5, usize>(0, 3, 1)
                .unfold::<6, usize>(1, 3, 1),
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
/// Note: This is the ``[..., H=3, W=3, Vy=3, Vx=3]`` view;
/// though the input is ``[..., Vy=3, Vx=3, H=3, W=3]``.
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
    state: Tensor<B, D>
) -> Tensor<B, D2> {
    // state: [..., Vy=3, Vx=3, H_KERN=3, W_KERN=3]
    // output: [..., Vy=3, Vx=3]

    assert_eq!(D - 2, D2, "D ({D}) - 2 must equal D2 ({D2})");

    #[cfg(debug_assertions)]
    bimm_contracts::assert_shape_contract_periodically!(
        [..., "Vy", "Vx", "HK", "WK"],
        &state.shape().dims,
        &[("Vy", 3), ("Vx", 3), ("HK", 3), ("WK", 3)]
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
        rows.push(Tensor::cat(columns, D - 3));
    }

    // Concatenate along V dimension
    Tensor::cat(rows, D - 4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;
    use burn::tensor::{Distribution, Tolerance};

    #[test]
    fn test_expand_vu_cell_sum() {
        let device = Default::default();

        let input: Tensor<Wgpu, 3> = Tensor::from_data(
            [
                [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                [[10., 20., 30.], [40., 50., 60.], [70., 80., 90.]],
            ],
            &device,
        );

        let result = sum_cell_2d::<Wgpu, 3>(input.clone());

        let expected: Tensor<Wgpu, 3> = Tensor::from_data([[[45.]], [[450.]]], &device);
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

        let partials = ops.vu_equi_partials::<3>(input.clone());

        partials.ev.clone().to_data().assert_eq(
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
        partials.eu.clone().to_data().assert_eq(
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

        partials.dvy.clone().to_data().assert_eq(
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

        partials.dvx.clone().to_data().assert_eq(
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
            &(partials.ev.clone() * partials.dvy.clone().expand(shape.clone())
                + partials.eu.clone() * partials.dvx.clone().expand(shape))
            .to_data(),
            Tolerance::default(),
        );
    }

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
        let _eq = ops.vu_cell_equilibrium(state.clone());
        let _col = ops.collision(state.clone(), 0.5);
    }

    #[test]
    fn test_streaming_updates() {
        let device = Default::default();

        let ops = LBMOperations::<Wgpu>::init(&device);

        let state =
            Tensor::<Wgpu, 4>::random([10, 10, 3, 3], Distribution::Normal(0., 1.), &device);

        let updates = ops.interior_streaming_updates(state.clone());
        assert_eq!(updates.shape().dims(), [10 - 2, 10 - 2, 3, 3]);
    }
}
