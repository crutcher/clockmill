//! # D2Q9 Lattice-Boltzmann Fluid Simulation

use crate::compat::operations::sum_dims;
use bimm_contracts::{assert_shape_contract_periodically, unpack_shape_contract};
use burn::Tensor;
use burn::prelude::{Backend, Bool};
use burn::tensor::Slice;

/// Population Density
///
/// # Arguments
///
/// - `dist`: a ``[H, W, Y=3, X=3]`` population distribution.
///
/// # Returns
///
/// A ``[H, W]`` population density.
pub fn population_density<B: Backend>(dist: Tensor<B, 4>) -> Tensor<B, 2> {
    sum_dims(dist, &[2, 3]).squeeze_dims::<2>(&[2, 3])
}

/// D2Q9 Direction Vectors
///
/// # Returns
///
/// The ``[Y=3, X=3, (Y, X)=2]`` direction vectors.
pub fn direction_vectors<B: Backend>(device: &B::Device) -> Tensor<B, 3> {
    Tensor::<B, 3>::from_data(
        [
            [[1., -1.], [1., 0.], [1., 1.]],
            [[0., -1.], [0., 0.], [0., 1.]],
            [[-1., -1.], [-1., 0.], [-1., 1.]],
        ],
        device,
    )
}

/// D2Q9 Equilibrium Weight Matrix
///
/// # Returns
///
/// The ``[Y=3, X=3]`` equilibrium weight matrix.
pub fn weight_matrix<B: Backend>(device: &B::Device) -> Tensor<B, 2> {
    Tensor::<B, 2>::from_data(
        [
            [1.0 / 36.0, 1.0 / 9.0, 1.0 / 36.0],
            [1.0 / 9.0, 4.0 / 9.0, 1.0 / 9.0],
            [1.0 / 36.0, 1.0 / 9.0, 1.0 / 36.0],
        ],
        device,
    )
}

/// Compute the directional macroscopic momentum.
///
/// This is the unnormalized macroscopic momentum.
///
/// # Arguments
///
/// - `dist`: a ``[H, W, Y, X]`` population distribution.
/// - `e`: the D2Q9 direction vectors.
///
/// # Returns
///
/// The ``[H, W, (Y, X)=2]`` momentum.
pub fn macroscopic_momentum<B: Backend>(
    dist: Tensor<B, 4>,
    e: Tensor<B, 3>,
) -> Tensor<B, 3> {
    sum_dims(
        dist.unsqueeze_dims::<5>(&[-1])
            .mul(e.clone().unsqueeze::<5>()),
        &[2, 3],
    )
    .squeeze_dims::<3>(&[2, 3])
}

/// Computes directional velocity from macroscopic momentum.
///
/// # Arguments
///
/// - `m`: ``[H, W, (Y, X)=2]`` macroscopic momentum.
/// - `rho`: ``[H, W]`` population density.
///
/// # Returns
///
/// The ``[H, W, (Y, X)=2]`` velocity.
pub fn normalize_velocity<B: Backend>(
    m: Tensor<B, 3>,
    rho: Tensor<B, 2>,
) -> Tensor<B, 3> {
    m / rho.unsqueeze_dim(2)
}

/// Compute the directional macroscopic velocity.
///
/// # Arguments
///
/// - `dist`: a ``[H, W, Y, X]`` population distribution.
/// - `e`: the D2Q9 direction vectors.
///
/// # Returns
///
/// The ``[H, W, (Y, X)=2]`` velocity.
pub fn macroscopic_velocity<B: Backend>(
    dist: Tensor<B, 4>,
    rho: Tensor<B, 2>,
    e: Tensor<B, 3>,
) -> Tensor<B, 3> {
    normalize_velocity(macroscopic_momentum(dist, e), rho)
}

/// Compute the squared magnitude of velocity field.
///
/// # Arguments
/// - `u`: `[H, W, (Y, X)=2]` macroscopic velocity
///
/// # Returns
/// - `[H, W]` velocity magnitude squared
pub fn velocity_squared<B: Backend>(u: Tensor<B, 3>) -> Tensor<B, 2> {
    // TODO: Benchmark:
    // Tensor::powi_scalar(2) is still a float pow operation.
    // * u.powi_scalar(2)
    // * u * u
    u.powi_scalar(2).sum_dim(2).squeeze_dims::<2>(&[2])
}

/// Compute e·u for each lattice direction
///
/// # Arguments
/// - `e`: `[Y=3, X=3, (Y,X)=2]` direction vectors
/// - `u`: `[H, W, (Y,X)=2]` macroscopic velocity
///
/// # Returns
/// - `[H, W, Y=3, X=3]` dot product at each grid point and direction
pub fn lattice_dot_velocity<B: Backend>(
    u: Tensor<B, 3>,
    e: Tensor<B, 3>,
) -> Tensor<B, 4> {
    // e * u[..., None, None, :] -> [H, W, Y, X, 2]
    // sum over component dimension -> [H, W, Y, X]
    (e.unsqueeze::<5>() * u.unsqueeze_dims::<5>(&[2, 3]))
        .sum_dim(4)
        .squeeze_dims::<4>(&[4])
}

/// Compute equilibrium distribution
///
/// # Arguments
/// - `rho`: `[H, W]` population density
/// - `u`: `[H, W, (Y,X)=2]` macroscopic velocity
/// - `e`: `[Y=3, X=3, (Y,X)=2]` direction vectors
/// - `w`: `[Y=3, X=3]` equilibrium weights
///
/// # Returns
/// - `[H, W, Y=3, X=3]` equilibrium distribution
#[rustfmt::skip]
pub fn equilibrium<B: Backend>(
    rho: Tensor<B, 2>,
    u: Tensor<B, 3>,
    e: Tensor<B, 3>,
    w: Tensor<B, 2>,
) -> Tensor<B, 4> {
    // [H, W, Y, X]
    let e_dot_u = lattice_dot_velocity(u.clone(), e);

    // [H, W]
    let u_sq = velocity_squared(u);

    // TODO: Benchmark:
    // Tensor::powi_scalar(2) is still a float pow operation.
    // * `3.0 * e_dot_u + 4.5 * e_dot_u^2`
    // * `e_dot_u * (3.0 + 4.5 * e_dot_u)`
    (w.unsqueeze() * rho.unsqueeze_dim(2)).mul(
        1
            + 3.0 * e_dot_u.clone()
            + 4.5 * e_dot_u.clone().powi_scalar(2)
            - 1.5 * u_sq.unsqueeze_dims::<4>(&[2, 3])
    )
}

/// Wrapper for the BGK collision operator.
pub enum RelaxationParam {
    /// Relaxation frequency (1/tau), typically in (0, 2)
    Omega(f64),

    /// Relaxation time (1/omega), typically > 0.5
    Tau(f64),
}

impl RelaxationParam {
    /// Get the relaxation frequency (1/tau), typically in (0, 2)
    pub fn as_omega(&self) -> f64 {
        match self {
            RelaxationParam::Omega(omega) => *omega,
            RelaxationParam::Tau(tau) => 1.0 / *tau,
        }
    }

    /// Get the relaxation time (1/omega), typically > 0.5
    pub fn as_tau(&self) -> f64 {
        match self {
            RelaxationParam::Omega(omega) => 1.0 / *omega,
            RelaxationParam::Tau(tau) => *tau,
        }
    }
}

/// Bhatnagar-Gross-Krook collision operator.
///
/// # Arguments
/// - `dist`: `[H, W, Y=3, X=3]` current distribution
/// - `dist_eq`: `[H, W, Y=3, X=3]` equilibrium distribution
/// - `param`: collision parameter.
///
/// # Returns
/// - `[H, W, Y=3, X=3]` post-collision distribution
pub fn bgk_collision<B: Backend>(
    dist: Tensor<B, 4>,
    dist_eq: Tensor<B, 4>,
    param: RelaxationParam,
) -> Tensor<B, 4> {
    dist.clone() + (dist_eq - dist) * param.as_omega()
}

/// Apply the streaming update step to the non-border cells of a population.
///
/// # Arguments
///
/// - `dist`: a ``[H, W, V=3, U=3]`` population distribution.
/// - `solid_mask`: a ``[H, W]`` solid mask.
///
/// # Returns
///
/// The updated distribution for the ``[H[1:-1], W[1:-1], V=3, U=3]`` interior.
pub fn stream_distribution_interior<B: Backend>(
    dist: Tensor<B, 4>,
    _solid_mask: Tensor<B, 2, Bool>,
) -> Tensor<B, 4> {
    let [h, w] = unpack_shape_contract!(
        ["H", "W", "UY", "UX"],
        &dist.shape().dims,
        &["H", "W"],
        &[("UY", 3), ("UX", 3)]
    );

    // Map the state into no-copy 3x3 neighborhood windows.
    let dist_windows = dist.unfold::<5, usize>(0, 3, 1).unfold::<6, usize>(1, 3, 1);

    // TODO: implement bounce.
    // This requires computing 3x3x2 columns;
    // and using a where operation on the solid_mask:
    // cell = where(mask_cell, bounce_cell, stream_cell)

    assert_shape_contract_periodically!(
        ["H" - "PAD", "W" - "PAD", "UY", "UX", "HK", "WK"],
        &dist_windows.shape().dims,
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
            let cell = dist_windows
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

#[cfg(test)]
mod tests {
    use super::*;

    use burn::Tensor;
    use burn::backend::Cuda;
    use burn::tensor::Tolerance;

    #[test]
    fn test_population_density() {
        type B = Cuda;
        let device = Default::default();

        let dist: Tensor<B, 4> = Tensor::from_data(
            [
                [
                [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                [[10., 20., 30.], [40., 50., 60.], [70., 80., 90.]],
            ],
                [
                    [[9., 10., 3.], [4., 5., 6.], [7., 8., 9.]],
                    [[0., -2., 0.], [0., 8., 0.], [0., 0., 0.]],
                ]
            ],
            &device,
        );

        let rho = population_density(dist.clone());

        rho.to_data().assert_approx_eq::<f32>(
            &Tensor::<B, 2>::from_data(
                [[45., 450.], [61., 6.]],
                &device,
            ).to_data(),
            Tolerance::default(),
        )
    }

    #[test]
    fn test_direction_vectors() {
        type B = Cuda;
        let device = Default::default();

        let e: Tensor<B, 3> = direction_vectors(&device);

        e.to_data().assert_eq(
            &Tensor::<B, 3>::from_data(
                [
                    [[1., -1.], [1., 0.], [1., 1.]],
                    [[0., -1.], [0., 0.], [0., 1.]],
                    [[-1., -1.], [-1., 0.], [-1., 1.]],
                ],
                &device,
            ).to_data(),
            false
        );
    }

    #[test]
    fn test_weight_matrix() {
        type B = Cuda;
        let device = Default::default();

        let w: Tensor<B, 2> = weight_matrix(&device);

        w.to_data().assert_eq(
            &Tensor::<B, 2>::from_data(
                [
                    [1.0 / 36.0, 1.0 / 9.0, 1.0 / 36.0],
                    [1.0 / 9.0, 4.0 / 9.0, 1.0 / 9.0],
                    [1.0 / 36.0, 1.0 / 9.0, 1.0 / 36.0],
                ],
                &device,
            ).to_data(),
            false
        );
    }

    #[test]
    fn test_macroscopic_momentum() {
        type B = Cuda;
        let device = Default::default();

        let dist: Tensor<B, 4> = Tensor::from_data(
            [
                [
                    [[1., 0., 0.], [0., 10., 0.], [0., 0., 0.]],
                    [[1., 2., 3.], [4., 10., 5.], [6., 7., 8.]],
                ],
            ],
            &device,
        );

        let e = direction_vectors(&device);

        let momentum = macroscopic_momentum(dist.clone(), e.clone());

        momentum.to_data().assert_approx_eq::<f32>(
            &Tensor::<B, 3>::from_data(
                [
                    [[1., -1.], [-15., 5.]],
                ],
                &device,
            ).to_data(),
            Tolerance::default(),
        )
    }

    #[test]
    fn test_eq() {
        type B = Cuda;
        let device = Default::default();
        // Population Distribution
        // [H, W, Y, X]
        let dist: Tensor<B, 4> = Tensor::from_data(
            [[
                [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]],
                [[10., 20., 30.], [40., 50., 60.], [70., 80., 90.]],
            ]],
            &device,
        );

        // Population Density
        // [H, W]
        let rho = population_density(dist.clone());

        // Directional Vectors
        // [H, W, (Y, X)]
        let e = direction_vectors(&device);

        // Weight Matrix
        // [Y, X]
        let w = weight_matrix::<B>(&device);

        // Macroscopic Momentum
        // [H, W, (Y, X)]
        let u = macroscopic_velocity(dist.clone(), rho.clone(), e.clone());

        // Velocity Squared
        // [H, W]
        let _u_sq = velocity_squared(u.clone());

        // Velocity Projection.
        let _e_u = lattice_dot_velocity(u.clone(), e.clone());

        let eq_dist = equilibrium(rho.clone(), u.clone(), e.clone(), w.clone());

        // Invariant: density(equilibrium(..., dist)) == density(dist)
        population_density(eq_dist.clone())
            .clone()
            .into_data()
            .assert_approx_eq::<f32>(&rho.clone().into_data(), Tolerance::default());

        let _update_dist = bgk_collision(dist.clone(), eq_dist.clone(), RelaxationParam::Tau(0.7));
    }
}
