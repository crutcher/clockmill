//! # D2Q9 Lattice-Boltzmann Method Operations
//!
//! This module contains the operations used by the D2Q9 LBM simulation.
//!
//! See:
//! * [Wikipedia](https://en.wikipedia.org/wiki/Lattice_Boltzmann_methods).
use crate::compat::operations::{fast_powi_2, sum_dims};
use bimm_contracts::{assert_shape_contract_periodically, unpack_shape_contract};
use burn::Tensor;
use burn::prelude::{s, Backend, Bool};
use burn::serde::{Deserialize, Serialize};
use crate::compat::FRAC_1_SQRT_3;

/// The speed of sound.
pub const SPEED_OF_SOUND: f64 = FRAC_1_SQRT_3;

/// Population Density
///
/// # Arguments
///
/// - `dist`: a ``[H, W, VY=3, VX=3]`` population distribution.
///
/// # Returns
///
/// A ``[H, W]`` population density.
pub fn density<B: Backend>(dist: Tensor<B, 4>) -> Tensor<B, 2> {
    sum_dims(dist, &[2, 3]).squeeze_dims::<2>(&[2, 3])
}

/// D2Q9 Direction Vectors
///
/// # Returns
///
/// The ``[VY=3, VX=3, (VY, VX)=2]`` direction vectors.
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
/// The ``[VY=3, VX=3]`` equilibrium weight matrix.
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
/// - `dist`: a ``[H, W, VY, VX]`` population distribution.
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
        dist.unsqueeze_dims::<5>(&[-1]).mul(e.unsqueeze::<5>()),
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
    // TODO: div-by-zero check?
    // .clamp_min(1e-15)?
    m.div(rho.unsqueeze_dim(2))
}

/// Compute the directional macroscopic velocity.
///
/// # Arguments
///
/// - `dist`: a ``[H, W, VY, VX]`` population distribution.
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

/// Compute the first (density) and second (macro velocity) moments.
///
/// # Arguments
///
/// - `dist`: a ``[H, W, VY, VX]`` population distribution.
/// - `e`: the D2Q9 direction vectors.
///
/// # Returns
///
/// A pair of:
/// - `density`: ``[H, W]``
/// - `velocity`: ``[H, W, (Y, X)=2]``
pub fn moments<B: Backend>(
    dist: Tensor<B, 4>,
    e: Tensor<B, 3>,
) -> (Tensor<B, 2>, Tensor<B, 3>) {
    let rho = density(dist.clone());
    let u = macroscopic_velocity(dist, rho.clone(), e);
    (rho, u)
}

/// Compute the squared magnitude of velocity field.
///
/// # Arguments
/// - `u`: ``[H, W, (Y, X)=2]`` macroscopic velocity
///
/// # Returns
/// - ``[H, W]`` velocity magnitude squared
pub fn velocity_squared<B: Backend>(u: Tensor<B, 3>) -> Tensor<B, 2> {
    // TODO: Benchmark:
    // Tensor::powi_scalar(2) is still a float pow operation.
    // * u.powi_scalar(2)
    // * u * u
    fast_powi_2(u).sum_dim(2).squeeze_dims::<2>(&[2])
}

/// Compute e·u for each lattice direction
///
/// # Arguments
/// - `e`: ``[Y=3, X=3, (Y, X)=2]`` direction vectors
/// - `u`: ``[H, W, (Y, X)=2]`` macroscopic velocity
///
/// # Returns
///
/// The ``[H, W, Y=3, X=3]`` dot product at each grid point and direction.
pub fn lattice_dot_velocity<B: Backend>(
    u: Tensor<B, 3>,
    e: Tensor<B, 3>,
) -> Tensor<B, 4> {
    ldv_projection(e, u).sum_dim(4).squeeze_dims::<4>(&[4])
}

/// The projection component of [`lattice_dot_velocity`].
///
/// # Arguments
/// - `e`: ``[Y=3, X=3, (Y, X)=2]`` direction vectors
/// - `u`: ``[H, W, (Y, X)=2]`` macroscopic velocity
///
/// # Returns
///
/// The ``[H, W, Y=3, X=3, (Y, X)=2]`` projection.
pub fn ldv_projection<B: Backend>(
    e: Tensor<B, 3>,
    u: Tensor<B, 3>,
) -> Tensor<B, 5> {
    // e[None, None, ... ] * u[..., None, :] -> [H, W, Y, X, 2]
    // e[1, 1, Y, X, (Y, X)=2] * u[H, W, 1, 1, (Y, X)=2]
    // -> [H, W, Y, X, (Y, X)]
    e.unsqueeze::<5>() * u.unsqueeze_dims::<5>(&[2, 3])
}

/// Compute equilibrium distribution
///
/// # Arguments
/// - `rho`: ``[H, W]`` population density
/// - `u`: ``[H, W, (Y, X)=2]`` macroscopic velocity
/// - `e`: ``[Y=3, X=3, (Y, X)=2]`` direction vectors
/// - `w`: ``[Y=3, X=3]`` equilibrium weights
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

    static C2: f64 = SPEED_OF_SOUND * SPEED_OF_SOUND;
    static C4: f64 = C2 * C2;

    // TODO: Benchmark:
    // Tensor::powi_scalar(2) is still a float pow operation.
    // * `3.0 * e_dot_u + 4.5 * e_dot_u^2`
    // * `e_dot_u * (3.0 + 4.5 * e_dot_u)`
    (w.unsqueeze() * rho.unsqueeze_dim(2)).mul(
        1
            + e_dot_u.clone() / C2
            + fast_powi_2(e_dot_u) / (2.0 * C4)
            - u_sq.unsqueeze_dims::<4>(&[2, 3]) / (2.0 * C2)
    )
}

/// Wrapper for the BGK collision operator.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RelaxationParam {
    /// Relaxation frequency (1/tau), typically in (0, 2)
    Omega(f64),

    /// Relaxation time (1/omega), typically > 0.5
    Tau(f64),
}

impl RelaxationParam {
    /// Validate the relaxation; or panic.
    pub fn validate(&self) {
        match self {
            RelaxationParam::Omega(omega) => {
                assert!(
                    (0.0..=2.0).contains(omega),
                    "omega ({omega}) must be in [0, 2.0] range"
                );
            }
            RelaxationParam::Tau(tau) => {
                assert!(*tau >= 0.5, "tau ({tau}) must be >= 0.5");
            }
        }
    }
    /// Get the relaxation frequency (1/tau), typically in (0, 2)
    pub fn as_omega_value(&self) -> f64 {
        match self {
            RelaxationParam::Omega(omega) => *omega,
            RelaxationParam::Tau(tau) => 1.0 / *tau,
        }
    }

    /// Get the relaxation time (1/omega), typically > 0.5
    pub fn as_tau_value(&self) -> f64 {
        match self {
            RelaxationParam::Omega(omega) => 1.0 / *omega,
            RelaxationParam::Tau(tau) => *tau,
        }
    }
}

/// The relaxation operator for [`naive_bgk_collision`].
///
/// Computes ``dist_a * (1 - omega) + dist_b * omega``.
///
/// # Arguments
/// - `dist_a`: a ``[H, W, VY=3, VX=3]`` distribution.
/// - `dist_b`: a ``[H, W, VY=3, VX=3]`` distribution.
/// - `relaxation`: relaxation parameter.
///
/// # Returns
///
/// The `[H, W, VY=3, VX=3]` relaxed sum.
pub fn relaxed_sum<B: Backend>(
    dist_a: Tensor<B, 4>,
    dist_b: Tensor<B, 4>,
    relaxation: RelaxationParam,
) -> Tensor<B, 4> {
    // A + (B - A) / T
    // A + (B - A) O
    // A + B O - A O
    // A - A O + B O
    // A ( 1 - O ) + B O
    let omega = relaxation.as_omega_value();
    assert!(
        (0.0..=2.0).contains(&omega),
        "omega must be in [0, 2.0] range"
    );
    dist_a * (1.0 - omega) + dist_b * omega
}

/// Naive aggregate Bhatnagar-Gross-Krook collision operator.
///
/// This operator makes no accounting for solids, reflection,
/// or boundaries.
///
/// # Arguments
/// - `dist`: ``[H, W, VY=3, VX=3]`` current distribution
/// - `dist_eq`: ``[H, W, VY=3, VX=3]`` equilibrium distribution
/// - `relaxation`: relaxation parameter.
///
/// # Returns
/// - `[H, W, VY=3, VX=3]` post-collision distribution
pub fn naive_bgk_collision<B: Backend>(
    dist: Tensor<B, 4>,
    e: Tensor<B, 3>,
    w: Tensor<B, 2>,
    relaxation: RelaxationParam,
) -> Tensor<B, 4> {
    let (rho, u) = moments(dist.clone(), e.clone());
    let dist_eq = equilibrium(rho, u, e, w);
    relaxed_sum(dist, dist_eq, relaxation)
}

/// Applies isotropic spherical solid reflection updates to [`naive_bgk_collision`].
///
/// This models every solid point as a sphere, normal to all directions.
///
/// # Arguments
/// - `pre_dist`: ``[H, W, VY=3, VX=3]`` pre-collision distribution
/// - `naive_dist`: ``[H, W, VY=3, VX=3]`` post-collision distribution
/// - `solid_mask`: ``[H, W]`` mask of solid locations.
///
/// # Returns
/// - ``[H, W, VY=3, VX=3]`` distribution.
pub fn isotropic_spherical_reflection<B: Backend>(
    pre_dist: Tensor<B, 4>,
    naive_dist: Tensor<B, 4>,
    solid_mask: Tensor<B, 2, Bool>,
) -> Tensor<B, 4> {
    naive_dist.mask_where(solid_mask.unsqueeze_dims::<4>(&[-1, -1]), pre_dist)
}

/// Combined bgk collision and isotropic reflection operator.
///
/// This combines:
/// - [`bgk_collision]
/// - [`isotropic_spherical_reflection`]
///
/// # Arguments
/// - `pre_dist`: ``[H, W, VY=3, VX=3]`` pre-collision distribution
/// - `naive_dist`: ``[H, W, VY=3, VX=3]`` post-collision distribution
/// - `solid_mask`: ``[H, W]`` mask of solid locations.
///
/// # Returns
/// - ``[H, W, VY=3, VX=3]`` distribution.
pub fn combined_isotropic_collision<B: Backend>(
    dist: Tensor<B, 4>,
    e: Tensor<B, 3>,
    w: Tensor<B, 2>,
    solid_mask: Tensor<B, 2, Bool>,
    relaxation: RelaxationParam,
) -> Tensor<B, 4> {
    let naive_dist = naive_bgk_collision(dist.clone(), e, w, relaxation);
    isotropic_spherical_reflection(dist, naive_dist, solid_mask)
}

/// Apply the streaming update step to the non-border cells of a population.
///
/// # Arguments
///
/// - `dist`: a ``[H, W, VY=3, VX=3]`` population distribution.
///
/// # Returns
/// - The updated ``[H[1:-1], W[1:-1], VY=3, VX=3]`` interior.
pub fn stream_interior_cells<B: Backend>(dist: Tensor<B, 4>) -> Tensor<B, 4> {
    let [h, w] = unpack_shape_contract!(
        ["H", "W", "VY", "VX"],
        &dist.shape().dims,
        &["H", "W"],
        &[("VY", 3), ("VX", 3)]
    );

    // Map the state into no-copy 3x3 neighborhood windows.
    // [H-2, W-2, V=3, U=3, VY=3, VX=3]
    let windows = dist.unfold::<5, _>(0, 3, 1).unfold::<6, _>(1, 3, 1);

    let result: Tensor<B, 4> = Tensor::cat(
        (0..3)
            .map(|vy| -> Tensor<B, 4> {
                let source_vy = 2 - vy;

                Tensor::cat(
                    (0..3)
                        .map(|vx| -> Tensor<B, 4> {
                            let source_vx = 2 - vx;

                            windows
                                .clone()
                                .slice(s![.., .., source_vy, source_vx, vy, vx])
                                .squeeze_dims::<4>(&[-2, -1])
                        })
                        .collect(),
                    3,
                )
            })
            .collect(),
        2,
    );

    assert_shape_contract_periodically!(
        ["H" - "PAD", "W" - "PAD", "VY", "VX"],
        &result.shape().dims,
        &[("H", h), ("W", w), ("PAD", 2), ("VY", 3), ("VX", 3)]
    );

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    use burn::Tensor;
    use burn::backend::{Cuda, Wgpu};
    use burn::tensor::{Distribution, Tolerance};

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
                ],
            ],
            &device,
        );

        let rho = density(dist.clone());

        rho.to_data().assert_approx_eq::<f32>(
            &Tensor::<B, 2>::from_data([[45., 450.], [61., 6.]], &device).to_data(),
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
            )
            .to_data(),
            false,
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
            )
            .to_data(),
            false,
        );
    }

    #[test]
    fn test_momentum_and_velocity() {
        type B = Cuda;
        let device = Default::default();

        let dist: Tensor<B, 4> = Tensor::from_data(
            [[
                [[1., 0., 0.], [0., 10., 0.], [0., 0., 0.]],
                [[1., 2., 3.], [4., 10., 5.], [6., 7., 8.]],
            ]],
            &device,
        );

        let e = direction_vectors(&device);

        let momentum = macroscopic_momentum(dist.clone(), e.clone());

        momentum.clone().to_data().assert_approx_eq::<f32>(
            &Tensor::<B, 3>::from_data([[[1., -1.], [-15., 5.]]], &device).to_data(),
            Tolerance::default(),
        );

        let (rho, u) = moments(dist.clone(), e.clone());

        let rho_data = rho.to_data().to_vec::<f32>().unwrap();

        u.clone().to_data().assert_approx_eq::<f32>(
            &Tensor::<B, 3>::from_data(
                [[
                    [1. / rho_data[0], -1. / rho_data[0]],
                    [-15. / rho_data[1], 5. / rho_data[1]],
                ]],
                &device,
            )
            .to_data(),
            Tolerance::default(),
        );

        let v_sq = velocity_squared(u.clone());

        v_sq.clone().to_data().assert_approx_eq::<f32>(
            &Tensor::<B, 2>::from_data(
                [[
                    (1. + 1.) / rho_data[0].powi(2),
                    (15. * 15. + 5. * 5.) / rho_data[1].powi(2),
                ]],
                &device,
            )
            .to_data(),
            Tolerance::default(),
        );
    }

    #[test]
    fn test_lattice_dot_velocity() {
        type B = Cuda;
        let device = Default::default();

        let e: Tensor<B, 3> = direction_vectors(&device);

        let u: Tensor<B, 3> = Tensor::from_data([[[0.1, -2.], [0.5, -1.5]]], &device);

        let parts = ldv_projection(e.clone(), u.clone());

        parts.clone().to_data().assert_approx_eq::<f32>(
            &Tensor::<B, 5>::from_data(
                [[
                    [
                        [[0.1, 2.], [0.1, 0.], [0.1, -2.]],
                        [[0., 2.], [0., 0.], [0., -2.]],
                        [[-0.1, 2.], [-0.1, 0.], [-0.1, -2.]],
                    ],
                    [
                        [[0.5, 1.5], [0.5, 0.], [0.5, -1.5]],
                        [[0., 1.5], [0., 0.], [0., -1.5]],
                        [[-0.5, 1.5], [-0.5, 0.], [-0.5, -1.5]],
                    ],
                ]],
                &device,
            )
            .to_data(),
            Tolerance::default(),
        );

        let e_u = lattice_dot_velocity(u.clone(), e.clone());

        e_u.clone().to_data().assert_approx_eq::<f32>(
            &parts.sum_dim(4).squeeze_dims::<4>(&[4]).to_data(),
            Tolerance::default(),
        );
    }

    #[test]
    fn test_equilibrium_invariants() {
        type B = Cuda;
        let device = Default::default();

        let e = direction_vectors(&device);
        let w = weight_matrix(&device);

        let dist = Tensor::<B, 4>::random([20, 20, 3, 3], Distribution::Uniform(0.1, 1.0), &device);

        let (rho, u) = moments(dist.clone(), e.clone());
        let dist_eq = equilibrium(rho.clone(), u.clone(), e.clone(), w.clone());

        // Invariant: density(equilibrium(dist)) == density(dist)
        density(dist_eq.clone())
            .to_data()
            .assert_approx_eq::<f32>(&rho.to_data(), Tolerance::default());
    }

    #[test]
    fn test_collision_invariants() {
        type B = Cuda;
        let device = Default::default();

        let e = direction_vectors(&device);
        let w = weight_matrix(&device);

        let dist = Tensor::<B, 4>::random([20, 20, 3, 3], Distribution::Uniform(0.1, 1.0), &device);

        let dist_col = naive_bgk_collision(
            dist.clone(),
            e.clone(),
            w.clone(),
            RelaxationParam::Omega(0.5),
        );

        // Invariant: density(collision(dist, param)) == density(dist)
        density(dist_col.clone())
            .to_data()
            .assert_approx_eq::<f32>(&density(dist).to_data(), Tolerance::default());
    }

    #[test]
    #[rustfmt::skip]
    fn test_equilibrium() {
        type B = Cuda;
        let device = Default::default();

        let dist = Tensor::<B, 4>::random([20, 20, 3, 3], Distribution::Default, &device);

        let e = direction_vectors(&device);
        let w = weight_matrix(&device);

        let (rho, u) = moments(dist.clone(), e.clone());

        let dist_eq = equilibrium(rho.clone(), u.clone(), e.clone(), w.clone());

        density(dist_eq.clone())
            .to_data()
            .assert_approx_eq::<f32>(&rho.to_data(), Tolerance::default());

        let e_dot_u = lattice_dot_velocity(u.clone(), e);
        let u_sq = velocity_squared(u);
        let expected_eq = (w.unsqueeze() * rho.unsqueeze_dim(2)).mul(
            1
                + 3.0 * e_dot_u.clone()
                + 4.5 * e_dot_u.clone().powi_scalar(2)
                - 1.5 * u_sq.unsqueeze_dims::<4>(&[2, 3])
        );

        dist_eq.clone().to_data().assert_approx_eq::<f32>(&expected_eq.to_data(), Tolerance::default());

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

        let result = stream_interior_cells(state.clone());

        assert_eq!(result.shape().dims, vec![1, 1, 3, 3]);

        let expected: Tensor<B, 4> = Tensor::from_data([[[
            [8., 16., 24.],
            [32., 40., 48.],
            [56., 64., 72.],
        ]]], &device);

        result.to_data().assert_eq(&expected.to_data(), false);
    }
}
