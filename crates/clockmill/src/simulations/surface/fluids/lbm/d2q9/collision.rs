//! # Collision Operators

use crate::simulations::surface::fluids::lbm::d2q9::relaxation::{OmegaSource, RelaxationParam};
use crate::simulations::surface::fluids::lbm::d2q9::{reflection, relaxation, space, thermal};
use burn::Tensor;
use burn::prelude::{Backend, Bool};

/// Naive aggregate Bhatnagar-Gross-Krook collision operator.
///
/// This operator makes no accounting for solids, reflection,
/// or boundaries.
///
/// ## Correction
///
/// The `correction` term is a scale applied to the result. It is provided
/// as a way for dynamic corrections to be computed as fused operations;
/// and will default to `1.0` for `None`.
///
/// # Arguments
/// - `dist`: ``[H, W, VY=3, VX=3]`` current distribution
/// - `equi_dist`: ``[H, W, VY=3, VX=3]`` equilibrium distribution
/// - `relaxation`: relaxation parameter.
/// - `correction`: fused correction factor for the relaxation operator;
///   defaults to 1.0.
///
/// # Returns
/// - `[H, W, VY=3, VX=3]` post-collision distribution
pub fn bgk_collision<B: Backend, S: Into<OmegaSource<B>>>(
    dist: Tensor<B, 4>,
    e: Tensor<B, 3>,
    w: Tensor<B, 2>,
    relaxation: S,
    correction: Option<f64>,
) -> Tensor<B, 4> {
    let (source_rho, u) = space::moments(dist.clone(), e.clone());
    let eq_dist = thermal::thermal_equilibrium(source_rho.clone(), u, e, w);
    relaxation::relaxed_sum(dist, eq_dist, relaxation, correction)
}

/// Combined bgk collision and isotropic reflection operator.
///
/// This combines:
/// - [`bgk_collision`]
/// - [`reflection::with_spherical_reflection`]
///
/// # Arguments
/// - `pre_dist`: ``[H, W, VY=3, VX=3]`` pre-collision distribution
/// - `naive_dist`: ``[H, W, VY=3, VX=3]`` post-collision distribution
/// - `solid_mask`: ``[H, W]`` mask of solid locations.
/// - `correction`: fused correction factor for the relaxation operator;
///   defaults to 1.0.
///
/// # Returns
/// - ``[H, W, VY=3, VX=3]`` distribution.
pub fn combined_isotropic_collision<B: Backend>(
    dist: Tensor<B, 4>,
    e: Tensor<B, 3>,
    w: Tensor<B, 2>,
    solid_mask: Tensor<B, 2, Bool>,
    relaxation: RelaxationParam,
    correction: Option<f64>,
) -> Tensor<B, 4> {
    let pre_dist = dist;

    let naive_dist = bgk_collision(pre_dist.clone(), e, w, relaxation, correction);
    reflection::with_spherical_reflection(pre_dist, naive_dist, solid_mask)
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::simulations::surface::fluids::lbm::d2q9::relaxation::RelaxationParam;
    use crate::simulations::surface::fluids::lbm::d2q9::space::{
        density, direction_vectors, weight_matrix,
    };
    use burn::Tensor;
    use burn::backend::Wgpu;
    use burn::tensor::DType::F32;
    use burn::tensor::{Distribution, Tolerance};

    #[test]
    fn test_collision_invariants() {
        type B = Wgpu;
        let device = Default::default();

        let dtype = F32;

        let e = direction_vectors(&device).cast(dtype);
        let w = weight_matrix(&device).cast(dtype);

        let dist = Tensor::<B, 4>::random([20, 20, 3, 3], Distribution::Uniform(0.1, 1.0), &device)
            .cast(dtype);
        let rho = density(dist.clone());

        let col_dist = bgk_collision(
            dist.clone(),
            e.clone(),
            w.clone(),
            RelaxationParam::Omega(0.5),
            None,
        );

        // Invariant: density(collision(dist, param)) == density(dist)
        density(col_dist.clone())
            .to_data()
            .assert_approx_eq::<f32>(&rho.clone().to_data(), Tolerance::default());

        // With Correction
        {
            let col_dist = bgk_collision(
                dist.clone(),
                e.clone(),
                w.clone(),
                RelaxationParam::Omega(0.5),
                Some(1.2),
            );

            // Invariant: density(collision(dist, param)) == density(dist)
            density(col_dist.clone())
                .to_data()
                .assert_approx_eq::<f32>(&(rho * 1.2).to_data(), Tolerance::default());
        }
    }
}
