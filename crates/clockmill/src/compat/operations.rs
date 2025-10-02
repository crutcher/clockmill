//! # Compat Tensor Operations

use burn::Tensor;
use burn::prelude::Backend;
use burn::tensor::AsIndex;
use burn::tensor::indexing::canonicalize_dim;

/// Aggregate sum over dims.
///
/// See ``Tensor::sum_dims(dims)`` in 0.19.0
pub fn sum_dims<B: Backend, const D: usize, I: AsIndex>(
    tensor: Tensor<B, D>,
    dims: &[I],
) -> Tensor<B, D> {
    dims.iter().fold(tensor, |tensor, &dim| {
        let dim = canonicalize_dim(dim, D, false);
        tensor.sum_dim(dim)
    })
}

/// Fast `tensor.powi(2)` implementation.
///
/// [`burn`] currently has no specialization for `tensor.powi(2)`.
pub fn fast_powi_2<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    tensor.clone().mul(tensor)
}
