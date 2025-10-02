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

/// Maps nan and infinities to numbers.
pub fn nan_to_num<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    nan_val: f64,
    neg_inf_val: f64,
    pos_inf_val: f64,
) -> Tensor<B, D> {
    let is_nan = tensor.clone().is_nan();
    let is_inf = tensor.clone().is_inf();
    let is_neg = tensor.clone().lower_elem(0.0);

    let pos_inf = is_inf.clone().bool_and(is_neg.clone().bool_not());
    let neg_inf = is_inf.clone().bool_and(is_neg);

    tensor
        .mask_fill(is_nan, nan_val)
        .mask_fill(neg_inf, neg_inf_val)
        .mask_fill(pos_inf, pos_inf_val)
}
