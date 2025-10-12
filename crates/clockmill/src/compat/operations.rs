//! # Compat Tensor Operations

use burn::Tensor;
use burn::prelude::Backend;

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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::Tensor;
    use burn::backend::Wgpu;

    #[test]
    fn test_fast_powi_2() {
        type B = Wgpu;
        let device = Default::default();

        let input: Tensor<B, 1> = Tensor::from_data([1.0, 2.0, 3.0], &device);

        fast_powi_2(input).to_data().assert_eq(
            &&Tensor::<B, 1>::from_data([1.0, 4.0, 9.0], &device).to_data(),
            false,
        );
    }
}
