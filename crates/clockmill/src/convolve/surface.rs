//! # Surface Convolutions (2D)

use bimm_contracts::{assert_shape_contract_periodically, unpack_shape_contract};
use burn::Tensor;
use burn::prelude::Backend;
use burn::tensor::BasicOps;
use burn::tensor::ops::unfold::calculate_unfold_windows;

/// Convolve a neighborhood function over a 2D tensor.
///
/// # Arguments
///
/// * `input` - a ``[batch, c_in, height, width]`` tensor.
/// * `kernel` - a kernel shape, e.g. ``[3, 3]``.
/// * `func` - a func from ``[batch, h_wins * w_wins, c_in, kernel[0], kernel[1]]``
///   to ``[batch, blocks, c_out]``
///
/// # Returns
///
/// A tensor in ``[batch, c_out, h_wins, w_wins]``
pub fn convolve_func_2d<B, KIn, KOut, F>(
    input: Tensor<B, 4, KIn>,
    kernel: [usize; 2],
    func: F,
) -> Tensor<B, 4, KOut>
where
    B: Backend,
    KIn: BasicOps<B>,
    KOut: BasicOps<B>,
    F: Fn(Tensor<B, 5, KIn>) -> Tensor<B, 3, KOut>,
{
    let [batch, c_in, height, width] =
        unpack_shape_contract!(["batch", "c_in", "height", "width"], &input.shape().dims,);

    let h_wins = calculate_unfold_windows(height, kernel[0], 1);
    let w_wins = calculate_unfold_windows(width, kernel[1], 1);

    let x: Tensor<B, 6, KIn> = input
        .unfold::<5, usize>(2, kernel[0], 1)
        .unfold::<6, usize>(3, kernel[1], 1);

    assert_shape_contract_periodically!(
        ["batch", "c_in", "h_wins", "w_wins", "kernel0", "kernel1"],
        &x.shape().dims,
        &[
            ("batch", batch),
            ("c_in", c_in),
            ("h_wins", h_wins),
            ("w_wins", w_wins),
            ("kernel0", kernel[0]),
            ("kernel1", kernel[1]),
        ]
    );

    let x: Tensor<B, 5, KIn> = x
        .reshape([batch, c_in, h_wins * w_wins, kernel[0], kernel[1]])
        .swap_dims(1, 2);
    // [batch, h_wins * w_wins, c_in, kernel[0], kernel[1]]

    assert_shape_contract_periodically!(
        ["batch", "h_wins" * "w_wins", "c_in", "kernel0", "kernel1"],
        &x.shape().dims,
        &[
            ("batch", batch),
            ("c_in", c_in),
            ("h_wins", h_wins),
            ("w_wins", w_wins),
            ("kernel0", kernel[0]),
            ("kernel1", kernel[1]),
        ]
    );

    let x = (func)(x);
    // [batch, h_wins * w_wins, c_out]

    assert_shape_contract_periodically!(
        ["batch", "h_wins" * "w_wins", "c_out"],
        &x.shape().dims,
        &[("batch", batch), ("h_wins", h_wins), ("w_wins", w_wins),]
    );

    let c_out = x.shape().dims[2];
    x.swap_dims(1, 2).reshape([batch, c_out, h_wins, w_wins])
}
