//! # Surface Convolutions (2D)
use burn::Tensor;
use burn::prelude::Backend;
use burn::tensor::BasicOps;

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
    let x: Tensor<B, 5, KIn> = input
        .unfold::<4, usize>(2, kernel[0], 1)
        .unfold::<5, usize>(3, kernel[1], 1);
    // [batch, c_in, h_wins, w_wins, kernel[0], kernel[1]]

    let dims = &x.shape().dims;
    let [batch, h_wins, w_wins] = [dims[0], dims[2], dims[3]];

    let x: Tensor<B, 5, KIn> = x.flatten::<5>(2, 3).swap_dims(1, 2);
    // [batch, h_wins * w_wins, c_in, kernel[0], kernel[1]]

    let x = (func)(x);
    // [batch, h_wins * w_wins, c_out]

    let c_out = x.shape().dims[2];
    x.swap_dims(1, 2).reshape([batch, h_wins, w_wins, c_out])
}
