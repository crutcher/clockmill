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
/// * `func` - a func from ``[batch, h_wins, w_wins, c_in, kernel[0], kernel[1]]``
///   to ``[batch, h_wins, w_wins, c_out]``
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
    F: Fn(Tensor<B, 6, KIn>) -> Tensor<B, 4, KOut>,
{
    #[cfg(debug_assertions)]
    let [batch, c_in, height, width] = bimm_contracts::unpack_shape_contract!(
        ["batch", "c_in", "height", "width"],
        &input.shape().dims
    );
    #[cfg(debug_assertions)]
    let h_wins = burn::tensor::ops::unfold::calculate_unfold_windows(height, kernel[0], 1);
    #[cfg(debug_assertions)]
    let w_wins = burn::tensor::ops::unfold::calculate_unfold_windows(width, kernel[1], 1);

    let x: Tensor<B, 6, KIn> = input
        .unfold::<5, usize>(2, kernel[0], 1)
        .unfold::<6, usize>(3, kernel[1], 1);

    #[cfg(debug_assertions)]
    bimm_contracts::assert_shape_contract_periodically!(
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

    let x: Tensor<B, 6, KIn> = x.permute([0, 2, 3, 1, 4, 5]);

    #[cfg(debug_assertions)]
    bimm_contracts::assert_shape_contract_periodically!(
        ["batch", "h_wins", "w_wins", "c_in", "kernel0", "kernel1"],
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

    #[cfg(debug_assertions)]
    bimm_contracts::assert_shape_contract_periodically!(
        ["batch", "h_wins", "w_wins", "c_out"],
        &x.shape().dims,
        &[("batch", batch), ("h_wins", h_wins), ("w_wins", w_wins),]
    );

    x.permute([0, 3, 1, 2])
}
