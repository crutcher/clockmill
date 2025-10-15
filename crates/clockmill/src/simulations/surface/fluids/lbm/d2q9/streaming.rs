//! # Streaming Operations

use crate::simulations::surface::fluids::lbm::d2q9::space::LbmTables;
use crate::simulations::surface::fluids::lbm::d2q9::{collision, reflection, space};
use bimm_contracts::unpack_shape_contract;
use burn::Tensor;
use burn::prelude::{Backend, Bool, ElementConversion, s};

/// Apply the streaming update step to the non-border cells of a population.
///
/// # Arguments
///
/// - `dist`: a ``[H, W, VY=3, VX=3]`` population distribution.
///
/// # Returns
/// - The updated ``[H[1:-1], W[1:-1], VY=3, VX=3]`` interior.
pub fn stream_interior_windows<B: Backend>(dist: Tensor<B, 4>) -> Tensor<B, 4> {
    #[cfg(debug_assertions)]
    let [h, w] = bimm_contracts::unpack_shape_contract!(
        ["H", "W", "VY", "VX"],
        &dist.shape().dims,
        &["H", "W"],
        &[("VY", 3), ("VX", 3)]
    );

    let windows = space::dist_windows(dist);

    // Timing: crutcher, Oct 2025:
    // cat([cat([tensor,]),]) is ~10% faster than cat([tensor,]).reshape([..., 3, 3])
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
                                .slice(s![.., .., vy, vx, source_vy, source_vx])
                                .squeeze_dims::<4>(&[-2, -1])
                        })
                        .collect(),
                    3,
                )
            })
            .collect(),
        2,
    );

    #[cfg(debug_assertions)]
    bimm_contracts::assert_shape_contract_periodically!(
        ["H" - "PAD", "W" - "PAD", "VY", "VX"],
        &result.shape().dims,
        &[("H", h), ("W", w), ("PAD", 2), ("VY", 3), ("VX", 3)]
    );

    result
}

/// Stream the edge flow values.
///
/// This is an inner utility flow function. It assumes that the first
/// and last spatial cells (``Z``) are perpendicular edges. Values
/// which flow "out" of the boundary (from the perpendicular edges)
/// are "lost".
///
/// This can be used on both the outflow values from the penultimate
/// inner rows; and the crossflow values from the cell perpendicular
/// flows.
///
/// # Arguments
///
/// - `source`: a ``[Z, V=3]`` source.
///
/// # Returns
/// - a ``[Z, V=3]`` outflow.
pub fn stream_partial_edge_flow<B: Backend>(source: Tensor<B, 2>) -> Tensor<B, 2> {
    let z = source.shape().dims[0];
    let device = source.device();
    let dtype = source.dtype();

    Tensor::<B, 2>::zeros([z, 3], &device)
        .cast(dtype)
        // v- flow
        .slice_assign(s![..-1, 0], source.clone().slice(s![1.., 0]))
        // v0 flow
        .slice_assign(s![.., 1], source.clone().slice(s![.., 1]))
        // v+ flow
        .slice_assign(s![1.., 2], source.clone().slice(s![..-1, 2]))
}

/// todo.
pub fn stream_edge_crossflow<B: Backend>(source: Tensor<B, 2>) -> Tensor<B, 2> {
    let z = source.shape().dims[0];
    let device = source.device();
    let dtype = source.dtype();

    Tensor::<B, 2>::zeros([z, 3], &device).cast(dtype)
}

/// Advance the world simulation by one step.
#[allow(unused)]
fn _advance_step<B: Backend>(
    dist: Tensor<B, 4>,
    solid_mask: Tensor<B, 2, Bool>,
    _e: Tensor<B, 3>,
    _w: Tensor<B, 2>,
    omega: Tensor<B, 2>,
    correction_term: Option<f64>,
) -> (Tensor<B, 4>, f64) {
    let [height, width] = unpack_shape_contract!(
        ["height", "width", "VY", "VX"],
        &dist.shape().dims,
        &["height", "width"],
        &[("VY", 3), ("VX", 3)]
    );

    let device = dist.device();
    let dtype = dist.dtype();

    let solid_mask = solid_mask
        .clone()
        .slice_fill(s![0, ..], true)
        .slice_fill(s![-1, ..], true)
        .slice_fill(s![.., 0], true)
        .slice_fill(s![.., -1], true);

    let lbm_tables = LbmTables::for_dist(&dist);

    // Local Updates:
    // 1. Internal cell collisions.
    let col_dist =
        collision::bgk_collision(dist.clone(), omega.clone(), correction_term, &lbm_tables);
    let thermal_dist = reflection::with_spherical_reflection(dist.clone(), col_dist, solid_mask);

    let energy_delta: f64 = -(thermal_dist.clone().slice(s![0, .., 0, ..]).sum()
        + thermal_dist.clone().slice(s![-1, .., -1, ..]).sum()
        + thermal_dist.clone().slice(s![.., 0, .., 0]).sum()
        + thermal_dist.clone().slice(s![.., -1, .., -1]).sum())
    .into_scalar()
    .elem::<f64>();
    println!("energy loss: {}", energy_delta);

    let inner_stream = stream_interior_windows(thermal_dist.clone());

    let horiz_inflow = Tensor::<B, 4>::zeros([1, width, 1, 3], &device).cast(dtype);
    let vert_inflow = Tensor::<B, 4>::zeros([height - 2, 1, 3, 1], &device).cast(dtype);

    let top = Tensor::cat(
        vec![
            // outflow
            stream_partial_edge_flow(
                thermal_dist
                    .clone()
                    .slice(s![1, .., 0, ..])
                    .squeeze_dims::<2>(&[0, 2]),
            )
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(2),
            // crossflow
            stream_partial_edge_flow(
                thermal_dist
                    .clone()
                    .slice(s![0, .., 0, ..])
                    .squeeze_dims::<2>(&[0, 2]),
            )
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(2),
            // inflow
            horiz_inflow.clone(),
        ],
        2,
    );
    let middle = Tensor::cat(
        vec![
            Tensor::cat(
                vec![
                    // outflow
                    stream_partial_edge_flow(
                        thermal_dist
                            .clone()
                            .slice(s![.., 1, .., 0])
                            .squeeze_dims::<2>(&[1, 3]),
                    )
                    .slice_dim(0, s![1..-1])
                    .unsqueeze_dim::<3>(1)
                    .unsqueeze_dim::<4>(3),
                    // crossflow
                    stream_partial_edge_flow(
                        thermal_dist
                            .clone()
                            .slice(s![.., 0, .., 0])
                            .squeeze_dims::<2>(&[1, 3]),
                    )
                    .slice_dim(0, s![1..-1])
                    .unsqueeze_dim::<3>(1)
                    .unsqueeze_dim::<4>(3),
                    // inflow
                    vert_inflow.clone(),
                ],
                3,
            ),
            inner_stream,
            Tensor::cat(
                vec![
                    // inflow
                    vert_inflow,
                    // crossflow
                    stream_partial_edge_flow(
                        thermal_dist
                            .clone()
                            .slice(s![.., -1, .., 0])
                            .squeeze_dims::<2>(&[1, 3]),
                    )
                    .slice_dim(0, s![1..-1])
                    .unsqueeze_dim::<3>(1)
                    .unsqueeze_dim::<4>(3),
                    // outflow
                    stream_partial_edge_flow(
                        thermal_dist
                            .clone()
                            .slice(s![.., -2, .., 0])
                            .squeeze_dims::<2>(&[1, 3]),
                    )
                    .slice_dim(0, s![1..-1])
                    .unsqueeze_dim::<3>(1)
                    .unsqueeze_dim::<4>(3),
                ],
                3,
            ),
        ],
        1,
    );
    let bottom = Tensor::cat(
        vec![
            // inflow
            horiz_inflow,
            // crossflow
            stream_partial_edge_flow(
                thermal_dist
                    .clone()
                    .slice(s![1, .., 0, ..])
                    .squeeze_dims::<2>(&[0, 2]),
            )
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(2),
            // outflow
            stream_partial_edge_flow(
                thermal_dist
                    .clone()
                    .slice(s![-2, .., 0, ..])
                    .squeeze_dims::<2>(&[0, 2]),
            )
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim::<4>(2),
        ],
        2,
    );

    let streaming_dist = Tensor::cat(vec![top, middle, bottom], 0);

    // TODO: better handle of numerical instability.
    // let dist = dist.clone().mask_fill(dist.is_finite().bool_not(), 0.0);

    (streaming_dist, energy_delta)
}

/// Apply the streaming update step to a population.
///
/// This combines [`stream_interior_windows`] with edge outflow-streaming.
///
/// All edge outflow is clipped to the boundary. Closed universe sims should
/// have zero edge outflow before calling this operation.
///
/// # Arguments
///
/// - `dist`: a ``[H, W, VY=3, VX=3]`` population distribution.
///
/// # Returns
/// - The updated ``[H, W, VY=3, VX=3]`` distribution.
pub fn outflow_clipping_stream<B: Backend>(thermal_dist: Tensor<B, 4>) -> Tensor<B, 4> {
    let mut streaming_dist = thermal_dist.zeros_like();
    streaming_dist = streaming_dist.slice_assign(
        s![1..-1, 1..-1],
        stream_interior_windows(thermal_dist.clone()),
    );

    streaming_dist =
        streaming_dist.slice_assign(s![0, .., 1, 1], thermal_dist.clone().slice(s![0, .., 1, 1]));
    streaming_dist =
        streaming_dist.slice_assign(s![.., 0, 1, 1], thermal_dist.clone().slice(s![.., 0, 1, 1]));
    streaming_dist = streaming_dist.slice_assign(
        s![-1, .., 1, 1],
        thermal_dist.clone().slice(s![-1, .., 1, 1]),
    );
    streaming_dist = streaming_dist.slice_assign(
        s![.., -1, 1, 1],
        thermal_dist.clone().slice(s![.., -1, 1, 1]),
    );

    // stream top edges.
    // e[-1, -1]; v[0, 0]
    streaming_dist = streaming_dist.slice_assign(
        s![0, ..-1, 0, 0],
        thermal_dist.clone().slice(s![1, 1.., 0, 0]),
    );
    // e[-1, 0]; v[0, 1]
    streaming_dist =
        streaming_dist.slice_assign(s![0, .., 0, 1], thermal_dist.clone().slice(s![1, .., 0, 1]));
    // e[-1, 1]; v[0, 2]
    streaming_dist = streaming_dist.slice_assign(
        s![0, 1.., 0, 2],
        thermal_dist.clone().slice(s![1, ..-1, 0, 2]),
    );

    // stream bottom edges.
    // e[1, -1]; v[2, 0]
    streaming_dist = streaming_dist.slice_assign(
        s![-1, ..-1, 2, 0],
        thermal_dist.clone().slice(s![-2, 1.., 2, 0]),
    );
    // e[1, 0]; v[2, 1]
    streaming_dist = streaming_dist.slice_assign(
        s![-1, .., 2, 1],
        thermal_dist.clone().slice(s![-2, .., 2, 1]),
    );
    // e[1, 1]; v[2, 2]
    streaming_dist = streaming_dist.slice_assign(
        s![-1, 1.., 2, 2],
        thermal_dist.clone().slice(s![-2, ..-1, 2, 2]),
    );

    // stream left edges.
    // e[-1, -1]; v[0, 0]
    streaming_dist = streaming_dist.slice_assign(
        s![..-1, 0, 0, 0],
        thermal_dist.clone().slice(s![1.., 1, 0, 0]),
    );
    // e[0, -1]; v[1, 0]
    streaming_dist =
        streaming_dist.slice_assign(s![.., 0, 1, 0], thermal_dist.clone().slice(s![.., 1, 1, 0]));
    // e[1, -1]; v[2, 0]
    streaming_dist = streaming_dist.slice_assign(
        s![1.., 0, 2, 0],
        thermal_dist.clone().slice(s![..-1, 1, 2, 0]),
    );

    // stream right edges.
    // e[-1, 1]; v[0, 2]
    streaming_dist = streaming_dist.slice_assign(
        s![..-1, -1, 0, 2],
        thermal_dist.clone().slice(s![1.., -2, 0, 2]),
    );
    // e[0, 1]; v[1, 2]
    streaming_dist = streaming_dist.slice_assign(
        s![.., -1, 1, 2],
        thermal_dist.clone().slice(s![.., -2, 1, 2]),
    );
    // e[1, 1]; v[2, 2]
    streaming_dist = streaming_dist.slice_assign(
        s![1.., -1, 2, 2],
        thermal_dist.clone().slice(s![..-1, -2, 2, 2]),
    );

    streaming_dist
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::Wgpu;

    #[test]
    #[rustfmt::skip]
    fn test_stream_interior_windows() {
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

        let result = stream_interior_windows(state.clone());

        assert_eq!(result.shape().dims, vec![1, 1, 3, 3]);

        let expected: Tensor<B, 4> = Tensor::from_data([[[
            [72., 64., 56.],
            [48., 40., 32.],
            [24., 16., 8.],
        ]]], &device);

        result.to_data().assert_eq(&expected.to_data(), false);
    }

    #[test]
    fn test_stream_partial_edge_flow() {
        type B = Wgpu;
        let device = Default::default();

        let source: Tensor<B, 2> =
            Tensor::from_data([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], &device);

        let result = stream_partial_edge_flow(source);

        result.to_data().assert_eq(
            &Tensor::<B, 2>::from_data([[4., 2., 0.], [7., 5., 3.], [0., 8., 6.]], &device)
                .to_data(),
            false,
        );
    }
}
