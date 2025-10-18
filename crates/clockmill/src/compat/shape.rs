//! # Shape Helpers

use burn::prelude::Shape;
use burn::tensor::AsIndex;
use burn::tensor::indexing::canonicalize_index;

/// Compute the ravel index for the given coordinates.
///
/// This returns the row-major order raveling.
///
/// # Arguments
/// - `coords`: must be the same size as `self.rank()`.
///
/// # Returns
/// - the ravel offset index.
pub fn ravel_shape<const R: usize, I: AsIndex>(
    shape: &Shape,
    coords: [I; R],
) -> usize {
    ravel_dims(&shape.dims, coords)
}

/// Compute the ravel index for the given coordinates.
///
/// This returns the row-major order raveling.
///
/// # Arguments
/// - `coords`: must be the same size as `self.rank()`.
///
/// # Returns
/// - the ravel offset index.
pub fn ravel_dims<const R: usize, I: AsIndex>(
    dims: &[usize],
    coords: [I; R],
) -> usize {
    assert_eq!(
        dims.len(),
        R,
        "Shape rank mismatch: expected {}, got {R}",
        dims.len(),
    );

    let mut ravel_idx = 0;
    let mut stride = 1;

    for i in (0..R).rev() {
        let dim = dims[i];
        let coord = canonicalize_index(coords[i], dim, false);

        ravel_idx += coord * stride;
        stride *= dim;
    }

    ravel_idx
}
