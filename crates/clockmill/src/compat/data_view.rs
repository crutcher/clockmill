//! # `TensorData` View Wrappers

use crate::compat::shape::ravel_dims;
use burn::prelude::TensorData;
use burn::tensor::{AsIndex, Element};
use std::ops::Index;

/// Ravel Index View for a `TensorData`.
#[derive(Debug)]
pub struct TensorDataIndexView<'a, E: Element, const R: usize> {
    data: &'a TensorData,
    _phantom: std::marker::PhantomData<&'a E>,
}

impl<'a, E: Element, const R: usize> TensorDataIndexView<'a, E, R> {
    /// Get an indexed view of the data.
    pub fn view(data: &'a TensorData) -> TensorDataIndexView<'a, E, R> {
        TensorDataIndexView {
            data,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, I: AsIndex, E: Element, const R: usize> Index<[I; R]> for TensorDataIndexView<'a, E, R> {
    type Output = E;
    fn index(
        &self,
        index: [I; R],
    ) -> &Self::Output {
        &self.data.as_slice::<E>().unwrap()[ravel_dims(&self.data.shape, index)]
    }
}
