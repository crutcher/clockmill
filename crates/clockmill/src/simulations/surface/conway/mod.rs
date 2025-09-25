//! # Conway's Game of Life

use burn::Tensor;
use burn::config::Config;
use burn::prelude::{Backend, Bool, Int, RangesArg, ToElement, s};
use burn::tensor::{Distribution, Slice};

/// Config for [`Conway`]
#[derive(Config, Debug)]
pub struct ConwayConfig {
    /// The shape of the board.
    pub shape: [usize; 2],
}

impl ConwayConfig {
    /// Initialize a [`Conway`] module.
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> Conway<B> {
        Conway {
            state: Tensor::<B, 2, Int>::zeros(self.shape, device).bool(),
            previous: None,
        }
    }
}

/// State module for Conway's Game of Life.
pub struct Conway<B: Backend> {
    /// The current state of the board.
    pub state: Tensor<B, 2, Bool>,

    /// The previous state of the board.
    pub previous: Option<Tensor<B, 2, Bool>>,
}

impl<B: Backend> Conway<B> {
    /// Get the device the module is on.
    pub fn device(&self) -> B::Device {
        self.state.device()
    }

    /// Get the board shape.
    pub fn shape(&self) -> [usize; 2] {
        self.state.shape().dims()
    }

    /// Add uniform positive noise to the board.
    pub fn fuzz(
        &mut self,
        density: f64,
    ) {
        if density == 0.0 {
            return;
        }

        let noise: Tensor<B, 2, Bool> = Tensor::<B, 2>::random(
            self.shape(),
            Distribution::Bernoulli(density),
            &self.device(),
        )
        .equal_elem(1.0);

        self.state = self.state.clone().bool_or(noise);
    }

    /// Wrap the board state.
    ///
    /// This simulates a toroidal space by copying the penultimate rows and columns
    /// to the edges of the opposite sides.
    pub fn wrap(&mut self) {
        let mut state = self.state.clone();
        state = state
            .clone()
            .slice_assign(s![0, ..], state.clone().slice(s![-2, ..]));
        state = state
            .clone()
            .slice_assign(s![-1, ..], state.clone().slice(s![1, ..]));
        state = state
            .clone()
            .slice_assign(s![1..-1, 0], state.clone().slice(s![1..-1, -2]));
        state = state
            .clone()
            .slice_assign(s![1..-1, -1], state.clone().slice(s![1..-1, 1]));
        self.state = state;
    }

    /// Advance the board state by one step; without applying wrapping.
    pub fn step_no_wrap(&mut self) {
        self.previous = Some(self.state.clone());
        self.state = next_inner(self.state.clone());
    }

    /// Read a slice of the previous board state.
    pub fn read_previous_slice<R>(
        &self,
        ranges: R,
    ) -> Option<Vec<Vec<bool>>>
    where
        R: RangesArg<2>,
    {
        self.previous
            .as_ref()
            .map(|previous| read_2d_slice(previous.clone(), ranges))
    }

    /// Read a slice of the current board state.
    pub fn read_slice<R>(
        &self,
        ranges: R,
    ) -> Vec<Vec<bool>>
    where
        R: RangesArg<2>,
    {
        read_2d_slice(self.state.clone(), ranges)
    }

    /// Write a slice to the current board state.
    pub fn write_slice<R>(
        &mut self,
        ranges: R,
        data: Vec<Vec<bool>>,
    ) where
        R: RangesArg<2>,
    {
        let slices = ranges.into_slices(self.state.shape());
        let [h, w] = slices_shape(&slices);

        assert_eq!(data.len(), h);
        for row in data.iter() {
            assert_eq!(row.len(), w);
        }

        let mut block = Vec::with_capacity(h * w);
        for row in data.iter() {
            for &cell in row.iter() {
                block.push(cell as u32);
            }
        }

        let data = Tensor::<B, 1, Int>::from_data(block.as_slice(), &self.device());
        let data = data.bool().reshape([h, w]);

        self.state = self.state.clone().slice_assign(slices, data);
    }
}

fn slice_size(slice: &Slice) -> usize {
    (slice.end.unwrap() - slice.start) as usize
}

fn slices_shape(slices: &[Slice; 2]) -> [usize; 2] {
    [slice_size(&slices[0]), slice_size(&slices[1])]
}

fn read_2d_slice<B: Backend, R>(
    state: Tensor<B, 2, Bool>,
    ranges: R,
) -> Vec<Vec<bool>>
where
    R: RangesArg<2>,
{
    let slices = ranges.into_slices(state.shape());
    let [h, w] = slices_shape(&slices);

    let block_data = state.clone().slice(slices).to_data();
    let block_slice = block_data.as_slice::<B::BoolElem>().unwrap();

    let mut result = Vec::with_capacity(h);
    for hidx in 0..h {
        let start = hidx * w;

        result.push(
            block_slice[start..start + w]
                .iter()
                .map(|&cell| cell.to_bool())
                .collect(),
        )
    }

    result
}

fn neighborhood_count<B: Backend>(state: Tensor<B, 2, Bool>) -> Tensor<B, 2, Int> {
    state
        .unfold::<3, usize>(0, 3, 1)
        .unfold::<4, usize>(1, 3, 1)
        .int()
        .sum_dim(2)
        .sum_dim(3)
        .squeeze_dims::<2>(&[2, 3])
}

fn next_inner<B: Backend>(state: Tensor<B, 2, Bool>) -> Tensor<B, 2, Bool> {
    let count: Tensor<B, 2, Int> = neighborhood_count(state.clone());

    let live: Tensor<B, 2, Bool> = state.clone().slice(s![1..-1, 1..-1]);

    let threes = count.clone().equal_elem(3);
    let fours = count.equal_elem(4);
    let update = threes.bool_or(fours.bool_and(live));

    state.slice_assign(s![1..-1, 1..-1], update)
}

#[cfg(test)]
mod tests {
    use crate::simulations::surface::conway::{
        Conway, ConwayConfig, neighborhood_count, next_inner,
    };
    use burn::backend::Wgpu;
    use burn::prelude::s;
    use burn::tensor::TensorData;

    #[test]
    fn test_setup() {
        let device = Default::default();
        let config = ConwayConfig { shape: [10, 10] };
        let mut _conway: Conway<Wgpu> = config.init(&device);
    }

    #[test]
    fn test_logic() {
        let device = Default::default();
        let config = ConwayConfig { shape: [5, 5] };
        let mut conway: Conway<Wgpu> = config.init(&device);

        assert_eq!(
            conway.read_slice(s![1..3, 1..3]),
            vec![vec![false, false], vec![false, false]]
        );

        conway.write_slice(s![1..3, 1..3], vec![vec![true, true], vec![true, false]]);

        conway.write_slice(s![-2.., -2..], vec![vec![false, true], vec![true, true]]);

        assert_eq!(
            conway.read_slice(s![1..3, 1..3]),
            vec![vec![true, true], vec![true, false]]
        );

        neighborhood_count(conway.state.clone())
            .to_data()
            .assert_eq(&TensorData::from([[3, 3, 1], [3, 3, 2], [1, 2, 3]]), false);

        next_inner(conway.state.clone()).to_data().assert_eq(
            &TensorData::from([
                [false, false, false, false, false],
                [false, true, true, false, false],
                [false, true, true, false, false],
                [false, false, false, true, true],
                [false, false, false, true, true],
            ]),
            false,
        )
    }
}
