use burn::config::Config;
use burn::prelude::{Backend, Bool, Int, Tensor, s};
use burn::tensor::{Distribution, RangesArg, Slice};

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

pub struct Conway<B: Backend> {
    pub state: Tensor<B, 2, Bool>,
    pub previous: Option<Tensor<B, 2, Bool>>,
}

impl<B: Backend> Conway<B> {
    pub fn device(&self) -> B::Device {
        self.state.device()
    }

    pub fn shape(&self) -> [usize; 2] {
        self.state.shape().dims()
    }

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

    pub fn step(&mut self) {
        self.previous = Some(self.state.clone());
        self.state = Self::next_inner(self.state.clone());
    }

    pub fn neighbor_count(state: Tensor<B, 2, Bool>) -> Tensor<B, 2, Int> {
        state
            .unfold::<3, usize>(0, 3, 1)
            .unfold::<4, usize>(1, 3, 1)
            .int()
            .sum_dim(2)
            .sum_dim(3)
            .squeeze_dims::<2>(&[2, 3])
    }

    pub fn next_inner(state: Tensor<B, 2, Bool>) -> Tensor<B, 2, Bool> {
        let neighbor_count: Tensor<B, 2, Int> = Self::neighbor_count(state.clone());

        let live: Tensor<B, 2, Bool> = state.clone().slice(s![1..-1, 1..-1]);

        let neighbor_count = neighbor_count - live.clone().int();

        let survivors = live.clone().bool_and(
            neighbor_count
                .clone()
                .equal_elem(2)
                .bool_or(neighbor_count.clone().equal_elem(3)),
        );

        let spawners = live.bool_not().bool_and(neighbor_count.equal_elem(3));

        let update = survivors.bool_or(spawners);

        state.slice_assign(s![1..-1, 1..-1], update)
    }

    fn slice_size(slice: &Slice) -> usize {
        (slice.end.unwrap() - slice.start) as usize
    }

    fn slices_shape(slices: &[Slice; 2]) -> [usize; 2] {
        [Self::slice_size(&slices[0]), Self::slice_size(&slices[1])]
    }

    pub fn read_2d_slice<R>(
        state: Tensor<B, 2, Bool>,
        ranges: R,
    ) -> Vec<Vec<bool>>
    where
        R: RangesArg<2>,
    {
        let slices = ranges.into_slices(state.shape());
        let [h, w] = Self::slices_shape(&slices);

        let block_data = state.clone().slice(slices).to_data();

        let block_slice: &[u32] = block_data.as_slice().unwrap();

        let mut result = Vec::with_capacity(h);
        for hidx in 0..h {
            let start = hidx * w;

            result.push(
                block_slice[start..start + w]
                    .iter()
                    .map(|&x| x != 0)
                    .collect(),
            )
        }

        result
    }

    pub fn read_previous_slice<R>(
        &self,
        ranges: R,
    ) -> Option<Vec<Vec<bool>>>
    where
        R: RangesArg<2>,
    {
        self.previous
            .as_ref()
            .map(|previous| Self::read_2d_slice(previous.clone(), ranges))
    }

    pub fn read_slice<R>(
        &self,
        ranges: R,
    ) -> Vec<Vec<bool>>
    where
        R: RangesArg<2>,
    {
        Self::read_2d_slice(self.state.clone(), ranges)
    }

    pub fn write_slice<R>(
        &mut self,
        ranges: R,
        data: Vec<Vec<bool>>,
    ) where
        R: RangesArg<2>,
    {
        let slices = ranges.into_slices(self.state.shape());
        let [h, w] = Self::slices_shape(&slices);

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

#[cfg(test)]
mod tests {
    use super::*;
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

        Conway::neighbor_count(conway.state.clone())
            .to_data()
            .assert_eq(&TensorData::from([[3, 3, 1], [3, 3, 2], [1, 2, 3]]), false);

        Conway::next_inner(conway.state.clone())
            .to_data()
            .assert_eq(
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
