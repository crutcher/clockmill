use std::time::Instant;
use burn::backend::Cuda;
use burn::prelude::{s, Backend, Bool, Tensor};
use burn::tensor::Distribution;
use burn::tensor::DType::F16;
use burn::tensor::module::unfold4d;
use burn::tensor::ops::UnfoldOptions;
use clap::Parser;
use indicatif::ProgressBar;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// The number of steps to run.
    #[arg(long, default_value = "1000")]
    pub steps: usize,

    /// The width and height of the grid.
    #[arg(long, default_value = "1000")]
    pub grid_size: usize,
}

fn main() {
    let args = Args::parse();

    run::<Cuda>(&args);
}

fn run<B: Backend>(args: &Args) {
    let device = Default::default();

    let n = args.grid_size;
    let k = args.steps;
    let warmup = k/10;

    let mut state: Tensor::<B, 2, Bool> = Tensor::<B, 2>::random([n, n], Distribution::Default, &device).greater_elem(0.5);

    let mut t0: Instant = Instant::now();
    let bar = ProgressBar::new(k as u64);
    for step in 0..k {
        if step == warmup {
            t0 = Instant::now();
        }
        state = conway(state);
        bar.inc(1);
    }
    let t1: Instant = Instant::now();
    bar.finish();

    let step_rate = (k - warmup) as f64 / (t1 - t0).as_secs_f64();
    println!("{:.2} steps/sec", step_rate);
}

fn conway<B: Backend>(state: Tensor<B, 2, Bool>) -> Tensor<B, 2, Bool> {
    let [h, w] = state.shape().dims();

    let blocks = unfold4d(
        state.clone().float().cast(F16).reshape([1, 1, h, w]),
        [3, 3],
        UnfoldOptions {
            stride: [1, 1],
            padding: [0, 0],
            dilation: [1, 1],
        }
    ).int();

    let block_sum = blocks.clone().sum_dim(1);
    let neighbor_count = block_sum - blocks.slice(s![0, 5, ..]);

    let neighbor_count = neighbor_count.reshape([h-2, w-2]);

    let inner = state.clone().slice(s![1..-1, 1..-1]);

    let survivors = inner.bool_and(neighbor_count.clone().equal_elem(2));
    let spawns = neighbor_count.equal_elem(3);

    let update = survivors.bool_or(spawns);

    state.slice_assign(s![1..-1, 1..-1], update)
}
