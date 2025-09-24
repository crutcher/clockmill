#![allow(unused)]
use bimm_contracts::{assert_shape_contract_periodically, unpack_shape_contract};
use burn::backend::Cuda;
use burn::prelude::{Backend, Bool, Int, Tensor, s};
use burn::tensor::DType::F16;
use burn::tensor::Distribution;
use burn::tensor::module::unfold4d;
use burn::tensor::ops::UnfoldOptions;
use clap::Parser;
use indicatif::ProgressBar;
use std::time::Instant;

/// Conway's Game of Life demo for Burn.
#[derive(Parser, Debug)]
#[command(long_about = None)]
pub struct Args {
    /// The number of steps to run.
    #[arg(long, default_value = "1000")]
    pub steps: usize,

    /// The width and height of the grid.
    #[arg(long, default_value = "100")]
    pub grid_size: usize,

    /// Use `Tensor::unfold()` views.
    #[arg(long, default_value = "false")]
    pub unfold_views: bool,

    /// The fraction of steps to use for warmup.
    #[arg(long, default_value_t = 10)]
    pub warmup_fraction: usize,

    /// Show progress bar.
    #[arg(short, long, default_value_t = false)]
    pub progress: bool,
}

fn main() {
    let args = Args::parse();

    run::<Cuda>(&args);
}

fn run<B: Backend>(args: &Args) {
    println!("Args: {:?}", args);

    let device = Default::default();

    let warmup = args.steps / args.warmup_fraction;

    let mut state: Tensor<B, 2, Bool> = Tensor::<B, 2>::random(
        [args.grid_size, args.grid_size],
        Distribution::Default,
        &device,
    )
    .greater_elem(0.5);

    let mut t0: Instant = Instant::now();
    let bar = if args.progress {
        Some(ProgressBar::new(args.steps as u64))
    } else {
        None
    };

    for step in 0..args.steps {
        if step == warmup {
            t0 = Instant::now();
        }
        if args.unfold_views {
            state = conway_unfold_views(state);
        } else {
            state = conway_unfold_copies(state);
        }

        if let Some(bar) = &bar {
            bar.inc(1);
        }
    }
    let t1: Instant = Instant::now();
    if let Some(bar) = &bar {
        bar.finish();
    }

    let step_rate = (args.steps - warmup) as f64 / (t1 - t0).as_secs_f64();
    println!("{:.2} steps/sec", step_rate);
}

fn conway_unfold_views<B: Backend>(state: Tensor<B, 2, Bool>) -> Tensor<B, 2, Bool> {
    let [h, w] = state.shape().dims();

    let h_blocks: Tensor<B, 3, Bool> = state.clone().unfold(0, 3, 1);
    assert_shape_contract_periodically!(
        ["h_wins" = "height" - "pad", "width", "kernel"],
        &h_blocks.shape().dims,
        &[("height", h), ("width", w), ("kernel", 3), ("pad", 2)]
    );

    let blocks: Tensor<B, 4, Bool> = h_blocks.unfold(1, 3, 1);
    assert_shape_contract_periodically!(
        [
            "h_wins" = "height" - "pad",
            "w_wins" = "width" - "pad",
            "kernel",
            "kernel"
        ],
        &blocks.shape().dims,
        &[("height", h), ("width", w), ("kernel", 3), ("pad", 2)]
    );

    let blocks: Tensor<B, 3, Int> = blocks
        .reshape([h - 2, w - 2, 3 * 3])
        .permute([2, 0, 1])
        .int();

    let block_sum = blocks.clone().sum_dim(0);
    let neighbor_count = block_sum - blocks.slice(s![5, .., ..]);

    let neighbor_count = neighbor_count.reshape([h - 2, w - 2]);

    conway_transition(state, neighbor_count)
}

fn conway_unfold_copies<B: Backend>(state: Tensor<B, 2, Bool>) -> Tensor<B, 2, Bool> {
    let [h, w] = state.shape().dims();

    let blocks = unfold4d(
        state.clone().float().cast(F16).reshape([1, 1, h, w]),
        [3, 3],
        UnfoldOptions {
            stride: [1, 1],
            padding: [0, 0],
            dilation: [1, 1],
        },
    )
    .int()
    .squeeze::<2>(0);

    assert_shape_contract_periodically!(
        [
            "kernel" ^ 2,
            "blocks" = ("height" - "pad") * ("width" - "pad")
        ],
        &blocks.shape().dims,
        &[("kernel", 3), ("height", h), ("width", w), ("pad", 2),],
    );

    let block_sum = blocks.clone().sum_dim(0);
    let neighbor_count = block_sum - blocks.slice(s![5, ..]);
    let neighbor_count = neighbor_count.reshape([h - 2, w - 2]);

    conway_transition(state, neighbor_count)
}

fn conway_transition<B: Backend>(
    state: Tensor<B, 2, Bool>,
    neighbor_count: Tensor<B, 2, Int>,
) -> Tensor<B, 2, Bool> {
    let inner = state.clone().slice(s![1..-1, 1..-1]);

    let survivors = inner.bool_and(neighbor_count.clone().equal_elem(2));
    let spawns = neighbor_count.equal_elem(3);

    let update = survivors.bool_or(spawns);

    state.slice_assign(s![1..-1, 1..-1], update)
}
