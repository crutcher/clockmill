use burn::prelude::Backend;
use clap::Parser;
use clockmill::simulations::surface::conway::life2d::{ConwayLife2DConfig, ConwayLife2DState};
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
    #[arg(long, default_value = "1000")]
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
    println!("{:#?}", args);

    #[cfg(feature = "cuda")]
    run::<burn::backend::Cuda>(&args);

    #[cfg(feature = "wgpu")]
    run::<burn::backend::Wgpu>(&args);

    #[cfg(feature = "metal")]
    run::<burn::backend::Metal>(&args);
}

fn run<B: Backend>(args: &Args) {
    let device = Default::default();

    let warmup = args.steps / args.warmup_fraction;

    let mut conway: ConwayLife2DState<B> = ConwayLife2DConfig {
        shape: [args.grid_size, args.grid_size],
    }
    .init(&device);
    conway.fuzz(0.2);

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
        conway.wrap();
        conway.step_no_wrap();

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
