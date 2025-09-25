# clockmill
Rust/Burn Based 2D and 3D Grid Sims

Early stages; will focus on frameworks for 2D and 3D Constant-Volume-Grid simulations;
leveraging `burn`'s and no-copy folded window views.

## Note: Burn Preview Dependency

This repo uses a direct link to a fixed revision of the dev branch of `burn`,

# Demos

## conway_speedtest

A pure speed test of the Conway's Game of Life simulation.

```terminaloutput
$ cargo run --release -p conway_speedtest
Args: Args { steps: 1000, grid_size: 1000, unfold_views: false, warmup_fraction: 10, progress: false }
2901.03 steps/sec
```

## fishbowl

A graphical demo of Conway's Game of Life.

```terminaloutput
$ cargo run --release -p fishbowl
```
