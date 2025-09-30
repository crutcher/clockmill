use burn::Tensor;
use burn::backend::Cuda;
use burn::tensor::DType::{F16, F32, F64};
use burn::tensor::Distribution;
use clockmill::simulations::surface::fluids::lattice_boltzmann::LBMD2Q9Operations;
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn bench_lbm(c: &mut Criterion) {
    type B = Cuda;
    let device = Default::default();

    let n = 1000;
    let tau = 2.0;

    let mut group = c.benchmark_group(format!("LBMD2Q9Operations: {n}x{n}"));
    group.measurement_time(std::time::Duration::from_secs(10));
    group.sample_size(10);

    for dtype in [F16, F32, F64] {
        let state =
            Tensor::<B, 4>::random([n, n, 3, 3], Distribution::Normal(0., 1.), &device).cast(dtype);
        let solid_mask = Tensor::<B, 2>::zeros([n, n], &device).bool();

        let ops = LBMD2Q9Operations::<B>::init(&device).cast(dtype);

        group.bench_function(format!("{:?} equilibrium", dtype).as_str(), |b| {
            b.iter(|| {
                black_box(ops.equilibrium(state.clone()));
            })
        });

        group.bench_function(format!("{:?} collision", dtype).as_str(), |b| {
            b.iter(|| {
                black_box(ops.collision(state.clone(), tau));
            })
        });

        group.bench_function(format!("{:?} streaming", dtype).as_str(), |b| {
            b.iter(|| {
                black_box(ops.interior_streaming(state.clone(), solid_mask.clone()));
            })
        });
    }
}

criterion_group!(benches, bench_lbm);
criterion_main!(benches);
