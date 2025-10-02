use burn::Tensor;
use burn::backend::Cuda;
use burn::tensor::DType::{F16, F32, F64};
use burn::tensor::Distribution;
use clockmill::simulations::surface::fluids::lbm::d2q9::operations::{
    RelaxationParam, bgk_collision, direction_vectors, equilibrium, macroscopic_velocity,
    population_density, weight_matrix,
};
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn bench_lbm_d2q9(c: &mut Criterion) {
    type B = Cuda;
    let device = Default::default();

    let n = 1000;

    let mut group = c.benchmark_group(format!("lbm:d2q9: {n}x{n}"));
    group.measurement_time(std::time::Duration::from_secs(10));
    group.sample_size(10);

    let relaxation = RelaxationParam::Omega(1.5);

    for dtype in [F16, F32, F64] {
        let dist = Tensor::<B, 4>::random([n, n, 3, 3], Distribution::Default, &device).cast(dtype);
        let e = direction_vectors(&device).cast(dtype);
        let w = weight_matrix(&device).cast(dtype);

        group.bench_function(format!("{:?} equilibrium", dtype).as_str(), |b| {
            b.iter(|| {
                let rho = population_density(dist.clone());
                let u = macroscopic_velocity(dist.clone(), rho.clone(), e.clone());
                let dist_eq = equilibrium(rho.clone(), u.clone(), e.clone(), w.clone());

                black_box(dist_eq);
            })
        });

        group.bench_function(format!("{:?} collision", dtype).as_str(), |b| {
            b.iter(|| {
                let rho = population_density(dist.clone());
                let u = macroscopic_velocity(dist.clone(), rho.clone(), e.clone());
                let dist_eq = equilibrium(rho.clone(), u.clone(), e.clone(), w.clone());
                let dist_col = bgk_collision(dist.clone(), dist_eq.clone(), relaxation);

                black_box(dist_col);
            })
        });
    }
}

criterion_group!(benches, bench_lbm_d2q9);
criterion_main!(benches);
