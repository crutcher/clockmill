use burn::Tensor;
use burn::backend::Wgpu;
use burn::prelude::{Bool, s};
use burn::tensor::DType::{F16, F32};
use burn::tensor::Distribution;
use clockmill::simulations::surface::fluids::lbm::d2q9::operations::{
    RelaxationParam, combined_isotropic_collision, direction_vectors,
    isotropic_spherical_reflection, naive_bgk_collision, stream_interior_cells, weight_matrix,
};
use criterion::{Criterion, criterion_group, criterion_main};
use std::hint::black_box;

fn bench_lbm_d2q9(c: &mut Criterion) {
    type B = Wgpu;
    let device = Default::default();

    let n = 1000;

    let mut group = c.benchmark_group(format!("lbm:d2q9: {n}x{n}"));

    let relaxation = RelaxationParam::Omega(1.5);

    for dtype in [F16, F32] {
        let e = direction_vectors(&device);
        let w = weight_matrix(&device);

        let dist = Tensor::<B, 4>::random([n, n, 3, 3], Distribution::Default, &device);
        let solid_mask = Tensor::<B, 2, Bool>::full([n, n], false, &device);

        let e = e.cast(dtype);
        let w = w.cast(dtype);
        let dist = dist.cast(dtype);

        group.bench_function(format!("{:?} bgk_collision", dtype).as_str(), |b| {
            b.iter(|| {
                let dist_col =
                    naive_bgk_collision(dist.clone(), e.clone(), w.clone(), relaxation, None);

                black_box(dist_col.mean().into_scalar());
            })
        });

        group.bench_function(format!("{:?} isotropic collision", dtype).as_str(), |b| {
            b.iter(|| {
                let dist_col = combined_isotropic_collision(
                    dist.clone(),
                    e.clone(),
                    w.clone(),
                    solid_mask.clone(),
                    relaxation,
                    None,
                );

                black_box(dist_col.mean().into_scalar());
            })
        });

        group.bench_function(format!("{:?} streaming", dtype).as_str(), |b| {
            b.iter(|| {
                let stream_result = stream_interior_cells(dist.clone());

                black_box(stream_result);
            })
        });

        group.bench_function(format!("{:?} update", dtype).as_str(), |b| {
            b.iter(|| {
                let dist = isotropic_spherical_reflection(
                    dist.clone(),
                    naive_bgk_collision(dist.clone(), e.clone(), w.clone(), relaxation, None),
                    solid_mask.clone(),
                );

                let stream_result = stream_interior_cells(dist.clone());
                let dist = dist.slice_assign(s![1..-1, 1..-1], stream_result);

                black_box(dist.mean().into_scalar());
            })
        });
    }
}

criterion_group!(benches, bench_lbm_d2q9);
criterion_main!(benches);
