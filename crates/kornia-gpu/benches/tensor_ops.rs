use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use kornia_gpu::*;

fn cpu_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x + y).collect()
}

fn cpu_mul(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b).map(|(x, y)| x * y).collect()
}

fn cpu_sum(a: &[f32]) -> f32 {
    a.iter().sum()
}

fn cpu_mean(a: &[f32]) -> f32 {
    a.iter().sum::<f32>() / a.len() as f32
}

fn cpu_min(a: &[f32]) -> f32 {
    a.iter().copied().fold(f32::MAX, f32::min)
}

fn cpu_max(a: &[f32]) -> f32 {
    a.iter().copied().fold(f32::MIN, f32::max)
}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
fn bench_element_wise_ops(c: &mut Criterion) {
    #[cfg(feature = "cuda")]
    let runtime = init_cuda_runtime();
    #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
    let runtime = init_wgpu_runtime();

    if runtime.is_err() {
        println!("GPU not available, skipping benchmarks");
        return;
    }
    let runtime = runtime.unwrap();

    let mut group = c.benchmark_group("elementwise_add");
    
    for size in [100, 1_000, 10_000, 100_000].iter() {
        let a: Vec<f32> = (0..*size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..*size).map(|i| (i * 2) as f32).collect();
        
        // CPU baseline
        group.bench_with_input(BenchmarkId::new("cpu", size), size, |bench, _| {
            bench.iter(|| {
                let result = cpu_add(black_box(&a), black_box(&b));
                black_box(result);
            });
        });
        
        // GPU
        let a_gpu = to_device(&a, vec![*size], &runtime).unwrap();
        let b_gpu = to_device(&b, vec![*size], &runtime).unwrap();
        let out_gpu = allocate::<_, f32>(vec![*size], &runtime).unwrap();
        
        group.bench_with_input(BenchmarkId::new("gpu", size), size, |bench, _| {
            bench.iter(|| {
                add_execute(black_box(&a_gpu), black_box(&b_gpu), black_box(&out_gpu), black_box(&runtime)).unwrap();
            });
        });
    }
    group.finish();

    let mut group = c.benchmark_group("elementwise_mul");
    for size in [100, 1_000, 10_000, 100_000].iter() {
        let a: Vec<f32> = (0..*size).map(|i| i as f32 + 1.0).collect();
        let b: Vec<f32> = (0..*size).map(|i| (i * 2) as f32 + 1.0).collect();
        
        group.bench_with_input(BenchmarkId::new("cpu", size), size, |bench, _| {
            bench.iter(|| {
                let result = cpu_mul(black_box(&a), black_box(&b));
                black_box(result);
            });
        });
        
        let a_gpu = to_device(&a, vec![*size], &runtime).unwrap();
        let b_gpu = to_device(&b, vec![*size], &runtime).unwrap();
        let out_gpu = allocate::<_, f32>(vec![*size], &runtime).unwrap();
        
        group.bench_with_input(BenchmarkId::new("gpu", size), size, |bench, _| {
            bench.iter(|| {
                mul_execute(black_box(&a_gpu), black_box(&b_gpu), black_box(&out_gpu), black_box(&runtime)).unwrap();
            });
        });
    }
    group.finish();
}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
fn bench_reductions(c: &mut Criterion) {
    #[cfg(feature = "cuda")]
    let runtime = init_cuda_runtime();
    #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
    let runtime = init_wgpu_runtime();

    if runtime.is_err() {
        return;
    }
    let runtime = runtime.unwrap();

    let mut group = c.benchmark_group("reduction_sum");
    for size in [100, 1_000, 10_000, 100_000].iter() {
        let data: Vec<f32> = (0..*size).map(|i| i as f32).collect();
        
        group.bench_with_input(BenchmarkId::new("cpu", size), size, |bench, _| {
            bench.iter(|| {
                let result = cpu_sum(black_box(&data));
                black_box(result);
            });
        });
        
        let data_gpu = to_device(&data, vec![*size], &runtime).unwrap();
        
        group.bench_with_input(BenchmarkId::new("gpu", size), size, |bench, _| {
            bench.iter(|| {
                let result = sum_execute::<_, f32>(black_box(&data_gpu), black_box(&runtime)).unwrap();
                black_box(result);
            });
        });
    }
    group.finish();

    let mut group = c.benchmark_group("reduction_mean");
    for size in [100, 1_000, 10_000, 100_000].iter() {
        let data: Vec<f32> = (0..*size).map(|i| i as f32).collect();
        
        group.bench_with_input(BenchmarkId::new("cpu", size), size, |bench, _| {
            bench.iter(|| {
                let result = cpu_mean(black_box(&data));
                black_box(result);
            });
        });
        
        let data_gpu = to_device(&data, vec![*size], &runtime).unwrap();
        
        group.bench_with_input(BenchmarkId::new("gpu", size), size, |bench, _| {
            bench.iter(|| {
                let result = mean_execute::<_, f32>(black_box(&data_gpu), black_box(&runtime)).unwrap();
                black_box(result);
            });
        });
    }
    group.finish();

    let mut group = c.benchmark_group("reduction_minmax");
    for size in [100, 1_000, 10_000, 100_000].iter() {
        let data: Vec<f32> = (0..*size).map(|i| i as f32).collect();
        
        group.bench_with_input(BenchmarkId::new("cpu_min", size), size, |bench, _| {
            bench.iter(|| {
                let result = cpu_min(black_box(&data));
                black_box(result);
            });
        });

        group.bench_with_input(BenchmarkId::new("cpu_max", size), size, |bench, _| {
            bench.iter(|| {
                let result = cpu_max(black_box(&data));
                black_box(result);
            });
        });
        
        let data_gpu = to_device(&data, vec![*size], &runtime).unwrap();
        
        group.bench_with_input(BenchmarkId::new("gpu_min", size), size, |bench, _| {
            bench.iter(|| {
                let result = min_execute::<_, f32>(black_box(&data_gpu), black_box(&runtime)).unwrap();
                black_box(result);
            });
        });

        group.bench_with_input(BenchmarkId::new("gpu_max", size), size, |bench, _| {
            bench.iter(|| {
                let result = max_execute::<_, f32>(black_box(&data_gpu), black_box(&runtime)).unwrap();
                black_box(result);
            });
        });
    }
    group.finish();
}

#[cfg(any(feature = "cuda", feature = "wgpu"))]
criterion_group!(benches, bench_element_wise_ops, bench_reductions);

#[cfg(not(any(feature = "cuda", feature = "wgpu")))]
criterion_group!(benches);

criterion_main!(benches);
