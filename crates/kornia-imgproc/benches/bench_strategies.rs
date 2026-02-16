use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use kornia_image::{Image, ImageError};
use kornia_imgproc::filter::{kernels, separable_filter};
use kornia_tensor::CpuAllocator;

// Serial execution
fn gaussian_blur_serial<const C: usize>(
    src: &Image<u8, C, CpuAllocator>,
    dst: &mut Image<u8, C, CpuAllocator>,
    kernel_x: &[f32],
    kernel_y: &[f32],
) -> Result<(), ImageError> {
    separable_filter(
        src,
        dst,
        kernel_x,
        kernel_y,
        kornia_imgproc::parallel::ExecutionStrategy::Serial,
    )
}

// Parallel elements execution
fn gaussian_blur_simd_parallel<const C: usize>(
    src: &Image<u8, C, CpuAllocator>,
    dst: &mut Image<u8, C, CpuAllocator>,
    kernel_x: &[f32],
    kernel_y: &[f32],
) -> Result<(), ImageError> {
    separable_filter(
        src,
        dst,
        kernel_x,
        kernel_y,
        kornia_imgproc::parallel::ExecutionStrategy::ParallelElements,
    )
}

fn bench_execution_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("Execution Strategies");

    // Test configurations: (width, height, kernel_size)
    let configs = [
        (1024, 896, 3),  // Large image, 3×3 kernel (fast path)
        (1024, 896, 5),  // Large image, 5×5 kernel (fast path)
    ];

    for (width, height, kernel_size) in configs.iter() {
        let parameter_string = format!("{width}x{height}x{kernel_size}");

        // Create test images
        let image_data = vec![128u8; width * height * 3];
        let image_size = [*width, *height].into();
        let image_u8 = Image::<_, 3, _>::new(image_size, image_data.clone(), CpuAllocator).unwrap();
        let output_u8 = Image::<_, 3, _>::from_size_val(image_size, 0u8, CpuAllocator).unwrap();

        // Calculate the Kornia filters BEFORE the stopwatch starts
        let kernel_x = kernels::gaussian_kernel_1d(*kernel_size, 1.5);
        let kernel_y = kernels::gaussian_kernel_1d(*kernel_size, 1.5);

        // Kornia Serial benchmark
        group.bench_with_input(
            BenchmarkId::new("serial", &parameter_string),
            &(&image_u8, &output_u8),
            |b, i| {
                let (src, mut dst) = (i.0, i.1.clone());
                // Pass the pre-calculated filters into the function
                b.iter(|| std::hint::black_box(gaussian_blur_serial(src, &mut dst, &kernel_x, &kernel_y)))
            },
        );

        // Kornia Parallel benchmark
        group.bench_with_input(
            BenchmarkId::new("simd_parallel", &parameter_string),
            &(&image_u8, &output_u8),
            |b, i| {
                let (src, mut dst) = (i.0, i.1.clone());
                // Pass the pre-calculated filters into the function
                b.iter(|| std::hint::black_box(gaussian_blur_simd_parallel(src, &mut dst, &kernel_x, &kernel_y)))
            },
        );

        // Opencv benchmark
        #[cfg(feature = "opencv_bench")]
        group.bench_with_input(
            BenchmarkId::new("opencv", &parameter_string),
            &image_data,
            |b, src_data| {
                use opencv::prelude::*;

                // Set up the memory and matrices BEFORE the stopwatch starts
                let src_flat = opencv::core::Mat::from_slice(src_data).unwrap();
                let src_mat = src_flat.reshape(3, *height as i32).unwrap();
                // Create dst_mat with the correct size and type to avoid allocation in the loop if possible,
                // or at least have the container ready.
                // Note: GaussianBlur in OpenCV might still reallocate if size doesn't match, 
                // but here we set it up to match.
                let mut dst_mat = opencv::core::Mat::new_rows_cols_with_default(
                    *height as i32,
                    *width as i32,
                    opencv::core::CV_8UC3,
                    opencv::core::Scalar::all(0.0)
                ).unwrap();

                // Run OpenCV once to lock in the memory and warm up
                opencv::imgproc::gaussian_blur(
                    &src_mat,
                    &mut dst_mat,
                    opencv::core::Size::new(*kernel_size as i32, *kernel_size as i32),
                    1.5,
                    1.5,
                    opencv::core::BORDER_DEFAULT,
                ).unwrap();

                // Now run the timed benchmark
                b.iter(|| {
                    opencv::imgproc::gaussian_blur(
                        &src_mat,
                        &mut dst_mat,
                        opencv::core::Size::new(*kernel_size as i32, *kernel_size as i32),
                        1.5,
                        1.5,
                        opencv::core::BORDER_DEFAULT,
                    ).unwrap();
                })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_execution_strategies);
criterion_main!(benches);
