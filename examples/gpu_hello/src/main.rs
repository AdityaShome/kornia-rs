use kornia_gpu::*;
use kornia_tensor::{CpuAllocator, Tensor};
use std::time::Instant;

fn main() -> Result<()> {
    println!("=== GPU Image Processing Pipeline ===\n");

    #[cfg(feature = "cuda")]
    {
        println!("Initializing CUDA runtime...");
        match init_cuda_runtime() {
            Ok(runtime) => {
                print_device_info(&runtime);
                run_pipeline(&runtime)?;
            }
            Err(e) => {
                println!("CUDA not available: {:?}", e);
            }
        }
    }

    #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
    {
        println!("Initializing WGPU runtime...");
        match init_wgpu_runtime() {
            Ok(runtime) => {
                print_device_info(&runtime);
                run_pipeline(&runtime)?;
            }
            Err(e) => {
                println!("WGPU not available: {:?}", e);
            }
        }
    }

    #[cfg(not(any(feature = "cuda", feature = "wgpu")))]
    {
        println!("No GPU features enabled!");
        println!("Build with: cargo run -p gpu_hello --features cuda");
        println!("        or: cargo run -p gpu_hello --features wgpu");
    }

    Ok(())
}

/// Print device/backend info to confirm GPU is being used.
#[allow(dead_code)]
fn print_device_info<R: CubeclRuntime>(runtime: &RuntimeContext<R>) {
    println!("  Backend:       {}", runtime.backend_name());
    let (max_page, align) = runtime.memory_properties();
    println!("  Max page size: {} bytes ({:.0} MB)", max_page, max_page as f64 / 1_048_576.0);
    println!("  Alignment:     {} bytes", align);
    println!();
}

#[allow(dead_code)]
fn run_pipeline<R: CubeclRuntime>(runtime: &RuntimeContext<R>) -> Result<()> {
    // --- Step 1: Create a synthetic 4x4 RGB image ---
    let (src_h, src_w, channels): (u32, u32, u32) = (4, 4, 3);
    let num_pixels = (src_h * src_w) as usize;

    let mut rgb_data = Vec::with_capacity(num_pixels * channels as usize);
    for y in 0..src_h {
        for x in 0..src_w {
            rgb_data.push(y as f32 * 64.0);
            rgb_data.push(x as f32 * 64.0);
            rgb_data.push(128.0);
        }
    }

    let src_tensor =
        Tensor::<f32, 1, _>::from_shape_vec([rgb_data.len()], rgb_data, CpuAllocator)
            .map_err(|e| GpuError::Other(e.to_string()))?;

    println!("Step 1: Created {}x{} RGB image ({} elements)", src_h, src_w, src_tensor.as_slice().len());

    // --- Step 2: Transfer to GPU ---
    let rgb_gpu = tensor_to_device(&src_tensor, runtime)?;
    println!("Step 2: Transferred to GPU");

    // --- Step 3: RGB to Grayscale on GPU ---
    let gray_gpu = allocate::<_, f32>(vec![num_pixels], runtime)?;
    rgb_to_gray_execute::<_, f32>(&rgb_gpu, &gray_gpu, runtime)?;
    runtime.sync();
    println!("Step 3: RGB -> Grayscale on GPU");

    let gray_result: Vec<f32> = to_cpu(&gray_gpu, runtime)?;
    println!("  First 4 gray values: [{:.1}, {:.1}, {:.1}, {:.1}]",
        gray_result[0], gray_result[1], gray_result[2], gray_result[3]);

    // --- Step 4: Resize on GPU (4x4 -> 2x2, bilinear) ---
    let (dst_h, dst_w): (u32, u32) = (2, 2);
    let dst_elements = (dst_h * dst_w * channels) as usize;
    let resized_gpu = allocate::<_, f32>(vec![dst_elements], runtime)?;

    resize_execute::<_, f32>(
        &rgb_gpu, &resized_gpu,
        src_h, src_w, dst_h, dst_w, channels,
        GpuInterpolation::Bilinear, runtime,
    )?;
    runtime.sync();
    println!("Step 4: Resize {}x{} -> {}x{} (bilinear)", src_h, src_w, dst_h, dst_w);

    let resized_tensor: Tensor<f32, 1, CpuAllocator> =
        tensor_from_device(&resized_gpu, [dst_elements], runtime)?;
    let resized = resized_tensor.as_slice();
    for y in 0..dst_h as usize {
        for x in 0..dst_w as usize {
            let i = (y * dst_w as usize + x) * channels as usize;
            println!("  pixel({},{}) = R:{:.1} G:{:.1} B:{:.1}",
                y, x, resized[i], resized[i + 1], resized[i + 2]);
        }
    }

    // --- Step 5: Elementwise + reductions ---
    println!("\nStep 5: Elementwise ops");
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let b_data = vec![10.0f32, 20.0, 30.0, 40.0];
    let a_gpu = to_device(&a_data, vec![4], runtime)?;
    let b_gpu = to_device(&b_data, vec![4], runtime)?;
    let out_gpu = allocate::<_, f32>(vec![4], runtime)?;

    add_execute::<_, f32>(&a_gpu, &b_gpu, &out_gpu, runtime)?;
    let add_result: Vec<f32> = to_cpu(&out_gpu, runtime)?;
    println!("  {:?} + {:?} = {:?}", a_data, b_data, add_result);

    let sum_val = sum_execute::<_, f32>(&a_gpu, runtime)?;
    println!("  sum({:?}) = {:.1}", a_data, sum_val);

    // --- Step 6: Large-scale GPU proof ---
    // Run a large operation on GPU. If this completes in <100ms on GPU but would
    // take much longer on CPU, it proves GPU execution.
    println!("\nStep 6: Large-scale GPU verification");
    let big_n: usize = 10_000_000; // 10M elements
    let big_data: Vec<f32> = (0..big_n).map(|i| i as f32).collect();
    println!("  Transferring {} elements ({:.1} MB) to GPU...", big_n, big_n as f64 * 4.0 / 1_048_576.0);

    let t0 = Instant::now();
    let big_a = to_device(&big_data, vec![big_n], runtime)?;
    let transfer_time = t0.elapsed();
    println!("  Transfer time: {:.2?}", transfer_time);

    let big_b = to_device(&big_data, vec![big_n], runtime)?;
    let big_out = allocate::<_, f32>(vec![big_n], runtime)?;

    // Warm up (first kernel launch has JIT overhead)
    add_execute::<_, f32>(&big_a, &big_b, &big_out, runtime)?;
    runtime.sync();

    // Timed run
    let t1 = Instant::now();
    for _ in 0..100 {
        add_execute::<_, f32>(&big_a, &big_b, &big_out, runtime)?;
    }
    runtime.sync();
    let gpu_time = t1.elapsed();
    println!("  100x add on 10M elements: {:.2?} ({:.2?}/iter)", gpu_time, gpu_time / 100);

    // CPU comparison
    let t2 = Instant::now();
    let mut cpu_out = vec![0.0f32; big_n];
    for _ in 0..100 {
        for i in 0..big_n {
            cpu_out[i] = big_data[i] + big_data[i];
        }
    }
    let cpu_time = t2.elapsed();
    println!("  100x add on 10M elements (CPU): {:.2?} ({:.2?}/iter)", cpu_time, cpu_time / 100);

    if gpu_time < cpu_time {
        println!("  GPU is {:.1}x faster than CPU", cpu_time.as_secs_f64() / gpu_time.as_secs_f64());
    } else {
        println!("  Note: GPU was slower (likely due to small problem size or transfer overhead)");
    }

    // Verify correctness on a sample
    let result_sample: Vec<f32> = to_cpu(&big_out, runtime)?;
    assert_eq!(result_sample[0], 0.0);
    assert_eq!(result_sample[1], 2.0);
    assert_eq!(result_sample[1000], 2000.0);
    println!("  Correctness verified (spot-checked 3 elements)");

    println!("\n=== All GPU operations completed successfully! ===");
    Ok(())
}
