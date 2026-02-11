//! Executable element-wise operations.
//!
//! This module provides execution wrappers around the CubeCL kernels.

use crate::error::{GpuError, Result};
use crate::kernels::elementwise::{add_kernel, mul_kernel, mul_scalar_kernel, sub_kernel, div_kernel};
use crate::kernels::reduction::{sum_kernel, min_kernel, max_kernel};
use crate::kernels::color::{rgb_to_gray_kernel, gray_to_rgb_kernel};
use crate::kernels::resize::{resize_bilinear_kernel, resize_nearest_kernel};
use crate::memory::{allocate, to_cpu};
use crate::runtime::{GpuBuffer, RuntimeContext};
use cubecl::prelude::*;

/// Execute element-wise addition on GPU: output = a + b
///
/// # Arguments
///
/// * `a` - First input buffer
/// * `b` - Second input buffer
/// * `output` - Output buffer (must be pre-allocated with same shape)
/// * `runtime` - Runtime context
///
/// # Example
///
/// ```ignore
/// let a_gpu = to_device(&vec![1.0, 2.0, 3.0], vec![3], &runtime)?;
/// let b_gpu = to_device(&vec![4.0, 5.0, 6.0], vec![3], &runtime)?;
/// let mut out_gpu = allocate::<_, f32>(vec![3], &runtime)?;
/// add_execute(&a_gpu, &b_gpu, &mut out_gpu, &runtime)?;
/// ```
pub fn add_execute<R: Runtime, F: Float + CubeElement>(
    a: &GpuBuffer<R>,
    b: &GpuBuffer<R>,
    output: &GpuBuffer<R>,
    runtime: &RuntimeContext<R>,
) -> Result<()> {
    // Validate shapes match
    if a.shape() != b.shape() || a.shape() != output.shape() {
        return Err(GpuError::Other(format!(
            "Shape mismatch: a={:?}, b={:?}, output={:?}",
            a.shape(),
            b.shape(),
            output.shape()
        )));
    }

    let len = a.len();
    if len == 0 {
        return Ok(());
    }

    // Configure kernel launch
    // Use 256 threads per block (common GPU warp size)
    let threads_per_block: u32 = 256;
    let num_blocks = (len as u32 + threads_per_block - 1) / threads_per_block;

    let cube_count = CubeCount::Static(num_blocks, 1, 1);
    let cube_dim = CubeDim::new(threads_per_block, 1, 1);

    let client = runtime.client();

    // Launch kernel using CubeCL's generated launch_unchecked function
    unsafe {
        add_kernel::launch_unchecked::<F, R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(a.handle(), a.len(), 1),
            ArrayArg::from_raw_parts(b.handle(), b.len(), 1),
            ArrayArg::from_raw_parts(output.handle(), output.len(), 1),
        );
    }

    Ok(())
}

/// Execute element-wise multiplication on GPU: output = a * b
pub fn mul_execute<R: Runtime, F: Float + CubeElement>(
    a: &GpuBuffer<R>,
    b: &GpuBuffer<R>,
    output: &GpuBuffer<R>,
    runtime: &RuntimeContext<R>,
) -> Result<()> {
    if a.shape() != b.shape() || a.shape() != output.shape() {
        return Err(GpuError::Other(format!(
            "Shape mismatch: a={:?}, b={:?}, output={:?}",
            a.shape(),
            b.shape(),
            output.shape()
        )));
    }

    let len = a.len();
    if len == 0 {
        return Ok(());
    }

    let threads_per_block: u32 = 256;
    let num_blocks = (len as u32 + threads_per_block - 1) / threads_per_block;

    let cube_count = CubeCount::Static(num_blocks, 1, 1);
    let cube_dim = CubeDim::new(threads_per_block, 1, 1);

    let client = runtime.client();

    unsafe {
        mul_kernel::launch_unchecked::<F, R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(a.handle(), a.len(), 1),
            ArrayArg::from_raw_parts(b.handle(), b.len(), 1),
            ArrayArg::from_raw_parts(output.handle(), output.len(), 1),
        );
    }

    Ok(())
}

/// Execute scalar multiplication on GPU: output = a * scalar
pub fn mul_scalar_execute<R: Runtime, F: Float + CubeElement>(
    a: &GpuBuffer<R>,
    scalar: F,
    output: &GpuBuffer<R>,
    runtime: &RuntimeContext<R>,
) -> Result<()> {
    if a.shape() != output.shape() {
        return Err(GpuError::Other(format!(
            "Shape mismatch: a={:?}, output={:?}",
            a.shape(),
            output.shape()
        )));
    }

    let len = a.len();
    if len == 0 {
        return Ok(());
    }

    let threads_per_block: u32 = 256;
    let num_blocks = (len as u32 + threads_per_block - 1) / threads_per_block;

    let cube_count = CubeCount::Static(num_blocks, 1, 1);
    let cube_dim = CubeDim::new(threads_per_block, 1, 1);

    let client = runtime.client();

    unsafe {
        mul_scalar_kernel::launch_unchecked::<F, R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(a.handle(), a.len(), 1),
            ScalarArg::new(scalar),
            ArrayArg::from_raw_parts(output.handle(), output.len(), 1),
        );
    }

    Ok(())
}

/// Execute element-wise subtraction on GPU: output = a - b
pub fn sub_execute<R: Runtime, F: Float + CubeElement>(
    a: &GpuBuffer<R>,
    b: &GpuBuffer<R>,
    output: &GpuBuffer<R>,
    runtime: &RuntimeContext<R>,
) -> Result<()> {
    if a.shape() != b.shape() || a.shape() != output.shape() {
        return Err(GpuError::Other(format!(
            "Shape mismatch: a={:?}, b={:?}, output={:?}",
            a.shape(),
            b.shape(),
            output.shape()
        )));
    }

    let len = a.len();
    if len == 0 {
        return Ok(());
    }

    let threads_per_block: u32 = 256;
    let num_blocks = (len as u32 + threads_per_block - 1) / threads_per_block;

    let cube_count = CubeCount::Static(num_blocks, 1, 1);
    let cube_dim = CubeDim::new(threads_per_block, 1, 1);

    let client = runtime.client();

    unsafe {
        sub_kernel::launch_unchecked::<F, R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(a.handle(), a.len(), 1),
            ArrayArg::from_raw_parts(b.handle(), b.len(), 1),
            ArrayArg::from_raw_parts(output.handle(), output.len(), 1),
        );
    }

    Ok(())
}

/// Execute element-wise division on GPU: output = a / b
pub fn div_execute<R: Runtime, F: Float + CubeElement>(
    a: &GpuBuffer<R>,
    b: &GpuBuffer<R>,
    output: &GpuBuffer<R>,
    runtime: &RuntimeContext<R>,
) -> Result<()> {
    if a.shape() != b.shape() || a.shape() != output.shape() {
        return Err(GpuError::Other(format!(
            "Shape mismatch: a={:?}, b={:?}, output={:?}",
            a.shape(),
            b.shape(),
            output.shape()
        )));
    }

    let len = a.len();
    if len == 0 {
        return Ok(());
    }

    let threads_per_block: u32 = 256;
    let num_blocks = (len as u32 + threads_per_block - 1) / threads_per_block;

    let cube_count = CubeCount::Static(num_blocks, 1, 1);
    let cube_dim = CubeDim::new(threads_per_block, 1, 1);

    let client = runtime.client();

    unsafe {
        div_kernel::launch_unchecked::<F, R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(a.handle(), a.len(), 1),
            ArrayArg::from_raw_parts(b.handle(), b.len(), 1),
            ArrayArg::from_raw_parts(output.handle(), output.len(), 1),
        );
    }

    Ok(())
}

/// Execute RGB to Grayscale conversion on GPU
///
/// Converts an RGB image to grayscale using: Y = 0.299*R + 0.587*G + 0.114*B
///
/// # Arguments
///
/// * `rgb` - Input RGB buffer (shape: [H, W, 3] flattened to H*W*3)
/// * `gray` - Output grayscale buffer (shape: [H, W, 1] flattened to H*W)
/// * `runtime` - GPU runtime context
pub fn rgb_to_gray_execute<R: Runtime, F: Float + CubeElement>(
    rgb: &GpuBuffer<R>,
    gray: &GpuBuffer<R>,
    runtime: &RuntimeContext<R>,
) -> Result<()> {
    // Validate buffer sizes
    let num_pixels = gray.len();
    if rgb.len() != num_pixels * 3 {
        return Err(GpuError::Other(format!(
            "RGB buffer size mismatch: expected {}, got {}",
            num_pixels * 3,
            rgb.len()
        )));
    }

    if num_pixels == 0 {
        return Ok(());
    }

    let threads_per_block: u32 = 256;
    let num_blocks = (num_pixels as u32 + threads_per_block - 1) / threads_per_block;

    let cube_count = CubeCount::Static(num_blocks, 1, 1);
    let cube_dim = CubeDim::new(threads_per_block, 1, 1);

    let client = runtime.client();

    unsafe {
        rgb_to_gray_kernel::launch_unchecked::<F, R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(rgb.handle(), rgb.len(), 1),
            ArrayArg::from_raw_parts(gray.handle(), gray.len(), 1),
        );
    }

    Ok(())
}

/// Execute Grayscale to RGB conversion on GPU
///
/// Converts a grayscale image to RGB by duplicating the gray value to all channels.
///
/// # Arguments
///
/// * `gray` - Input grayscale buffer (shape: [H, W, 1] flattened to H*W)
/// * `rgb` - Output RGB buffer (shape: [H, W, 3] flattened to H*W*3)
/// * `runtime` - GPU runtime context
pub fn gray_to_rgb_execute<R: Runtime, F: Float + CubeElement>(
    gray: &GpuBuffer<R>,
    rgb: &GpuBuffer<R>,
    runtime: &RuntimeContext<R>,
) -> Result<()> {
    // Validate buffer sizes
    let num_pixels = gray.len();
    if rgb.len() != num_pixels * 3 {
        return Err(GpuError::Other(format!(
            "RGB buffer size mismatch: expected {}, got {}",
            num_pixels * 3,
            rgb.len()
        )));
    }

    if num_pixels == 0 {
        return Ok(());
    }

    let threads_per_block: u32 = 256;
    let num_blocks = (num_pixels as u32 + threads_per_block - 1) / threads_per_block;

    let cube_count = CubeCount::Static(num_blocks, 1, 1);
    let cube_dim = CubeDim::new(threads_per_block, 1, 1);

    let client = runtime.client();

    unsafe {
        gray_to_rgb_kernel::launch_unchecked::<F, R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(gray.handle(), gray.len(), 1),
            ArrayArg::from_raw_parts(rgb.handle(), rgb.len(), 1),
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::{allocate, to_cpu, to_device};

    #[test]
    #[cfg(feature = "cuda")]
    fn test_add_execute_cuda() {
        use crate::runtime::init_cuda_runtime;

        // Skip if CUDA not available
        let Ok(runtime) = init_cuda_runtime() else {
            return;
        };

        // Test data
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let b_data = vec![5.0f32, 6.0, 7.0, 8.0];
        let shape = vec![4];

        // Transfer to GPU
        let a_gpu = to_device(&a_data, shape.clone(), &runtime).unwrap();
        let b_gpu = to_device(&b_data, shape.clone(), &runtime).unwrap();
        let out_gpu = allocate::<_, f32>(shape, &runtime).unwrap();

        // Execute on GPU
        add_execute::<_, f32>(&a_gpu, &b_gpu, &out_gpu, &runtime).unwrap();

        // Transfer back
        let result: Vec<f32> = to_cpu(&out_gpu, &runtime).unwrap();

        // Verify
        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn test_mul_execute_wgpu() {
        use crate::runtime::init_wgpu_runtime;

        let Ok(runtime) = init_wgpu_runtime() else {
            return;
        };

        let a_data = vec![2.0f32, 3.0, 4.0];
        let b_data = vec![5.0f32, 6.0, 7.0];
        let shape = vec![3];

        let a_gpu = to_device(&a_data, shape.clone(), &runtime).unwrap();
        let b_gpu = to_device(&b_data, shape.clone(), &runtime).unwrap();
        let out_gpu = allocate::<_, f32>(shape, &runtime).unwrap();

        mul_execute::<_, f32>(&a_gpu, &b_gpu, &out_gpu, &runtime).unwrap();

        let result: Vec<f32> = to_cpu(&out_gpu, &runtime).unwrap();

        assert_eq!(result, vec![10.0, 18.0, 28.0]);
    }
}


/// Execute sum reduction on GPU.
///
/// Returns the sum of all elements in the input tensor.
///
/// # Arguments
///
/// * `input` - Input buffer to reduce
/// * `runtime` - Runtime context
///
/// # Returns
///
/// The sum as a single scalar value
///
/// # Example
///
/// ```ignore
/// let data_gpu = to_device(&vec![1.0, 2.0, 3.0, 4.0], vec![4], &runtime)?;
/// let sum = sum_execute(&data_gpu, &runtime)?; // Returns 10.0
/// ```
pub fn sum_execute<R: Runtime, F: Float + CubeElement + bytemuck::Pod + Copy>(
    input: &GpuBuffer<R>,
    runtime: &RuntimeContext<R>,
) -> Result<F> {
    let len = input.len();
    if len == 0 {
        return Ok(F::new(0.0));
    }

    // For small inputs, just do a single pass
    if len <= 256 {
        let threads_per_block: u32 = 256;
        let num_blocks = 1u32;

        let cube_count = CubeCount::Static(num_blocks, 1, 1);
        let cube_dim = CubeDim::new(threads_per_block, 1, 1);

        // Output buffer for partial sums (one per block)
        let output_gpu = allocate::<_, F>(vec![num_blocks as usize], runtime)?;

        let client = runtime.client();

        unsafe {
            sum_kernel::launch_unchecked::<F, R>(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(input.handle(), input.len(), 1),
                ArrayArg::from_raw_parts(output_gpu.handle(), num_blocks as usize, 1),
            );
        }

        // Read result back
        let partial_sums: Vec<F> = to_cpu(&output_gpu, runtime)?;
        return Ok(partial_sums[0]);
    }

    // For larger inputs, use two-phase reduction
    let threads_per_block: u32 = 256;
    let num_blocks = Ord::min((len as u32 + threads_per_block - 1) / threads_per_block, 1024);

    let cube_count = CubeCount::Static(num_blocks, 1, 1);
    let cube_dim = CubeDim::new(threads_per_block, 1, 1);

    // First phase: reduce to partial sums
    let partial_output = allocate::<_, F>(vec![num_blocks as usize], runtime)?;

    let client = runtime.client();

    unsafe {
        sum_kernel::launch_unchecked::<F, R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(input.handle(), input.len(), 1),
            ArrayArg::from_raw_parts(partial_output.handle(), num_blocks as usize, 1),
        );
    }

    // Second phase: reduce partial sums on CPU (simpler for now)
    let partial_sums: Vec<F> = to_cpu(&partial_output, runtime)?;
    let mut final_sum = F::new(0.0);
    for &val in &partial_sums {
        final_sum = final_sum + val;
    }

    Ok(final_sum)
}

/// Execute mean reduction on GPU.
///
/// Returns the mean (average) of all elements.
pub fn mean_execute<R: Runtime, F: Float + CubeElement + bytemuck::Pod + Copy>(
    input: &GpuBuffer<R>,
    runtime: &RuntimeContext<R>,
) -> Result<F> {
    let len = input.len();
    if len == 0 {
        return Ok(F::new(0.0));
    }

    // Sum first, then divide by count
    let sum = sum_execute::<R, F>(input, runtime)?;
    Ok(sum / F::new(len as f32))
}

/// Execute min reduction on GPU.
///
/// Returns the minimum value in the tensor.
pub fn min_execute<R: Runtime, F: Float + CubeElement + bytemuck::Pod + Copy>(
    input: &GpuBuffer<R>,
    runtime: &RuntimeContext<R>,
) -> Result<F> {
    let len = input.len();
    if len == 0 {
        return Err(GpuError::Other("Cannot compute min of empty tensor".into()));
    }

    if len <= 256 {
        let threads_per_block: u32 = 256;
        let num_blocks = 1u32;

        let cube_count = CubeCount::Static(num_blocks, 1, 1);
        let cube_dim = CubeDim::new(threads_per_block, 1, 1);

        let output_gpu = allocate::<_, F>(vec![num_blocks as usize], runtime)?;

        let client = runtime.client();

        unsafe {
            min_kernel::launch_unchecked::<F, R>(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(input.handle(), input.len(), 1),
                ArrayArg::from_raw_parts(output_gpu.handle(), num_blocks as usize, 1),
            );
        }

        let partial_mins: Vec<F> = to_cpu(&output_gpu, runtime)?;
        return Ok(partial_mins[0]);
    }

    // Two-phase reduction
    let threads_per_block: u32 = 256;
    let num_blocks = Ord::min((len as u32 + threads_per_block - 1) / threads_per_block, 1024);

    let cube_count = CubeCount::Static(num_blocks, 1, 1);
    let cube_dim = CubeDim::new(threads_per_block, 1, 1);

    let partial_output = allocate::<_, F>(vec![num_blocks as usize], runtime)?;

    let client = runtime.client();

    unsafe {
        min_kernel::launch_unchecked::<F, R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(input.handle(), input.len(), 1),
            ArrayArg::from_raw_parts(partial_output.handle(), num_blocks as usize, 1),
        );
    }

    // Find min of partial results
    let partial_mins: Vec<F> = to_cpu(&partial_output, runtime)?;
    let mut final_min = F::new(f32::MAX);
    for &val in &partial_mins {
        if val < final_min {
            final_min = val;
        }
    }

    Ok(final_min)
}

/// Execute max reduction on GPU.
///
/// Returns the maximum value in the tensor.
pub fn max_execute<R: Runtime, F: Float + CubeElement + bytemuck::Pod + Copy>(
    input: &GpuBuffer<R>,
    runtime: &RuntimeContext<R>,
) -> Result<F> {
    let len = input.len();
    if len == 0 {
        return Err(GpuError::Other("Cannot compute max of empty tensor".into()));
    }

    if len <= 256 {
        let threads_per_block: u32 = 256;
        let num_blocks = 1u32;

        let cube_count = CubeCount::Static(num_blocks, 1, 1);
        let cube_dim = CubeDim::new(threads_per_block, 1, 1);

        let output_gpu = allocate::<_, F>(vec![num_blocks as usize], runtime)?;

        let client = runtime.client();

        unsafe {
            max_kernel::launch_unchecked::<F, R>(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(input.handle(), input.len(), 1),
                ArrayArg::from_raw_parts(output_gpu.handle(), num_blocks as usize, 1),
            );
        }

        let partial_maxs: Vec<F> = to_cpu(&output_gpu, runtime)?;
        return Ok(partial_maxs[0]);
    }

    // Two-phase reduction
    let threads_per_block: u32 = 256;
    let num_blocks = Ord::min((len as u32 + threads_per_block - 1) / threads_per_block, 1024);

    let cube_count = CubeCount::Static(num_blocks, 1, 1);
    let cube_dim = CubeDim::new(threads_per_block, 1, 1);

    let partial_output = allocate::<_, F>(vec![num_blocks as usize], runtime)?;

    let client = runtime.client();

    unsafe {
        max_kernel::launch_unchecked::<F, R>(
            client,
            cube_count,
            cube_dim,
            ArrayArg::from_raw_parts(input.handle(), input.len(), 1),
            ArrayArg::from_raw_parts(partial_output.handle(), num_blocks as usize, 1),
        );
    }

    // Find max of partial results
    let partial_maxs: Vec<F> = to_cpu(&partial_output, runtime)?;
    let mut final_max = F::new(f32::MIN);
    for &val in &partial_maxs {
        if val > final_max {
            final_max = val;
        }
    }

    Ok(final_max)
}

/// Interpolation mode for GPU resize operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuInterpolation {
    /// Bilinear interpolation (smooth, recommended for downscaling)
    Bilinear,
    /// Nearest-neighbor interpolation (fast, preserves hard edges)
    Nearest,
}

/// Execute image resize on GPU.
///
/// Resizes an image from `[src_h, src_w, channels]` to `[dst_h, dst_w, channels]`.
/// Data is in HWC (height, width, channels) layout, flattened to 1-D.
///
/// # Arguments
///
/// * `src` - Source image buffer (size: `src_h * src_w * channels`)
/// * `dst` - Destination image buffer (size: `dst_h * dst_w * channels`, pre-allocated)
/// * `src_h` - Source image height
/// * `src_w` - Source image width
/// * `dst_h` - Destination image height
/// * `dst_w` - Destination image width
/// * `channels` - Number of channels per pixel
/// * `interpolation` - Interpolation method
/// * `runtime` - GPU runtime context
pub fn resize_execute<R: Runtime, F: Float + CubeElement>(
    src: &GpuBuffer<R>,
    dst: &GpuBuffer<R>,
    src_h: u32,
    src_w: u32,
    dst_h: u32,
    dst_w: u32,
    channels: u32,
    interpolation: GpuInterpolation,
    runtime: &RuntimeContext<R>,
) -> Result<()> {
    // Validate buffer sizes
    let expected_src = (src_h * src_w * channels) as usize;
    let expected_dst = (dst_h * dst_w * channels) as usize;

    if src.len() != expected_src {
        return Err(GpuError::InvalidBufferSize {
            expected: expected_src,
            actual: src.len(),
        });
    }
    if dst.len() != expected_dst {
        return Err(GpuError::InvalidBufferSize {
            expected: expected_dst,
            actual: dst.len(),
        });
    }

    let num_pixels = (dst_h * dst_w) as usize;
    if num_pixels == 0 {
        return Ok(());
    }

    let threads_per_block: u32 = 256;
    let num_blocks = (num_pixels as u32 + threads_per_block - 1) / threads_per_block;

    let cube_count = CubeCount::Static(num_blocks, 1, 1);
    let cube_dim = CubeDim::new(threads_per_block, 1, 1);

    let client = runtime.client();

    match interpolation {
        GpuInterpolation::Bilinear => unsafe {
            resize_bilinear_kernel::launch_unchecked::<F, R>(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(src.handle(), src.len(), 1),
                ArrayArg::from_raw_parts(dst.handle(), dst.len(), 1),
                ScalarArg::new(src_h),
                ScalarArg::new(src_w),
                ScalarArg::new(dst_h),
                ScalarArg::new(dst_w),
                ScalarArg::new(channels),
            );
        },
        GpuInterpolation::Nearest => unsafe {
            resize_nearest_kernel::launch_unchecked::<F, R>(
                client,
                cube_count,
                cube_dim,
                ArrayArg::from_raw_parts(src.handle(), src.len(), 1),
                ArrayArg::from_raw_parts(dst.handle(), dst.len(), 1),
                ScalarArg::new(src_h),
                ScalarArg::new(src_w),
                ScalarArg::new(dst_h),
                ScalarArg::new(dst_w),
                ScalarArg::new(channels),
            );
        },
    }

    Ok(())
}
