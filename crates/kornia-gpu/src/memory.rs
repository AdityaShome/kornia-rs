//! Memory transfer operations between CPU and GPU.
//!
//! This module provides functions for moving data between CPU and GPU memory.

use crate::error::{GpuError, Result};
use crate::runtime::{GpuBuffer, RuntimeContext};
use cubecl::prelude::*;
use kornia_tensor::{CpuAllocator, Tensor};

/// Transfer data from CPU to GPU.
///
/// # Arguments
///
/// * `data` - Slice of data to transfer
/// * `shape` - Shape of the tensor
/// * `runtime` - Runtime context with GPU client
///
/// # Returns
///
/// A GPU buffer containing the transferred data
///
/// # Example
///
/// ```ignore
/// let data = vec![1.0f32, 2.0, 3.0, 4.0];
/// let shape = vec![2, 2];
/// let buffer = to_device(&data, shape, &runtime)?;
/// ```
pub fn to_device<R: Runtime, T: CubePrimitive + bytemuck::Pod + Copy>(
    data: &[T],
    shape: Vec<usize>,
    runtime: &RuntimeContext<R>,
) -> Result<GpuBuffer<R>> {
    // Verify shape matches data length
    let expected_len: usize = shape.iter().product();
    if data.len() != expected_len {
        return Err(GpuError::Other(format!(
            "Data length {} does not match shape {:?} (expected {})",
            data.len(),
            shape,
            expected_len
        )));
    }

    // Create buffer on GPU from CPU data
    let client = runtime.client();
    let bytes = bytemuck::cast_slice(data);
    let handle = client.create(bytes);

    Ok(GpuBuffer::from_handle(handle, shape))
}

/// Transfer data from GPU to CPU.
///
/// # Arguments
///
/// * `buffer` - GPU buffer to read from
/// * `runtime` - Runtime context with GPU client
///
/// # Returns
///
/// A Vec containing the data from GPU
///
/// # Example
///
/// ```ignore
/// let data = to_cpu(&gpu_buffer, &runtime)?;
/// ```
pub fn to_cpu<R: Runtime, T: CubePrimitive + bytemuck::Pod + Copy>(
    buffer: &GpuBuffer<R>,
    runtime: &RuntimeContext<R>,
) -> Result<Vec<T>> {
    let client = runtime.client();
    
    // Read bytes from GPU (clone handle for binding as it consumes self)
    let bytes = client.read(buffer.handle().clone().binding());
    
    // Convert bytes to Vec<T>
    let data: Vec<T> = bytemuck::cast_slice(&bytes).to_vec();
    
    Ok(data)
}

/// Allocate an empty GPU buffer with the given shape.
///
/// # Arguments
///
/// * `shape` - Shape of the tensor
/// * `runtime` - Runtime context with GPU client
///
/// # Returns
///
/// An empty GPU buffer ready for use
pub fn allocate<R: Runtime, T: CubePrimitive>(
    shape: Vec<usize>,
    runtime: &RuntimeContext<R>,
) -> Result<GpuBuffer<R>> {
    let len: usize = shape.iter().product();
    let client = runtime.client();
    
    // Allocate empty buffer
    let handle = client.empty(len * std::mem::size_of::<T>());
    
    Ok(GpuBuffer::from_handle(handle, shape))
}

/// Transfer a CPU tensor to the GPU, returning a `GpuBuffer`.
///
/// The tensor's data is copied to GPU memory. The shape is preserved in the
/// returned `GpuBuffer` for later reconstruction.
///
/// # Arguments
///
/// * `tensor` - A CPU tensor to transfer
/// * `runtime` - Runtime context with GPU client
///
/// # Example
///
/// ```ignore
/// let tensor = Tensor::<f32, 2, _>::from_shape_vec([2, 3], data, CpuAllocator)?;
/// let gpu_buf = tensor_to_device(&tensor, &runtime)?;
/// ```
pub fn tensor_to_device<R: Runtime, T: CubePrimitive + bytemuck::Pod + Copy, const N: usize>(
    tensor: &Tensor<T, N, CpuAllocator>,
    runtime: &RuntimeContext<R>,
) -> Result<GpuBuffer<R>> {
    let data = tensor.as_slice();
    let shape: Vec<usize> = tensor.shape.to_vec();
    to_device(data, shape, runtime)
}

/// Transfer a `GpuBuffer` back to a CPU tensor.
///
/// Reads GPU memory back to the host and constructs a `Tensor` with the given shape.
///
/// # Arguments
///
/// * `buffer` - GPU buffer to read from
/// * `shape` - The shape for the resulting tensor (must match buffer element count)
/// * `runtime` - Runtime context with GPU client
///
/// # Example
///
/// ```ignore
/// let tensor: Tensor<f32, 2, CpuAllocator> =
///     tensor_from_device(&gpu_buf, [2, 3], &runtime)?;
/// ```
pub fn tensor_from_device<R: Runtime, T: CubePrimitive + bytemuck::Pod + Copy, const N: usize>(
    buffer: &GpuBuffer<R>,
    shape: [usize; N],
    runtime: &RuntimeContext<R>,
) -> Result<Tensor<T, N, CpuAllocator>> {
    let data: Vec<T> = to_cpu(buffer, runtime)?;
    let tensor = Tensor::from_shape_vec(shape, data, CpuAllocator)
        .map_err(|e| GpuError::Other(format!("Failed to create tensor from GPU data: {}", e)))?;
    Ok(tensor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "cuda")]
    fn test_memory_transfer_roundtrip() {
        use crate::runtime::init_cuda_runtime;
        
        // Skip if CUDA not available
        let Ok(runtime) = init_cuda_runtime() else {
            return;
        };

        // Test data
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];

        // Transfer to GPU
        let gpu_buffer = to_device(&data, shape.clone(), &runtime).unwrap();
        
        // Transfer back to CPU
        let result: Vec<f32> = to_cpu(&gpu_buffer, &runtime).unwrap();

        // Verify
        assert_eq!(result, data);
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn test_memory_transfer_wgpu() {
        use crate::runtime::init_wgpu_runtime;
        
        // Skip if WGPU not available
        let Ok(runtime) = init_wgpu_runtime() else {
            return;
        };

        let data = vec![10.0f32, 20.0, 30.0, 40.0];
        let shape = vec![4];

        let gpu_buffer = to_device(&data, shape.clone(), &runtime).unwrap();
        let result: Vec<f32> = to_cpu(&gpu_buffer, &runtime).unwrap();

        assert_eq!(result, data);
    }

    #[test]
    fn test_shape_mismatch() {
        let data = vec![1.0f32, 2.0, 3.0];
        let wrong_shape = vec![2, 2];
        assert_eq!(data.len(), 3);
        assert_eq!(wrong_shape.iter().product::<usize>(), 4);
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_tensor_roundtrip_cuda() {
        use crate::runtime::init_cuda_runtime;

        let Ok(runtime) = init_cuda_runtime() else {
            return;
        };

        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tensor =
            Tensor::<f32, 2, _>::from_shape_vec([2, 3], data.clone(), CpuAllocator).unwrap();

        // CPU tensor -> GPU buffer -> CPU tensor
        let gpu_buf = tensor_to_device(&tensor, &runtime).unwrap();
        let result: Tensor<f32, 2, CpuAllocator> =
            tensor_from_device(&gpu_buf, [2, 3], &runtime).unwrap();

        assert_eq!(result.as_slice(), data.as_slice());
        assert_eq!(result.shape, [2, 3]);
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn test_tensor_roundtrip_wgpu() {
        use crate::runtime::init_wgpu_runtime;

        let Ok(runtime) = init_wgpu_runtime() else {
            return;
        };

        let data = vec![10.0f32, 20.0, 30.0, 40.0];
        let tensor =
            Tensor::<f32, 1, _>::from_shape_vec([4], data.clone(), CpuAllocator).unwrap();

        let gpu_buf = tensor_to_device(&tensor, &runtime).unwrap();
        let result: Tensor<f32, 1, CpuAllocator> =
            tensor_from_device(&gpu_buf, [4], &runtime).unwrap();

        assert_eq!(result.as_slice(), data.as_slice());
    }
}
