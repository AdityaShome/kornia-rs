//! CubeCL runtime wrapper for actual GPU execution.
//!
//! This module provides a wrapper around CubeCL's runtime for executing kernels.

use crate::error::Result;
use cubecl::prelude::*;
use cubecl::server::Handle;
use std::marker::PhantomData;

// Re-export CubeCL's Runtime trait so downstream crates don't need cubecl directly
pub use cubecl::Runtime as CubeclRuntime;

/// Runtime backend types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// CUDA backend (NVIDIA GPUs)
    #[cfg(feature = "cuda")]
    Cuda,
    /// WGPU backend (Vulkan/Metal/DirectX12)
    #[cfg(feature = "wgpu")]
    Wgpu,
    /// CPU fallback
    Cpu,
}

/// GPU runtime context with CubeCL Client.
///
/// Wraps CubeCL's Client and provides a simplified interface for kernel execution
/// and memory management.
pub struct RuntimeContext<R: Runtime> {
    client: ComputeClient<R::Server, R::Channel>,
    _phantom: PhantomData<R>,
}

impl<R: Runtime> RuntimeContext<R> {
    /// Create a new runtime context with the specified device.
    ///
    /// # Arguments
    ///
    /// * `device` - The CubeCL device to use
    pub fn new(device: R::Device) -> Self {
        Self {
            client: R::client(&device),
            _phantom: PhantomData,
        }
    }

    /// Get a reference to the compute client.
    pub fn client(&self) -> &ComputeClient<R::Server, R::Channel> {
        &self.client
    }

    /// Get the backend name (e.g. "cuda", "wgpu<wgsl>").
    pub fn backend_name(&self) -> &'static str {
        R::name()
    }

    /// Get memory device properties (max page size, alignment).
    pub fn memory_properties(&self) -> (u64, u64) {
        let props = self.client.properties();
        let mem = props.memory_properties();
        (mem.max_page_size, mem.alignment)
    }

    /// Synchronize all pending GPU operations.
    ///
    /// Reads a tiny buffer to force all enqueued kernels to complete.
    /// Useful for timing and correctness verification.
    pub fn sync(&self) {
        // client.read() is a blocking sync point in CubeCL.
        // Create a 1-byte buffer, read it back to force a full pipeline flush.
        let handle = self.client.empty(1);
        let _ = self.client.read(handle.binding());
    }
}

/// GPU buffer wrapping CubeCL's tensor handle.
///
/// This provides a safe wrapper around CubeCL's memory handles for GPU tensors.
pub struct GpuBuffer<R: Runtime> {
    handle: Handle,
    shape: Vec<usize>,
    _phantom: PhantomData<R>,
}

impl<R: Runtime> GpuBuffer<R> {
    /// Create a new GPU buffer from a handle and shape.
    pub fn from_handle(handle: Handle, shape: Vec<usize>) -> Self {
        Self {
            handle,
            shape,
            _phantom: PhantomData,
        }
    }

    /// Get the shape of the buffer.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the total number of elements.
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.shape.iter().product::<usize>() == 0
    }

    /// Get a reference to the underlying handle.
    pub fn handle(&self) -> &Handle {
        &self.handle
    }
}

/// Initialize a CubeCL runtime with default device.
///
/// # Returns
///
/// A runtime context ready for kernel execution.
///
/// # Errors
///
/// Returns an error if the runtime cannot be initialized (e.g., no GPU found).
#[cfg(feature = "cuda")]
pub fn init_cuda_runtime() -> Result<RuntimeContext<cubecl_cuda::CudaRuntime>> {
    use cubecl_cuda::CudaDevice;
    
    let device = CudaDevice::new(0);
    Ok(RuntimeContext::new(device))
}

/// Initialize a WGPU runtime with default device.
#[cfg(feature = "wgpu")]
pub fn init_wgpu_runtime() -> Result<RuntimeContext<cubecl_wgpu::WgpuRuntime>> {
    use cubecl_wgpu::WgpuDevice;
    
    let device = WgpuDevice::BestAvailable;
    Ok(RuntimeContext::new(device))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(feature = "cuda")]
    fn test_cuda_runtime_init() {
        if let Ok(runtime) = init_cuda_runtime() {
            println!("Backend: {}", runtime.backend_name());
            let (max_page, align) = runtime.memory_properties();
            println!("Memory: max_page={}B, alignment={}B", max_page, align);
            runtime.sync();
        }
    }

    #[test]
    #[cfg(feature = "wgpu")]
    fn test_wgpu_runtime_init() {
        if let Ok(runtime) = init_wgpu_runtime() {
            println!("Backend: {}", runtime.backend_name());
            let (max_page, align) = runtime.memory_properties();
            println!("Memory: max_page={}B, alignment={}B", max_page, align);
            runtime.sync();
        }
    }
}
