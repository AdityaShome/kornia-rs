//! GPU device handle and allocator.
//!
//! This module provides `GpuAllocator` as a lightweight handle to a GPU device,
//! used for allocating GPU memory via CubeCL. Unlike `CpuAllocator`, GPU memory
//! is managed through opaque `Handle` objects (via `GpuBuffer<R>`) rather than
//! raw pointers, so `GpuAllocator` does NOT implement `TensorAllocator`.
//!
//! To move data between `Tensor<T, N, CpuAllocator>` and `GpuBuffer<R>`, use
//! the conversion functions in the [`memory`](crate::memory) module.

use crate::device::GpuDevice;

/// GPU memory allocator handle.
///
/// A lightweight reference to a GPU device used for memory operations.
/// This is used alongside `RuntimeContext<R>` to allocate and manage GPU buffers.
///
/// # Examples
///
/// ```rust,ignore
/// use kornia_gpu::{GpuDevice, GpuAllocator};
///
/// let device = GpuDevice::new(0)?;
/// let allocator = GpuAllocator::new(&device);
/// ```
#[derive(Debug, Clone)]
pub struct GpuAllocator {
    device_id: usize,
}

impl GpuAllocator {
    /// Create a new GPU allocator for the specified device.
    pub fn new(device: &GpuDevice) -> Self {
        Self {
            device_id: device.device_id(),
        }
    }

    /// Get the device ID for this allocator.
    pub fn device_id(&self) -> usize {
        self.device_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocator_creation() {
        let device = GpuDevice::new(0).unwrap();
        let allocator = GpuAllocator::new(&device);
        assert_eq!(allocator.device_id(), 0);
    }
}
