//! Integration tests for GPU functionality.

#[cfg(test)]
mod integration {
    use crate::{GpuAllocator, GpuDevice};

    #[test]
    fn test_gpu_device_allocator_integration() {
        let device = GpuDevice::new(0).expect("Failed to create GPU device");
        let allocator = GpuAllocator::new(&device);
        assert_eq!(allocator.device_id(), device.device_id());
    }
}
