//! Error types for GPU operations.

use thiserror::Error;

/// Result type for GPU operations.
pub type Result<T> = std::result::Result<T, GpuError>;

/// Error types that can occur during GPU operations.
#[derive(Error, Debug)]
pub enum GpuError {
    /// GPU device not available or not found.
    #[error("GPU device not available: {0}")]
    DeviceNotAvailable(String),

    /// Out of GPU memory.
    #[error("Out of GPU memory: {0}")]
    OutOfMemory(String),

    /// Invalid buffer size or dimensions.
    #[error("Invalid buffer size: expected {expected}, got {actual}")]
    InvalidBufferSize {
        /// Expected buffer size
        expected: usize,
        /// Actual buffer size
        actual: usize,
    },

    /// Kernel launch failed.
    #[error("Kernel launch failed: {0}")]
    KernelLaunchFailed(String),

    /// Memory transfer failed (host ↔ device).
    #[error("Memory transfer failed: {0}")]
    MemoryTransferFailed(String),

    /// Device synchronization failed.
    #[error("Device synchronization failed: {0}")]
    SynchronizationFailed(String),

    /// Unsupported operation on this backend.
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    /// CubeCL runtime error.
    #[error("CubeCL error: {0}")]
    CubeCLError(String),

    /// Image error from kornia-image.
    #[error("Image error: {0}")]
    ImageError(#[from] kornia_image::ImageError),

    /// Tensor error from kornia-tensor.
    #[error("Tensor error: {0}")]
    TensorError(#[from] kornia_tensor::TensorError),

    /// Generic error.
    #[error("{0}")]
    Other(String),
}
