//! GPU acceleration backend for kornia using CubeCL.
//!
//! This crate provides GPU-accelerated implementations of tensor operations
//! and image processing transforms using CubeCL as the primary compute framework.
//!
//! # Features
//!
//! - **Multi-platform**: Supports CUDA, WGPU (Vulkan/Metal/DX12), and CPU fallback
//! - **Type-safe**: Leverages Rust's type system for safe GPU programming
//! - **Zero-cost abstractions**: Minimal overhead for GPU operations
//!
//! # Feature Flags
//!
//! - `cuda`: Enable CUDA backend (NVIDIA GPUs)
//! - `wgpu`: Enable WGPU backend (Vulkan/Metal/DirectX12)
//! - `cuda-native`: Enable native CUDA optimizations (requires CUDA toolkit)
//!
//! # Examples
//!
//! ```rust,ignore
//! use kornia_gpu::{GpuDevice, Runtime};
//!
//! // Initialize GPU device
//! let device = GpuDevice::new(0)?;
//!
//! // Transfer image to GPU
//! let image_gpu = image.to_device(&device)?;
//!
//! // Perform GPU operations
//! let resized = resize_gpu(&image_gpu, new_size)?;
//! ```

#![deny(missing_docs)]
#![warn(clippy::all)]

pub mod allocator;
pub mod device;
pub mod error;

#[cfg(feature = "gpu")]
pub mod runtime;

#[cfg(feature = "gpu")]
pub mod memory;

#[cfg(feature = "gpu")]
pub mod kernels;

#[cfg(feature = "gpu")]
pub mod ops;

#[cfg(test)]
mod tests;

// Re-exports
pub use allocator::GpuAllocator;
pub use device::{GpuDevice, Runtime};
pub use error::{GpuError, Result};

#[cfg(feature = "gpu")]
pub use runtime::*;

#[cfg(feature = "gpu")]
pub use memory::*;

#[cfg(feature = "gpu")]
pub use kernels::*;

#[cfg(feature = "gpu")]
pub use ops::*;
