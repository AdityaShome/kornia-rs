//! GPU kernel implementations using CubeCL.
//!
//! This module contains all GPU kernels for tensor operations.
//! Each kernel is annotated with `#[cube(launch_unchecked)]` for CubeCL compilation.

// CubeCL's #[cube] macro generates code without docs
#![allow(missing_docs)]

pub mod elementwise;
pub mod reduction;
pub mod color;
pub mod resize;

// Re-export commonly used kernels
pub use elementwise::*;
pub use reduction::*;
pub use color::*;
pub use resize::*;
