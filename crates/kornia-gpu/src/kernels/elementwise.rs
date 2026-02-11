//! Element-wise GPU operations.
//!
//! This module provides basic element-wise operations on GPU arrays using CubeCL.

// CubeCL's #[cube] macro generates additional code that doesn't have docs
#![allow(missing_docs)]

use cubecl::prelude::*;

/// Element-wise addition kernel.
///
/// This is a simple "hello world" kernel that demonstrates CubeCL's basic usage.
/// Each GPU thread processes one element: output[i] = a[i] + b[i]
///
/// # CubeCL Attributes
///
/// - `#[cube(launch_unchecked)]`: Marks this function as a GPU kernel
/// - The function will be compiled to GPU code at runtime
///
/// # Arguments
///
/// * `a` - First input array
/// * `b` - Second input array  
/// * `output` - Output array (must be pre-allocated)
///
/// # Thread Model
///
/// - `ABSOLUTE_POS`: Global thread index (provided by CubeCL)
/// - Each thread computes one element
#[allow(missing_docs)]
#[cube(launch_unchecked)]
pub fn add_kernel<F: Float>(
    a: &Array<F>,
    b: &Array<F>,
    output: &mut Array<F>,
) {
    // Get the global thread index
    // ABSOLUTE_POS is a CubeCL built-in that gives us the thread's position
    let pos = ABSOLUTE_POS;
    
    // Safety check: don't access beyond array bounds
    if pos < a.len() {
        // Element-wise addition
        output[pos] = a[pos] + b[pos];
    }
}

/// Element-wise multiplication kernel.
///
/// Each GPU thread processes one element: output[i] = a[i] * b[i]
#[allow(missing_docs)]
#[cube(launch_unchecked)]
pub fn mul_kernel<F: Float>(
    a: &Array<F>,
    b: &Array<F>,
    output: &mut Array<F>,
) {
    let pos = ABSOLUTE_POS;
    
    if pos < a.len() {
        output[pos] = a[pos] * b[pos];
    }
}

/// Scalar multiplication kernel.
///
/// Each GPU thread processes one element: output[i] = a[i] * scalar
#[allow(missing_docs)]
#[cube(launch_unchecked)]
pub fn mul_scalar_kernel<F: Float>(
    a: &Array<F>,
    scalar: F,
    output: &mut Array<F>,
) {
    let pos = ABSOLUTE_POS;
    
    if pos < a.len() {
        output[pos] = a[pos] * scalar;
    }
}

/// Element-wise subtraction kernel.
///
/// Each GPU thread processes one element: output[i] = a[i] - b[i]
#[allow(missing_docs)]
#[cube(launch_unchecked)]
pub fn sub_kernel<F: Float>(
    a: &Array<F>,
    b: &Array<F>,
    output: &mut Array<F>,
) {
    let pos = ABSOLUTE_POS;
    
    if pos < a.len() {
        output[pos] = a[pos] - b[pos];
    }
}

/// Element-wise division kernel.
///
/// Each GPU thread processes one element: output[i] = a[i] / b[i]
#[allow(missing_docs)]
#[cube(launch_unchecked)]
pub fn div_kernel<F: Float>(
    a: &Array<F>,
    b: &Array<F>,
    output: &mut Array<F>,
) {
    let pos = ABSOLUTE_POS;
    
    if pos < a.len() {
        output[pos] = a[pos] / b[pos];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    // Note: These tests require CubeCL runtime to be initialized
    // For now, they verify that the code compiles
    
    #[test]
    fn test_kernel_compilation() {
        // This test ensures the kernels compile with CubeCL
        // Actual runtime testing requires GPU or CPU backend
        // TODO: Add runtime tests with CubeCL CPU backend
    }
}
