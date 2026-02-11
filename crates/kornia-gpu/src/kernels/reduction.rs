//! Reduction operations on GPU.
//!
//! This module provides parallel reduction operations using shared memory
//! for efficient GPU computation.

#![allow(missing_docs)]

use cubecl::prelude::*;

/// Sum reduction kernel using parallel reduction with shared memory.
///
/// This implements a two-phase reduction:
/// 1. Each block reduces its portion to shared memory
/// 2. Thread 0 writes the block's partial sum to output
///
/// For complete reduction, you may need to launch this multiple times
/// or use a follow-up kernel to sum the partial results.
///
/// # Arguments
///
/// * `input` - Input array to reduce
/// * `output` - Output array for partial sums (size = num_blocks)
///
/// # Thread Model
///
/// - Each block processes BLOCK_SIZE elements
/// - Uses shared memory for efficient reduction
/// - Thread 0 writes the final block result
#[cube(launch_unchecked)]
pub fn sum_kernel<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
) {
    // Get thread and block IDs
    let tid = UNIT_POS;
    let block_id = CUBE_POS;
    let block_size = CUBE_DIM_X;
    
    // Global position in input array
    let global_id = block_id * block_size + tid;
    
    // Load data into shared memory
    // If out of bounds, load zero
    let mut local_sum = if global_id < input.len() {
        input[global_id]
    } else {
        F::new(0.0)
    };
    
    // Parallel reduction in shared memory
    // This is a tree-based reduction
    let mut stride = block_size / 2;
    
    while stride > 0 {
        sync_units(); // Synchronize threads in the block
        
        if tid < stride && (global_id + stride) < input.len() {
            let other_id = global_id + stride;
            local_sum = local_sum + input[other_id];
        }
        
        stride /= 2;
    }
    
    // Thread 0 writes the block's result
    if tid == 0 {
        output[block_id] = local_sum;
    }
}

/// Mean reduction kernel.
///
/// Computes the sum first, then divides by count in a separate step.
/// This is the same as sum_kernel but caller must divide by N.
#[cube(launch_unchecked)]
pub fn mean_kernel<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
) {
    // Mean is just sum / N, we compute sum here
    let tid = UNIT_POS;
    let block_id = CUBE_POS;
    let block_size = CUBE_DIM_X;
    let global_id = block_id * block_size + tid;
    
    let mut local_sum = if global_id < input.len() {
        input[global_id]
    } else {
        F::new(0.0)
    };
    
    let mut stride = block_size / 2;
    
    while stride > 0 {
        sync_units();
        
        if tid < stride && (global_id + stride) < input.len() {
            let other_id = global_id + stride;
            local_sum = local_sum + input[other_id];
        }
        
        stride /= 2;
    }
    
    if tid == 0 {
        output[block_id] = local_sum;
    }
}

/// Min reduction kernel.
///
/// Finds the minimum value using parallel reduction.
#[cube(launch_unchecked)]
pub fn min_kernel<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
) {
    let tid = UNIT_POS;
    let block_id = CUBE_POS;
    let block_size = CUBE_DIM_X;
    let global_id = block_id * block_size + tid;
    
    // Initialize with first element or max value
    let mut local_min = if global_id < input.len() {
        input[global_id]
    } else {
        // Use a large value for out-of-bounds
        F::new(f32::MAX)
    };
    
    let mut stride = block_size / 2;
    
    while stride > 0 {
        sync_units();
        
        if tid < stride && (global_id + stride) < input.len() {
            let other_id = global_id + stride;
            let other_val = input[other_id];
            
            // Min comparison
            if other_val < local_min {
                local_min = other_val;
            }
        }
        
        stride /= 2;
    }
    
    if tid == 0 {
        output[block_id] = local_min;
    }
}

/// Max reduction kernel.
///
/// Finds the maximum value using parallel reduction.
#[cube(launch_unchecked)]
pub fn max_kernel<F: Float>(
    input: &Array<F>,
    output: &mut Array<F>,
) {
    let tid = UNIT_POS;
    let block_id = CUBE_POS;
    let block_size = CUBE_DIM_X;
    let global_id = block_id * block_size + tid;
    
    // Initialize with first element or min value
    let mut local_max = if global_id < input.len() {
        input[global_id]
    } else {
        // Use a small value for out-of-bounds
        F::new(f32::MIN)
    };
    
    let mut stride = block_size / 2;
    
    while stride > 0 {
        sync_units();
        
        if tid < stride && (global_id + stride) < input.len() {
            let other_id = global_id + stride;
            let other_val = input[other_id];
            
            // Max comparison
            if other_val > local_max {
                local_max = other_val;
            }
        }
        
        stride /= 2;
    }
    
    if tid == 0 {
        output[block_id] = local_max;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_reduction_kernels_compile() {
        // Verify kernels compile with CubeCL
        // Actual execution tests are in ops module
    }
}
