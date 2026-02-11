//! Color conversion GPU kernels.
//!
//! This module provides GPU-accelerated color space conversions.

#![allow(missing_docs)]

use cubecl::prelude::*;

/// RGB to Grayscale GPU kernel.
///
/// Converts RGB image to grayscale using the formula:
/// Y = 0.299 * R + 0.587 * G + 0.114 * B
///
/// # Arguments
///
/// * `rgb` - Input RGB buffer (flat array: [R0, G0, B0, R1, G1, B1, ...])
/// * `gray` - Output grayscale buffer (flat array: [Y0, Y1, Y2, ...])
///
/// # Thread Model
///
/// - Each thread processes one pixel
/// - Thread reads 3 consecutive RGB values
/// - Thread writes 1 grayscale value
#[cube(launch_unchecked)]
pub fn rgb_to_gray_kernel<F: Float>(
    rgb: &Array<F>,
    gray: &mut Array<F>,
) {
    let pos = ABSOLUTE_POS;
    let gray_len = gray.len();
    
    if pos < gray_len {
        // Calculate RGB buffer index (pos * 3)
        let idx_base = pos * 3u32;

        let r = rgb[idx_base];
        let g = rgb[idx_base + 1u32];
        let b = rgb[idx_base + 2u32];

        // ITU-R BT.601 luma coefficients
        gray[pos] = F::new(0.299) * r + F::new(0.587) * g + F::new(0.114) * b;
    }
}

/// Grayscale to RGB GPU kernel.
///
/// Converts grayscale image to RGB by duplicating the gray value to all channels.
///
/// # Arguments
///
/// * `gray` - Input grayscale buffer (flat array: [Y0, Y1, Y2, ...])
/// * `rgb` - Output RGB buffer (flat array: [R0, G0, B0, R1, G1, B1, ...])
///
/// # Thread Model
///
/// - Each thread processes one pixel
/// - Thread reads 1 grayscale value
/// - Thread writes 3 RGB values (all equal to gray)
#[cube(launch_unchecked)]
pub fn gray_to_rgb_kernel<F: Float>(
    gray: &Array<F>,
    rgb: &mut Array<F>,
) {
    let pos = ABSOLUTE_POS;
    let gray_len = gray.len();
    
    if pos < gray_len {
        let gray_val = gray[pos];
        
        // Calculate RGB buffer index (pos * 3)
        let idx_base = pos * 3u32;
        
        // Duplicate gray value to R, G, B channels
        rgb[idx_base] = gray_val;
        rgb[idx_base + 1u32] = gray_val;
        rgb[idx_base + 2u32] = gray_val;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_kernels_compile() {
        // Just test that kernels compile
        // Actual execution tests are in integration tests
    }
}
