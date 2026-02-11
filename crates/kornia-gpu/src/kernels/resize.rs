//! GPU resize kernel using bilinear interpolation.
//!
//! Layout: HWC (height, width, channels) — matching kornia-imgproc convention.

#![allow(missing_docs)]

use cubecl::prelude::*;

/// Bilinear interpolation resize kernel.
///
/// Each thread computes one output pixel (all channels) by sampling 4 neighbors
/// from the source image.
///
/// # Arguments
///
/// * `src` - Source image data, flattened HWC: `[src_h * src_w * channels]`
/// * `dst` - Destination image data, flattened HWC: `[dst_h * dst_w * channels]`
/// * `src_h` - Source image height
/// * `src_w` - Source image width
/// * `dst_h` - Destination image height
/// * `dst_w` - Destination image width
/// * `channels` - Number of channels (1, 3, etc.)
#[cube(launch_unchecked)]
pub fn resize_bilinear_kernel<F: Float>(
    src: &Array<F>,
    dst: &mut Array<F>,
    src_h: u32,
    src_w: u32,
    dst_h: u32,
    dst_w: u32,
    channels: u32,
) {
    let pos = ABSOLUTE_POS;
    let num_pixels = dst_h * dst_w;

    if pos >= num_pixels {
        return;
    }

    // Output pixel coordinates
    let dst_y = pos / dst_w;
    let dst_x = pos % dst_w;

    // Map destination pixel to source coordinates
    // Use (src_dim - 1) / (dst_dim - 1) scaling to align corners
    let scale_x = if dst_w > 1 {
        F::cast_from(src_w - 1) / F::cast_from(dst_w - 1)
    } else {
        F::new(0.0)
    };
    let scale_y = if dst_h > 1 {
        F::cast_from(src_h - 1) / F::cast_from(dst_h - 1)
    } else {
        F::new(0.0)
    };

    let src_x_f = F::cast_from(dst_x) * scale_x;
    let src_y_f = F::cast_from(dst_y) * scale_y;

    // Integer coordinates of the 4 neighbors
    let x0 = u32::cast_from(F::floor(src_x_f));
    let y0 = u32::cast_from(F::floor(src_y_f));
    let x1 = if x0 + 1 < src_w { x0 + 1 } else { x0 };
    let y1 = if y0 + 1 < src_h { y0 + 1 } else { y0 };

    // Fractional part for blending
    let fx = src_x_f - F::cast_from(x0);
    let fy = src_y_f - F::cast_from(y0);
    let one = F::new(1.0);

    // Bilinear weights
    let w00 = (one - fx) * (one - fy);
    let w01 = fx * (one - fy);
    let w10 = (one - fx) * fy;
    let w11 = fx * fy;

    // Source pixel base indices (HWC layout)
    let s00 = (y0 * src_w + x0) * channels;
    let s01 = (y0 * src_w + x1) * channels;
    let s10 = (y1 * src_w + x0) * channels;
    let s11 = (y1 * src_w + x1) * channels;

    // Destination pixel base index
    let d = (dst_y * dst_w + dst_x) * channels;

    // Interpolate each channel
    let mut c: u32 = 0;
    while c < channels {
        let val = w00 * src[s00 + c]
            + w01 * src[s01 + c]
            + w10 * src[s10 + c]
            + w11 * src[s11 + c];
        dst[d + c] = val;
        c += 1;
    }
}

/// Nearest-neighbor resize kernel.
///
/// Each thread computes one output pixel by copying the nearest source pixel.
#[cube(launch_unchecked)]
pub fn resize_nearest_kernel<F: Float>(
    src: &Array<F>,
    dst: &mut Array<F>,
    src_h: u32,
    src_w: u32,
    dst_h: u32,
    dst_w: u32,
    channels: u32,
) {
    let pos = ABSOLUTE_POS;
    let num_pixels = dst_h * dst_w;

    if pos >= num_pixels {
        return;
    }

    let dst_y = pos / dst_w;
    let dst_x = pos % dst_w;

    let scale_x = if dst_w > 1 {
        F::cast_from(src_w - 1) / F::cast_from(dst_w - 1)
    } else {
        F::new(0.0)
    };
    let scale_y = if dst_h > 1 {
        F::cast_from(src_h - 1) / F::cast_from(dst_h - 1)
    } else {
        F::new(0.0)
    };

    let src_x = u32::cast_from(F::round(F::cast_from(dst_x) * scale_x));
    let src_y = u32::cast_from(F::round(F::cast_from(dst_y) * scale_y));

    let s = (src_y * src_w + src_x) * channels;
    let d = (dst_y * dst_w + dst_x) * channels;

    let mut c: u32 = 0;
    while c < channels {
        dst[d + c] = src[s + c];
        c += 1;
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_resize_kernels_compile() {
        // Verify kernels compile with CubeCL
    }
}
