//! Tests for GPU color conversion operations

use kornia_gpu::*;

const EPSILON: f32 = 1e-5;

#[allow(dead_code)]
fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < EPSILON
}

#[allow(dead_code)]
fn vec_approx_eq(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| approx_eq(*x, *y))
}

#[cfg(feature = "cuda")]
mod cuda_tests {
    use super::*;

    fn get_runtime() -> Option<RuntimeContext<cubecl_cuda::CudaRuntime>> {
        init_cuda_runtime().ok()
    }

    #[test]
    fn test_rgb_to_gray_basic() {
        let Some(runtime) = get_runtime() else { return };

        // Create a simple RGB image: 2x2 pixels
        // Red pixel, Green pixel, Blue pixel, White pixel
        #[rustfmt::skip]
        let rgb_data = vec![
            1.0, 0.0, 0.0,  // Red (0,0)
            0.0, 1.0, 0.0,  // Green (0,1)
            0.0, 0.0, 1.0,  // Blue (1,0)
            1.0, 1.0, 1.0,  // White (1,1)
        ];

        let num_pixels = 4;
        let rgb_gpu = to_device(&rgb_data, vec![num_pixels * 3], &runtime).unwrap();
        let gray_gpu = allocate::<_, f32>(vec![num_pixels], &runtime).unwrap();

        rgb_to_gray_execute::<_, f32>(&rgb_gpu, &gray_gpu, &runtime).unwrap();

        let result: Vec<f32> = to_cpu(&gray_gpu, &runtime).unwrap();

        // Expected values using: Y = 0.299*R + 0.587*G + 0.114*B
        let expected = vec![
            0.299,  // Red
            0.587,  // Green
            0.114,  // Blue
            1.0,    // White
        ];

        assert!(vec_approx_eq(&result, &expected), "RGB to Gray conversion incorrect");
    }

    #[test]
    fn test_gray_to_rgb_basic() {
        let Some(runtime) = get_runtime() else { return };

        // Grayscale values
        let gray_data = vec![0.0, 0.5, 1.0, 0.25];
        let num_pixels = 4;

        let gray_gpu = to_device(&gray_data, vec![num_pixels], &runtime).unwrap();
        let rgb_gpu = allocate::<_, f32>(vec![num_pixels * 3], &runtime).unwrap();

        gray_to_rgb_execute::<_, f32>(&gray_gpu, &rgb_gpu, &runtime).unwrap();

        let result: Vec<f32> = to_cpu(&rgb_gpu, &runtime).unwrap();

        // Each gray value should be duplicated to R, G, B
        #[rustfmt::skip]
        let expected = vec![
            0.0, 0.0, 0.0,      // Black
            0.5, 0.5, 0.5,      // Mid-gray
            1.0, 1.0, 1.0,      // White
            0.25, 0.25, 0.25,   // Dark gray
        ];

        assert!(vec_approx_eq(&result, &expected), "Gray to RGB conversion incorrect");
    }

    #[test]
    fn test_rgb_to_gray_large() {
        let Some(runtime) = get_runtime() else { return };

        // Large image: 256x256 = 65536 pixels
        let num_pixels = 256 * 256;
        let mut rgb_data = Vec::with_capacity(num_pixels * 3);
        
        // Create gradient pattern
        for i in 0..num_pixels {
            let val = (i % 256) as f32 / 255.0;
            rgb_data.push(val);      // R
            rgb_data.push(val * 0.5); // G
            rgb_data.push(val * 0.25); // B
        }

        let rgb_gpu = to_device(&rgb_data, vec![num_pixels * 3], &runtime).unwrap();
        let gray_gpu = allocate::<_, f32>(vec![num_pixels], &runtime).unwrap();

        rgb_to_gray_execute::<_, f32>(&rgb_gpu, &gray_gpu, &runtime).unwrap();

        let result: Vec<f32> = to_cpu(&gray_gpu, &runtime).unwrap();

        // Compute expected on CPU
        let expected: Vec<f32> = (0..num_pixels)
            .map(|i| {
                let r = rgb_data[i * 3];
                let g = rgb_data[i * 3 + 1];
                let b = rgb_data[i * 3 + 2];
                0.299 * r + 0.587 * g + 0.114 * b
            })
            .collect();

        assert!(vec_approx_eq(&result, &expected), "Large RGB to Gray conversion incorrect");
    }

    #[test]
    fn test_roundtrip_gray_rgb_gray() {
        let Some(runtime) = get_runtime() else { return };

        // Original grayscale image
        let gray_original = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        let num_pixels = 8;

        let gray1_gpu = to_device(&gray_original, vec![num_pixels], &runtime).unwrap();
        let rgb_gpu = allocate::<_, f32>(vec![num_pixels * 3], &runtime).unwrap();
        let gray2_gpu = allocate::<_, f32>(vec![num_pixels], &runtime).unwrap();

        // Gray -> RGB
        gray_to_rgb_execute::<_, f32>(&gray1_gpu, &rgb_gpu, &runtime).unwrap();

        // RGB -> Gray
        rgb_to_gray_execute::<_, f32>(&rgb_gpu, &gray2_gpu, &runtime).unwrap();

        let result: Vec<f32> = to_cpu(&gray2_gpu, &runtime).unwrap();

        // Should be identical to original (within epsilon)
        assert!(vec_approx_eq(&result, &gray_original), "Roundtrip conversion failed");
    }

    #[test]
    fn test_empty_buffers() {
        let Some(runtime) = get_runtime() else { return };

        let empty_rgb: Vec<f32> = vec![];
        let empty_gray: Vec<f32> = vec![];

        let rgb_gpu = to_device(&empty_rgb, vec![0], &runtime).unwrap();
        let gray_gpu = to_device(&empty_gray, vec![0], &runtime).unwrap();

        // Should not crash
        assert!(rgb_to_gray_execute::<_, f32>(&rgb_gpu, &gray_gpu, &runtime).is_ok());
        assert!(gray_to_rgb_execute::<_, f32>(&gray_gpu, &rgb_gpu, &runtime).is_ok());
    }

    #[test]
    fn test_buffer_size_mismatch() {
        let Some(runtime) = get_runtime() else { return };

        let rgb_data = vec![1.0; 12]; // 4 pixels
        let gray_data = vec![0.0; 5];  // 5 pixels - mismatch!

        let rgb_gpu = to_device(&rgb_data, vec![12], &runtime).unwrap();
        let gray_gpu = to_device(&gray_data, vec![5], &runtime).unwrap();

        // Should error due to size mismatch
        assert!(rgb_to_gray_execute::<_, f32>(&rgb_gpu, &gray_gpu, &runtime).is_err());
    }

    #[test]
    fn test_known_values() {
        let Some(runtime) = get_runtime() else { return };

        // Test with known color values
        #[rustfmt::skip]
        let rgb_data = vec![
            0.5, 0.5, 0.5,    // Mid-gray from gray input
            1.0, 0.0, 0.0,    // Pure red
            0.0, 1.0, 0.0,    // Pure green
            0.0, 0.0, 1.0,    // Pure blue
            0.5, 0.25, 0.125, // Custom color
        ];

        let num_pixels = 5;
        let rgb_gpu = to_device(&rgb_data, vec![num_pixels * 3], &runtime).unwrap();
        let gray_gpu = allocate::<_, f32>(vec![num_pixels], &runtime).unwrap();

        rgb_to_gray_execute::<_, f32>(&rgb_gpu, &gray_gpu, &runtime).unwrap();

        let result: Vec<f32> = to_cpu(&gray_gpu, &runtime).unwrap();

        // Manually calculated expected values
        let expected = vec![
            0.5,                                    // Mid-gray
            0.299,                                  // Red
            0.587,                                  // Green
            0.114,                                  // Blue
            0.299 * 0.5 + 0.587 * 0.25 + 0.114 * 0.125, // Custom
        ];

        assert!(vec_approx_eq(&result, &expected), "Known values test failed");
    }
}

#[cfg(feature = "wgpu")]
mod wgpu_tests {
    use super::*;

    fn get_runtime() -> Option<RuntimeContext<cubecl_wgpu::WgpuRuntime>> {
        init_wgpu_runtime().ok()
    }

    #[test]
    fn test_wgpu_color_ops() {
        let Some(runtime) = get_runtime() else { return };

        #[rustfmt::skip]
        let rgb_data = vec![
            1.0, 0.0, 0.0,  // Red
            0.0, 1.0, 0.0,  // Green
            0.0, 0.0, 1.0,  // Blue
        ];

        let num_pixels = 3;
        let rgb_gpu = to_device(&rgb_data, vec![num_pixels * 3], &runtime).unwrap();
        let gray_gpu = allocate::<_, f32>(vec![num_pixels], &runtime).unwrap();

        rgb_to_gray_execute::<_, f32>(&rgb_gpu, &gray_gpu, &runtime).unwrap();

        let result: Vec<f32> = to_cpu(&gray_gpu, &runtime).unwrap();

        let expected = vec![0.299, 0.587, 0.114];
        assert!(vec_approx_eq(&result, &expected));
    }
}
