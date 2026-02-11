// Quick debug test for GPU color conversions
use kornia_gpu::*;

#[test]
fn debug_gpu_color() {
    let runtime = init_wgpu_runtime().expect("WGPU init failed");
    
    // Simple RGB colors
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
    
    println!("RGB Input: {:?}", rgb_data);
    println!("Gray Output: {:?}", result);
    println!("Expected: [0.299, 0.587, 0.114]");
    
    // Check each value
    for (i, (&r, &e)) in result.iter().zip(&[0.299f32, 0.587, 0.114]).enumerate() {
        let diff = (r - e).abs();
        println!("Pixel {}: got {:.6}, expected {:.6}, diff {:.6}", i, r, e, diff);
        assert!(diff < 1e-4, "Pixel {} failed: {} vs {}", i, r, e);
    }
}
