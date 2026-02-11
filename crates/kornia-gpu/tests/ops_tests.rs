//! Comprehensive tests for GPU tensor operations

use kornia_gpu::*;

#[allow(dead_code)]
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
    fn test_add_various_sizes() {
        let Some(runtime) = get_runtime() else { return };

        // Small
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let a_gpu = to_device(&a, vec![3], &runtime).unwrap();
        let b_gpu = to_device(&b, vec![3], &runtime).unwrap();
        let out_gpu = allocate::<_, f32>(vec![3], &runtime).unwrap();
        add_execute(&a_gpu, &b_gpu, &out_gpu, &runtime).unwrap();
        let result: Vec<f32> = to_cpu(&out_gpu, &runtime).unwrap();
        assert_eq!(result, vec![5.0, 7.0, 9.0]);

        // Medium (1024 elements)
        let size = 1024;
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();
        let expected: Vec<f32> = (0..size).map(|i| (i * 3) as f32).collect();
        
        let a_gpu = to_device(&a, vec![size], &runtime).unwrap();
        let b_gpu = to_device(&b, vec![size], &runtime).unwrap();
        let out_gpu = allocate::<_, f32>(vec![size], &runtime).unwrap();
        add_execute(&a_gpu, &b_gpu, &out_gpu, &runtime).unwrap();
        let result: Vec<f32> = to_cpu(&out_gpu, &runtime).unwrap();
        assert!(vec_approx_eq(&result, &expected));

        // Large (10000 elements)
        let size = 10000;
        let a: Vec<f32> = vec![1.5; size];
        let b: Vec<f32> = vec![2.5; size];
        
        let a_gpu = to_device(&a, vec![size], &runtime).unwrap();
        let b_gpu = to_device(&b, vec![size], &runtime).unwrap();
        let out_gpu = allocate::<_, f32>(vec![size], &runtime).unwrap();
        add_execute(&a_gpu, &b_gpu, &out_gpu, &runtime).unwrap();
        let result: Vec<f32> = to_cpu(&out_gpu, &runtime).unwrap();
        assert!(result.iter().all(|&x| approx_eq(x, 4.0)));
    }

    #[test]
    fn test_sub_execute() {
        let Some(runtime) = get_runtime() else { return };

        let a = vec![10.0, 20.0, 30.0, 40.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        
        let a_gpu = to_device(&a, vec![4], &runtime).unwrap();
        let b_gpu = to_device(&b, vec![4], &runtime).unwrap();
        let out_gpu = allocate::<_, f32>(vec![4], &runtime).unwrap();
        
        sub_execute(&a_gpu, &b_gpu, &out_gpu, &runtime).unwrap();
        
        let result: Vec<f32> = to_cpu(&out_gpu, &runtime).unwrap();
        assert_eq!(result, vec![9.0, 18.0, 27.0, 36.0]);
    }

    #[test]
    fn test_mul_execute() {
        let Some(runtime) = get_runtime() else { return };

        let a = vec![2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0];
        
        let a_gpu = to_device(&a, vec![3], &runtime).unwrap();
        let b_gpu = to_device(&b, vec![3], &runtime).unwrap();
        let out_gpu = allocate::<_, f32>(vec![3], &runtime).unwrap();
        
        mul_execute(&a_gpu, &b_gpu, &out_gpu, &runtime).unwrap();
        
        let result: Vec<f32> = to_cpu(&out_gpu, &runtime).unwrap();
        assert_eq!(result, vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_div_execute() {
        let Some(runtime) = get_runtime() else { return };

        let a = vec![10.0, 20.0, 30.0, 40.0];
        let b = vec![2.0, 4.0, 5.0, 8.0];
        
        let a_gpu = to_device(&a, vec![4], &runtime).unwrap();
        let b_gpu = to_device(&b, vec![4], &runtime).unwrap();
        let out_gpu = allocate::<_, f32>(vec![4], &runtime).unwrap();
        
        div_execute(&a_gpu, &b_gpu, &out_gpu, &runtime).unwrap();
        
        let result: Vec<f32> = to_cpu(&out_gpu, &runtime).unwrap();
        assert_eq!(result, vec![5.0, 5.0, 6.0, 5.0]);
    }

    #[test]
    fn test_mul_scalar_execute() {
        let Some(runtime) = get_runtime() else { return };

        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let scalar = 2.5f32;
        
        let a_gpu = to_device(&a, vec![5], &runtime).unwrap();
        let out_gpu = allocate::<_, f32>(vec![5], &runtime).unwrap();
        
        mul_scalar_execute(&a_gpu, scalar, &out_gpu, &runtime).unwrap();
        
        let result: Vec<f32> = to_cpu(&out_gpu, &runtime).unwrap();
        assert_eq!(result, vec![2.5, 5.0, 7.5, 10.0, 12.5]);
    }

    #[test]
    fn test_sum_execute_various_sizes() {
        let Some(runtime) = get_runtime() else { return };

        // Small
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let data_gpu = to_device(&data, vec![4], &runtime).unwrap();
        let sum = sum_execute::<_, f32>(&data_gpu, &runtime).unwrap();
        assert!(approx_eq(sum, 10.0));

        // Medium (256 elements exactly - single block)
        let data: Vec<f32> = (1..=256).map(|i| i as f32).collect();
        let expected_sum: f32 = (256.0 * 257.0) / 2.0; // Sum of 1 to 256
        let data_gpu = to_device(&data, vec![256], &runtime).unwrap();
        let sum = sum_execute::<_, f32>(&data_gpu, &runtime).unwrap();
        assert!(approx_eq(sum, expected_sum));

        // Large (1000 elements - multi-block)
        let data: Vec<f32> = vec![1.5; 1000];
        let expected_sum = 1500.0;
        let data_gpu = to_device(&data, vec![1000], &runtime).unwrap();
        let sum = sum_execute::<_, f32>(&data_gpu, &runtime).unwrap();
        assert!(approx_eq(sum, expected_sum));
    }

    #[test]
    fn test_mean_execute() {
        let Some(runtime) = get_runtime() else { return };

        let data = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let data_gpu = to_device(&data, vec![5], &runtime).unwrap();
        let mean = mean_execute::<_, f32>(&data_gpu, &runtime).unwrap();
        assert!(approx_eq(mean, 6.0));

        // Large array
        let data: Vec<f32> = vec![5.0; 10000];
        let data_gpu = to_device(&data, vec![10000], &runtime).unwrap();
        let mean = mean_execute::<_, f32>(&data_gpu, &runtime).unwrap();
        assert!(approx_eq(mean, 5.0));
    }

    #[test]
    fn test_min_execute() {
        let Some(runtime) = get_runtime() else { return };

        let data = vec![5.0, 2.0, 8.0, 1.0, 9.0, 3.0];
        let data_gpu = to_device(&data, vec![6], &runtime).unwrap();
        let min = min_execute::<_, f32>(&data_gpu, &runtime).unwrap();
        assert!(approx_eq(min, 1.0));

        // Large array with known min
        let mut data: Vec<f32> = vec![10.0; 5000];
        data[2500] = 0.5; // Minimum in the middle
        let data_gpu = to_device(&data, vec![5000], &runtime).unwrap();
        let min = min_execute::<_, f32>(&data_gpu, &runtime).unwrap();
        assert!(approx_eq(min, 0.5));
    }

    #[test]
    fn test_max_execute() {
        let Some(runtime) = get_runtime() else { return };

        let data = vec![5.0, 2.0, 8.0, 1.0, 9.0, 3.0];
        let data_gpu = to_device(&data, vec![6], &runtime).unwrap();
        let max = max_execute::<_, f32>(&data_gpu, &runtime).unwrap();
        assert!(approx_eq(max, 9.0));

        // Large array with known max
        let mut data: Vec<f32> = vec![1.0; 5000];
        data[3333] = 99.5; // Maximum  
        let data_gpu = to_device(&data, vec![5000], &runtime).unwrap();
        let max = max_execute::<_, f32>(&data_gpu, &runtime).unwrap();
        assert!(approx_eq(max, 99.5));
    }

    #[test]
    fn test_empty_tensors() {
        let Some(runtime) = get_runtime() else { return };

        // Empty add should succeed
        let empty: Vec<f32> = vec![];
        let a_gpu = to_device(&empty, vec![0], &runtime).unwrap();
        let b_gpu = to_device(&empty, vec![0], &runtime).unwrap();
        let out_gpu = allocate::<_, f32>(vec![0], &runtime).unwrap();
        assert!(add_execute(&a_gpu, &b_gpu, &out_gpu, &runtime).is_ok());

        // Empty sum should return 0
        let sum = sum_execute::<_, f32>(&a_gpu, &runtime).unwrap();
        assert!(approx_eq(sum, 0.0));

        // Empty min/max should error
        assert!(min_execute::<_, f32>(&a_gpu, &runtime).is_err());
        assert!(max_execute::<_, f32>(&a_gpu, &runtime).is_err());
    }

    #[test]
    fn test_shape_mismatch() {
        let Some(runtime) = get_runtime() else { return };

        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0]; // Different size
        
        let a_gpu = to_device(&a, vec![3], &runtime).unwrap();
        let b_gpu = to_device(&b, vec![2], &runtime).unwrap();
        let out_gpu = allocate::<_, f32>(vec![3], &runtime).unwrap();
        
        // Should error due to shape mismatch
        assert!(add_execute(&a_gpu, &b_gpu, &out_gpu, &runtime).is_err());
    }
}

#[cfg(feature = "wgpu")]
mod wgpu_tests {
    use super::*;

    fn get_runtime() -> Option<RuntimeContext<cubecl_wgpu::WgpuRuntime>> {
        init_wgpu_runtime().ok()
    }

    #[test]
    fn test_wgpu_basic_ops() {
        let Some(runtime) = get_runtime() else { return };

        // Test add
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let a_gpu = to_device(&a, vec![3], &runtime).unwrap();
        let b_gpu = to_device(&b, vec![3], &runtime).unwrap();
        let out_gpu = allocate::<_, f32>(vec![3], &runtime).unwrap();
        add_execute(&a_gpu, &b_gpu, &out_gpu, &runtime).unwrap();
        let result: Vec<f32> = to_cpu(&out_gpu, &runtime).unwrap();
        assert_eq!(result, vec![5.0, 7.0, 9.0]);

        // Test reductions
        let data = vec![2.0, 4.0, 6.0, 8.0];
        let data_gpu = to_device(&data, vec![4], &runtime).unwrap();
        
        let sum = sum_execute::<_, f32>(&data_gpu, &runtime).unwrap();
        assert!(approx_eq(sum, 20.0));
        
        let mean = mean_execute::<_, f32>(&data_gpu, &runtime).unwrap();
        assert!(approx_eq(mean, 5.0));
    }
}
