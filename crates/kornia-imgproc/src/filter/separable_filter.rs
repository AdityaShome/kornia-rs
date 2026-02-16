use crate::parallel::ExecutionStrategy;
use kornia_image::{allocator::ImageAllocator, Image, ImageError};
use num_traits::Zero;
use rayon::prelude::*;

/// Trait for floating point casting
pub trait FloatConversion {
    /// Convert the type to f32
    fn to_f32(&self) -> f32;
    /// Convert the type from f32
    fn from_f32(val: f32) -> Self;
}

impl FloatConversion for f32 {
    fn to_f32(&self) -> f32 {
        *self
    }

    fn from_f32(val: f32) -> Self {
        val
    }
}

impl FloatConversion for f64 {
    fn to_f32(&self) -> f32 {
        *self as f32
    }

    fn from_f32(val: f32) -> Self {
        val as f64
    }
}

impl FloatConversion for u8 {
    fn to_f32(&self) -> f32 {
        *self as f32
    }

    fn from_f32(val: f32) -> Self {
        val.clamp(0.0, 255.0) as u8
    }
}

/// A separable 2D filter that applies horizontal and vertical 1D convolutions sequentially.
///
/// This struct caches the kernel data and precomputed offsets for efficient filtering.
struct SeparableFilter {
    kernel_x: Vec<f32>,
    kernel_y: Vec<f32>,
    offsets_x: Vec<isize>,
}



impl SeparableFilter {
    /// Create a new separable filter with the given kernels.
    ///
    /// # Arguments
    ///
    /// * `kernel_x` - The horizontal convolution kernel
    /// * `kernel_y` - The vertical convolution kernel
    fn new(kernel_x: &[f32], kernel_y: &[f32]) -> Self {
        let half_x = kernel_x.len() / 2;


        let offsets_x = (0..kernel_x.len())
            .map(|i| i as isize - half_x as isize)
            .collect();

        Self {
            kernel_x: kernel_x.to_vec(),
            kernel_y: kernel_y.to_vec(),
            offsets_x,
        }
    }



    /// Apply the filter to an image.
    ///
    /// Performs horizontal filtering followed by vertical filtering using a temporary buffer.
    ///
    /// # Arguments
    ///
    /// * `src` - The source image
    /// * `dst` - The destination image (must be same size as source)
    /// * `strategy` - The execution strategy to use
    fn apply<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
        &self,
        src: &Image<T, C, A1>,
        dst: &mut Image<T, C, A2>,
        strategy: ExecutionStrategy,
    ) -> Result<(), ImageError>
    where
        T: FloatConversion + Clone + Zero + Send + Sync,
    {
        if src.size() != dst.size() {
            return Err(ImageError::InvalidImageSize(
                src.cols(),
                src.rows(),
                dst.cols(),
                dst.rows(),
            ));
        }

        if src.cols() == 0 || src.rows() == 0 {
            return Ok(());
        }

        match strategy {
            ExecutionStrategy::Serial => {
                self.apply_pipeline(src, dst)
            }
            ExecutionStrategy::Fixed(n) => {
                if n == 0 {
                    return Err(ImageError::Parallel("thread count must be > 0".to_string()));
                }
                let rows = src.rows();
                let chunk_height = rows.div_ceil(n);
                 // Is div_ceil stable? It is in Rust 1.73+.
                 // If not, (rows + n - 1) / n
                 // Let's use the manual calculation to be safe or check version.
                 // safe way: (rows + n - 1) / n
                 self.apply_pipeline_parallel(src, dst, chunk_height)
            }

            ExecutionStrategy::AutoRows(stride) => {
                if stride == 0 {
                    return Err(ImageError::Parallel("row stride must be > 0".to_string()));
                }
                self.apply_pipeline_parallel(src, dst, stride)
            }

            ExecutionStrategy::ParallelElements => {
                 self.apply_pipeline_parallel(src, dst, 32)
            }
        }
    }

    /// 3×3 horizontal fast path (unrolled, FMA-optimized)
    fn horizontal_row_3x3<T: FloatConversion, const C: usize>(
        src_data: &[T],
        out_row: &mut [f32],
        scratch: &mut [f32],
        row: usize,
        cols: usize,
        k: &[f32; 3],
    ) {
        let row_len = cols * C;
        let row_offset = row * row_len;
        
        for (i, val) in src_data[row_offset..row_offset + row_len].iter().enumerate() {
            scratch[i] = val.to_f32();
        }
        
        for e in 0..(C.min(row_len)) {
            let c = e / C;
            let ch = e % C;
            let mut acc = 0.0f32;
            for (i, &k_val) in k.iter().enumerate() {
                let x = c as isize + (i as isize - 1);
                if x >= 0 && x < cols as isize {
                    acc += scratch[x as usize * C + ch] * k_val;
                }
            }
            out_row[e] = acc;
        }
        
        let h_safe_start = C;
        let h_safe_end = row_len.saturating_sub(C);
        
        if h_safe_start < h_safe_end {
            use wide::f32x8;
            let k0 = f32x8::splat(k[0]);
            let k1 = f32x8::splat(k[1]);
            let k2 = f32x8::splat(k[2]);
            
            let mut e = h_safe_start;
            while e + 8 <= h_safe_end {
                let s0: [f32; 8] = scratch[e - C..e - C + 8].try_into().unwrap();
                let s1: [f32; 8] = scratch[e..e + 8].try_into().unwrap();
                let s2: [f32; 8] = scratch[e + C..e + C + 8].try_into().unwrap();
                let acc = f32x8::new(s0) * k0 + f32x8::new(s1) * k1 + f32x8::new(s2) * k2;
                out_row[e..e + 8].copy_from_slice(&<[f32; 8]>::from(acc));
                e += 8;
            }
            while e < h_safe_end {
                out_row[e] = scratch[e - C] * k[0] + scratch[e] * k[1] + scratch[e + C] * k[2];
                e += 1;
            }
        }
        
        for e in h_safe_end..row_len {
            let c = e / C;
            let ch = e % C;
            let mut acc = 0.0f32;
            for (i, &k_val) in k.iter().enumerate() {
                let x = c as isize + (i as isize - 1);
                if x >= 0 && x < cols as isize {
                    acc += scratch[x as usize * C + ch] * k_val;
                }
            }
            out_row[e] = acc;
        }
    }

    /// 3×3 vertical fast path (unrolled, FMA-optimized)
    fn vertical_row_3x3(
        ring_buffer: &[Vec<f32>],
        out_row: &mut [f32],
        out_r: usize,
        total_rows: usize,
        row_len: usize,
        k: &[f32; 3],
    ) {
        let mut e = 0;
        let use_r0 = out_r >= 1;
        let use_r2 = out_r + 1 < total_rows;

        {
            use wide::f32x8;
            let k0 = f32x8::splat(k[0]);
            let k1 = f32x8::splat(k[1]);
            let k2 = f32x8::splat(k[2]);

            while e + 8 <= row_len {
                let s1: [f32; 8] = ring_buffer[out_r % 3][e..e + 8].try_into().unwrap();
                let mut acc = f32x8::new(s1) * k1;

                if use_r0 {
                    let s0: [f32; 8] = ring_buffer[(out_r - 1) % 3][e..e + 8].try_into().unwrap();
                    acc = k0.mul_add(f32x8::new(s0), acc);
                }
                if use_r2 {
                    let s2: [f32; 8] = ring_buffer[(out_r + 1) % 3][e..e + 8].try_into().unwrap();
                    acc = k2.mul_add(f32x8::new(s2), acc);
                }
                out_row[e..e + 8].copy_from_slice(&<[f32; 8]>::from(acc));
                e += 8;
            }
        }

        while e < row_len {
            let mut acc = ring_buffer[out_r % 3][e] * k[1];
            if use_r0 {
                acc += ring_buffer[(out_r - 1) % 3][e] * k[0];
            }
            if use_r2 {
                acc += ring_buffer[(out_r + 1) % 3][e] * k[2];
            }
            out_row[e] = acc;
            e += 1;
        }
    }

    /// 5×5 horizontal fast path  (unrolled, FMA-optimized)
    fn horizontal_row_5x5<T: FloatConversion, const C: usize>(
        src_data: &[T],
        out_row: &mut [f32],
        scratch: &mut [f32],
        row: usize,
        cols: usize,
        k: &[f32; 5],
    ) {
        let row_len = cols * C;
        let row_offset = row * row_len;
        
        for (i, val) in src_data[row_offset..row_offset + row_len].iter().enumerate() {
            scratch[i] = val.to_f32();
        }
        
        for e in 0..((2 * C).min(row_len)) {
            let c = e / C;
            let ch = e % C;
            let mut acc = 0.0f32;
            for (i, &k_val) in k.iter().enumerate() {
                let x = c as isize + (i as isize - 2);
                if x >= 0 && x < cols as isize {
                    acc += scratch[x as usize * C + ch] * k_val;
                }
            }
            out_row[e] = acc;
        }
        
        let h_safe_start = 2 * C;
        let h_safe_end = row_len.saturating_sub(2 * C);
        
        if h_safe_start < h_safe_end {
            use wide::f32x8;
            let k0 = f32x8::splat(k[0]);
            let k1 = f32x8::splat(k[1]);
            let k2 = f32x8::splat(k[2]);
            let k3 = f32x8::splat(k[3]);
            let k4 = f32x8::splat(k[4]);
            
            let mut e = h_safe_start;
            while e + 8 <= h_safe_end {
                let s0: [f32; 8] = scratch[e - 2 * C..e - 2 * C + 8].try_into().unwrap();
                let s1: [f32; 8] = scratch[e - C..e - C + 8].try_into().unwrap();
                let s2: [f32; 8] = scratch[e..e + 8].try_into().unwrap();
                let s3: [f32; 8] = scratch[e + C..e + C + 8].try_into().unwrap();
                let s4: [f32; 8] = scratch[e + 2 * C..e + 2 * C + 8].try_into().unwrap();
                let acc = f32x8::new(s0) * k0 + f32x8::new(s1) * k1 + f32x8::new(s2) * k2 + f32x8::new(s3) * k3 + f32x8::new(s4) * k4;
                out_row[e..e + 8].copy_from_slice(&<[f32; 8]>::from(acc));
                e += 8;
            }
            while e < h_safe_end {
                out_row[e] = scratch[e - 2 * C] * k[0] + scratch[e - C] * k[1] + scratch[e] * k[2] + scratch[e + C] * k[3] + scratch[e + 2 * C] * k[4];
                e += 1;
            }
        }
        
        for e in h_safe_end..row_len {
            let c = e / C;
            let ch = e % C;
            let mut acc = 0.0f32;
            for (i, &k_val) in k.iter().enumerate() {
                let x = c as isize + (i as isize - 2);
                if x >= 0 && x < cols as isize {
                    acc += scratch[x as usize * C + ch] * k_val;
                }
            }
            out_row[e] = acc;
        }
    }

    /// 5×5 vertical fast path (unrolled, FMA-optimized)
    fn vertical_row_5x5(
        ring_buffer: &[Vec<f32>],
        out_row: &mut [f32],
        out_r: usize,
        total_rows: usize,
        row_len: usize,
        k: &[f32; 5],
    ) {
        let mut e = 0;
        let use_r0 = out_r >= 2;
        let use_r1 = out_r >= 1;
        let use_r3 = out_r + 1 < total_rows;
        let use_r4 = out_r + 2 < total_rows;

        {
            use wide::f32x8;
            let k0 = f32x8::splat(k[0]);
            let k1 = f32x8::splat(k[1]);
            let k2 = f32x8::splat(k[2]);
            let k3 = f32x8::splat(k[3]);
            let k4 = f32x8::splat(k[4]);

            while e + 8 <= row_len {
                let s2: [f32; 8] = ring_buffer[out_r % 5][e..e + 8].try_into().unwrap();
                let mut acc = f32x8::new(s2) * k2;

                if use_r0 {
                    let s0: [f32; 8] = ring_buffer[(out_r - 2) % 5][e..e + 8].try_into().unwrap();
                    acc = k0.mul_add(f32x8::new(s0), acc);
                }
                if use_r1 {
                    let s1: [f32; 8] = ring_buffer[(out_r - 1) % 5][e..e + 8].try_into().unwrap();
                    acc = k1.mul_add(f32x8::new(s1), acc);
                }
                if use_r3 {
                    let s3: [f32; 8] = ring_buffer[(out_r + 1) % 5][e..e + 8].try_into().unwrap();
                    acc = k3.mul_add(f32x8::new(s3), acc);
                }
                if use_r4 {
                    let s4: [f32; 8] = ring_buffer[(out_r + 2) % 5][e..e + 8].try_into().unwrap();
                    acc = k4.mul_add(f32x8::new(s4), acc);
                }
                out_row[e..e + 8].copy_from_slice(&<[f32; 8]>::from(acc));
                e += 8;
            }
        }

        while e < row_len {
            let mut acc = ring_buffer[out_r % 5][e] * k[2];
            if use_r0 {
                acc += ring_buffer[(out_r - 2) % 5][e] * k[0];
            }
            if use_r1 {
                acc += ring_buffer[(out_r - 1) % 5][e] * k[1];
            }
            if use_r3 {
                acc += ring_buffer[(out_r + 1) % 5][e] * k[3];
            }
            if use_r4 {
                acc += ring_buffer[(out_r + 2) % 5][e] * k[4];
            }
            out_row[e] = acc;
            e += 1;
        }
    }

    /// Ring-buffer horizontal pass with SIMD
    fn horizontal_row_simd<T: FloatConversion, const C: usize>(
        &self,
        src_data: &[T],
        out_row: &mut [f32],
        scratch: &mut [f32],
        row: usize,
        cols: usize,
    ) {
        use wide::f32x8;
        let row_len = cols * C;
        let row_offset = row * row_len;
        let half_x = self.kernel_x.len() / 2;
        let h_safe_start = half_x * C;
        let h_safe_end = row_len.saturating_sub(half_x * C);

        for (i, val) in src_data[row_offset..row_offset + row_len].iter().enumerate() {
            scratch[i] = val.to_f32();
        }

        for e in 0..h_safe_start.min(row_len) {
            let c = e / C;
            let ch = e % C;
            let mut acc = 0.0f32;
            for (&k, &off) in self.kernel_x.iter().zip(self.offsets_x.iter()) {
                let x = c as isize + off;
                if x >= 0 && x < cols as isize {
                    acc += scratch[x as usize * C + ch] * k;
                }
            }
            out_row[e] = acc;
        }

        if h_safe_start < h_safe_end {
            for v in out_row[h_safe_start..h_safe_end].iter_mut() {
                *v = 0.0;
            }
            
            {
                for (&k, &off) in self.kernel_x.iter().zip(self.offsets_x.iter()) {
                    let elem_off = off * C as isize;
                    let k_vec = f32x8::splat(k);
                    let mut e = h_safe_start;
                    while e + 8 <= h_safe_end {
                        let si = (e as isize + elem_off) as usize;
                        let src_arr: [f32; 8] = scratch[si..si + 8].try_into().unwrap();
                        let acc_arr: [f32; 8] = out_row[e..e + 8].try_into().unwrap();
                        let result = f32x8::new(acc_arr) + f32x8::new(src_arr) * k_vec;
                        out_row[e..e + 8].copy_from_slice(&<[f32; 8]>::from(result));
                        e += 8;
                    }
                    while e < h_safe_end {
                        out_row[e] += scratch[(e as isize + elem_off) as usize] * k;
                        e += 1;
                    }
                }
            }
        }

        for e in h_safe_end..row_len {
            let c = e / C;
            let ch = e % C;
            let mut acc = 0.0f32;
            for (&k, &off) in self.kernel_x.iter().zip(self.offsets_x.iter()) {
                let x = c as isize + off;
                if x >= 0 && x < cols as isize {
                    acc += scratch[x as usize * C + ch] * k;
                }
            }
            out_row[e] = acc;
        }
    }

    /// Ring-buffer vertical pass with SIMD
    fn vertical_row_simd(
        ring_buffer: &[Vec<f32>],
        out_row: &mut [f32],
        out_r: usize,
        total_rows: usize,
        row_len: usize,
        kernel_y: &[f32],
    ) {
        use wide::f32x8;
        let ky_size = kernel_y.len();
        let ky_half = ky_size / 2;
        let mut e = 0;

        let k_start = if out_r >= ky_half { 0 } else { ky_half - out_r };
        let k_end_arg = (total_rows + ky_half).saturating_sub(out_r);
        let k_end = ky_size.min(k_end_arg);

        {
            while e + 8 <= row_len {
                let mut acc = f32x8::splat(0.0);
                for k_idx in k_start..k_end {
                    let k_val = unsafe { *kernel_y.get_unchecked(k_idx) };
                    // We know src_r is valid by construction of k_start/k_end
                    let src_r = (out_r + k_idx) - ky_half;
                    let buf_idx = src_r % ky_size;
                    let arr: [f32; 8] = ring_buffer[buf_idx][e..e + 8].try_into().unwrap();
                    acc = f32x8::splat(k_val).mul_add(f32x8::new(arr), acc);
                }
                out_row[e..e + 8].copy_from_slice(&<[f32; 8]>::from(acc));
                e += 8;
            }
        }

        while e < row_len {
            let mut acc = 0.0f32;
            for k_idx in k_start..k_end {
                let k_val = unsafe { *kernel_y.get_unchecked(k_idx) };
                let src_r = (out_r + k_idx) - ky_half;
                let buf_idx = src_r % ky_size;
                acc += ring_buffer[buf_idx][e] * k_val;
            }
            out_row[e] = acc;
            e += 1;
        }
    }

    /// Ring-buffer pipeline (serial)
    fn apply_pipeline<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
        &self,
        src: &Image<T, C, A1>,
        dst: &mut Image<T, C, A2>,
    ) -> Result<(), ImageError>
    where
        T: FloatConversion + Clone + Zero,
    {
        let rows = src.rows();
        let cols = src.cols();
        let row_len = cols * C;
        let src_data = src.as_slice();
        let dst_data = dst.as_slice_mut();

        if rows == 0 || cols == 0 {
            return Ok(());
        }

        let ky_size = self.kernel_y.len();
        let ky_half = ky_size / 2;
        let mut ring = vec![vec![0.0f32; row_len]; ky_size];
        let mut scratch = vec![0.0f32; row_len];
        let mut vert_buf = vec![0.0f32; row_len];

        for r in 0..ky_half.min(rows) {
            let idx = r % ky_size;
            if ky_size == 3 && self.kernel_x.len() == 3 {
                let kx: [f32; 3] = [self.kernel_x[0], self.kernel_x[1], self.kernel_x[2]];
                Self::horizontal_row_3x3::<T, C>(src_data, &mut ring[idx], &mut scratch, r, cols, &kx);
            } else if ky_size == 5 && self.kernel_x.len() == 5 {
                let kx: [f32; 5] = [self.kernel_x[0], self.kernel_x[1], self.kernel_x[2], self.kernel_x[3], self.kernel_x[4]];
                Self::horizontal_row_5x5::<T, C>(src_data, &mut ring[idx], &mut scratch, r, cols, &kx);
            } else {
                self.horizontal_row_simd::<T, C>(src_data, &mut ring[idx], &mut scratch, r, cols);
            }
        }

        for r in 0..rows {
            let read_r = r + ky_half;
            if read_r < rows {
                let idx = read_r % ky_size;
                if ky_size == 3 && self.kernel_x.len() == 3 {
                    let kx: [f32; 3] = [self.kernel_x[0], self.kernel_x[1], self.kernel_x[2]];
                    Self::horizontal_row_3x3::<T, C>(src_data, &mut ring[idx], &mut scratch, read_r, cols, &kx);
                } else if ky_size == 5 && self.kernel_x.len() == 5 {
                    let kx: [f32; 5] = [self.kernel_x[0], self.kernel_x[1], self.kernel_x[2], self.kernel_x[3], self.kernel_x[4]];
                    Self::horizontal_row_5x5::<T, C>(src_data, &mut ring[idx], &mut scratch, read_r, cols, &kx);
                } else {
                    self.horizontal_row_simd::<T, C>(src_data, &mut ring[idx], &mut scratch, read_r, cols);
                }
            }

            if ky_size == 3 {
                let ky: [f32; 3] = [self.kernel_y[0], self.kernel_y[1], self.kernel_y[2]];
                Self::vertical_row_3x3(&ring, &mut vert_buf, r, rows, row_len, &ky);
            } else if ky_size == 5 {
                let ky: [f32; 5] = [self.kernel_y[0], self.kernel_y[1], self.kernel_y[2], self.kernel_y[3], self.kernel_y[4]];
                Self::vertical_row_5x5(&ring, &mut vert_buf, r, rows, row_len, &ky);
            } else {
                Self::vertical_row_simd(&ring, &mut vert_buf, r, rows, row_len, &self.kernel_y);
            }

            let off = r * row_len;
            for (e, &v) in vert_buf.iter().enumerate() {
                dst_data[off + e] = T::from_f32(v);
            }
        }

        Ok(())
    }

    /// Ring-buffer pipeline (parallel)
    fn apply_pipeline_parallel<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
        &self,
        src: &Image<T, C, A1>,
        dst: &mut Image<T, C, A2>,
        chunk_height: usize,
    ) -> Result<(), ImageError>
    where
        T: FloatConversion + Clone + Zero + Send + Sync,
    {
        let rows = src.rows();
        let cols = src.cols();
        let row_len = cols * C;
        let src_data = src.as_slice();
        let dst_data = dst.as_slice_mut();

        if rows == 0 || cols == 0 {
            return Ok(());
        }

        let ky_size = self.kernel_y.len();
        let ky_half = ky_size / 2;

        dst_data.par_chunks_mut(chunk_height * row_len).enumerate().for_each(|(chunk_idx, dst_chunk)| {
            let start_r = chunk_idx * chunk_height;
            let end_r = (start_r + chunk_height).min(rows);
            let actual_h = end_r - start_r;

            let mut ring = vec![vec![0.0f32; row_len]; ky_size];
            let mut scratch = vec![0.0f32; row_len];
            let mut vert_buf = vec![0.0f32; row_len];

            let prime_start = start_r.saturating_sub(ky_half);
            let loop_read_start = start_r + ky_half;
            let prime_end = loop_read_start.min(rows);

            for r in prime_start..prime_end {
                let idx = r % ky_size;
                if ky_size == 3 && self.kernel_x.len() == 3 {
                    let kx: [f32; 3] = [self.kernel_x[0], self.kernel_x[1], self.kernel_x[2]];
                    Self::horizontal_row_3x3::<T, C>(src_data, &mut ring[idx], &mut scratch, r, cols, &kx);
                } else if ky_size == 5 && self.kernel_x.len() == 5 {
                    let kx: [f32; 5] = [self.kernel_x[0], self.kernel_x[1], self.kernel_x[2], self.kernel_x[3], self.kernel_x[4]];
                    Self::horizontal_row_5x5::<T, C>(src_data, &mut ring[idx], &mut scratch, r, cols, &kx);
                } else {
                    self.horizontal_row_simd::<T, C>(src_data, &mut ring[idx], &mut scratch, r, cols);
                }
            }

            for local_r in 0..actual_h {
                let abs_r = start_r + local_r;
                let read_r = abs_r + ky_half;

                if read_r < rows {
                    let idx = read_r % ky_size;
                    if ky_size == 3 && self.kernel_x.len() == 3 {
                        let kx: [f32; 3] = [self.kernel_x[0], self.kernel_x[1], self.kernel_x[2]];
                        Self::horizontal_row_3x3::<T, C>(src_data, &mut ring[idx], &mut scratch, read_r, cols, &kx);
                    } else if ky_size == 5 && self.kernel_x.len() == 5 {
                        let kx: [f32; 5] = [self.kernel_x[0], self.kernel_x[1], self.kernel_x[2], self.kernel_x[3], self.kernel_x[4]];
                        Self::horizontal_row_5x5::<T, C>(src_data, &mut ring[idx], &mut scratch, read_r, cols, &kx);
                    } else {
                        self.horizontal_row_simd::<T, C>(src_data, &mut ring[idx], &mut scratch, read_r, cols);
                    }
                }

                if ky_size == 3 {
                    let ky: [f32; 3] = [self.kernel_y[0], self.kernel_y[1], self.kernel_y[2]];
                    Self::vertical_row_3x3(&ring, &mut vert_buf, abs_r, rows, row_len, &ky);
                } else if ky_size == 5 {
                    let ky: [f32; 5] = [self.kernel_y[0], self.kernel_y[1], self.kernel_y[2], self.kernel_y[3], self.kernel_y[4]];
                    Self::vertical_row_5x5(&ring, &mut vert_buf, abs_r, rows, row_len, &ky);
                } else {
                    Self::vertical_row_simd(&ring, &mut vert_buf, abs_r, rows, row_len, &self.kernel_y);
                }

                let out_off = local_r * row_len;
                for (e, &v) in vert_buf.iter().enumerate() {
                    dst_chunk[out_off + e] = T::from_f32(v);
                }
            }
        });

        Ok(())
    }
}

/// Apply a separable filter to an image.
///
/// # Arguments
///
/// * `src` - The source image with shape (H, W, C).
/// * `dst` - The destination image with shape (H, W, C).
/// * `kernel_x` - The horizontal kernel.
/// * `kernel_y` - The vertical kernel.
pub fn separable_filter<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, C, A1>,
    dst: &mut Image<T, C, A2>,
    kernel_x: &[f32],
    kernel_y: &[f32],
    strategy: ExecutionStrategy,
) -> Result<(), ImageError>
where
    T: FloatConversion + Clone + Zero + Send + Sync,
{
    if kernel_x.is_empty() || kernel_y.is_empty() {
        return Err(ImageError::InvalidKernelLength(
            kernel_x.len(),
            kernel_y.len(),
        ));
    }

    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    let filter = SeparableFilter::new(kernel_x, kernel_y);
    filter.apply(src, dst, strategy)
}

/// Apply a 1D separable filter (serial execution only, no parallelism).
///
/// This version does not require `Send + Sync` bounds on the pixel type,
///
/// # Arguments
///
/// * `src` - Source image
/// * `dst` - Destination image (must have same size as source)
/// * `kernel_x` - Horizontal filter kernel  
/// * `kernel_y` - Vertical filter kernel
pub fn separable_filter_serial<T, const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<T, C, A1>,
    dst: &mut Image<T, C, A2>,
    kernel_x: &[f32],
    kernel_y: &[f32],
) -> Result<(), ImageError>
where
    T: FloatConversion + Clone + Zero,
{
    if kernel_x.is_empty() || kernel_y.is_empty() {
        return Err(ImageError::InvalidKernelLength(
            kernel_x.len(),
            kernel_y.len(),
        ));
    }

    if src.size() != dst.size() {
        return Err(ImageError::InvalidImageSize(
            src.cols(),
            src.rows(),
            dst.cols(),
            dst.rows(),
        ));
    }

    let filter = SeparableFilter::new(kernel_x, kernel_y);
    filter.apply_pipeline(src, dst)
}

/// Apply a fast filter horizontally using cumulative kernel
///
/// # Arguments
///
/// * `src` - The source image with shape (H, W, C).
/// * `dst` - The destination image with transposed shape (W, H, C).
/// * `half_kernel_x_size` - Half of the kernel at weight 1. The total size would be 2*this+1
pub(crate) fn fast_horizontal_filter<const C: usize, A1: ImageAllocator, A2: ImageAllocator>(
    src: &Image<f32, C, A1>,
    dst: &mut Image<f32, C, A2>,
    half_kernel_x_size: usize,
) -> Result<(), ImageError> {
    let src_data = src.as_slice();
    let dst_data = dst.as_slice_mut();
    let mut row_acc = [0.0; C];

    let mut leftmost_pixel = [0.0; C];
    let mut rightmost_pixel = [0.0; C];

    let pixels_between_first_last_cols = (src.cols() - 1) * C;
    let kernel_pix_offset_diffs: Vec<usize> =
        (0..half_kernel_x_size).map(|p| (p + 1) * C).collect();
    for (pix_offset, source_pixel) in src_data.iter().enumerate() {
        let ch = pix_offset % C;
        let rc = pix_offset / C;
        let c = rc % src.cols();
        let r = rc / src.cols();

        let transposed_r = c;
        let transposed_c = r;
        let transposed_pix_offset = transposed_r * src.rows() * C + transposed_c * C + ch;

        if c == 0 {
            row_acc[ch] = *source_pixel * (half_kernel_x_size + 1) as f32;
            for pix_diff in &kernel_pix_offset_diffs {
                row_acc[ch] += src_data[pix_offset + pix_diff]
            }
            leftmost_pixel[ch] = *source_pixel;
            rightmost_pixel[ch] = src_data[pix_offset + pixels_between_first_last_cols];
        } else {
            row_acc[ch] -= match c.checked_sub(half_kernel_x_size + 1) {
                Some(_) => {
                    let prv_leftmost_pix_offset = pix_offset - C * (half_kernel_x_size + 1);
                    src_data[prv_leftmost_pix_offset]
                }
                None => leftmost_pixel[ch],
            };

            let rightmost_x = c + half_kernel_x_size;

            row_acc[ch] += match rightmost_x {
                x if x < src.cols() => {
                    let rightmost_pix_offset = pix_offset + C * half_kernel_x_size;
                    src_data[rightmost_pix_offset]
                }
                _ => rightmost_pixel[ch],
            };
        }
        dst_data[transposed_pix_offset] = row_acc[ch] / (half_kernel_x_size * 2 + 1) as f32;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use kornia_image::ImageSize;
    use kornia_tensor::CpuAllocator;

    #[test]
    fn test_separable_filter_f32() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 5,
            height: 5,
        };

        #[rustfmt::skip]
        let img = Image::new(
            size,
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            CpuAllocator
        )?;

        let mut dst = Image::<_, 1, _>::from_size_val(img.size(), 0f32, CpuAllocator)?;
        let kernel_x = vec![1.0, 1.0, 1.0];
        let kernel_y = vec![1.0, 1.0, 1.0];
        separable_filter(
            &img,
            &mut dst,
            &kernel_x,
            &kernel_y,
            ExecutionStrategy::Serial,
        )?;

        #[rustfmt::skip]
        assert_eq!(
            dst.as_slice(),
            &[
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 1.0, 1.0, 0.0,
                0.0, 1.0, 1.0, 1.0, 0.0,
                0.0, 1.0, 1.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
            ]
        );

        let xsum = dst.as_slice().iter().sum::<f32>();
        assert_eq!(xsum, 9.0);

        Ok(())
    }

    #[test]
    fn test_separable_filter_u8() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 5,
            height: 5,
        };

        #[rustfmt::skip]
        let img = Image::new(
            size,
            vec![
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 255, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
            ],
            CpuAllocator
        )?;

        let mut dst = Image::<u8, 1, _>::from_size_val(img.size(), 0, CpuAllocator)?;
        let kernel_x = vec![1.0, 1.0, 1.0];
        let kernel_y = vec![1.0, 1.0, 1.0];
        separable_filter(
            &img,
            &mut dst,
            &kernel_x,
            &kernel_y,
            ExecutionStrategy::Serial,
        )?;

        #[rustfmt::skip]
        assert_eq!(
            dst.as_slice(),
            &[
                0, 0, 0, 0, 0,
                0, 255, 255, 255, 0,
                0, 255, 255, 255, 0,
                0, 255, 255, 255, 0,
                0, 0, 0, 0, 0,
            ]
        );
        Ok(())
    }

    #[test]
    fn test_separable_filter_u8_max_val() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 5,
            height: 5,
        };

        let kernel_x = vec![1.0, 1.0, 1.0];
        let kernel_y = vec![1.0, 1.0, 1.0];

        let mut img = Image::<u8, 1, _>::from_size_val(size, 0, CpuAllocator)?;
        img.as_slice_mut()[12] = 255;

        let mut dst = Image::<u8, 1, _>::from_size_val(size, 0, CpuAllocator)?;
        separable_filter(
            &img,
            &mut dst,
            &kernel_x,
            &kernel_y,
            ExecutionStrategy::Serial,
        )?;

        #[rustfmt::skip]
        assert_eq!(
            dst.as_slice(),
            &[0, 0, 0, 0, 0,
            0, 255, 255, 255, 0,
            0, 255, 255, 255, 0,
            0, 255, 255, 255, 0,
            0, 0, 0, 0, 0]
        );
        Ok(())
    }

    #[test]
    fn test_fast_horizontal_filter() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 5,
            height: 5,
        };

        #[rustfmt::skip]
        let img = Image::new(
            size,
            vec![
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 9.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
            ],
            CpuAllocator
        )?;

        let mut transposed = Image::<_, 1, _>::from_size_val(size, 0.0, CpuAllocator)?;

        fast_horizontal_filter(&img, &mut transposed, 1)?;

        #[rustfmt::skip]
        assert_eq!(
            transposed.as_slice(),
            &[
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 3.0, 0.0, 0.0,
                0.0, 0.0, 3.0, 0.0, 0.0,
                0.0, 0.0, 3.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
            ]
        );

        let mut dst = Image::<_, 1, _>::from_size_val(size, 0.0, CpuAllocator)?;

        fast_horizontal_filter(&transposed, &mut dst, 1)?;

        #[rustfmt::skip]
        assert_eq!(
            dst.as_slice(),
            &[
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 1.0, 1.0, 0.0,
                0.0, 1.0, 1.0, 1.0, 0.0,
                0.0, 1.0, 1.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
            ]
        );
        let xsum = dst.as_slice().iter().sum::<f32>();
        assert_eq!(xsum, 9.0);

        Ok(())
    }

    #[test]
    fn test_parallel_strategies_consistency() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 10,
            height: 10,
        };

        // Create test image with pattern
        let mut data = vec![0.0f32; 100];
        data[44] = 1.0; // Center pixel
        data[33] = 0.5; // Another pixel
        data[67] = 0.8;

        let img = Image::new(size, data, CpuAllocator)?;
        let kernel_x = vec![0.25, 0.5, 0.25];
        let kernel_y = vec![0.25, 0.5, 0.25];

        // Serial (reference)
        let mut dst_serial = Image::<f32, 1, _>::from_size_val(size, 0.0, CpuAllocator)?;
        separable_filter(
            &img,
            &mut dst_serial,
            &kernel_x,
            &kernel_y,
            ExecutionStrategy::Serial,
        )?;

        // Fixed(4)
        let mut dst_fixed = Image::<f32, 1, _>::from_size_val(size, 0.0, CpuAllocator)?;
        separable_filter(
            &img,
            &mut dst_fixed,
            &kernel_x,
            &kernel_y,
            ExecutionStrategy::Fixed(4),
        )?;

        // AutoRows
        let mut dst_auto = Image::<f32, 1, _>::from_size_val(size, 0.0, CpuAllocator)?;
        separable_filter(
            &img,
            &mut dst_auto,
            &kernel_x,
            &kernel_y,
            ExecutionStrategy::AutoRows(0),
        )?;

        // ParallelElements
        let mut dst_elements = Image::<f32, 1, _>::from_size_val(size, 0.0, CpuAllocator)?;
        separable_filter(
            &img,
            &mut dst_elements,
            &kernel_x,
            &kernel_y,
            ExecutionStrategy::ParallelElements,
        )?;

        // All strategies should produce identical results
        assert_eq!(
            dst_serial.as_slice(),
            dst_fixed.as_slice(),
            "Fixed strategy mismatch"
        );
        assert_eq!(
            dst_serial.as_slice(),
            dst_auto.as_slice(),
            "AutoRows strategy mismatch"
        );
        assert_eq!(
            dst_serial.as_slice(),
            dst_elements.as_slice(),
            "ParallelElements strategy mismatch"
        );

        Ok(())
    }

    #[test]
    fn test_parallel_strategies_u8() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 8,
            height: 8,
        };

        let mut data = vec![0u8; 64];
        data[27] = 255;
        data[36] = 128;

        let img = Image::new(size, data, CpuAllocator)?;
        let kernel_x = vec![1.0, 1.0, 1.0];
        let kernel_y = vec![1.0, 1.0, 1.0];

        // Test strategies
        let mut dst_serial = Image::<u8, 1, _>::from_size_val(size, 0, CpuAllocator)?;
        separable_filter(
            &img,
            &mut dst_serial,
            &kernel_x,
            &kernel_y,
            ExecutionStrategy::Serial,
        )?;

        let mut dst_fixed = Image::<u8, 1, _>::from_size_val(size, 0, CpuAllocator)?;
        separable_filter(
            &img,
            &mut dst_fixed,
            &kernel_x,
            &kernel_y,
            ExecutionStrategy::Fixed(2),
        )?;

        let mut dst_auto = Image::<u8, 1, _>::from_size_val(size, 0, CpuAllocator)?;
        separable_filter(
            &img,
            &mut dst_auto,
            &kernel_x,
            &kernel_y,
            ExecutionStrategy::AutoRows(0),
        )?;

        let mut dst_elements = Image::<u8, 1, _>::from_size_val(size, 0, CpuAllocator)?;
        separable_filter(
            &img,
            &mut dst_elements,
            &kernel_x,
            &kernel_y,
            ExecutionStrategy::ParallelElements,
        )?;

        assert_eq!(dst_serial.as_slice(), dst_fixed.as_slice());
        assert_eq!(dst_serial.as_slice(), dst_auto.as_slice());
        assert_eq!(dst_serial.as_slice(), dst_elements.as_slice());

        Ok(())
    }

    #[test]
    fn test_fixed_threadpool_validation() {
        let size = ImageSize {
            width: 5,
            height: 5,
        };
        let img = Image::<f32, 1, _>::from_size_val(size, 0.5, CpuAllocator).unwrap();
        let mut dst = Image::<f32, 1, _>::from_size_val(size, 0.0, CpuAllocator).unwrap();
        let kernel = vec![1.0];

        // Fixed(0) should error
        let result = separable_filter(
            &img,
            &mut dst,
            &kernel,
            &kernel,
            ExecutionStrategy::Fixed(0),
        );
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("thread count must be > 0"));
    }
    #[test]
    fn test_separable_filter_rgb_simd() -> Result<(), ImageError> {
        let size = ImageSize {
            width: 4,
            height: 4,
        };
        // R G B interleaved
        // We will make R=1, G=2, B=3 for all pixels
        let mut data = vec![];
        for _ in 0..16 {
            data.push(1.0); // R
            data.push(2.0); // G
            data.push(3.0); // B
        }
        
        let img = Image::<f32, 3, _>::new(size, data, CpuAllocator)?;
        let mut dst = Image::<f32, 3, _>::from_size_val(size, 0.0, CpuAllocator)?;
        
        // Identity kernel (should preserve values)
        // 3x3 kernel with center=1
        let kernel_x = vec![0.0, 1.0, 0.0];
        let kernel_y = vec![0.0, 1.0, 0.0];
        
        // Use Serial strategy which uses the same low-level SIMD functions
        separable_filter(
            &img,
            &mut dst,
            &kernel_x,
            &kernel_y,
            ExecutionStrategy::Serial,
        )?;
        
        // Check output
        for (i, val) in dst.as_slice().iter().enumerate() {
            let channel = i % 3;
            let expected = match channel {
                0 => 1.0,
                1 => 2.0,
                2 => 3.0,
                _ => unreachable!(),
            };
            
            // Checking valid region (1,1) to (2,2)
            let rc = i / 3;
            let r = rc / 4;
            let c = rc % 4;
            
            if r >= 1 && r < 3 && c >= 1 && c < 3 {
               assert!((val - expected).abs() < 1e-4, "Failed at r={}, c={}, ch={}. Expected {}, got {}", r, c, channel, expected, val);
            }
        }
        
        Ok(())
    }
}
