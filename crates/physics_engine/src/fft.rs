//! Common utilities for Fast Fourier Transforms (FFTs) on 3D grids.

use rustfft::{FftPlanner, num_complex::Complex};
use rustfft::num_traits::Zero;
use nalgebra::Vector3;
use anyhow::Result;

/// Solves the Poisson equation ∇²Φ = S in Fourier space.
///
/// Takes a source field `S` (like density), FFTs it, applies the Green's
/// function for the Laplacian (1/(-k²)), and inverse FFTs to get the potential Φ.
pub fn solve_poisson_fft(source_field: &mut [Complex<f64>], n_grid: usize) -> Result<()> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_grid * n_grid * n_grid);
    let ifft = planner.plan_fft_inverse(n_grid * n_grid * n_grid);

    // Forward FFT
    fft.process(source_field);

    // Apply Green's function in Fourier space: 1 / -k^2
    let k_factor = 2.0 * std::f64::consts::PI / (n_grid as f64);
    for kx in 0..n_grid {
        for ky in 0..n_grid {
            for kz in 0..n_grid {
                let k_sq = (kx as f64 * k_factor).powi(2) +
                           (ky as f64 * k_factor).powi(2) +
                           (kz as f64 * k_factor).powi(2);

                if k_sq > 1e-9 { // Avoid division by zero at k=0
                    let flat_idx = kx + ky * n_grid + kz * n_grid * n_grid;
                    source_field[flat_idx] /= -k_sq;
                }
            }
        }
    }
    source_field[0] = Complex::zero(); // Set DC mode (k=0) to zero

    // Inverse FFT to get potential
    ifft.process(source_field);

    Ok(())
}

/// Calculates the gradient of a scalar field in Fourier space.
///
/// Takes a scalar field `phi_k` in Fourier space, multiplies by `i*k` for each
/// component to get the gradient components, and then inverse FFTs them back
/// to real space.
pub fn gradient_fft(phi_k: &[Complex<f64>], n_grid: usize) -> Result<Vec<Vector3<f64>>> {
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(n_grid * n_grid * n_grid);
    let grid_vol = (n_grid * n_grid * n_grid) as f64;

    let mut grad_x_k = phi_k.to_vec();
    let mut grad_y_k = phi_k.to_vec();
    let mut grad_z_k = phi_k.to_vec();

    let k_factor = 2.0 * std::f64::consts::PI / (n_grid as f64);

    for kx in 0..n_grid {
        for ky in 0..n_grid {
            for kz in 0..n_grid {
                let idx = kx + ky * n_grid + kz * n_grid * n_grid;
                let k_vec = Vector3::new(kx as f64, ky as f64, kz as f64) * k_factor;

                // Gradient in Fourier space is multiplication by i*k
                let i = Complex::new(0.0, 1.0);

                grad_x_k[idx] *= i * k_vec.x;
                grad_y_k[idx] *= i * k_vec.y;
                grad_z_k[idx] *= i * k_vec.z;
            }
        }
    }

    // Inverse FFT each component
    ifft.process(&mut grad_x_k);
    ifft.process(&mut grad_y_k);
    ifft.process(&mut grad_z_k);

    let mut gradient_field = vec![Vector3::zeros(); phi_k.len()];
    for i in 0..phi_k.len() {
        gradient_field[i] = Vector3::new(
            grad_x_k[i].re / grid_vol,
            grad_y_k[i].re / grid_vol,
            grad_z_k[i].re / grid_vol,
        );
    }

    Ok(gradient_field)
} 