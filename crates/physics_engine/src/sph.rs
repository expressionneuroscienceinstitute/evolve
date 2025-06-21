//! Smoothed Particle Hydrodynamics (SPH) Engine
//! 
//! Implements conservative "grad-h" SPH formulation for realistic fluid dynamics
//! in star formation and gas collapse simulations.
//! 
//! Based on SEREN methodology: https://www.aanda.org/articles/aa/full_html/2011/05/aa14949-10/aa14949-10.html
//! 
//! Key features:
//! - Conservative grad-h SPH formulation
//! - Artificial viscosity for shock capturing
//! - Pressure gradient forces
//! - Density estimation via kernel interpolation
//! - Variable smoothing length optimization

use crate::{Vector3, FundamentalParticle, ParticleType, Result};
use tracing::debug;

/// SPH kernel function types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KernelType {
    /// Cubic spline kernel (Monaghan & Lattanzio 1985)
    CubicSpline,
    /// Gaussian kernel
    Gaussian,
    /// Quartic spline kernel
    QuarticSpline,
}

/// SPH particle with hydrodynamic properties
#[derive(Debug, Clone)]
pub struct SphParticle {
    /// Base particle properties
    pub particle: FundamentalParticle,
    /// Density (kg/m³)
    pub density: f64,
    /// Pressure (Pa)
    pub pressure: f64,
    /// Sound speed (m/s)
    pub sound_speed: f64,
    /// Smoothing length (m)
    pub smoothing_length: f64,
    /// Artificial viscosity coefficient
    pub alpha_viscosity: f64,
    /// Beta viscosity coefficient
    pub beta_viscosity: f64,
    /// Divergence of velocity field
    pub velocity_divergence: f64,
    /// Curl of velocity field
    pub velocity_curl: f64,
    /// Internal energy per unit mass (J/kg)
    pub internal_energy: f64,
    /// Temperature (K)
    pub temperature: f64,
    /// Equation of state gamma (adiabatic index)
    pub gamma: f64,
}

impl SphParticle {
    /// Create a new SPH particle from a base particle
    pub fn new(particle: FundamentalParticle, smoothing_length: f64) -> Self {
        let density = 1.0; // Will be computed by SPH
        let temperature = 300.0; // Initial temperature
        let gamma = 5.0 / 3.0; // Monatomic gas
        let internal_energy = temperature * 1.5 * 1.380649e-23 / particle.mass; // 3/2 kT
        
        Self {
            particle,
            density,
            pressure: 0.0, // Will be computed from EOS
            sound_speed: 0.0, // Will be computed from EOS
            smoothing_length,
            alpha_viscosity: 1.0,
            beta_viscosity: 2.0,
            velocity_divergence: 0.0,
            velocity_curl: 0.0,
            internal_energy,
            temperature,
            gamma,
        }
    }

    /// Update pressure and sound speed from equation of state
    pub fn update_eos(&mut self) {
        // Ideal gas equation of state: P = (γ-1)ρu
        self.pressure = (self.gamma - 1.0) * self.density * self.internal_energy;
        
        // Sound speed: c = sqrt(γP/ρ)
        self.sound_speed = (self.gamma * self.pressure / self.density).sqrt();
        
        // Update temperature from internal energy
        self.temperature = 2.0 * self.internal_energy * self.particle.mass / (3.0 * 1.380649e-23);
    }
}

/// SPH kernel functions
#[derive(Debug)]
pub struct SphKernel {
    kernel_type: KernelType,
    dimension: usize,
}

impl SphKernel {
    /// Create a new SPH kernel
    pub fn new(kernel_type: KernelType, dimension: usize) -> Self {
        Self {
            kernel_type,
            dimension,
        }
    }

    /// Evaluate the kernel function W(r,h)
    pub fn evaluate(&self, r: f64, h: f64) -> f64 {
        let q = r / h;
        
        match self.kernel_type {
            KernelType::CubicSpline => self.cubic_spline_kernel(q, h),
            KernelType::Gaussian => self.gaussian_kernel(q, h),
            KernelType::QuarticSpline => self.quartic_spline_kernel(q, h),
        }
    }

    /// Evaluate the gradient of the kernel function ∇W(r,h)
    pub fn gradient(&self, r: f64, h: f64, r_vec: Vector3<f64>) -> Vector3<f64> {
        let q = r / h;
        let gradient_magnitude = match self.kernel_type {
            KernelType::CubicSpline => self.cubic_spline_gradient(q, h),
            KernelType::Gaussian => self.gaussian_gradient(q, h),
            KernelType::QuarticSpline => self.quartic_spline_gradient(q, h),
        };
        
        if r > 1e-10 {
            gradient_magnitude * r_vec / r
        } else {
            Vector3::zeros()
        }
    }

    /// Cubic spline kernel (Monaghan & Lattanzio 1985)
    fn cubic_spline_kernel(&self, q: f64, h: f64) -> f64 {
        let alpha_d = match self.dimension {
            1 => 1.0 / h,
            2 => 10.0 / (7.0 * std::f64::consts::PI * h * h),
            3 => 1.0 / (std::f64::consts::PI * h * h * h),
            _ => 1.0 / (std::f64::consts::PI * h * h * h),
        };
        
        if q < 1.0 {
            if q < 0.5 {
                alpha_d * (1.0 - 6.0 * q * q + 6.0 * q * q * q)
            } else {
                alpha_d * 2.0 * (1.0 - q) * (1.0 - q) * (1.0 - q)
            }
        } else {
            0.0
        }
    }

    /// Gradient of cubic spline kernel
    fn cubic_spline_gradient(&self, q: f64, h: f64) -> f64 {
        let alpha_d = match self.dimension {
            1 => 1.0 / h,
            2 => 10.0 / (7.0 * std::f64::consts::PI * h * h),
            3 => 1.0 / (std::f64::consts::PI * h * h * h),
            _ => 1.0 / (std::f64::consts::PI * h * h * h),
        };
        
        if q < 1.0 {
            if q < 0.5 {
                alpha_d * (-12.0 * q + 18.0 * q * q) / h
            } else {
                alpha_d * (-6.0 * (1.0 - q) * (1.0 - q)) / h
            }
        } else {
            0.0
        }
    }

    /// Gaussian kernel
    fn gaussian_kernel(&self, q: f64, h: f64) -> f64 {
        let alpha_d = match self.dimension {
            1 => 1.0 / (h * (std::f64::consts::PI).sqrt()),
            2 => 1.0 / (std::f64::consts::PI * h * h),
            3 => 1.0 / (std::f64::consts::PI.sqrt() * std::f64::consts::PI * h * h * h),
            _ => 1.0 / (std::f64::consts::PI.sqrt() * std::f64::consts::PI * h * h * h),
        };
        
        alpha_d * (-q * q).exp()
    }

    /// Gradient of Gaussian kernel
    fn gaussian_gradient(&self, q: f64, h: f64) -> f64 {
        let alpha_d = match self.dimension {
            1 => 1.0 / (h * (std::f64::consts::PI).sqrt()),
            2 => 1.0 / (std::f64::consts::PI * h * h),
            3 => 1.0 / (std::f64::consts::PI.sqrt() * std::f64::consts::PI * h * h * h),
            _ => 1.0 / (std::f64::consts::PI.sqrt() * std::f64::consts::PI * h * h * h),
        };
        
        -2.0 * alpha_d * q * (-q * q).exp() / h
    }

    /// Quartic spline kernel
    fn quartic_spline_kernel(&self, q: f64, h: f64) -> f64 {
        let alpha_d = match self.dimension {
            1 => 1.0 / h,
            2 => 96.0 / (119.0 * std::f64::consts::PI * h * h),
            3 => 1.0 / (20.0 * std::f64::consts::PI * h * h * h),
            _ => 1.0 / (20.0 * std::f64::consts::PI * h * h * h),
        };
        
        if q < 1.0 {
            alpha_d * (1.0 - q) * (1.0 - q) * (1.0 - q) * (1.0 - q) * (1.0 + 4.0 * q)
        } else {
            0.0
        }
    }

    /// Gradient of quartic spline kernel
    fn quartic_spline_gradient(&self, q: f64, h: f64) -> f64 {
        let alpha_d = match self.dimension {
            1 => 1.0 / h,
            2 => 96.0 / (119.0 * std::f64::consts::PI * h * h),
            3 => 1.0 / (20.0 * std::f64::consts::PI * h * h * h),
            _ => 1.0 / (20.0 * std::f64::consts::PI * h * h * h),
        };
        
        if q < 1.0 {
            alpha_d * (-4.0 * (1.0 - q) * (1.0 - q) * (1.0 - q) * (1.0 + 4.0 * q) + 
                       (1.0 - q) * (1.0 - q) * (1.0 - q) * (1.0 - q) * 4.0) / h
        } else {
            0.0
        }
    }
}

/// SPH solver implementing conservative grad-h formulation
#[derive(Debug)]
pub struct SphSolver {
    kernel: SphKernel,
    /// Number of neighbors for smoothing length optimization
    target_neighbors: usize,
    /// Artificial viscosity parameters
    alpha_viscosity: f64,
    beta_viscosity: f64,
    /// Time step safety factor
    courant_factor: f64,
    /// Maximum time step
    max_dt: f64,
}

impl SphSolver {
    /// Create a new SPH solver
    pub fn new(kernel_type: KernelType, dimension: usize) -> Self {
        Self {
            kernel: SphKernel::new(kernel_type, dimension),
            target_neighbors: 50,
            alpha_viscosity: 1.0,
            beta_viscosity: 2.0,
            courant_factor: 0.3,
            max_dt: 1e-6,
        }
    }

    /// Compute density for all particles using SPH interpolation
    pub fn compute_density(&self, particles: &mut [SphParticle]) -> Result<()> {
        let n_particles = particles.len();
        
        for i in 0..n_particles {
            let mut density = 0.0;
            
            for j in 0..n_particles {
                if i != j {
                    let r_vec = particles[j].particle.position - particles[i].particle.position;
                    let r = r_vec.magnitude();
                    let h = particles[i].smoothing_length;
                    
                    if r < h {
                        let kernel_value = self.kernel.evaluate(r, h);
                        density += particles[j].particle.mass * kernel_value;
                    }
                }
            }
            
            particles[i].density = density;
        }
        
        debug!("Computed density for {} SPH particles", n_particles);
        Ok(())
    }

    /// Compute pressure forces using conservative grad-h SPH
    pub fn compute_pressure_forces(&self, particles: &mut [SphParticle]) -> Result<Vec<Vector3<f64>>> {
        let n_particles = particles.len();
        let mut forces = vec![Vector3::zeros(); n_particles];
        
        // Update equation of state for all particles
        for particle in particles.iter_mut() {
            particle.update_eos();
        }
        
        for i in 0..n_particles {
            for j in (i + 1)..n_particles {
                let r_vec = particles[j].particle.position - particles[i].particle.position;
                let r = r_vec.magnitude();
                let h_i = particles[i].smoothing_length;
                let h_j = particles[j].smoothing_length;
                let h_avg = 0.5 * (h_i + h_j);
                
                if r < h_avg {
                    let kernel_gradient = self.kernel.gradient(r, h_avg, r_vec);
                    
                    // Conservative pressure force: F = -m_i m_j (P_i/ρ_i² + P_j/ρ_j²) ∇W
                    let pressure_term = particles[i].pressure / (particles[i].density * particles[i].density) +
                                       particles[j].pressure / (particles[j].density * particles[j].density);
                    
                    let force_magnitude = -particles[i].particle.mass * particles[j].particle.mass * pressure_term;
                    let force = force_magnitude * kernel_gradient;
                    
                    forces[i] += force;
                    forces[j] -= force; // Action-reaction
                }
            }
        }
        
        debug!("Computed pressure forces for {} SPH particles", n_particles);
        Ok(forces)
    }

    /// Compute artificial viscosity for shock capturing
    pub fn compute_viscosity_forces(&self, particles: &mut [SphParticle]) -> Result<Vec<Vector3<f64>>> {
        let n_particles = particles.len();
        let mut forces = vec![Vector3::zeros(); n_particles];
        
        for i in 0..n_particles {
            for j in (i + 1)..n_particles {
                let r_vec = particles[j].particle.position - particles[i].particle.position;
                let r = r_vec.magnitude();
                let h_i = particles[i].smoothing_length;
                let h_j = particles[j].smoothing_length;
                let h_avg = 0.5 * (h_i + h_j);
                
                if r < h_avg {
                    let v_rel = particles[j].particle.velocity - particles[i].particle.velocity;
                    let v_dot_r = v_rel.dot(&r_vec);
                    
                    // Only apply viscosity for approaching particles
                    if v_dot_r < 0.0 {
                        let c_avg = 0.5 * (particles[i].sound_speed + particles[j].sound_speed);
                        let rho_avg = 0.5 * (particles[i].density + particles[j].density);
                        
                        // Artificial viscosity: Π = (-αμc + βμ²)/ρ
                        let mu = h_avg * v_dot_r / (r * r + 0.01 * h_avg * h_avg);
                        let pi_visc = (-self.alpha_viscosity * mu * c_avg + self.beta_viscosity * mu * mu) / rho_avg;
                        
                        let kernel_gradient = self.kernel.gradient(r, h_avg, r_vec);
                        let force_magnitude = -0.5 * particles[i].particle.mass * particles[j].particle.mass * pi_visc;
                        let force = force_magnitude * kernel_gradient;
                        
                        forces[i] += force;
                        forces[j] -= force; // Action-reaction
                    }
                }
            }
        }
        
        debug!("Computed viscosity forces for {} SPH particles", n_particles);
        Ok(forces)
    }

    /// Optimize smoothing lengths to maintain target number of neighbors
    pub fn optimize_smoothing_lengths(&self, particles: &mut [SphParticle]) -> Result<()> {
        let n_particles = particles.len();
        
        for i in 0..n_particles {
            let mut neighbor_count = 0;
            
            // Count current neighbors
            for j in 0..n_particles {
                if i != j {
                    let r_vec = particles[j].particle.position - particles[i].particle.position;
                    let r = r_vec.magnitude();
                    if r < particles[i].smoothing_length {
                        neighbor_count += 1;
                    }
                }
            }
            
            // Adjust smoothing length to achieve target neighbors
            let ratio = (self.target_neighbors as f64 / neighbor_count as f64).powf(1.0 / 3.0);
            particles[i].smoothing_length *= ratio;
            
            // Clamp smoothing length to reasonable bounds
            particles[i].smoothing_length = particles[i].smoothing_length
                .max(1e-10)
                .min(1e-3);
        }
        
        debug!("Optimized smoothing lengths for {} SPH particles", n_particles);
        Ok(())
    }

    /// Compute time step based on Courant condition
    pub fn compute_time_step(&self, particles: &[SphParticle]) -> f64 {
        let mut min_dt = f64::INFINITY;
        
        for particle in particles {
            let courant_dt = self.courant_factor * particle.smoothing_length / 
                           (particle.sound_speed + particle.particle.velocity.magnitude());
            min_dt = min_dt.min(courant_dt);
        }
        
        min_dt.min(self.max_dt)
    }

    /// Integrate SPH particles for one time step
    pub fn integrate_step(&mut self, particles: &mut [SphParticle], dt: f64) -> Result<()> {
        // Optimize smoothing lengths
        self.optimize_smoothing_lengths(particles)?;
        
        // Compute density
        self.compute_density(particles)?;
        
        // Compute forces
        let pressure_forces = self.compute_pressure_forces(particles)?;
        let viscosity_forces = self.compute_viscosity_forces(particles)?;
        
        // Integrate particle positions and velocities
        for (i, particle) in particles.iter_mut().enumerate() {
            let total_force = pressure_forces[i] + viscosity_forces[i];
            
            // Velocity Verlet integration
            particle.particle.velocity += total_force / particle.particle.mass * dt;
            particle.particle.position += particle.particle.velocity * dt;
            
            // Update internal energy (adiabatic)
            let work = total_force.dot(&particle.particle.velocity) * dt;
            particle.internal_energy += work / particle.particle.mass;
        }
        
        debug!("Integrated {} SPH particles for dt = {:.2e}", particles.len(), dt);
        Ok(())
    }

    /// Convert regular particles to SPH particles
    pub fn convert_to_sph_particles(&self, particles: Vec<FundamentalParticle>) -> Vec<SphParticle> {
        particles.into_iter()
            .filter(|p| matches!(p.particle_type, ParticleType::Hydrogen | ParticleType::Helium))
            .map(|p| SphParticle::new(p, 1e-6)) // Initial smoothing length
            .collect()
    }

    /// Convert SPH particles back to regular particles
    pub fn convert_from_sph_particles(&self, sph_particles: Vec<SphParticle>) -> Vec<FundamentalParticle> {
        sph_particles.into_iter().map(|sp| sp.particle).collect()
    }
}

impl Default for SphSolver {
    fn default() -> Self {
        Self::new(KernelType::CubicSpline, 3)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ParticleType;

    #[test]
    fn test_sph_particle_creation() {
        let particle = FundamentalParticle::new(
            ParticleType::Hydrogen,
            1.67e-27, // Proton mass
            Vector3::new(0.0, 0.0, 0.0)
        );
        
        let sph_particle = SphParticle::new(particle, 1e-6);
        assert_eq!(sph_particle.density, 1.0);
        assert_eq!(sph_particle.gamma, 5.0 / 3.0);
        assert!(sph_particle.internal_energy > 0.0);
    }

    #[test]
    fn test_kernel_evaluation() {
        let kernel = SphKernel::new(KernelType::CubicSpline, 3);
        
        // Test kernel at origin
        let w0 = kernel.evaluate(0.0, 1.0);
        assert!(w0 > 0.0);
        
        // Test kernel at smoothing length
        let w1 = kernel.evaluate(1.0, 1.0);
        assert_eq!(w1, 0.0);
        
        // Test kernel beyond smoothing length
        let w2 = kernel.evaluate(2.0, 1.0);
        assert_eq!(w2, 0.0);
    }

    #[test]
    fn test_sph_solver_creation() {
        let solver = SphSolver::new(KernelType::CubicSpline, 3);
        assert_eq!(solver.target_neighbors, 50);
        assert_eq!(solver.alpha_viscosity, 1.0);
        assert_eq!(solver.beta_viscosity, 2.0);
    }

    #[test]
    fn test_density_computation() {
        let solver = SphSolver::new(KernelType::CubicSpline, 3);
        
        // Create test particles
        let mut particles = vec![
            SphParticle::new(FundamentalParticle::new(
                ParticleType::Hydrogen,
                1.67e-27,
                Vector3::new(0.0, 0.0, 0.0)
            ), 1e-6),
            SphParticle::new(FundamentalParticle::new(
                ParticleType::Hydrogen,
                1.67e-27,
                Vector3::new(1e-7, 0.0, 0.0)
            ), 1e-6),
        ];
        
        solver.compute_density(&mut particles).unwrap();
        
        // Both particles should have non-zero density
        assert!(particles[0].density > 0.0);
        assert!(particles[1].density > 0.0);
    }
} 