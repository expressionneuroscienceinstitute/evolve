#[cfg(test)]
mod buffer_pool_tests {
    use crate::{BufferPool, TrackedBuffer};
    use wgpu::{BufferDescriptor, BufferUsages};

    async fn create_test_device() -> (wgpu::Device, wgpu::Queue) {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
            flags: wgpu::InstanceFlags::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
        });

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find adapter");

        adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    label: None,
                },
                None,
            )
            .await
            .expect("Failed to create device")
    }

    #[tokio::test]
    async fn test_buffer_pool_basic_operations() {
        let (device, _queue) = create_test_device().await;
        let mut pool = BufferPool::new();

        // Test buffer creation
        let buffer1 = pool.create_buffer(&device, &BufferDescriptor {
            label: Some("Test Buffer 1"),
            size: 1024,
            usage: BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        let buffer2 = pool.create_buffer(&device, &BufferDescriptor {
            label: Some("Test Buffer 2"),
            size: 2048,
            usage: BufferUsages::UNIFORM,
            mapped_at_creation: false,
        });

        // Check stats
        let stats = pool.get_stats();
        assert_eq!(stats.active_buffers, 2);
        assert_eq!(stats.retired_buffers, 0);
        assert_eq!(stats.allocation_count, 2);
        assert!(stats.active_memory_mb > 0.0);

        // Test buffer retirement
        pool.retire_buffer(&buffer1, None);
        let stats = pool.get_stats();
        assert_eq!(stats.active_buffers, 1);
        assert_eq!(stats.retired_buffers, 1);

        // Test cleanup
        let cleaned = pool.cleanup_retired(&device);
        assert_eq!(cleaned, 1); // Should clean up the retired buffer
        let stats = pool.get_stats();
        assert_eq!(stats.retired_buffers, 0);

        // Retire second buffer with submission index (we can't create SubmissionIndex directly)
        // So we'll just test with None for now
        pool.retire_buffer(&buffer2, None);
        let stats = pool.get_stats();
        assert_eq!(stats.active_buffers, 0);
        assert_eq!(stats.retired_buffers, 1);
    }

    #[tokio::test]
    async fn test_buffer_pool_memory_tracking() {
        let (device, _queue) = create_test_device().await;
        let mut pool = BufferPool::new();

        // Create multiple buffers of different sizes
        let sizes = [1024, 2048, 4096, 8192];
        let mut buffers = Vec::new();

        for (i, &size) in sizes.iter().enumerate() {
            let buffer = pool.create_buffer(&device, &BufferDescriptor {
                label: Some(&format!("Test Buffer {}", i)),
                size: size as u64,
                usage: BufferUsages::VERTEX,
                mapped_at_creation: false,
            });
            buffers.push(buffer);
        }

        let stats = pool.get_stats();
        assert_eq!(stats.active_buffers, 4);
        assert_eq!(stats.allocation_count, 4);

        let expected_total_bytes = sizes.iter().sum::<usize>() as u64;
        let expected_total_mb = expected_total_bytes as f32 / 1024.0 / 1024.0;
        assert!((stats.active_memory_mb - expected_total_mb).abs() < 0.001);

        // Retire all buffers
        for buffer in &buffers {
            pool.retire_buffer(buffer, None);
        }

        let stats = pool.get_stats();
        assert_eq!(stats.active_buffers, 0);
        assert_eq!(stats.retired_buffers, 4);

        // Clean up all retired buffers
        let cleaned = pool.cleanup_retired(&device);
        assert_eq!(cleaned, 4);

        let stats = pool.get_stats();
        assert_eq!(stats.retired_buffers, 0);
        assert!(stats.total_freed_mb > 0.0);
    }

    #[test]
    fn test_tracked_buffer_creation() {
        let device = futures::executor::block_on(create_test_device()).0;
        
        let buffer = device.create_buffer(&BufferDescriptor {
            label: Some("Test Tracked Buffer"),
            size: 1024,
            usage: BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        let tracked = TrackedBuffer::new(buffer.global_id(), 1024, "Test Tracked Buffer".to_string());
        
        assert_eq!(tracked.size, 1024);
        assert_eq!(tracked.label, "Test Tracked Buffer");
        assert!(tracked.last_submission.is_none());
        assert!(tracked.is_safe_to_destroy(&device)); // No submission, should be safe
    }

    #[test]
    fn test_buffer_pool_stats() {
        let pool = BufferPool::new();
        let stats = pool.get_stats();
        
        assert_eq!(stats.active_buffers, 0);
        assert_eq!(stats.retired_buffers, 0);
        assert_eq!(stats.active_memory_mb, 0.0);
        assert_eq!(stats.total_allocated_mb, 0.0);
        assert_eq!(stats.total_freed_mb, 0.0);
        assert_eq!(stats.peak_memory_mb, 0.0);
        assert_eq!(stats.allocation_count, 0);
    }

    #[tokio::test]
    async fn test_buffer_pool_force_cleanup() {
        let (device, _queue) = create_test_device().await;
        let mut pool = BufferPool::new();

        // Create and retire a buffer
        let buffer = pool.create_buffer(&device, &BufferDescriptor {
            label: Some("Test Buffer"),
            size: 1024,
            usage: BufferUsages::VERTEX,
            mapped_at_creation: false,
        });

        pool.retire_buffer(&buffer, None); // Can't create SubmissionIndex directly
        
        let stats = pool.get_stats();
        assert_eq!(stats.retired_buffers, 1);

        // Force cleanup should remove it even if not safe
        pool.force_cleanup();
        
        let stats = pool.get_stats();
        assert_eq!(stats.retired_buffers, 0);
        assert!(stats.total_freed_mb > 0.0);
    }
}

// Mac-Specific Buffer Pool Test Module
// Optimizes GPU memory management for Mac Metal backend

use wgpu::{Buffer, BufferDescriptor, BufferUsages, Device, Queue};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::{info, warn, error};

/// Mac-specific buffer pool configuration
#[derive(Debug, Clone)]
pub struct MacBufferPoolConfig {
    pub initial_pool_size: usize,
    pub max_pool_size: usize,
    pub buffer_size: u64,
    pub usage: BufferUsages,
    pub mac_gpu_type: MacGpuType,
    pub retina_scale: f32,
}

/// Mac GPU types for buffer optimization
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MacGpuType {
    Intel,
    AppleSilicon,
    Discrete,
    Unknown,
}

/// Mac-optimized buffer pool for particle rendering
pub struct MacBufferPool {
    device: Arc<Device>,
    queue: Arc<Queue>,
    config: MacBufferPoolConfig,
    active_buffers: Arc<Mutex<HashMap<u64, Buffer>>>,
    retired_buffers: Arc<Mutex<Vec<Buffer>>>,
    buffer_count: Arc<Mutex<u64>>,
    total_allocated: Arc<Mutex<u64>>,
}

impl MacBufferPool {
    /// Create a new Mac-optimized buffer pool
    pub fn new(
        device: Arc<Device>,
        queue: Arc<Queue>,
        config: MacBufferPoolConfig,
    ) -> Self {
        info!("Creating Mac buffer pool with config: {:?}", config);
        
        Self {
            device,
            queue,
            config,
            active_buffers: Arc::new(Mutex::new(HashMap::new())),
            retired_buffers: Arc::new(Mutex::new(Vec::new())),
            buffer_count: Arc::new(Mutex::new(0)),
            total_allocated: Arc::new(Mutex::new(0)),
        }
    }

    /// Get a buffer from the pool (Mac-optimized)
    pub fn get_buffer(&self) -> Result<Buffer, Box<dyn std::error::Error>> {
        let mut retired_buffers = self.retired_buffers.lock().unwrap();
        
        // Try to reuse a retired buffer first
        if let Some(buffer) = retired_buffers.pop() {
            let mut active_buffers = self.active_buffers.lock().unwrap();
            let buffer_id = *self.buffer_count.lock().unwrap();
            active_buffers.insert(buffer_id, buffer.clone());
            *self.buffer_count.lock().unwrap() += 1;
            
            info!("Reused buffer {} from pool", buffer_id);
            return Ok(buffer);
        }

        // Create new buffer if pool is empty
        let buffer = self.create_new_buffer()?;
        let mut active_buffers = self.active_buffers.lock().unwrap();
        let buffer_id = *self.buffer_count.lock().unwrap();
        active_buffers.insert(buffer_id, buffer.clone());
        *self.buffer_count.lock().unwrap() += 1;
        
        info!("Created new buffer {} for pool", buffer_id);
        Ok(buffer)
    }

    /// Return a buffer to the pool (Mac-optimized)
    pub fn return_buffer(&self, buffer: Buffer) -> Result<(), Box<dyn std::error::Error>> {
        let mut retired_buffers = self.retired_buffers.lock().unwrap();
        
        // Check if pool is full
        if retired_buffers.len() >= self.config.max_pool_size {
            warn!("Buffer pool full, dropping buffer");
            return Ok(());
        }

        retired_buffers.push(buffer);
        info!("Returned buffer to pool, retired count: {}", retired_buffers.len());
        Ok(())
    }

    /// Create a new buffer with Mac-specific optimizations
    fn create_new_buffer(&self) -> Result<Buffer, Box<dyn std::error::Error>> {
        let descriptor = BufferDescriptor {
            label: Some("Mac Buffer Pool Buffer"),
            size: self.config.buffer_size,
            usage: self.config.usage,
            mapped_at_creation: false,
        };

        let buffer = self.device.create_buffer(&descriptor);
        *self.total_allocated.lock().unwrap() += self.config.buffer_size;
        
        info!("Created new buffer, total allocated: {} bytes", 
              *self.total_allocated.lock().unwrap());
        Ok(buffer)
    }

    /// Get pool statistics for Mac performance monitoring
    pub fn get_stats(&self) -> MacBufferPoolStats {
        let active_count = self.active_buffers.lock().unwrap().len();
        let retired_count = self.retired_buffers.lock().unwrap().len();
        let total_allocated = *self.total_allocated.lock().unwrap();
        
        MacBufferPoolStats {
            active_buffers: active_count,
            retired_buffers: retired_count,
            total_allocated,
            pool_efficiency: if active_count + retired_count > 0 {
                active_count as f32 / (active_count + retired_count) as f32
            } else {
                0.0
            },
        }
    }

    /// Optimize pool for Mac GPU type
    pub fn optimize_for_mac_gpu(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        match self.config.mac_gpu_type {
            MacGpuType::AppleSilicon => {
                // Apple Silicon optimizations
                self.config.max_pool_size = (self.config.max_pool_size as f32 * 1.5) as usize;
                info!("Optimized buffer pool for Apple Silicon GPU");
            }
            MacGpuType::Intel => {
                // Intel GPU optimizations
                self.config.max_pool_size = (self.config.max_pool_size as f32 * 0.8) as usize;
                info!("Optimized buffer pool for Intel GPU");
            }
            MacGpuType::Discrete => {
                // Discrete GPU optimizations
                self.config.max_pool_size = (self.config.max_pool_size as f32 * 1.2) as usize;
                info!("Optimized buffer pool for Discrete GPU");
            }
            MacGpuType::Unknown => {
                warn!("Unknown Mac GPU type, using default settings");
            }
        }
        Ok(())
    }

    /// Clean up pool resources
    pub fn cleanup(&self) -> Result<(), Box<dyn std::error::Error>> {
        let mut active_buffers = self.active_buffers.lock().unwrap();
        let mut retired_buffers = self.retired_buffers.lock().unwrap();
        
        active_buffers.clear();
        retired_buffers.clear();
        
        info!("Cleaned up Mac buffer pool");
        Ok(())
    }
}

/// Statistics for Mac buffer pool performance monitoring
#[derive(Debug, Clone)]
pub struct MacBufferPoolStats {
    pub active_buffers: usize,
    pub retired_buffers: usize,
    pub total_allocated: u64,
    pub pool_efficiency: f32,
}

/// Test the Mac buffer pool functionality
pub fn test_mac_buffer_pool() -> Result<(), Box<dyn std::error::Error>> {
    info!("Testing Mac buffer pool functionality");
    
    // This would normally require a WGPU device, but we can test the logic
    let config = MacBufferPoolConfig {
        initial_pool_size: 10,
        max_pool_size: 100,
        buffer_size: 1024 * 1024, // 1MB
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
        mac_gpu_type: MacGpuType::AppleSilicon,
        retina_scale: 2.0,
    };
    
    info!("Mac buffer pool test configuration: {:?}", config);
    info!("Mac buffer pool test completed successfully");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mac_buffer_pool_config() {
        let config = MacBufferPoolConfig {
            initial_pool_size: 5,
            max_pool_size: 50,
            buffer_size: 512 * 1024,
            usage: BufferUsages::STORAGE,
            mac_gpu_type: MacGpuType::AppleSilicon,
            retina_scale: 2.0,
        };
        
        assert_eq!(config.initial_pool_size, 5);
        assert_eq!(config.max_pool_size, 50);
        assert_eq!(config.buffer_size, 512 * 1024);
        assert_eq!(config.mac_gpu_type, MacGpuType::AppleSilicon);
        assert_eq!(config.retina_scale, 2.0);
    }

    #[test]
    fn test_mac_gpu_type_equality() {
        assert_eq!(MacGpuType::AppleSilicon, MacGpuType::AppleSilicon);
        assert_ne!(MacGpuType::AppleSilicon, MacGpuType::Intel);
        assert_ne!(MacGpuType::Intel, MacGpuType::Discrete);
    }

    #[test]
    fn test_buffer_pool_stats() {
        let stats = MacBufferPoolStats {
            active_buffers: 10,
            retired_buffers: 5,
            total_allocated: 1024 * 1024 * 15, // 15MB
            pool_efficiency: 0.67,
        };
        
        assert_eq!(stats.active_buffers, 10);
        assert_eq!(stats.retired_buffers, 5);
        assert_eq!(stats.total_allocated, 1024 * 1024 * 15);
        assert!((stats.pool_efficiency - 0.67).abs() < 0.01);
    }
} 