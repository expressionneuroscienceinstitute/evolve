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