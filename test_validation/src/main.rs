mod basic_integration_test;

use basic_integration_test::{BasicIntegrationTestManager, BasicIntegrationTestConfig};
use std::time::Duration;

#[tokio::main]
async fn main() {
    println!("=== EVOLUTION Universe Simulation - Integration Testing ===");
    println!("Based on TestRail's integration testing best practices");
    println!("https://www.testrail.com/blog/integration-testing/");
    println!();

    // Configure integration tests
    let config = BasicIntegrationTestConfig {
        test_duration: Duration::from_secs(5), // Shorter for demo
        max_iterations: 100,
        validation_thresholds: basic_integration_test::BasicValidationThresholds {
            max_execution_time: Duration::from_secs(10),
            min_success_rate: 0.95,
            max_memory_usage_mb: 512.0,
        },
    };

    let mut test_manager = BasicIntegrationTestManager::new(config);

    println!("Running basic integration test suite...");
    println!("This validates core component interactions without requiring full compilation.");
    println!();

    // Run the integration test suite
    let results = test_manager.run_basic_integration_suite().await;

    // Display results
    println!("=== Integration Test Results ===");
    println!();

    let mut total_tests = 0;
    let mut passed_tests = 0;
    let mut total_duration = Duration::from_nanos(0);

    for result in &results {
        total_tests += 1;
        total_duration += result.duration;

        let status = if result.success { "✅ PASS" } else { "❌ FAIL" };
        println!("{} - {}", status, result.test_name);
        println!("  Duration: {:?}", result.duration);
        println!("  Success Rate: {:.2}%", result.success_rate * 100.0);
        println!("  Iterations: {}", result.iterations_completed);
        println!("  Memory Usage: {:.2} MB", result.memory_usage_mb);
        println!("  Operations/sec: {:.0}", result.performance_metrics.operations_per_second);

        if !result.errors.is_empty() {
            println!("  Errors:");
            for error in &result.errors {
                println!("    - {}", error);
            }
        }

        if !result.warnings.is_empty() {
            println!("  Warnings:");
            for warning in &result.warnings {
                println!("    - {}", warning);
            }
        }

        if result.success {
            passed_tests += 1;
        }

        println!();
    }

    // Summary
    println!("=== Summary ===");
    println!("Total Tests: {}", total_tests);
    println!("Passed: {}", passed_tests);
    println!("Failed: {}", total_tests - passed_tests);
    println!("Success Rate: {:.1}%", (passed_tests as f64 / total_tests as f64) * 100.0);
    println!("Total Duration: {:?}", total_duration);
    println!();

    if passed_tests == total_tests {
        println!("🎉 All integration tests passed!");
        println!("The EVOLUTION codebase demonstrates solid component integration.");
    } else {
        println!("⚠️  Some integration tests failed.");
        println!("Review the errors above to identify integration issues.");
    }

    println!();
    println!("=== Integration Testing Strategy ===");
    println!("This basic integration test validates:");
    println!("1. Core Physics Integration - Basic physics calculations");
    println!("2. Data Flow Integration - Component data exchange");
    println!("3. Performance Integration - Real-time performance requirements");
    println!("4. Memory Management Integration - Memory usage patterns");
    println!("5. Error Handling Integration - Graceful error recovery");
    println!();
    println!("For full integration testing with the complete EVOLUTION codebase,");
    println!("the comprehensive framework in physics_engine/tests/integration_test_framework.rs");
    println!("provides detailed testing of physics-engine ↔ universe-simulation interactions,");
    println!("agent-evolution ↔ physics-engine coupling, and visualization ↔ data pipeline integration.");
    println!();
    println!("=== Current Status ===");
    println!("The physics engine has compilation issues that need to be resolved");
    println!("before full integration testing can be performed. The basic integration");
    println!("test framework is ready and demonstrates the testing approach.");
} 