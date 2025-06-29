use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use tempfile::TempDir;

#[tokio::test]
async fn test_cli_help() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.arg("--help");
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Command-line interface for the Evolve universe simulation engine"));
}

#[tokio::test]
async fn test_status_command() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["status"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("UNIVERSE SIMULATION STATUS"));
}

#[tokio::test]
async fn test_status_command_json_format() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["status", "--format", "json"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("tick"));
}

#[tokio::test]
async fn test_status_command_compact_format() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["status", "--format", "compact"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("UNIVERSE"));
}

#[tokio::test]
async fn test_list_planets_command() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["list-planets"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("PLANETS"));
}

#[tokio::test]
async fn test_list_planets_with_class_filter() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["list-planets", "--class", "E"]);
    
    cmd.assert()
        .success();
}

#[tokio::test]
async fn test_list_planets_invalid_class() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["list-planets", "--class", "X"]);
    
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Invalid planet class"));
}

#[tokio::test]
async fn test_list_planets_json_format() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["list-planets", "--format", "json"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("["));
}

#[tokio::test]
async fn test_list_planets_csv_format() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["list-planets", "--format", "csv"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Name,Class,Habitability"));
}

#[tokio::test]
async fn test_list_planets_with_limit() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["list-planets", "--limit", "5"]);
    
    cmd.assert()
        .success();
}

#[tokio::test]
async fn test_list_planets_with_habitability_filter() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["list-planets", "--min-habitability", "0.5"]);
    
    cmd.assert()
        .success();
}

#[tokio::test]
async fn test_inspect_planet_command() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["inspect", "planet", "test-planet-1"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("PLANET INSPECTION"));
}

#[tokio::test]
async fn test_inspect_planet_with_environment_flag() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["inspect", "planet", "test-planet-1", "--environment"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("ENVIRONMENT PROFILE"));
}

#[tokio::test]
async fn test_inspect_planet_with_lineages_flag() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["inspect", "planet", "test-planet-1", "--lineages"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("ACTIVE LINEAGES"));
}

#[tokio::test]
async fn test_inspect_planet_with_energy_flag() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["inspect", "planet", "test-planet-1", "--energy"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("ENERGY BUDGET"));
}

#[tokio::test]
async fn test_inspect_planet_with_resources_flag() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["inspect", "planet", "test-planet-1", "--resources"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("RESOURCE COMPOSITION"));
}

#[tokio::test]
async fn test_inspect_lineage_command() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["inspect", "lineage", "test-lineage-1"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("LINEAGE INSPECTION"));
}

#[tokio::test]
async fn test_inspect_lineage_with_fitness_flag() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["inspect", "lineage", "test-lineage-1", "--fitness"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("FITNESS HISTORY"));
}

#[tokio::test]
async fn test_inspect_lineage_with_code_flag() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["inspect", "lineage", "test-lineage-1", "--code"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("CODE EVOLUTION"));
}

#[tokio::test]
async fn test_inspect_lineage_with_genealogy_flag() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["inspect", "lineage", "test-lineage-1", "--genealogy"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("GENEALOGY"));
}

#[tokio::test]
async fn test_inspect_system_command() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["inspect", "system", "test-system-1"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("STAR SYSTEM INSPECTION"));
}

#[tokio::test]
async fn test_inspect_system_with_orbits_flag() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["inspect", "system", "test-system-1", "--orbits"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("ORBITAL MECHANICS"));
}

#[tokio::test]
async fn test_inspect_system_with_stellar_flag() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["inspect", "system", "test-system-1", "--stellar"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("STELLAR PROPERTIES"));
}

#[tokio::test]
async fn test_inspect_performance_command() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["inspect", "performance"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("PERFORMANCE INSPECTION"));
}

#[tokio::test]
async fn test_inspect_performance_cpu_metric() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["inspect", "performance", "--metric", "cpu"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("CPU METRICS"));
}

#[tokio::test]
async fn test_inspect_performance_memory_metric() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["inspect", "performance", "--metric", "memory"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("MEMORY METRICS"));
}

#[tokio::test]
async fn test_map_command_stars() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["map", "3", "--map-type", "stars"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("UNIVERSE MAP: STARS"));
}

#[tokio::test]
async fn test_map_command_entropy() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["map", "2", "--map-type", "entropy"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("UNIVERSE MAP: ENTROPY"));
}

#[tokio::test]
async fn test_map_command_temperature() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["map", "4", "--map-type", "temperature"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("UNIVERSE MAP: TEMPERATURE"));
}

#[tokio::test]
async fn test_map_command_with_dimensions() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["map", "2", "--width", "60", "--height", "30"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("LEGEND"));
}

#[tokio::test]
async fn test_map_command_invalid_type() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["map", "2", "--map-type", "invalid"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Unknown map type"));
}

#[tokio::test]
async fn test_snapshot_command_toml() {
    let temp_dir = TempDir::new().unwrap();
    let snapshot_path = temp_dir.path().join("test_snapshot");
    
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["snapshot", snapshot_path.to_str().unwrap(), "--format", "toml"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("SNAPSHOT SUMMARY"));
    
    // Verify file was created
    let expected_file = format!("{}.toml", snapshot_path.to_str().unwrap());
    assert!(std::path::Path::new(&expected_file).exists());
}

#[tokio::test]
async fn test_snapshot_command_json() {
    let temp_dir = TempDir::new().unwrap();
    let snapshot_path = temp_dir.path().join("test_snapshot_json");
    
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["snapshot", snapshot_path.to_str().unwrap(), "--format", "json"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Snapshot successfully created!"));
    
    let expected_file = format!("{}.json", snapshot_path.to_str().unwrap());
    assert!(std::path::Path::new(&expected_file).exists());
}

#[tokio::test]
async fn test_snapshot_command_yaml() {
    let temp_dir = TempDir::new().unwrap();
    let snapshot_path = temp_dir.path().join("test_snapshot_yaml");
    
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["snapshot", snapshot_path.to_str().unwrap(), "--format", "yaml"]);
    
    cmd.assert()
        .success();
    
    let expected_file = format!("{}.yaml", snapshot_path.to_str().unwrap());
    assert!(std::path::Path::new(&expected_file).exists());
}

#[tokio::test]
async fn test_snapshot_command_compressed() {
    let temp_dir = TempDir::new().unwrap();
    let snapshot_path = temp_dir.path().join("test_snapshot_compressed");
    
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["snapshot", snapshot_path.to_str().unwrap(), "--compress"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Compressed: Yes"));
    
    let expected_file = format!("{}.toml.gz", snapshot_path.to_str().unwrap());
    assert!(std::path::Path::new(&expected_file).exists());
}

#[tokio::test]
async fn test_snapshot_command_full_state() {
    let temp_dir = TempDir::new().unwrap();
    let snapshot_path = temp_dir.path().join("test_snapshot_full");
    
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["snapshot", snapshot_path.to_str().unwrap(), "--full"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Full State"));
}

#[tokio::test]
async fn test_snapshot_command_invalid_format() {
    let temp_dir = TempDir::new().unwrap();
    let snapshot_path = temp_dir.path().join("test_snapshot_invalid");
    
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["snapshot", snapshot_path.to_str().unwrap(), "--format", "invalid"]);
    
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Unsupported format"));
}

#[tokio::test]
async fn test_stress_test_cpu() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["stress-test", "cpu", "--duration", "2", "--intensity", "3"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("CPU STRESS TEST"));
}

#[tokio::test]
async fn test_stress_test_memory() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["stress-test", "memory", "--duration", "2", "--intensity", "2"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("MEMORY STRESS TEST"));
}

#[tokio::test]
async fn test_stress_test_io() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["stress-test", "io", "--duration", "2", "--intensity", "2"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("I/O STRESS TEST"));
}

#[tokio::test]
async fn test_stress_test_gpu() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["stress-test", "gpu", "--duration", "2", "--intensity", "3"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("GPU STRESS TEST"));
}

#[tokio::test]
async fn test_stress_test_invalid_type() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["stress-test", "invalid", "--duration", "1"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Unknown test type"));
}

#[tokio::test]
async fn test_resources_queue_command() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["resources", "queue"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Resource request queue"));
}

#[tokio::test]
async fn test_resources_usage_command() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["resources", "usage"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Resource usage statistics"));
}

#[tokio::test]
async fn test_resources_grant_command() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["resources", "grant", "test-request-123"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Granting resources"));
}

#[tokio::test]
async fn test_oracle_inbox_command() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["oracle", "inbox"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Oracle inbox"));
}

#[tokio::test]
async fn test_oracle_inbox_unread_only() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["oracle", "inbox", "--unread"]);
    
    cmd.assert()
        .success();
}

#[tokio::test]
async fn test_oracle_inbox_with_limit() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["oracle", "inbox", "--limit", "10"]);
    
    cmd.assert()
        .success();
}

#[tokio::test]
async fn test_oracle_reply_command() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["oracle", "reply", "petition-123", "ack"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Replying to petition"));
}

#[tokio::test]
async fn test_oracle_reply_with_message() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["oracle", "reply", "petition-123", "message", "--message", "Hello lineage!"]);
    
    cmd.assert()
        .success();
}

#[tokio::test]
async fn test_oracle_stats_command() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["oracle", "stats"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Oracle communication stats"));
}

#[tokio::test]
async fn test_divine_commands_require_godmode() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["divine", "create-body", "planet", "1.0"]);
    
    cmd.assert()
        .failure()
        .stderr(predicate::str::contains("Divine commands require --godmode flag"));
}

#[tokio::test]
async fn test_divine_create_body_with_godmode() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["--godmode", "divine", "create-body", "planet", "1.0"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Creating planet"));
}

#[tokio::test]
async fn test_divine_delete_body_with_force() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["--godmode", "divine", "delete-body", "test-body-123", "--force"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Deleting body"));
}

#[tokio::test]
async fn test_divine_miracle_command() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["--godmode", "divine", "miracle", "planet-123", "rain"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Performing rain miracle"));
}

#[tokio::test]
async fn test_divine_time_warp_command() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["--godmode", "divine", "time-warp", "2.0"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Setting time warp to 2x"));
}

#[tokio::test]
async fn test_divine_spawn_lineage_command() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["--godmode", "divine", "spawn-lineage", "sha256:abc123", "planet-456"]);
    
    cmd.assert()
        .success()
        .stdout(predicate::str::contains("Spawning lineage"));
}

#[tokio::test]
async fn test_verbose_logging() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["--verbose", "status"]);
    
    cmd.assert()
        .success();
}

#[tokio::test]
async fn test_custom_socket_path() {
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["--socket", "/tmp/custom_universe.sock", "status"]);
    
    cmd.assert()
        .success(); // Should work with mock data even if socket doesn't exist
}

#[tokio::test]
async fn test_config_file_option() {
    let temp_dir = TempDir::new().unwrap();
    let config_path = temp_dir.path().join("test_config.toml");
    fs::write(&config_path, "[universe]\nsocket = \"/tmp/test.sock\"").unwrap();
    
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["--config", config_path.to_str().unwrap(), "status"]);
    
    cmd.assert()
        .success();
}

// Integration test for command chaining and complex scenarios
#[tokio::test]
async fn test_full_workflow_simulation() {
    // Test a complete workflow: status -> list planets -> inspect planet -> snapshot
    
    // 1. Check status
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["status", "--format", "compact"]);
    cmd.assert().success();
    
    // 2. List planets with filter
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["list-planets", "--class", "E", "--limit", "5"]);
    cmd.assert().success();
    
    // 3. Inspect a planet
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["inspect", "planet", "earth-alpha", "--environment", "--energy"]);
    cmd.assert().success();
    
    // 4. Create a snapshot
    let temp_dir = TempDir::new().unwrap();
    let snapshot_path = temp_dir.path().join("workflow_test");
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["snapshot", snapshot_path.to_str().unwrap(), "--format", "json", "--compress"]);
    cmd.assert().success();
    
    // Verify snapshot file exists
    let expected_file = format!("{}.json.gz", snapshot_path.to_str().unwrap());
    assert!(std::path::Path::new(&expected_file).exists());
}

#[tokio::test]
async fn test_error_handling_and_recovery() {
    // Test that CLI handles various error conditions gracefully
    
    // Invalid command
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["invalid-command"]);
    cmd.assert().failure();
    
    // Invalid arguments
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["status", "--invalid-arg"]);
    cmd.assert().failure();
    
    // Missing required arguments
    let mut cmd = Command::cargo_bin("universectl").unwrap();
    cmd.args(&["inspect"]);
    cmd.assert().failure();
}

#[tokio::test]
async fn test_concurrent_commands() {
    // Test that multiple CLI commands can run concurrently
    use std::sync::Arc;
    use tokio::sync::Semaphore;
    
    let semaphore = Arc::new(Semaphore::new(4));
    let mut handles = vec![];
    
    for i in 0..4 {
        let permit = semaphore.clone().acquire_owned().await.unwrap();
        let handle = tokio::spawn(async move {
            let _permit = permit;
            let mut cmd = Command::cargo_bin("universectl").unwrap();
            cmd.args(&["status", "--format", "compact"]);
            cmd.assert().success();
        });
        handles.push(handle);
    }
    
    // Wait for all commands to complete
    for handle in handles {
        handle.await.unwrap();
    }
}