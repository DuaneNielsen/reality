# Testing Framework

This directory contains the testing infrastructure for the Madrona Escape Room project.

## Test Organization

- `cpp/` - C++ tests using GoogleTest framework
- `python/` - Python tests using pytest framework
- `test_recordings/` - Recorded test sessions for replay testing

## Main Testing Entrypoint

**`test_tracker.py`** - The primary test runner that executes both C++ and Python tests, tracks individual test results across commits, and provides regression analysis.

## Additional Test Scripts

- `run_perf_test.sh` - Performance testing script for benchmarking
- `test_headless_loop.sh` - Headless simulation testing loop

## Test Tracker Usage

The test tracker is the main entrypoint for running tests. It automatically runs both C++ and Python tests, tracks results, and identifies regressions.

### Basic Usage
```bash
# Track tests for current commit
uv run python tests/test_tracker.py

# Dry run (don't save results)
uv run python tests/test_tracker.py --dry-run

# Run full test suite (includes GPU and slow tests)
uv run python tests/test_tracker.py --full
```

### Command-Line Options

| Option | Description |
|--------|-------------|
| `--dry-run` | Run tests but don't save results to CSV. Useful for testing or checking current state without recording. |
| `--full` | Run the full test suite including GPU tests and slow tests. Without this flag, only standard CPU tests are run. |
| `--commit HASH` | Analyze a specific commit hash instead of the current HEAD. |

### Test Suite Differences

**Standard Mode** (default):
- Runs `./build/mad_escape_tests` (C++ CPU tests)
- Runs Python tests with `-m not slow` (excludes slow tests)
- Suitable for quick iteration during development

**Full Mode** (`--full`):
- Runs all C++ test executables:
  - `./build/mad_escape_tests`
  - `./build/mad_escape_gpu_tests` 
  - `./build/mad_escape_gpu_stress_tests`
- Runs Python tests including slow tests but excluding skipped tests
- Sets `ALLOW_GPU_TESTS_IN_SUITE=1` environment variable
- Suitable for comprehensive testing before merging

### Output Files
- `individual-test-history.csv` - Every individual test result per commit with branch info

### Finding Test Regressions
```bash
# Find when a specific test started failing
grep "test_collision_behavior_differences" individual-test-history.csv

# See all tests broken by a specific commit
grep "abc123.*FAIL" individual-test-history.csv

# Track a test's history over multiple commits  
grep "TestName" individual-test-history.csv | tail -10

# Filter by branch
grep "main.*test_collision_behavior_differences" individual-test-history.csv
```

### Git Hook Integration
To automatically track tests on every commit:
```bash
# Install as post-commit hook
echo '#!/bin/bash\nuv run python tests/test_tracker.py' > .git/hooks/post-commit
chmod +x .git/hooks/post-commit
```

### Features
- **Individual test tracking** - Records every C++ and Python test result
- **Regression detection** - Shows newly broken vs pre-existing failures  
- **Commit attribution** - Links test failures to specific commits
- **Historical analysis** - Track test status changes over time

### Example Output
```
🧪 Testing commit a04fbb4: fix: adjust test_trajectory_logging_to_file

❌ Test Results for commit a04fbb4:
   C++: PASS (168 passed, 0 failed)
   Python: FAIL (167 passed, 33 failed, 7 skipped)
   Overall: FAIL

📋 Failed Python tests (27 total):
     - tests/python/test_collision_termination.py::TestCollisionTermination::test_north_collision_terminates
     - tests/python/test_collision_termination.py::TestCollisionTermination::test_west_collision_terminates
     [... all failed tests listed ...]

   ✅ No newly broken tests (all failures were pre-existing)

💡 To find when a specific test started failing:
   grep 'test_name' individual-test-history.csv
```

## Output Files

- `individual-test-history.csv` - Complete test history database with per-test results for every commit