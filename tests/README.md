# Testing Framework

This directory contains the testing infrastructure for the Madrona Escape Room project.

## Test Organization

- `cpp/` - C++ tests using GoogleTest framework
- `python/` - Python tests using pytest framework

## Test Scripts

### Quick Testing
- `quicktest.sh` - Runs both C++ and Python tests quickly (brief mode, no slow tests)

### Performance Testing
- `run_perf_test.sh` - Performance testing script
- `test_headless_loop.sh` - Headless testing loop

### Test Result Tracking
- `test_tracker.py` - **Individual test result tracker** - tracks every test result across commits

## Test Tracker Usage

The test tracker provides commit-level tracking of individual test results for regression analysis.

### Basic Usage
```bash
# Track tests for current commit
uv run python tests/test_tracker.py

# Dry run (don't save results)
uv run python tests/test_tracker.py --dry-run
```

### Output Files
- `individual-test-history.csv` - Every individual test result per commit with branch info
- `test-reports/` - Detailed test output logs per commit

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

## Test Data
- `test_recordings/` - Recorded test sessions for replay testing