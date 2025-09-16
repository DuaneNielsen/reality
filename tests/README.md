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

## Test Requirement Traceability

This project uses pytest markers to link tests to specification documents, enabling requirement traceability. When tests fail in verbose mode, the relevant specification sections are automatically displayed.

### Quick Start

#### 1. Mark Tests with Specifications

Add `@pytest.mark.spec()` decorators to link tests to spec documents:

```python
import pytest

@pytest.mark.spec("docs/specs/sim.md", "rewardSystem")
def test_step_zero_reward_is_zero(cpu_manager):
    """Test that step 0 reward is always 0"""
    # test implementation
```

#### 2. Generate Traceability Report

Run the report generator to see which specs are tested:

```bash
uv run python tests/traceability_report.py
```

This creates `tests/traceability_report.md` showing:
- All tests linked to specifications
- Coverage summary for each system
- Missing test coverage areas

#### 3. Run Tests for Specific Specifications

Filter tests by specification markers:

```bash
# Run all tests with spec markers
uv run pytest -m "spec"

# Run tests for a specific system
uv run pytest -k "reward" -m "spec"

# Show which tests would run without executing
uv run pytest -m "spec" --co -q

# Run with spec display on failures (verbose mode)
uv run pytest -m "spec" -v
```

### Specification Display on Failure

When tests fail in **verbose mode** (`-v`), the relevant specification section is automatically displayed:

```
FAILED test_reward.py::test_step_zero - AssertionError
------------------------------- 📋 Specification --------------------------------
📋 Specification: rewardSystem
────────────────────────────────────────────────────────────
- **Step 0**: Always 0.0 reward (no reward on reset)
- **Forward only**: Only Y-axis forward movement gives rewards
...
────────────────────────────────────────────────────────────
```

This helps developers immediately see what the expected behavior should be according to the specification.

### Marker Syntax

#### Basic Usage
```python
@pytest.mark.spec("path/to/spec.md", "sectionName")
```

- First argument: Path to specification document
- Second argument: Section or system name in the spec

#### Multiple Markers
Tests can have multiple markers for different aspects:

```python
@pytest.mark.spec("docs/specs/sim.md", "rewardSystem")
@pytest.mark.requirement("REQ-001")
@pytest.mark.slow
def test_complex_reward_scenario():
    pass
```

### Benefits

1. **Verification**: Ensures tests match documented specifications
2. **Coverage**: Identifies which specs lack test coverage
3. **Navigation**: Links tests directly to spec sections
4. **Filtering**: Run only tests for specific systems
5. **Documentation**: Self-documenting test purpose
6. **Failure Context**: Shows relevant specs when tests fail (in verbose mode)

### Current Coverage

Run `uv run python tests/traceability_report.py` to see current coverage.

Systems with test coverage:
- ✅ movementSystem
- ✅ agentCollisionSystem
- ✅ stepTrackerSystem
- ✅ rewardSystem
- ✅ resetSystem
- ✅ compassSystem

Systems needing tests:
- ❌ agentZeroVelSystem
- ❌ initProgressAfterReset
- ❌ collectObservationsSystem
- ❌ lidarSystem

### Adding New Tests

When adding tests for a new system:

1. Read the specification in `docs/specs/sim.md`
2. Write tests that verify the documented behavior
3. Add `@pytest.mark.spec("docs/specs/sim.md", "systemName")`
4. Run the traceability report to verify coverage

### Future Enhancements

Potential improvements to the traceability system:

- Generate HTML reports with clickable links
- Integrate with CI to track coverage trends
- Add requirement IDs for formal requirement tracking
- Export traceability matrix to CSV/Excel
- Validate that spec documents exist
- Parse spec documents to auto-detect sections