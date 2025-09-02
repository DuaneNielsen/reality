# Performance Testing Guide

This guide covers the performance testing framework for the Madrona Escape Room, including how to run tests, interpret results, and maintain performance baselines.

## Quick Start

### Running Performance Tests

```bash
# Quick CPU test (manual mode)
uv run python tests/performance/run_perf.py --worlds 1024 --steps 1000

# GPU test with CUDA
uv run python tests/performance/run_perf.py --worlds 1024 --steps 1000 --cuda 0

# Test specific commit
uv run python tests/performance/run_perf.py --commit abc123 --worlds 2048 --steps 500

# Nightly mode - test all untested commits
uv run python tests/performance/run_perf.py --nightly

# Legacy: Detailed profiling with sim_bench.py
uv run python tests/performance/sim_bench.py \
    --num-worlds 2048 \
    --num-steps 500 \
    --gpu-id 0 \
    --check-baseline \
    --save-profile
```

## Performance Testing Framework

### Core Components

1. **tests/performance/run_perf.py** - Main performance testing script
   - **Manual mode**: Test specific commits with custom parameters
   - **Nightly mode**: Automated testing of all untested commits
   - Uses headless executable for cross-branch compatibility
   - Saves comprehensive raw data for each test run
   - Compares against baseline thresholds

2. **tests/performance/sim_bench.py** - Advanced benchmark script
   - Detailed Python-based profiling with pyinstrument
   - PyTorch tensor operations timing
   - Interactive profiling analysis
   - Used for detailed performance investigation

3. **tests/performance/performance_baselines.json** - Performance thresholds
   - Defines minimum and warning FPS levels
   - Separate baselines for CPU and GPU configurations
   - Easy to update when hardware or optimizations change

4. **tests/performance/perf_results/** - Performance data storage
   - `runs/{commit_hash}/` - Individual test run data
   - `history.csv` - Summary performance tracking
   - `latest.txt` - Morning reports for nightly runs

## Command-Line Options

### run_perf.py Arguments (Primary Tool)

| Argument | Default | Description |
|----------|---------|-------------|
| `--nightly` | False | Nightly mode: test all untested commits |
| `--commit` | HEAD | Test specific commit (default: current HEAD) |
| `--worlds` | 1024 | Number of parallel worlds to simulate |
| `--steps` | 1000 | Number of simulation steps to run |
| `--cuda` | -1 | GPU device ID for CUDA mode (-1 for CPU) |

### sim_bench.py Arguments (Advanced Profiling)

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-worlds` | 1024 | Number of parallel worlds to simulate |
| `--num-steps` | 1000 | Number of simulation steps to run |
| `--gpu-id` | -1 | GPU device ID (-1 for CPU mode) |
| `--profile-renderer` | False | Enable batch renderer profiling |
| `--check-baseline` | False | Compare performance against baselines |
| `--save-profile` | False | Save profiling data to files |
| `--output-dir` | Auto | Directory for saving results |

### Exit Codes

- **0**: Performance meets or exceeds warning threshold (PASS)
- **1**: Performance below minimum threshold (FAIL)
- **2**: Performance between minimum and warning thresholds (WARN)

## Understanding Results

### Performance Metrics

```
Performance Summary:
============================================================
FPS: 500,720
World FPS: 512,737,280

Baseline Check:
============================================================
✓ PASS: 500,720 FPS >= 500,000 FPS (warning threshold)
```

- **FPS**: Total frames per second across all worlds
- **World FPS**: FPS × number of worlds (total entity updates/sec)
- **Baseline Status**: PASS ✓ / WARN ⚠ / FAIL ✗

### Output Files

#### run_perf.py Output Structure

```
tests/performance/perf_results/
├── runs/{commit_hash}/
│   ├── benchmark_output.txt    # Raw headless executable output
│   ├── build_log.txt          # Compilation logs
│   ├── metadata.json          # Structured test results
│   └── profile_*.html         # Performance profiles (if available)
├── history.csv                # Summary performance tracking
├── latest.txt                 # Morning reports for nightly runs
└── last_tested_commit.txt     # State tracking for nightly mode
```

#### sim_bench.py Output Structure (Advanced Profiling)

```
tests/performance/perf_results/20250112_143022/
├── cpu/
│   ├── profile.html      # Interactive profiler visualization
│   ├── profile.json      # Raw profiling data for analysis
│   └── results.json      # Performance metrics and configuration
└── gpu/
    └── [same structure]
```

#### metadata.json Structure (run_perf.py)

```json
{
  "timestamp": "2025-01-12T14:30:22",
  "commit": "eb3ab236482b0bcd6380b3b36e2357c133b68a84",
  "message": "Enhance performance testing system",
  "cpu_fps": 1101246,
  "cpu_status": "PASS",
  "command": "./build/headless -n 1024 -s 1000",
  "configuration": {
    "num_worlds": 1024,
    "num_steps": 1000,
    "device": "CPU",
    "gpu_id": null
  }
}
```

#### results.json Structure (sim_bench.py)

```json
{
  "timestamp": "2025-01-12T14:30:22",
  "config": {
    "num_worlds": 1024,
    "num_steps": 1000,
    "device": "CPU",
    "gpu_id": -1,
    "renderer": false
  },
  "performance": {
    "fps": 500720,
    "total_frames": 1024000,
    "total_time": 2.045,
    "frame_time_ms": 2.045
  },
  "baseline_check": {
    "status": "PASS",
    "min_fps": 450000,
    "warn_fps": 500000
  }
}
```

## Profiling Analysis

### Using pyinstrument Profiles

The framework uses [pyinstrument](https://github.com/joerick/pyinstrument) for profiling:

1. **HTML Profile** (`profile.html`)
   - Open in browser for interactive exploration
   - Shows call tree with time percentages
   - Identifies performance bottlenecks

2. **JSON Profile** (`profile.json`)
   - Machine-readable format
   - Can be loaded for custom analysis
   - Useful for comparing profiles over time

### Common Performance Bottlenecks

Look for these patterns in profiles:

- **High `SimManager.step` time**: Core simulation overhead
- **High `Tensor.to_torch/to_numpy` time**: Data transfer overhead
- **High `randint_like` time**: Action generation overhead
- **Unbalanced system times**: Specific ECS systems taking too long

## Updating Baselines

### When to Update

Update baselines when:
- Hardware changes (new GPU, CPU upgrade)
- Major optimizations are merged
- Switching between debug/release builds
- Changing default configurations

### How to Update

1. Run benchmarks multiple times to ensure consistency:
   ```bash
   # Using run_perf.py (recommended)
   for i in {1..5}; do
     uv run python tests/performance/run_perf.py --worlds 1024 --steps 1000
   done
   
   # Using sim_bench.py (for detailed profiling)
   for i in {1..5}; do
     uv run python tests/performance/sim_bench.py --num-worlds 1024 --num-steps 1000
   done
   ```

2. Calculate average FPS from runs

3. Edit `tests/performance/performance_baselines.json`:
   ```json
   {
     "cpu_1024": {
       "min_fps": 450000,    // Set to 90% of average
       "warn_fps": 500000,   // Set to 95% of average
       "description": "CPU benchmark with 1024 worlds, 1000 steps"
     }
   }
   ```

4. Commit the updated baselines with explanation

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Performance Tests
on: [push, pull_request]

jobs:
  perf-test:
    runs-on: [self-hosted, gpu]  # Requires GPU runner
    steps:
      - uses: actions/checkout@v3
      
      - name: Build
        run: |
          mkdir build
          cmake -B build
          make -C build -j8
      
      - name: Run Performance Tests
        run: ./tests/run_perf_test.sh
      
      - name: Upload Results
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: performance-results
          path: tests/performance/perf_results/
```

### Local Pre-commit Hook

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Run quick CPU performance check
uv run python tests/performance/run_perf.py \
    --worlds 100 \
    --steps 100

# Check if test passed by looking at output
if echo "$result" | grep -q "Result: FAIL"; then
    echo "Performance regression detected! Check before committing."
    exit 1
fi
```

## Troubleshooting

### Common Issues

1. **"No baseline for configuration"**
   - Add new baseline to `performance_baselines.json`
   - Or use standard configurations (1024 worlds for CPU, 8192 for GPU)
   - `run_perf.py` uses simplified baseline checking

2. **Inconsistent Results**
   - Ensure no other heavy processes running
   - Use `--profile-renderer` to check if rendering affects performance
   - Run multiple times and average results

3. **GPU Tests Failing**
   - Check CUDA installation: `nvidia-smi`
   - Verify GPU memory available
   - Try smaller world count

4. **Build or Import Errors**
   - `run_perf.py` handles build failures gracefully and reports them
   - For `sim_bench.py` import errors: rebuild and reinstall
   - Rebuild: `make -C build -j8`
   - Reinstall package: `uv pip install -e .`

### Performance Debugging

```bash
# Quick performance test
uv run python tests/performance/run_perf.py --worlds 10 --steps 100

# View saved raw data
cat tests/performance/perf_results/runs/{commit_hash}/benchmark_output.txt
cat tests/performance/perf_results/runs/{commit_hash}/metadata.json

# Generate detailed profile for analysis
uv run python tests/performance/sim_bench.py \
    --num-worlds 10 \
    --num-steps 100 \
    --save-profile \
    --output-dir debug_profile

# Open profile in browser
firefox debug_profile/cpu/profile.html

# Compare two profiles
diff debug_profile/cpu/results.json baseline_profile/cpu/results.json
```

## Best Practices

1. **Choose the Right Tool**:
   - Use `run_perf.py` for routine performance testing and monitoring
   - Use `sim_bench.py` for detailed profiling and performance investigation
   - Use nightly mode for automated regression detection

2. **Consistent Environment**: 
   - Close unnecessary applications
   - Disable CPU frequency scaling if possible
   - Use consistent GPU power settings

3. **Statistical Significance**:
   - Run tests multiple times
   - Look for > 5% performance changes
   - Consider variance in results

4. **Profile Regularly**:
   - Use `run_perf.py` for routine monitoring
   - Use `sim_bench.py` after major changes for detailed analysis
   - Save profiles for historical comparison
   - Look for unexpected bottlenecks

## Advanced Usage

### Custom Benchmarks

Create specialized benchmarks for specific scenarios:

```python
# benchmark_complex_level.py
import madrona_escape_room
from tests.performance.sim_bench import run_benchmark

# Load complex level
with open("levels/complex.lvl") as f:
    level_data = f.read()

# Run with custom configuration
results = run_benchmark(
    num_worlds=512,
    num_steps=2000,
    level_ascii=level_data,
    check_baseline=True
)
```

### Comparative Analysis

```bash
# Compare branches using run_perf.py
git checkout main
uv run python tests/performance/run_perf.py --worlds 1024 --steps 1000
main_fps=$(grep "FPS:" tests/performance/perf_results/runs/*/metadata.json | tail -1)

git checkout feature-branch  
uv run python tests/performance/run_perf.py --worlds 1024 --steps 1000
feature_fps=$(grep "FPS:" tests/performance/perf_results/runs/*/metadata.json | tail -1)

echo "Main: $main_fps"
echo "Feature: $feature_fps"

# Compare using sim_bench.py for detailed analysis
git checkout main
./tests/run_perf_test.sh
mv tests/performance/perf_results tests/performance/perf_results_main

git checkout feature-branch  
./tests/run_perf_test.sh
mv tests/performance/perf_results tests/performance/perf_results_feature

# Analyze differences
python tests/performance/compare_perf.py \
    tests/performance/perf_results_main \
    tests/performance/perf_results_feature
```

## Usage Examples

### Manual Testing

```bash
# Test current commit with default settings
uv run python tests/performance/run_perf.py

# Test specific commit with custom parameters
uv run python tests/performance/run_perf.py --commit abc123 --worlds 2048 --steps 500

# GPU performance test
uv run python tests/performance/run_perf.py --cuda 0 --worlds 8192 --steps 1000

# Quick smoke test
uv run python tests/performance/run_perf.py --worlds 10 --steps 50
```

### Nightly Testing

```bash
# Run nightly performance testing (tests all untested commits)
uv run python tests/performance/run_perf.py --nightly

# View morning report
cat tests/performance/perf_results/latest.txt

# Check performance history
cat tests/performance/perf_results/history.csv
```

## Related Documentation

- [Testing Guide](TESTING_GUIDE.md) - General testing practices
- [run_perf.py](run_perf.py) - Main performance testing script
- [sim_bench.py](sim_bench.py) - Advanced benchmark script
- [Headless Mode](../../deployment/headless/HEADLESS_MODE.md) - Running without graphics