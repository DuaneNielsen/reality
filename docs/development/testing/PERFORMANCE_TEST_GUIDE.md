# Performance Testing Guide

This guide covers the performance testing framework for the Madrona Escape Room, including how to run tests, interpret results, and maintain performance baselines.

## Quick Start

### Running Performance Tests

```bash
# Quick CPU test
uv run python scripts/sim_bench.py --check-baseline

# Full test suite (CPU + GPU)
./tests/run_perf_test.sh

# Custom configuration with profiling
uv run python scripts/sim_bench.py \
    --num-worlds 2048 \
    --num-steps 500 \
    --gpu-id 0 \
    --check-baseline \
    --save-profile
```

## Performance Testing Framework

### Core Components

1. **scripts/sim_bench.py** - Enhanced benchmark script
   - Measures FPS (frames per second) for simulation steps
   - Compares against baseline thresholds
   - Saves detailed profiling data
   - Returns appropriate exit codes

2. **scripts/performance_baselines.json** - Performance thresholds
   - Defines minimum and warning FPS levels
   - Separate baselines for CPU and GPU configurations
   - Easy to update when hardware or optimizations change

3. **tests/run_perf_test.sh** - Automated test runner
   - Runs standard CPU and GPU benchmarks
   - Organizes results by timestamp
   - Provides summary pass/fail status

## Command-Line Options

### sim_bench.py Arguments

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

When using `--save-profile`, the following files are created:

```
build/perf_results/20250112_143022/
├── cpu/
│   ├── profile.html      # Interactive profiler visualization
│   ├── profile.json      # Raw profiling data for analysis
│   └── results.json      # Performance metrics and configuration
└── gpu/
    └── [same structure]
```

#### results.json Structure

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
   for i in {1..5}; do
     uv run python scripts/sim_bench.py --num-worlds 1024 --num-steps 1000
   done
   ```

2. Calculate average FPS from runs

3. Edit `scripts/performance_baselines.json`:
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
          path: build/perf_results/
```

### Local Pre-commit Hook

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Run quick CPU performance check
uv run python scripts/sim_bench.py \
    --num-worlds 100 \
    --num-steps 100 \
    --check-baseline

if [ $? -ne 0 ]; then
    echo "Performance regression detected! Check before committing."
    exit 1
fi
```

## Troubleshooting

### Common Issues

1. **"No baseline for configuration"**
   - Add new baseline to `performance_baselines.json`
   - Or use standard configurations (1024 CPU, 8192 GPU)

2. **Inconsistent Results**
   - Ensure no other heavy processes running
   - Use `--profile-renderer` to check if rendering affects performance
   - Run multiple times and average results

3. **GPU Tests Failing**
   - Check CUDA installation: `nvidia-smi`
   - Verify GPU memory available
   - Try smaller world count

4. **Import Errors**
   - Rebuild: `make -C build -j8`
   - Reinstall package: `uv pip install -e .`

### Performance Debugging

```bash
# Generate detailed profile for analysis
uv run python scripts/sim_bench.py \
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

1. **Warm-up Runs**: The benchmark includes 10 warm-up steps to ensure stable timings

2. **Consistent Environment**: 
   - Close unnecessary applications
   - Disable CPU frequency scaling if possible
   - Use consistent GPU power settings

3. **Statistical Significance**:
   - Run tests multiple times
   - Look for > 5% performance changes
   - Consider variance in results

4. **Profile Regularly**:
   - Profile after major changes
   - Save profiles for historical comparison
   - Look for unexpected bottlenecks

## Advanced Usage

### Custom Benchmarks

Create specialized benchmarks for specific scenarios:

```python
# benchmark_complex_level.py
import madrona_escape_room
from scripts.sim_bench import run_benchmark

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
# Compare branches
git checkout main
./tests/run_perf_test.sh
mv build/perf_results build/perf_results_main

git checkout feature-branch  
./tests/run_perf_test.sh
mv build/perf_results build/perf_results_feature

# Analyze differences
python scripts/compare_perf.py \
    build/perf_results_main \
    build/perf_results_feature
```

## Related Documentation

- [Testing Guide](TESTING_GUIDE.md) - General testing practices
- [sim_bench.py](../../../scripts/sim_bench.py) - Benchmark script source
- [Headless Mode](../../deployment/headless/HEADLESS_MODE.md) - Running without graphics