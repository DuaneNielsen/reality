# GPU Performance Baseline Documentation

## Test Environment
- **GPU**: NVIDIA GeForce RTX 4090
- **CUDA Version**: 12.8
- **Driver Version**: 570.172.08

## Test Parameters
- **Standard GPU Benchmark**: 8192 worlds, 1000 steps
- **Headless Command**: `./build/headless --cuda 0 -n 8192 -s 1000 --rand-actions`

## Expected Performance (Minimal Branch - Before Level Loading)

### CPU Performance
- **Test**: 1024 worlds, 1000 steps
- **Result**: 1,070,427 FPS
- **Per-world**: ~1,045 FPS per world

### GPU Performance  
- **Test**: 8192 worlds, 1000 steps
- **Result**: 6,586,204 FPS
- **Per-world**: ~804 FPS per world

## Actual Performance (Feature Branch - With Level Loading)

### CPU Performance
- **Test**: 1024 worlds, 1000 steps  
- **Result**: 822,734 FPS (23% slower)
- **Per-world**: ~803 FPS per world

### GPU Performance
- **Test**: 8192 worlds, 1000 steps
- **Result**: 190,415 FPS (97% slower!)
- **Per-world**: ~23 FPS per world (35x slower than baseline)

## Performance Regression Summary

| Mode | Baseline FPS | With Level Loading | Slowdown | Per-World Impact |
|------|--------------|-------------------|----------|------------------|
| CPU (1024 worlds) | 1,070,427 | 822,734 | 23% | 1045 → 803 FPS |
| GPU (8192 worlds) | 6,586,204 | 190,415 | 97% | 804 → 23 FPS |

## Key Observations

1. **GPU is disproportionately affected**: 97% performance loss vs 23% on CPU
2. **Per-world performance**: GPU drops from 804 to 23 FPS per world (35x slower)
3. **The slowdown is NOT proportional to geometry**: Even simple levels cause massive GPU slowdown

## Test Levels Used

### Simple Room (levels/simple_room.lvl)
```
#####
#S..#
#...#
#####
```
- 16 wall tiles
- max_entities = 52

### Default 32x32 Level (when no level specified)
```
################################
#S.............................#
#..............................#
[... 28 more rows ...]
#.............................S#
################################
```
- 124 wall tiles  
- max_entities = 160

## Investigation Notes

The performance regression appears to be related to the level loading system itself rather than the amount of geometry, as even the simple 5x5 room causes the same massive GPU slowdown.

Potential causes to investigate:
- Memory access patterns in compiled level data structures
- GPU divergence in level generation code
- Cache coherency issues with dynamic vs static level data
- Warp efficiency problems with conditional level loading

## Root Cause Analysis: --rand-actions Flag Issue

### Discovery
After extensive bisection, we discovered that the massive GPU performance regression only occurs when using the `--rand-actions` flag in the headless executable. Without this flag, performance remains acceptable.

### Testing on Main Branch (2025-08-12)
Testing confirmed this is NOT a feature branch issue:

| Branch | Test | FPS | Per-World |
|--------|------|-----|-----------|
| main | 1024 worlds, no --rand-actions | 551,350 | 538 FPS |
| main | 1024 worlds, WITH --rand-actions | 91,764 | 90 FPS |
| **Slowdown** | | **6x** | |

### Feature Branch Performance (same conditions)
| Branch | Test | FPS | Per-World |
|--------|------|-----|-----------|
| feature/test-driven-levels | 8192 worlds, no --rand-actions | ~5,600,000 | ~684 FPS |
| feature/test-driven-levels | 8192 worlds, WITH --rand-actions | ~207,000 | ~25 FPS |
| **Slowdown** | | **27x** | |

### Conclusion
The `--rand-actions` flag causes a significant GPU performance regression in both main and feature branches. This is a **pre-existing issue** in the Madrona codebase, not introduced by the level loading feature. The regression is more severe with higher world counts (27x at 8192 worlds vs 6x at 1024 worlds), suggesting a scaling issue with the random action generation implementation on GPU.

The level loading feature itself does not cause significant GPU performance issues when tested without the --rand-actions flag.