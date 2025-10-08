# Plan: Add Sensor Configuration to Replay System

## Required Reading

**IMPORTANT**: Read these documents before starting implementation:
1. **docs/specs/mgr.md** - Manager specification, replay format documentation
2. **codegen/README.md** - Pahole codegen system, how Python bindings are auto-generated

## Problem
Replay recordings don't preserve sensor configuration (lidar beam count, FOV, noise parameters). When replaying, the system uses default values (128 beams, 120° FOV, no noise) instead of the original training configuration.

## Current State
- **ReplayMetadata** (src/mgr.hpp:39-79): Contains basic replay info (num_worlds, seed, auto_reset) but no sensor config
- **Manager::startRecording()** (src/mgr.cpp:1425-1511): Writes metadata + levels + actions, doesn't save sensor config
- **Manager::fromReplay()** (src/mgr.cpp:1692-1791): Loads metadata + levels + actions, uses default sensor config
- **Replay Format v4**: `[ReplayMetadata][CompiledLevel1...N][Actions/Checksums]`

## Solution: Add SensorConfig to ReplayMetadata (v5 format)

No backward compatibility needed - all replay files will be regenerated.

---

## Implementation Plan

### Phase 0: Branch Setup
- [ ] Switch to feature/beam_spread branch
  ```bash
  git checkout feature/beam_spread
  ```

### Phase 1: C++ Core Changes

- [ ] **1.1** Update ReplayMetadata struct (src/mgr.hpp:39-79)
  - Add `SensorConfig sensor_config;` field (replace part of reserved array)
  - Bump `REPLAY_VERSION` from 4 to 5
  - Update `isValid()` to only accept version 5 (remove v3/v4 support)
  - Update `createDefault()` to initialize sensor_config with default values:
    ```cpp
    meta.sensor_config.lidar_num_samples = 128;
    meta.sensor_config.lidar_fov_degrees = 120.0f;
    meta.sensor_config.lidar_noise_factor = 0.0f;
    meta.sensor_config.lidar_base_sigma = 0.0f;
    ```

- [ ] **1.2** Update Manager::startRecording() (src/mgr.cpp:1425-1511)
  - After line 1470, add:
    ```cpp
    impl_->recordingMetadata.sensor_config = impl_->cfg.sensorConfig;
    ```
  - Sensor config automatically written as part of ReplayMetadata header

- [ ] **1.3** Update Manager::fromReplay() (src/mgr.cpp:1692-1791)
  - After line 1720, add:
    ```cpp
    cfg.sensorConfig = metadata->sensor_config;
    ```
  - Sensor config will be loaded from replay file

### Phase 2: Python Bindings (Automatic)

- [ ] **2.1** Rebuild to regenerate Python bindings
  - Run `./build.sh` to trigger pahole codegen
  - Python `ReplayMetadata` class auto-updates with `sensor_config` field
  - Verify `madrona_escape_room/generated_dataclasses.py` contains sensor_config

### Phase 3: Documentation

- [ ] **3.1** Update docs/specs/mgr.md
  - Document v5 replay format
  - Add SensorConfig field to ReplayMetadata field list
  - Note that v3/v4 replays are no longer supported

### Phase 4: Testing

**📖 Required Reading**:
- `tests/README.md` - Testing procedures and test execution guide
- `tests/python/README.md` - Python testing guide with pytest

- [ ] **4.0** Read testing documentation
  - Read `tests/README.md` to understand test structure
  - Read `tests/python/README.md` to understand pytest conventions

- [ ] **4.1** Create pytest tests for replay sensor config
  - File: `tests/python/test_replay_sensor_config.py`
  - Test cases:
    - `test_replay_default_sensor_config` - Record/replay with default (128 beams, 120° FOV)
    - `test_replay_custom_32_beam_180_fov` - Record/replay with 32 beams, 180° FOV
    - `test_replay_custom_256_beam_360_fov` - Record/replay with 256 beams, 360° FOV
    - `test_replay_with_noise` - Record/replay with noise enabled (64 beams, 120° FOV, noise)
    - `test_replay_metadata_sensor_config` - Verify read_replay_metadata() includes sensor_config
  - Each test should:
    1. Create manager with specific sensor config
    2. Record a few steps
    3. Load replay using from_replay()
    4. Verify lidar tensor shape matches sensor config (not defaults)
    5. Verify replay steps produce identical observations

- [ ] **4.2** Manual verification with file_inspector
  - Create test recording with 32 beams, 180° FOV, noise enabled
  - Use `./build/file_inspector <recording.bin>` to verify sensor_config in metadata

- [ ] **4.3** Run new replay sensor config tests
  ```bash
  uv run --group dev pytest tests/python/test_replay_sensor_config.py -v
  ```

- [ ] **4.4** Run full test suite
  ```bash
  uv run python tests/test_tracker.py --dry-run
  ```

---

## Files Modified

### C++ Core
- `src/mgr.hpp` - ReplayMetadata struct, REPLAY_VERSION constant
- `src/mgr.cpp` - startRecording(), fromReplay()

### Documentation
- `docs/specs/mgr.md` - Replay format v5 specification

### Auto-Generated (via build)
- `madrona_escape_room/generated_dataclasses.py` - Python ReplayMetadata class

---

## Key Technical Details

### SensorConfig Structure (src/types.hpp:323-333)
```cpp
struct SensorConfig {
    int32_t lidar_num_samples = 128;      // Number of lidar beams (1-256)
    float lidar_fov_degrees = 120.0f;     // Lidar FOV in degrees (1.0-360.0)
    float lidar_noise_factor = 0.0f;      // Proportional noise (0.0=disabled)
    float lidar_base_sigma = 0.0f;        // Base noise floor (0.0=disabled)
};
```

### Replay Format v5
```
[ReplayMetadata (with SensorConfig)]
[CompiledLevel1]
[CompiledLevel2]
...
[CompiledLevelN]
[Actions/Checksums]
```

### Breaking Change
- Old v3/v4 replay files will **fail to load** with error message
- All existing recordings must be regenerated
- This is acceptable per user requirement

---

## Success Criteria

✅ ReplayMetadata includes SensorConfig field
✅ Recordings save sensor config to file
✅ Replay loads sensor config and creates manager with correct settings
✅ Lidar tensor shape matches recorded config (not defaults)
✅ All tests pass
✅ Documentation updated
