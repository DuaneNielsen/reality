# Refactoring Plan: Extract Sensor Configuration from CompiledLevel

## Problem Statement
Currently, sensor configuration (lidar_num_samples, lidar_fov_degrees, lidar_noise_factor, lidar_base_sigma) is embedded in the `CompiledLevel` struct. This creates several issues:
1. **Tight coupling**: Level geometry is mixed with sensor configuration
2. **Inflexible training**: Cannot easily sweep sensor parameters without regenerating level files
3. **Confusing API**: Sensor settings don't conceptually belong in level data

## Proposed Solution
Create a new `SensorConfig` C struct that holds all sensor-related parameters and configure it separately from levels. Use the existing pahole codegen system to automatically generate Python bindings.

---

## Implementation Plan

### Phase 1: C++ Core Changes ✅ COMPLETE

**📖 Required Reading**: `docs/specs/sim.md` - Understand the simulation architecture, ECS components, and how sensors integrate with the system

- [x] **1.0** Read `docs/specs/sim.md` to understand simulation architecture
- [x] **1.1** Create SensorConfig struct in `src/types.hpp` (line 320-333)
  ```cpp
  // [GAME_SPECIFIC] Sensor configuration structure
  // Defines parameters for all agent sensors (lidar, compass, etc.)
  // This struct is exported to Python via pahole codegen
  struct SensorConfig {
      // Lidar sensor configuration
      int32_t lidar_num_samples = 128;           // Number of lidar beams (1-256, default: 128)
      float lidar_fov_degrees = 120.0f;          // Lidar field of view in degrees (1.0-360.0, default: 120.0)

      // Sensor noise configuration
      float lidar_noise_factor = 0.0f;           // Proportional noise (0.001-0.01 typical, 0.0=disabled)
      float lidar_base_sigma = 0.0f;             // Base noise floor in world units (0.02 typical, 0.0=disabled)

      // Future expansion: RGB camera, depth sensor, etc.
  };
  ```

- [x] **1.2** Remove lidar fields from CompiledLevel in `src/types.hpp` (lines 319-324)
  - Removed: `float lidar_noise_factor = 0.0f;`
  - Removed: `float lidar_base_sigma = 0.0f;`
  - Removed: `int32_t lidar_num_samples = 128;`
  - Removed: `float lidar_fov_degrees = 120.0f;`

- [x] **1.3** Export SensorConfig via `src/struct_export.cpp`
  - Added to anonymous namespace: `volatile SensorConfig dummy_sensor_config = {};`
  - Added to extern "C" block: `void* _export_SensorConfig() { return (void*)&dummy_sensor_config; }`

- [x] **1.4** Update `codegen/generate_dataclass_structs.py`
  - Added `"SensorConfig"` to `DEFAULT_STRUCTS_TO_EXTRACT` list (line 65)

- [x] **1.5** Add SensorConfig to configuration structs
  - Added to `ManagerConfig` in `src/types.hpp` (line 379)
  - Added to `Manager::Config` in `src/mgr.hpp` (line 238)
  - Added to `WorldInit` in `src/sim.hpp` (line 64)

- [x] **1.6** Update Manager implementation in `src/mgr.cpp`
  - Pass `sensorConfig` to WorldInit in GPU path (line 580)
  - Pass `sensorConfig` to WorldInit in CPU path (line 657)

- [x] **1.7** Update Simulation in `src/sim.cpp`
  - Register SensorConfig as singleton component (line 76)
  - Initialize SensorConfig singleton in constructor (lines 1123-1125)
  - Update lidarSystem to read from `ctx.singleton<SensorConfig>()` (lines 605-611)
  - Update compassSystem to read from SensorConfig (lines 365-366)

- [x] **1.8** Build and verify pahole codegen
  - Ran `./build.sh fullbuild` - BUILD SUCCESSFUL ✅
  - Verified `madrona_escape_room/generated_dataclasses.py` contains SensorConfig ✅
  - SensorConfig size: 16 bytes (4 fields × 4 bytes) ✅
  - CompiledLevel no longer contains lidar fields ✅

---

### Phase 2: Clean Up Level System (Remove Lidar Config) ✅ COMPLETE

**📖 Required Reading**: `codegen/README.md` - Understand the pahole codegen system and how Python bindings are automatically generated from C++ structs

- [x] **2.0** Read `codegen/README.md` to understand pahole codegen workflow
- [x] **2.1** Update `madrona_escape_room/level_compiler.py`
  - Removed `lidar_num_samples` field handling (validation, processing, assignment)
  - Removed `lidar_fov_degrees` field handling
  - Removed `lidar_noise_factor` field handling
  - Removed `lidar_base_sigma` field handling
  - Removed all validation for these fields from both single-level and multi-level formats

- [x] **2.2** Update `madrona_escape_room/default_level.py`
  - Removed lidar configuration assignments from `create_base_level_template()` function
  - Removed lines setting `lidar_num_samples`, `lidar_fov_degrees`, `lidar_noise_factor`, `lidar_base_sigma`

- [x] **2.3** Update `levels/generate_full_progression.py`
  - Removed `--lidar-num-samples` argument from CLI
  - Removed `--lidar-fov-degrees` argument from CLI
  - Removed `--lidar-noise-factor` argument from CLI
  - Removed `--lidar-base-sigma` argument from CLI
  - Removed lidar fields from `generate_full_progression()` function signature
  - Removed lidar fields from multi-level JSON generation

- [x] **2.4** Delete old level JSON files with embedded lidar config
  - Deleted all 5 lidar-specific JSON files with embedded configuration
  - Deleted `levels/lidar_64_beam_360.json`

- [x] **2.5** Delete old compiled level files with embedded lidar config
  - Deleted all 5 lidar-specific compiled .lvl files
  - Deleted `levels/lidar_64_beam_360.lvl`

- [x] **2.6** Regenerate clean level file
  - Generated: `levels/full_progression_159_levels_spawn_random.json` (159 levels, no lidar config)
  - Compiled: `levels/full_progression_159_levels_spawn_random.lvl` ✅ SUCCESS

**Codegen Enhancement Note**: Enhanced `codegen/generate_dataclass_structs.py` to support nested struct types:
- Added type mapping for `struct SensorConfig` → Python `SensorConfig` class
- Added late-binding factory functions for struct instances using `globals()[typename]()`
- Updated struct extraction order to ensure dependencies are defined first
- **Known Limitation**: cdataclass doesn't support nested structs in metadata - using `c_byte * 16` as temporary workaround
- **Future Work**: Replace temporary workaround with proper nested struct support when cdataclass is enhanced

---

### Phase 3: Python SensorConfig Integration ✅ COMPLETE

**📖 Required Reading**: `docs/specs/level_compiler.md` - Understand the level compilation pipeline and how sensor config was previously embedded in CompiledLevel

- [x] **3.0** Read `docs/specs/level_compiler.md` to understand level compiler architecture ✅
- [x] **3.1** Update `madrona_escape_room/sensor_config.py` ✅
  - Created new `LidarConfig` class (separate from visual SensorConfig to avoid naming conflicts)
  - Added lidar fields matching C struct:
    - `lidar_num_samples: int = 128` (1-256 range)
    - `lidar_fov_degrees: float = 120.0` (1.0-360.0 range)
    - `lidar_noise_factor: float = 0.0` (proportional noise)
    - `lidar_base_sigma: float = 0.0` (base noise floor)
  - Implemented `to_c_struct()` method to convert to generated CSensorConfig
  - Added `validate()` method with proper range checks
  - Created preset factory methods: `default()`, `wide_fov()`, `narrow_fov()`, `with_noise()`
  - Added convenience exports: `LIDAR_CONFIG_DEFAULT`, `LIDAR_CONFIG_WIDE`, etc.

- [x] **3.2** Update `madrona_escape_room/manager.py` (SimManager) ✅
  - Imported generated `SensorConfig as CSensorConfig` from generated_dataclasses
  - Added `lidar_config` parameter to `__init__` (accepts `LidarConfig` instance or None)
  - Implemented conversion: Python `LidarConfig` → C `CSensorConfig` → bytes for ManagerConfig
  - Configured default values (128 beams, 120° FOV, no noise) when lidar_config=None
  - Validated config before conversion using `lidar_config.validate()`

- [x] **3.3** Integration Testing ✅
  - Created temporary `scratch/test_lidar_config.py` test script for validation
  - Verified default config (None) works correctly
  - Verified custom config with all parameters
  - Verified all preset configs (default, wide, narrow, noisy)
  - Verified validation catches invalid ranges
  - All basic integration tests passed ✅
  - **Note**: Deleted scratch test - proper integration with existing test suite happens in Phase 5

**Design Decision**: Used `LidarConfig` name instead of `SensorConfig` to avoid collision with existing visual sensor config class. This maintains backward compatibility and keeps visual/lidar sensor configs cleanly separated.

**Existing Tests to Migrate in Phase 5**:
- `test_configurable_lidar.py` - Currently uses `@pytest.mark.json_level` with embedded lidar config (11 tests)
- `test_lidar_noise.py` - Currently uses level attributes for noise config (6 tests)
- These tests need migration from JSON-embedded config to `lidar_config` parameter

---

### Phase 4: Training Scripts ✅ COMPLETE

- [x] **4.1** Update `scripts/train.py` ✅
  - Added `--lidar-num-samples` argument (default: 128)
  - Added `--lidar-fov-degrees` argument (default: 120.0)
  - Added `--lidar-noise-factor` argument (default: 0.0)
  - Added `--lidar-base-sigma` argument (default: 0.0)
  - Created LidarConfig from arguments with validation
  - Passed lidar_config to SimManager via setup_lidar_training_environment()
  - Updated training config dict for wandb logging with sensor parameters
  - Updated EvaluationRunner to accept and use lidar_config

- [x] **4.2** Update `scripts/policy.py` ✅
  - Already dynamic! Uses `tensor.shape[2]` to get actual feature size
  - Automatically adapts to any lidar sample count (32, 64, 128, 256)
  - No changes needed - existing implementation is correct

- [x] **4.3** Verify `scripts/sweep_config.yaml` ✅
  - Already configured with lidar parameters as sweep variables:
    - `lidar-num-samples: [32, 64, 128, 256]`
    - `lidar-fov-degrees: [120.0, 180.0, 360.0]`
    - `lidar-noise-factor: 0.005`
    - `lidar-base-sigma: 0.02`
  - Grid search configured across beam counts and FOV
  - No changes needed

- [x] **4.4** Update `scripts/inference_core.py` ✅
  - Added `lidar_config` parameter to InferenceConfig
  - Updated InferenceRunner.setup_simulation() to pass lidar_config
  - Ensures evaluation uses same sensor config as training

- [x] **4.5** Update `train_src/madrona_escape_room_learn/sim_interface_adapter.py` ✅
  - Added `lidar_config` parameter to setup_lidar_training_environment()
  - Passes lidar_config to SimManager initialization
  - Backward compatible with existing code (lidar_config=None)

---

### Phase 5: Test Updates

**📖 Required Reading**: `tests/README.md` - Understand the testing framework, pytest markers, and test organization

- [ ] **5.0** Read `tests/README.md` to understand test framework and markers
- [ ] **5.1** Update `tests/python/conftest.py`
  - Add `sensor_config` marker support to `cpu_manager` fixture
  - Parse sensor_config kwargs from marker
  - Create SensorConfig instance
  - Pass to SimManager initialization

- [ ] **5.2** Update `tests/python/test_configurable_lidar.py`
  - Migrate all tests from `@pytest.mark.json_level` with embedded config
  - To: `@pytest.mark.sensor_config(lidar_num_samples=X, lidar_fov_degrees=Y)`
  - Update test functions to work with new marker pattern
  - Remove lidar fields from json_level markers

- [ ] **5.3** Update `tests/python/test_lidar_noise.py`
  - Migrate from embedded config in json_level
  - To: `@pytest.mark.sensor_config(lidar_noise_factor=X, lidar_base_sigma=Y)`
  - Update test assertions

- [ ] **5.4** Create `tests/python/test_sensor_config.py`
  - Test SensorConfig defaults
  - Test to_c_struct() conversion
  - Test validation (invalid ranges)
  - Test integration with SimManager
  - Test config persistence across resets

- [ ] **5.5** Run full test suite
  - `uv run python tests/test_tracker.py --dry-run`
  - Verify all tests pass

---

### Phase 6: Documentation Updates

**📖 Required Reading**: Read the existing spec files to understand documentation structure and style

- [ ] **6.0** Read `docs/specs/sim.md` and `docs/specs/mgr.md` to understand spec format
- [ ] **6.1** Update `docs/specs/sim.md`
  - Add SensorConfig section documenting the new struct
  - Update lidarSystem section to note config comes from SensorConfig singleton
  - Update CompassObservation section to note bucket count from SensorConfig

- [ ] **6.2** Update `docs/specs/mgr.md`
  - Document SensorConfig field in ManagerConfig
  - Document default values and valid ranges

- [ ] **6.3** Update `docs/specs/level_compiler.md`
  - Remove Sensor Noise Configuration section (lines 433-475)
  - Update CompiledLevel structure documentation to remove lidar fields
  - Add note that sensor config is now separate from level geometry

- [ ] **6.4** Update `CLAUDE.md`
  - Add note about SensorConfig in relevant sections
  - Document that sensor config is separate from level geometry

---

## Key Advantages of This Approach

1. **Zero manual C API work** - Pahole generates Python bindings automatically
2. **Automatic synchronization** - Python structs updated on every build
3. **Layout validation** - Size assertions catch mismatches immediately
4. **Clean separation** - Level geometry completely separate from sensor config
5. **Flexible training** - Easy to sweep sensor parameters without regenerating levels
6. **Single source of truth** - C++ struct is the only definition

---

## Migration Strategy

### Breaking Changes
1. **CompiledLevel struct layout** changes - requires full rebuild
2. **All level files must be recompiled** (binary format changed)
3. **Level JSON format** no longer supports lidar fields
4. **Test markers** change from embedded config to sensor_config markers

### Rollout Steps
1. ✅ Phase 1: C++ changes + struct export + rebuild + verify codegen
2. ✅ Phase 2: Clean up level system (remove all lidar-related code and files)
3. ✅ Phase 3: Update Python SensorConfig integration
4. ✅ Phase 4: Update training scripts
5. 🔄 Phase 5: Migrate and run tests
6. 🔄 Phase 6: Update documentation

---

## Testing Checkpoints

### After Phase 1
- [ ] Build succeeds
- [ ] `generated_dataclasses.py` contains SensorConfig
- [ ] Size assertions pass

### After Phase 2 ✅
- [x] Level compiler no longer references lidar fields ✅
- [x] Old level files deleted (12 files: 6 JSON + 6 LVL) ✅
- [x] New clean level file generated and compiles ✅
  - `levels/full_progression_159_levels_spawn_random.json` (159 levels)
  - `levels/full_progression_159_levels_spawn_random.lvl` (compiled successfully)

### After Phase 3
- [ ] Python SensorConfig has lidar fields
- [ ] to_c_struct() works correctly
- [ ] SimManager accepts sensor_config parameter

### After Phase 4
- [ ] Training script accepts sensor config arguments
- [ ] Policy setup uses dynamic sample counts
- [ ] Sweep config works with new parameters

### After Phase 5
- [ ] All existing tests pass with new markers
- [ ] New sensor_config tests pass
- [ ] Full test suite passes

---

## File Checklist

### C++ Files (Modified)
- [ ] src/types.hpp
- [ ] src/struct_export.cpp
- [ ] src/mgr.cpp
- [ ] src/sim.cpp

### Codegen Files (Modified)
- [x] codegen/generate_dataclass_structs.py ✅

### Python Files (Modified)
- [ ] madrona_escape_room/sensor_config.py
- [ ] madrona_escape_room/manager.py
- [x] madrona_escape_room/level_compiler.py ✅
- [x] madrona_escape_room/default_level.py ✅

### Python Files (Auto-Generated)
- [x] madrona_escape_room/generated_dataclasses.py ✅ (automatic - includes SensorConfig, nested struct workaround)

### Script Files (Modified)
- [ ] scripts/train.py
- [ ] scripts/policy.py
- [x] levels/generate_full_progression.py ✅

### Test Files (Modified)
- [ ] tests/python/conftest.py
- [ ] tests/python/test_configurable_lidar.py
- [ ] tests/python/test_lidar_noise.py

### Test Files (New)
- [ ] tests/python/test_sensor_config.py

### Documentation Files (Modified)
- [ ] docs/specs/sim.md
- [ ] docs/specs/mgr.md
- [ ] docs/specs/level_compiler.md
- [ ] CLAUDE.md

### Files Deleted ✅
- [x] levels/full_progression_159_levels_spawn_random_lidar_32beam_180fov_noise_prop0.005_base0.02.json ✅
- [x] levels/full_progression_159_levels_spawn_random_lidar_32beam_180fov_noise_prop0.005_base0.02.lvl ✅
- [x] levels/full_progression_159_levels_spawn_random_lidar_64beam_120fov_noise_prop0.005_base0.02.json ✅
- [x] levels/full_progression_159_levels_spawn_random_lidar_64beam_120fov_noise_prop0.005_base0.02.lvl ✅
- [x] levels/full_progression_159_levels_spawn_random_lidar_64beam_360fov_noise_prop0.005_base0.02.json ✅
- [x] levels/full_progression_159_levels_spawn_random_lidar_64beam_360fov_noise_prop0.005_base0.02.lvl ✅
- [x] levels/full_progression_159_levels_spawn_random_lidar_128beam_180fov_noise_prop0.005_base0.02.json ✅
- [x] levels/full_progression_159_levels_spawn_random_lidar_128beam_180fov_noise_prop0.05_base0.02.lvl ✅
- [x] levels/full_progression_159_levels_spawn_random_lidar_256beam_360fov_noise_prop0.005_base0.02.json ✅
- [x] levels/full_progression_159_levels_spawn_random_lidar_256beam_360fov_noise_prop0.005_base0.02.lvl ✅
- [x] levels/lidar_64_beam_360.json ✅
- [x] levels/lidar_64_beam_360.lvl ✅

---

## Estimated Impact

- **Files Modified**: ~14 files
- **Files Auto-Generated**: 1 file (generated_dataclasses.py updated)
- **Files Deleted**: ~12 files
- **New Files**: 1 test file + 1 plan file
- **Lines Changed**: ~450 lines
- **Risk Level**: Medium (struct layout change + codegen dependency)
- **Benefit**: High (cleaner architecture + automatic Python sync + flexible training)
