# Plan: Add v5 Replay Format Support to Viewer and File Inspector

## Problem

The viewer and file_inspector tools only support replay format v3 and v4, but the current codebase generates v5 format recordings. When trying to load v5 recordings, the tools fail with "Unsupported version: 5" and attempting to parse v5 data as v4 causes position corruption (reading garbage values like position=(4838354.50,-3218428.50,0.00) instead of valid coordinates).

**Impact:**
- Cannot inspect or view recently recorded replays
- Training recordings (e.g., `wandb/run-20251007_220644-r8xbfzur/files/checkpoints/100.rec`) are unreadable
- No way to debug agent behavior or verify training progress visually

## Required Reading

**IMPORTANT**: Read these files before starting implementation:

1. **docs/specs/mgr.md** - Lines 515-614: Version 5 format specification
   - v5 metadata structure (208 bytes with SensorConfig field)
   - Mixed record types (ACTION and CHECKSUM)
   - Sensor config preservation fields
   - Breaking changes from v3/v4

2. **src/file_inspector.cpp** - Lines 102-280: Current v3/v4 parsing logic
   - Line 130: Version validation (currently rejects v5)
   - Lines 139-165: v3 vs v4 format handling
   - Lines 190-280: Display logic for metadata

3. **src/mgr.cpp** - Lines 1693-1860: Manager::fromReplay implementation
   - Lines 1748-1760: Metadata validation (rejects v5)
   - Lines 1789-1842: v3 vs v4 record parsing
   - Line 1722: SensorConfig extraction from metadata

4. **src/types.hpp** - Search for "ReplayMetadata" and "SensorConfig"
   - ReplayMetadata structure definition
   - SensorConfig struct (16 bytes: lidar_num_samples, lidar_fov_degrees, lidar_noise_factor, lidar_base_sigma)

5. **src/viewer_core.cpp** - Lines 1-150: ViewerCore initialization
   - How viewer uses Manager::fromReplay
   - No direct replay parsing (delegates to Manager)

## What Changed in v5

### Key Differences from v4:
1. **Metadata size increased**: 208 bytes (was ~184 bytes in v4)
2. **SensorConfig field added**: 16 bytes containing:
   - `lidar_num_samples` (uint32_t, 4 bytes)
   - `lidar_fov_degrees` (float, 4 bytes)
   - `lidar_noise_factor` (float, 4 bytes)
   - `lidar_base_sigma` (float, 4 bytes)
3. **Reserved space added**: 24 bytes for future extensions
4. **Same record format as v4**: Mixed ACTION/CHECKSUM records (no changes to action data)

### Why This Broke Viewers:
- Version check at line 130 in file_inspector.cpp explicitly rejects v5
- Version check at line 1755 in mgr.cpp explicitly rejects v5
- If version check is bypassed, metadata size mismatch causes offset errors
- Reading v5 metadata with v4 assumptions puts file pointer at wrong position
- This causes action data to be read from wrong offset → garbage positions

## Task List

### Phase 0: Understand Current Code ✓
- [x] Read v5 format specification in docs/specs/mgr.md
- [x] Understand ReplayMetadata structure in src/types.hpp
- [x] Study v3/v4 parsing in file_inspector.cpp
- [x] Study v3/v4 parsing in mgr.cpp::fromReplay
- [x] Identify all version validation checks

### Phase 1: Update file_inspector.cpp
- [ ] **Update version validation** (line 130)
  - Change condition from `if (metadata.version != 3 && metadata.version != 4)`
  - To: `if (metadata.version < 3 || metadata.version > 5)`
  - Update error message to reflect v3-v5 support

- [ ] **Add v5 display logic** (lines 256-264)
  - After v4 check, add `else if (metadata.version == 5)` branch
  - Display: "✓ Valid format (v5 with sensor config)"

- [ ] **Add SensorConfig display** (around line 280)
  - Check if version >= 5
  - Display sensor config fields:
    ```cpp
    if (metadata.version >= 5) {
        std::cout << "\nSensor Configuration:\n";
        std::cout << "  Lidar Beams: " << metadata.sensor_config.lidar_num_samples << "\n";
        std::cout << "  Lidar FOV: " << metadata.sensor_config.lidar_fov_degrees << "°\n";
        std::cout << "  Noise Factor: " << metadata.sensor_config.lidar_noise_factor << "\n";
        std::cout << "  Base Sigma: " << metadata.sensor_config.lidar_base_sigma << "\n";
    }
    ```

- [ ] **Update size calculation** (lines 137-165)
  - v5 uses same record format as v4 (mixed ACTION/CHECKSUM)
  - No changes needed to size validation (already handles variable-size v4 format)

- [ ] **Add JSON output for sensor config** (if JSON mode enabled)
  - Add sensor_config object to JSON output
  - Include all 4 sensor config fields

### Phase 2: Update mgr.cpp::fromReplay
- [ ] **Update version validation** (lines 1748-1760)
  - Change error message from "Only replay format v3 and v4 are supported"
  - To: "Only replay format v3, v4, and v5 are supported"
  - Update condition: `if (!metadata_inline.isValid() || metadata_inline.version > 5)`

- [ ] **Add v5 record parsing** (lines 1789-1842)
  - v5 uses same ACTION/CHECKSUM format as v4
  - Add `else if (metadata_inline.version == 5)` with same parsing as v4:
    ```cpp
    } else if (metadata_inline.version == 5) {
        // v5 format: Mixed ACTION and CHECKSUM records (same as v4, but with sensor config in metadata)
        actions = HeapArray<int32_t>(metadata_inline.num_steps * metadata_inline.num_worlds * metadata_inline.actions_per_step);
        // ... (same parsing logic as v4)
    }
    ```

- [ ] **Verify SensorConfig extraction** (line 1722)
  - Already extracts: `cfg.sensorConfig = metadata->sensor_config;`
  - This should work for v5 without changes (field exists in v5 metadata)
  - Verify this line is executed for v5 files

- [ ] **Update comments** (lines 1748-1760, 1789)
  - Change "v3 and v4 formats supported" to "v3, v4, and v5 formats supported"
  - Update format description comments

### Phase 3: Verify ReplayMetadata.isValid()
- [ ] **Read types.hpp ReplayMetadata definition**
  - Locate `isValid()` method implementation
  - Check if it validates version == 3 || version == 4
  - Update to: `version >= 3 && version <= 5`

- [ ] **Check for other version checks**
  - Search codebase for other hardcoded version checks
  - Update all instances to support v5

### Phase 4: Update viewer.cpp (if needed)
- [ ] **Check if viewer.cpp has direct replay parsing**
  - Search for version checks in viewer.cpp
  - Most likely delegates to Manager::fromReplay (no changes needed)

- [ ] **Verify ViewerCore compatibility**
  - ViewerCore uses Manager::fromReplay
  - Once Manager supports v5, ViewerCore should work automatically

### Phase 5: Testing
- [ ] **Test file_inspector with v5 recording**
  - Run: `./build/file_inspector wandb/run-20251007_220644-r8xbfzur/files/checkpoints/100.rec`
  - Expected: Shows metadata without "Unsupported version" error
  - Expected: Displays sensor config (32 beams, 360° FOV)
  - Expected: Shows correct step count, world count, etc.

- [ ] **Test viewer with v5 recording**
  - Run: `./build/viewer --replay wandb/run-20251007_220644-r8xbfzur/files/checkpoints/100.rec`
  - Expected: Loads without error
  - Expected: Agent positions are reasonable (e.g., -50 to +50 range, not millions)
  - Expected: Can step through replay frame by frame

- [ ] **Test with v4 recording (backward compatibility)**
  - Find or create a v4 recording
  - Verify file_inspector still works
  - Verify viewer still works

- [ ] **Test sensor config propagation**
  - Create v5 recording with custom sensor config (e.g., 64 beams, 180° FOV)
  - Verify file_inspector displays correct values
  - Verify viewer creates simulation with correct sensor config

### Phase 6: Build and Integration
- [ ] **Rebuild project**
  - Run: `./build.sh`
  - Ensure no compilation errors

- [ ] **Test end-to-end workflow**
  - Record new replay during training
  - Inspect with file_inspector
  - View with viewer
  - Verify all tools work together

### Phase 7: Documentation
- [ ] **Update file_inspector help text** (optional)
  - Mention v3, v4, v5 support in help message

- [ ] **Update viewer documentation** (optional)
  - Note v5 format support

- [ ] **Document v5 format differences**
  - Already documented in docs/specs/mgr.md (no changes needed)

## Implementation Details

### Version Validation Pattern

**Old (v3/v4 only):**
```cpp
if (metadata.version != 3 && metadata.version != 4) {
    // Error: unsupported version
}
```

**New (v3/v4/v5):**
```cpp
if (metadata.version < 3 || metadata.version > 5) {
    // Error: unsupported version (v3-v5 supported)
}
```

### Record Parsing Pattern

v5 uses the **exact same** ACTION/CHECKSUM record format as v4:
- No changes to record parsing loop
- Only metadata structure changed (added SensorConfig field)

**Implementation approach:**
```cpp
if (metadata.version == 3) {
    // v3: Pure action data
    // ... existing v3 parsing ...
} else if (metadata.version == 4 || metadata.version == 5) {
    // v4 and v5: Mixed ACTION/CHECKSUM records (same format)
    // v5 just has sensor config in metadata, not in record stream
    // ... existing v4 parsing logic works for v5 too ...
}
```

### SensorConfig Display

Only display sensor config for v5+ files:
```cpp
if (metadata.version >= 5) {
    std::cout << "\nSensor Configuration:\n";
    std::cout << "  Lidar Beams: " << metadata.sensor_config.lidar_num_samples << "\n";
    std::cout << "  Lidar FOV: " << metadata.sensor_config.lidar_fov_degrees << "°\n";

    // Only show noise if non-zero
    if (metadata.sensor_config.lidar_noise_factor > 0.0f ||
        metadata.sensor_config.lidar_base_sigma > 0.0f) {
        std::cout << "  Noise Factor: " << metadata.sensor_config.lidar_noise_factor << "\n";
        std::cout << "  Base Sigma: " << metadata.sensor_config.lidar_base_sigma << "\n";
    }
}
```

## Files Modified

### Core Changes (Required)
- `src/file_inspector.cpp` - Add v5 validation and display logic
- `src/mgr.cpp` - Add v5 parsing in fromReplay()
- `src/types.hpp` - Update ReplayMetadata::isValid() (if needed)

### No Changes Needed
- `src/viewer_core.cpp` - Uses Manager::fromReplay (no direct parsing)
- `src/viewer.cpp` - Delegates to ViewerCore (no direct parsing)
- `docs/specs/mgr.md` - Already documents v5 format ✓

## Backward Compatibility

### v3 and v4 Support Maintained
- All v3/v4 parsing logic remains unchanged
- Version validation expanded (not replaced)
- v3/v4 recordings continue to work

### Forward Compatibility
- Reserved fields in v5 metadata allow future extensions
- Version check uses range (v3-v5) instead of equality
- Easy to add v6 support later by extending range

## Testing Plan

### Test Cases

**TC1: v5 Recording with Default Sensor Config**
- File: Any v5 recording with 128 beams, 120° FOV
- file_inspector: Should show sensor config
- viewer: Should load and display correctly

**TC2: v5 Recording with Custom Sensor Config**
- File: `wandb/run-20251007_220644-r8xbfzur/files/checkpoints/100.rec` (32 beams, 360° FOV)
- file_inspector: Should show "Lidar Beams: 32, Lidar FOV: 360.0°"
- viewer: Should create 32-beam lidar simulation

**TC3: v4 Recording (Backward Compatibility)**
- File: Find or create v4 recording
- file_inspector: Should work without changes
- viewer: Should work without changes
- Sensor config: Should use defaults (128 beams, 120° FOV)

**TC4: Invalid/Corrupted v5 Recording**
- File: Truncated or corrupted v5 file
- file_inspector: Should show validation errors
- viewer: Should fail gracefully with error message

### Success Criteria

✅ file_inspector accepts v5 recordings without "Unsupported version" error
✅ file_inspector displays sensor config for v5 recordings
✅ viewer loads v5 recordings without position corruption
✅ Agent positions are reasonable (within -1000 to +1000 range per limits::maxCoordinate)
✅ Backward compatibility: v3 and v4 recordings still work
✅ Sensor config from v5 metadata propagates to simulation

## Expected Output Examples

### file_inspector on v5 Recording

**Before fix:**
```
Recording File: 100.rec
✓ Valid magic number (MESR)
✗ Unsupported version: 5 (only v3 and v4 supported)
```

**After fix:**
```
Recording File: 100.rec
✓ Valid magic number (MESR)
✓ Valid format (v5 with sensor config)
Number of worlds: 1024
Number of steps: 100
Agents per world: 1

Sensor Configuration:
  Lidar Beams: 32
  Lidar FOV: 360.0°
  Noise Factor: 0.0
  Base Sigma: 0.0

Level Information:
  World 0 -> Level (static_randomized_target)
  ...
```

### viewer on v5 Recording

**Before fix:**
```
Step  52: World 0 Agent 0: [pos=(4838354.50,-3218428.50,0.00) ...
```

**After fix:**
```
Loaded replay: 100.rec (v5 format)
Sensor config: 32 beams, 360.0° FOV
Step   0: World 0 Agent 0: [pos=(-6.25,8.42,0.00) rot=45.2° ...
Step   1: World 0 Agent 0: [pos=(-5.83,8.91,0.00) rot=45.2° ...
Step  52: World 0 Agent 0: [pos=(12.45,-3.21,0.00) rot=78.6° ...
```

## Notes

- v5 record format is **identical to v4** (mixed ACTION/CHECKSUM records)
- Only the **metadata** changed (added SensorConfig field + reserved space)
- This means parsing logic is mostly copy-paste from v4 handling
- Main work is updating version checks and adding display logic
- No changes to action data structure or simulation behavior
