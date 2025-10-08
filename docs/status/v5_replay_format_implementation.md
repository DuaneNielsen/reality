# V5 Replay Format Implementation Status

**Date:** 2025-10-07
**Branch:** `feature/beam_spread`
**Commit:** `ffde88ac`

## Changes Made

### 1. File Inspector Updates (`src/file_inspector.cpp`)

**Version Validation:**
- Changed from accepting v3/v4 to **v5 only**
- Updated error messages to reflect v5-only support
- Both JSON and text output modes updated

**Sensor Config Display:**
- Added sensor configuration output for v5 recordings:
  - Lidar beam count (`lidar_num_samples`)
  - Lidar FOV in degrees (`lidar_fov_degrees`)
  - Noise factor (`lidar_noise_factor`)
  - Base sigma (`lidar_base_sigma`)
- JSON output includes `sensor_config` object when version >= 5
- Text output includes "Sensor Configuration" section

**File Size Validation:**
- Simplified validation for v5's variable-length format
- Only checks minimum size: `sizeof(ReplayMetadata) + (num_worlds * sizeof(CompiledLevel))`
- Removed complex action data size calculations (not reliable with checksums)

### 2. Manager Updates (`src/mgr.cpp`)

**Recording (writeReplayAction):**
- Fixed condition from `version == 4` to `version >= 4`
- This ensures v5 recordings write ACTION record headers correctly

**Replay Loading (fromReplay):**
- Updated version validation error message to mention v5 support
- Record parsing already handled v4/v5 identically (same ACTION/CHECKSUM structure)
- Only metadata changed between v4 and v5

### 3. Manager Header Updates (`src/mgr.hpp`)

**ReplayMetadata::isValid():**
- Kept at `version == 5` (no backward compatibility)
- Only accepts v5 format recordings

### 4. Python Inference Scripts

**`scripts/infer_from_wandb.py`:**
- Added sensor config display during inference
- Shows lidar configuration extracted from wandb run

**`scripts/inference_utils.py`:**
- Added sensor config extraction from wandb run.config
- Creates `LidarConfig` from saved training parameters
- Falls back to defaults if not found

## Test Failures Introduced

### C++ Test Failures (4 total)

1. **`FileInspectorTest.InspectorHandlesRecordingFile`** (2 instances)
   - **Likely cause:** Test uses v3 or v4 format recording
   - **Fix needed:** Update test to create/use v5 format recording

2. **`ManagerIntegrationTest.ChecksumVerificationDetectsDivergence`** (2 instances)
   - **Likely cause:** Test creates v3 or v4 format recording
   - **Fix needed:** Update test recording format to v5

### Python Test Failures (11 total)

1. **Format Validation Tests:**
   - `test_multi_level_recording_replay.py::test_v3_format_validation`
   - `test_native_recording.py::test_recording_file_format`
   - `test_native_recording.py::test_current_format_specification_compliance`
   - `test_native_recording.py::test_field_alignment_and_padding`
   - **Likely cause:** Tests explicitly check for v3/v4 format or create v3/v4 recordings
   - **Fix needed:** Update to v5 format expectations

2. **Binary Format Tests:**
   - `test_recording_binary_format.py::test_replay_metadata_complete_structure`
   - `test_recording_binary_format.py::test_action_data_verification_step_by_step`
   - `test_recording_binary_format.py::test_format_specification_compliance`
   - **Likely cause:** Tests validate v3/v4 metadata structure or record format
   - **Fix needed:** Update to expect v5 metadata structure (with SensorConfig field)

3. **File Structure Test:**
   - `test_native_recording_replay_roundtrip.py::test_file_structure_integrity_validation`
   - **Likely cause:** Size validation expects v3/v4 format calculations
   - **Fix needed:** Update size validation to match v5's variable-length format

4. **Sensor Config Replay Tests:**
   - `test_replay_sensor_config.py::test_replay_custom_32_beam_180_fov`
   - `test_replay_sensor_config.py::test_replay_with_noise`
   - `test_replay_sensor_config.py::test_replay_metadata_sensor_config`
   - **Likely cause:** These are new tests for v5 sensor config feature
   - **Possible issue:**
     - Recording may not be creating v5 format correctly
     - Replay may not be extracting sensor config correctly
     - Tests may have incorrect expectations

## Root Cause Analysis

The primary cause of failures is **removal of backward compatibility** for v3/v4 formats:

1. **Version Check Changed:** `version == 5` instead of accepting multiple versions
2. **Recording Code Fixed:** Now correctly writes v5 format (was missing `version >= 4` check)
3. **Test Assumptions:** Many tests assume v3/v4 format is valid/supported

## Recommended Fix Strategy

### Phase 1: Update Existing Tests (Priority: High)
1. Identify all tests that create recordings
2. Ensure they create v5 format (should happen automatically with current code)
3. Update version validation assertions to expect v5

### Phase 2: Fix Format Validation Tests (Priority: High)
1. Update `test_v3_format_validation` → `test_v5_format_validation`
2. Update metadata structure assertions to include SensorConfig field
3. Update size calculations to match simplified v5 validation

### Phase 3: Fix Sensor Config Tests (Priority: Medium)
1. Debug why sensor config replay tests are failing
2. Verify recordings include correct sensor config in metadata
3. Verify replay extraction works correctly

### Phase 4: Update Documentation (Priority: Low)
1. Update test documentation to mention v5-only support
2. Document v5 format in test comments
3. Add migration guide for old v3/v4 recordings (if needed)

## Testing Success Criteria

✅ **Already Working:**
- Creating new v5 recordings with headless tool
- file_inspector displays sensor config correctly
- viewer loads and plays v5 recordings

❌ **Needs Fixing:**
- All recording/replay tests pass with v5 format
- Sensor config propagates correctly through recording → replay → simulation
- Backward compatibility removed cleanly (no v3/v4 code paths remain)

## Next Steps

1. Run individual failing tests to see specific error messages
2. Update test fixtures/factories to create v5 recordings
3. Update test assertions to expect v5 metadata structure
4. Re-run test suite and verify all tests pass

## Notes

- v5 format is **identical to v4** for record structure (ACTION/CHECKSUM records)
- Only **metadata** changed (added SensorConfig field + reserved space)
- This means action data parsing doesn't need changes
- Main work is updating version checks and metadata validation
