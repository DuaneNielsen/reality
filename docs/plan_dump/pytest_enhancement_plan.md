# Plan: Enhance pytest Suite with Binary Format Validation

## Prerequisite Reading List

Before implementing these changes, review these files to understand the current architecture:

### Core Source Files (C++ Implementation)
1. **`src/replay_metadata.hpp`** - Complete replay file format specification (ReplayMetadata struct, CompiledLevel sizes, file layout)
2. **`src/types.hpp`** - CompiledLevel struct definition (MAX_TILES=1024, field layouts)
3. **`src/mgr.cpp`** - Recording implementation and binary file writing logic

### Current Test Infrastructure
4. **`tests/python/test_native_recording.py`** - Current binary format testing (lines 159-260) 
5. **`tests/python/test_native_recording_replay_roundtrip.py`** - Existing replay validation patterns
6. **`tests/python/test_c_api_struct_validation.py`** - C API struct validation patterns
7. **`tests/python/conftest.py`** - Fixture patterns for recording/replay testing

### Root Files to Extract From
8. **`analyze_recording_v2.py`** - Authoritative binary format parsing (136-byte ReplayMetadata + 8216-byte CompiledLevel)
9. **`analyze_recording.py`** - Legacy format parsing (for understanding format evolution) 
10. **`debug_replay_actions.py`** - Action data parsing and step-by-step validation

## Implementation Plan

### Phase 1: Enhanced Binary Format Validation
**Target:** Create comprehensive recording file format tests that exceed current capabilities

**New Test File:** `tests/python/test_recording_binary_format.py`

**Enhancements over existing `test_recording_file_format()`:**

1. **Complete ReplayMetadata Validation**
   - Extract format parsing logic from `analyze_recording_v2.py` (lines 11-43)
   - Validate all 14 fields vs current 9 fields in existing test
   - Add level_name field validation (new in version 2)
   - Test current format (version 2) only

2. **CompiledLevel Structure Validation**
   - Add CompiledLevel size verification (8216 bytes expected)
   - Validate embedded level data matches original (currently missing)
   - Test tile data integrity within MAX_TILES bounds

3. **Action Data Verification**
   - Extract step-by-step action validation from `debug_replay_actions.py`
   - Verify action data offsets and boundaries
   - Add action sequence validation (currently only checks existence)

### Phase 2: Advanced Action Data Testing
**Target:** Enhance replay tests with detailed action verification

**Enhance:** `tests/python/test_native_recording_replay_roundtrip.py`

1. **Action Sequence Validation**
   - Add logic from `debug_replay_actions.py` (lines 42-63) for step-by-step verification
   - Validate action data types and ranges
   - Test action data corruption detection

2. **File Structure Integrity**
   - Add boundary checking for action data sections
   - Verify file size calculations match expected formats
   - Test partial file scenarios

### Phase 3: Current Format Validation
**Target:** Ensure robust validation of current recording format (version 2)

**New Test Class:** Add to existing `test_native_recording.py`

1. **Format Specification Testing**
   - Test ReplayMetadata struct matches C++ definition exactly
   - Validate magic number (0x4D455352 "MESR") and version (2)
   - Test field alignment and padding
   - Verify level_name field handling

2. **Error Condition Testing**
   - Test invalid magic numbers
   - Test corrupted headers
   - Test incomplete files

### Phase 4: Cleanup and Documentation
**Target:** Remove redundant root files after successful integration

1. **Remove Root Files After Integration:**
   - `analyze_recording.py` (functionality moved to pytest)
   - `analyze_recording_v2.py` (functionality moved to pytest) 
   - `debug_replay_actions.py` (functionality moved to pytest)

2. **Keep:** `setup.py`, `setup_build.py` (legitimate packaging files)

## Key Improvements Over Current Tests

### Current Limitations in `test_recording_file_format()`:
- Only validates 9/14 ReplayMetadata fields
- Hardcoded 128-byte metadata assumption (actually 136 bytes)
- No CompiledLevel validation
- Basic action data presence check only
- No comprehensive format specification testing

### New Capabilities:
- **Complete Format Validation:** All 14 ReplayMetadata fields + CompiledLevel structure
- **Current Format Focus:** Comprehensive version 2 format support
- **Action Data Integrity:** Step-by-step validation vs simple existence check
- **File Structure Verification:** Precise offset and size validation
- **Corruption Detection:** Test malformed file handling
- **C++ Struct Alignment:** Verify Python parsing matches C++ struct layout

## Implementation Notes

1. **Replace Current Implementation:** Update existing format tests with comprehensive validation
2. **Leverage Existing Fixtures:** Use `cpu_manager` and recording fixtures from conftest.py
3. **Extract, Don't Duplicate:** Move logic from root files, don't copy
4. **Follow Pytest Patterns:** Use existing test organization and naming conventions
5. **Comprehensive Coverage:** Test both happy path and error conditions
6. **Modern Format Only:** Focus on current version 2 format specification

## Validation Criteria

- [ ] All existing recording tests continue to pass
- [ ] New binary format tests cover 100% of ReplayMetadata fields (14/14)
- [ ] Action data validation detects corruption/modification
- [ ] Current format (version 2) tests comprehensively validate spec compliance
- [ ] CompiledLevel embedded data validation matches source
- [ ] Root directory files can be safely removed after integration
- [ ] Test execution time remains reasonable (<30s for full recording test suite)

## Files to be Enhanced/Created

1. **Create:** `tests/python/test_recording_binary_format.py` - Comprehensive binary format validation
2. **Enhance:** `tests/python/test_native_recording.py` - Update `test_recording_file_format()` with complete validation
3. **Enhance:** `tests/python/test_native_recording_replay_roundtrip.py` - Add action data validation
4. **Remove:** Root directory analysis scripts after successful integration

This plan transforms ad-hoc debug scripts into robust, maintainable test infrastructure while focusing on current format requirements and eliminating technical debt.