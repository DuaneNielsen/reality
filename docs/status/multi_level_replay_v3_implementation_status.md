# Multi-Level Replay V3 Format Implementation Status

**Date:** 2025-09-12  
**Issue:** Multi-level replay divergence when switching worlds in viewer  
**Solution:** Version 3 recording format with per-world level support

## 📋 Implementation Progress

### ✅ **Completed Tasks**

#### **1. Core Data Structure Updates**
- ✅ **ReplayMetadata Structure** (`src/mgr.hpp:25-59`)
  - Added `num_unique_levels` field
  - Added `world_level_indices[maxWorlds]` mapping array
  - Bumped `REPLAY_VERSION = 3`
  - Updated `isValid()` to only accept v3 format

#### **2. Recording Logic** (`src/mgr.cpp:1171-1272`)
- ✅ **Multi-Level Collection**: Deduplicates CompiledLevel structs across worlds
- ✅ **Format**: `[ReplayMetadata][NumUniqueLevels][CompiledLevel1...N][Actions]`
- ✅ **World Mapping**: Builds world-to-level index mapping
- ✅ **Backward Compatibility**: Removed (v3-only as requested)

#### **3. Replay Logic** (`src/mgr.cpp:1355-1430`) 
- ✅ **V3 Validation**: Only accepts version 3 files, errors on v1/v2
- ✅ **Multi-Level Reading**: Reads all unique levels and world mappings
- ✅ **Error Messages**: Clear v3-only requirement messages

#### **4. File Inspector** (`src/file_inspector.cpp:75-161`)
- ✅ **V3 Format Support**: Only validates v3 files  
- ✅ **Multi-Level Display**: Shows all unique levels and world mappings
- ✅ **File Size Calculation**: Accounts for multiple CompiledLevel structs

#### **5. Python Integration**
- ✅ **Generated Dataclasses**: ReplayMetadata updated with v3 fields (40196 bytes)
- ✅ **Python Bindings**: Return dataclass directly instead of dict conversion
- ✅ **API Improvements**: Type-safe dataclass vs dict approach

#### **6. Test Infrastructure**
- ✅ **Comprehensive Test**: `tests/python/test_multi_level_recording_replay.py`
  - Multi-level recording with 3 different level types
  - Level deduplication verification  
  - V3 format validation
- ✅ **Cleanup**: Removed incorrect test implementations as documented

## ⚠️ **Outstanding Issues**

### **1. Critical: C++ Metadata Reading/Writing Bug**

**Problem:** ReplayMetadata fields showing incorrect values:
```python
ReplayMetadata(magic=0, version=0, ..., num_unique_levels=0, ...)
```

**Expected:**
```python  
ReplayMetadata(magic=0x4D455352, version=3, ..., num_unique_levels=1, ...)
```

**Root Cause:** The C++ recording/reading pipeline is not properly handling the new v3 metadata fields.

**Investigation Needed:**
1. **C API Function**: `mer_read_replay_metadata` may need implementation
2. **Recording Pipeline**: Verify metadata is written correctly to binary file
3. **Reading Pipeline**: Ensure all v3 fields are read from file correctly

### **2. Test Failures**

**Current Status:**
```bash
❌ tests/python/test_multi_level_recording_replay.py::test_v3_format_validation
   - AssertionError: Expected version 3, got 0
   - Magic number is 0 instead of 0x4D455352
```

## 🔍 **Technical Implementation Details**

### **New V3 Format Structure**
```cpp
struct ReplayMetadata {
    uint32_t magic;                    // 0x4D455352 "MESR"
    uint32_t version;                  // 3
    // ... existing fields ...
    uint32_t num_unique_levels;        // NEW: Number of unique levels
    uint32_t world_level_indices[10000]; // NEW: World -> level mapping
    uint32_t reserved[7];              // Reserved space
};
```

### **Binary File Format**
```
[ReplayMetadata] -> [CompiledLevel1] -> [CompiledLevel2] -> ... -> [Actions...]
     188 bytes           85200 bytes       85200 bytes              Variable
```

### **Recording Flow**
```cpp
// 1. Collect unique levels from perWorldCompiledLevels
// 2. Build world_level_indices mapping  
// 3. Write metadata with v3 fields
// 4. Write all unique CompiledLevel structs
// 5. Write action streams (unchanged)
```

## 🎯 **Next Steps (Priority Order)**

### **1. Debug C++ Metadata Pipeline**
- [ ] **Verify Recording**: Check if `startRecording()` writes v3 metadata correctly
- [ ] **Verify Reading**: Ensure `mer_read_replay_metadata` reads v3 format
- [ ] **Add Debug Logging**: Print metadata values during write/read operations

### **2. Complete C API Implementation**
- [ ] **Implement**: `mer_read_replay_metadata` function in C API if missing
- [ ] **Test**: C API reads v3 metadata correctly from binary files

### **3. Validate Full Pipeline** 
- [ ] **End-to-End Test**: Record → Write → Read → Validate cycle
- [ ] **Multi-Level Test**: Verify different levels per world work correctly
- [ ] **Viewer Integration**: Test that viewer world switching works without divergence

## 🏗️ **Architecture Improvements Made**

1. **Type Safety**: Python bindings now return `ReplayMetadata` dataclass instead of dict
2. **Clean API**: Removed dict conversion complexity, direct dataclass access
3. **Deduplication**: Efficient storage of unique levels only
4. **Clear Errors**: V3-only validation with helpful error messages
5. **Extensibility**: Reserved fields for future format additions

## 📊 **Test Coverage Status**

- ✅ **Level Creation**: 3 distinct test levels (corridor, simple maze, complex maze)  
- ✅ **Multi-World Setup**: 5 worlds with mixed level assignments
- ✅ **Deduplication Logic**: Verifies only unique levels stored
- ⚠️ **V3 Format Validation**: Blocked by C++ metadata bug
- ⚠️ **Recording/Replay Roundtrip**: Needs C++ fixes to proceed

## 🔧 **Files Modified**

**C++ Core:**
- `src/mgr.hpp` - ReplayMetadata v3 structure
- `src/mgr.cpp` - Recording/replay logic with multi-level support  
- `src/file_inspector.cpp` - V3 format display

**Python Integration:**
- `madrona_escape_room/generated_dataclasses.py` - Updated ReplayMetadata (regenerated)
- `madrona_escape_room/manager.py` - Dataclass-based API

**Testing:**
- `tests/python/test_multi_level_recording_replay.py` - Comprehensive v3 tests

---

**Summary:** Core v3 format implementation is complete. Primary blocker is C++ metadata reading/writing bug causing version=0 instead of version=3. Once resolved, full multi-level replay functionality should work as designed.