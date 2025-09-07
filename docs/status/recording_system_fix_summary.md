# Recording System Fix Summary

## Problem Identified
The recording system had a design bug where it accepted a seed parameter but didn't use it to control the simulation. This caused confusion and required workarounds where users had to manually ensure the same seed was used for both inference and recording.

**Root Cause**: The recording system stored a seed in metadata but never applied it to reset the simulation, leading to two different random sequences and divergent trajectories.

## Solution Implemented
Simplified the recording system to enforce that recording can **only start from step 0** of a fresh simulation, eliminating the misleading seed parameter entirely.

### Key Changes

#### 1. Manager Class Changes (`src/mgr.cpp`, `src/mgr.hpp`)
- **Added step tracking**: `bool hasSteppedSinceInit` and `bool isInitializationStep` to track simulation state
- **Updated `startRecording()`**: 
  - Removed `uint32_t seed` parameter
  - Added step 0 validation - returns `false` if any user steps have been taken
  - Changed return type from `void` to `bool` for error handling
  - Uses Manager's original `randSeed` for metadata instead of separate parameter

#### 2. C API Changes (`src/madrona_escape_room_c_api.h`, `src/madrona_escape_room_c_api.cpp`)
- **Updated `mer_start_recording()`**: Removed `uint32_t seed` parameter
- **Enhanced error handling**: Returns `MER_ERROR_INVALID_PARAMETER` when recording cannot start

#### 3. Python API Changes (`madrona_escape_room/manager.py`, `madrona_escape_room/ctypes_bindings.py`)
- **Updated `start_recording()`**: Removed `seed` parameter
- **Updated documentation**: Clarifies that recording only works from fresh simulation
- **Updated ctypes bindings**: Removed `c_uint32` parameter

#### 4. Application Updates
- **Inference script** (`scripts/infer.py`): Removed `--recording-seed` argument
- **Headless runner** (`src/headless.cpp`): Updated to handle new return value
- **Viewer core** (`src/viewer_core.cpp`): Added error handling for recording failures

#### 5. Test Updates
- **All Python tests**: Removed seed parameters from `start_recording()` calls
- **All C++ tests**: Updated `startRecording()` and `mer_start_recording()` calls
- **Mock components**: Updated to match new API signature

## Step 0 Enforcement Logic
```cpp
// In Manager constructor - this step is allowed for initialization
step(); // isInitializationStep = true, so hasSteppedSinceInit stays false

// In subsequent step() calls
if (!impl_->isInitializationStep) {
    impl_->hasSteppedSinceInit = true; // Mark that user steps have occurred
}
impl_->isInitializationStep = false;

// In startRecording()
if (impl_->hasSteppedSinceInit) {
    std::cerr << "ERROR: Recording can only be started from the beginning of a fresh simulation\n";
    return false;
}
```

## Benefits of Fix
1. **Eliminates confusion**: No more misleading seed parameter that doesn't work
2. **Prevents bugs**: Impossible to accidentally record mid-episode 
3. **Clearer intent**: Recording is explicitly for capturing full episodes from start
4. **Simpler API**: Fewer parameters to manage
5. **Better error handling**: Clear error messages when used incorrectly

## Current Test Status
- **Build**: ✅ Successful compilation
- **Basic functionality**: ✅ Recording works when started immediately after Manager creation
- **Step 0 enforcement**: ✅ Correctly rejects recording after user steps
- **Test failures**: ❌ 12 tests need adaptation to new API constraints

## Files Modified
```
src/mgr.cpp                                    - Core recording logic
src/mgr.hpp                                    - Manager interface
src/madrona_escape_room_c_api.cpp              - C API implementation  
src/madrona_escape_room_c_api.h                - C API header
src/headless.cpp                               - Headless app
src/viewer_core.cpp                            - Viewer core
madrona_escape_room/manager.py                 - Python manager
madrona_escape_room/ctypes_bindings.py         - Python ctypes
scripts/infer.py                               - Inference script
tests/cpp/fixtures/mock_components.hpp         - Mock components
tests/cpp/integration/test_viewer_integration.cpp - C++ integration tests
tests/cpp/integration/test_viewer_errors.cpp   - C++ error tests
tests/cpp/e2e/test_viewer_workflows.cpp        - C++ workflow tests
tests/python/test_native_recording*.py         - Python recording tests
```

## Next Steps
1. Fix remaining C++ test failures (4 tests)
2. Fix remaining Python test failures (12 tests)
3. Update tests to work with step 0 constraint or create fresh managers as needed