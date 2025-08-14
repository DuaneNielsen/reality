# Viewer Test Implementation Pre-Reading Guide

## Essential Files to Read First

### 1. Core Implementation
- `src/viewer.cpp` - Main viewer (focus on lines 71-584: main() function)
- `src/mgr.hpp` - Manager interface (recording/replay methods)
- `include/madrona_escape_room_c_api.h` - C API functions available for testing

### 2. Existing Test Infrastructure  
- `tests/cpp/fixtures/test_base.hpp` - Base test patterns
- `tests/cpp/fixtures/viewer_test_base.hpp` - Current mock helpers
- `tests/cpp/unit/test_option_parser.cpp` - Option parsing test examples

### 3. Documentation
- `docs/development/testing/CPP_TESTING_GUIDE.md` - Testing constraints
- `CLAUDE.md` - Build commands and conventions

## Key Concepts to Understand

### GPU Testing Limitation
```cpp
// Only ONE GPU manager per process!
// Tests must use ALLOW_GPU_TESTS_IN_SUITE=1 or run separately
```

### Recording Format
```
[Metadata: 12 bytes] → [Embedded Level: 12,308 bytes] → [Actions: variable]
```

### Input → Action Mapping
- WASD → move_amount (0-3) + move_angle (0-7)  
- Q/E → rotate (0-4)
- R → triggerReset()
- T → toggle trajectory
- SPACE → pause/resume

## Critical Functions for Testing

### C API Functions
```cpp
mer_create_manager()      // Create with config
mer_step()               // Advance simulation
mer_set_action()         // Set agent actions
mer_start_recording()    // Begin recording
mer_load_replay()        // Load recording
mer_enable_trajectory_logging()  // Track agent
```

### Mock Pattern Example
```cpp
class MockWindowManager {
    void* makeWindow(...) { return (void*)1; }  // Return valid dummy pointer
    void* initGPU(...) { return (void*)1; }
};
```

## Build & Run
```bash
# Build tests
make -C build mad_escape_tests -j8

# Run new integration tests
./build/mad_escape_tests --gtest_filter="*Integration*"
```

## Start Here
1. Read `src/viewer.cpp` main() to understand flow
2. Study `test_base.hpp` for test patterns
3. Review C API for available functions
4. Implement mocks first, then integration tests