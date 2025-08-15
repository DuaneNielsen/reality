# Plan: Remove C API Dependency from C++ Tests

## Problem Statement
The current C++ test infrastructure incorrectly uses the C API (`madrona_escape_room_c_api.h`) as the primary interface for testing. This creates unnecessary indirection and is an anti-pattern. The C API is designed for Python bindings, not internal C++ testing.

## Current Issues

### 1. Architecture Problems
- All test fixtures inherit from `MadronaTestBase` which uses `MER_ManagerHandle` (C API)
- `TestManagerWrapper` wraps C API calls instead of using Manager directly
- Tests can't access internal C++ state or methods
- Extra layer of indirection for every operation
- Can't test C++ specific features that aren't exposed through C API

### 2. Linking Problems
- When attempting to include real C++ headers (`mgr.hpp`, `viewer_core.hpp`), we get linking errors
- GoogleTest is compiled with system standard library
- Madrona uses custom libc++ toolchain (std::__mad1 namespace)
- Symbol mismatch: `std::__mad1::basic_string` vs `std::__1::basic_string`

### 3. Current Test Structure
```
tests/cpp/
├── fixtures/
│   ├── test_base.hpp        # Uses MER_ManagerHandle (C API)
│   ├── mock_components.hpp  # TestManagerWrapper wraps C API
│   └── viewer_test_base.hpp # Inherits from MadronaTestBase
└── unit/
    ├── test_c_api_*.cpp     # Legitimately test C API
    └── test_viewer_*.cpp    # Shouldn't use C API but do
```

## Investigation Steps

### Phase 1: Understand Why viewer.cpp Works
1. **Analyze viewer.cpp build**
   - Check how viewer.cpp successfully includes `mgr.hpp`
   - Examine CMakeLists.txt for viewer target
   - Look at compiler flags and link libraries
   - Identify what makes it compatible with Madrona headers

2. **Compare with test build**
   - Diff the compiler flags between viewer and tests
   - Check if viewer uses different standard library settings
   - See if viewer has special CMake configuration

### Phase 2: Analyze GoogleTest Integration
1. **Current GoogleTest setup**
   - Located in `external/madrona/external/googletest`
   - Built in `tests/cpp/CMakeLists.txt` with `add_subdirectory`
   - Uses system compiler settings

2. **Potential fixes**
   - Build GoogleTest with Madrona toolchain
   - Use pre-built GoogleTest that's compatible
   - Build tests with same toolchain as main project

### Phase 3: Document Test Dependencies
1. **Map current dependencies**
   ```
   Test -> TestBase -> C API -> Manager (C++)
   ```

2. **Desired architecture**
   ```
   Test -> Manager (C++)
   Test -> ViewerCore (C++)
   ```

## Implementation Plan

### Step 1: Create New Test Base Classes
1. **Create `MadronaCppTestBase`**
   - Directly create `madEscape::Manager` instances
   - No C API wrapper
   - Direct access to C++ methods and state

2. **Create `ViewerCoreTestBase`**
   - Direct `ViewerCore` creation and manipulation
   - Access to internal components (StateMachine, ActionManager)

### Step 2: Fix Toolchain/Linking Issues

#### Option A: Build GoogleTest with Madrona Toolchain
```cmake
# In tests/cpp/CMakeLists.txt
set(CMAKE_CXX_COMPILER ${MADRONA_COMPILER})
set(CMAKE_CXX_FLAGS "${MADRONA_CXX_FLAGS}")
add_subdirectory(googletest)
```

#### Option B: Separate Test Executable
- Build tests as separate executable like viewer
- Link against same libraries as viewer
- Use same CMake configuration as viewer

#### Option C: Header-Only Testing
- Use header-only test framework (e.g., Catch2)
- Avoids ABI compatibility issues
- Compiles with same settings as project

### Step 3: Migrate Existing Tests
1. **Keep C API tests** (`test_c_api_*.cpp`)
   - These legitimately test the C API
   - Should continue using `MER_ManagerHandle`

2. **Migrate viewer tests**
   - Remove C API usage
   - Use direct C++ classes
   - Access internal state directly

3. **Create proper integration tests**
   - Test ViewerCore with real Manager
   - Test recording/replay at C++ level
   - Test trajectory logging directly

### Step 4: Implement ViewerCore Test Properly
```cpp
// Instead of:
MER_ManagerHandle handle;
mer_create_manager(&handle, ...);
TestManagerWrapper mgr(handle);

// Do this:
madEscape::Manager mgr({
    .execMode = madrona::ExecMode::CPU,
    .numWorlds = 1,
    ...
});

madEscape::ViewerCore core(config, &mgr);
```

## Success Criteria
1. C++ tests can directly instantiate and test C++ classes
2. No unnecessary C API indirection except for C API tests
3. Tests compile and link successfully
4. Can access internal state for thorough testing
5. ViewerCore trajectory test works with real Manager

## Risks and Mitigations
1. **Risk**: Toolchain incompatibility
   - **Mitigation**: Build everything with same toolchain

2. **Risk**: ABI incompatibility 
   - **Mitigation**: Use header-only test framework or build GoogleTest with project

3. **Risk**: Breaking existing tests
   - **Mitigation**: Migrate incrementally, keep C API tests separate

## Next Steps
1. Start new conversation with this plan
2. Investigate viewer.cpp build configuration
3. Test GoogleTest with Madrona toolchain
4. Create proper C++ test base classes
5. Implement ViewerCore trajectory test without C API