# C++ Testing Guide

This guide covers writing and running C++ unit tests for the Madrona Escape Room project using GoogleTest.

## Overview

The project uses GoogleTest (gtest) for C++ unit testing, leveraging Madrona's pre-configured GoogleTest build to ensure ABI compatibility with the custom toolchain. Tests are located in `tests/cpp/` and are built separately from the main project to keep the build clean.

## Project Constraints

The Madrona project has specific constraints that affect testing:
- **No exceptions** (`-fno-exceptions`)
- **No RTTI** (`-fno-rtti`)
- **GPU code requires CUDA**

These constraints are automatically applied to all test builds.

## Test Structure

```
tests/cpp/
├── CMakeLists.txt               # Test build configuration
├── fixtures/                    # Shared test fixtures and utilities
│   ├── test_base.hpp           # C API test fixture classes
│   ├── cpp_test_base.hpp       # Direct C++ test fixture classes
│   ├── viewer_test_base.hpp    # ViewerCore test fixtures
│   ├── mock_components.hpp     # Mock objects for testing
│   └── test_levels.hpp         # Test level data and helpers
├── unit/                        # Unit test files
│   ├── test_c_api_cpu.cpp     # C API CPU tests
│   ├── test_c_api_gpu.cpp     # C API GPU tests
│   ├── test_direct_cpp.cpp    # Direct C++ Manager tests
│   └── test_viewer_core.cpp   # ViewerCore tests
├── integration/                 # Integration test files
│   └── test_viewer_integration.cpp
└── e2e/                        # End-to-end test files
    └── test_viewer_workflows.cpp
```

## Building Tests

Tests are not built by default. To enable test building:

```bash
# Configure with tests enabled
cmake -B build -DBUILD_TESTS=ON

# Build the tests
make -C build mad_escape_tests -j8
```

**Note**: When `BUILD_TESTS=ON` is set, the build system automatically sets `MADRONA_ENABLE_TESTS=ON` to ensure GoogleTest is built with Madrona's custom toolchain, avoiding ABI compatibility issues.

## Running Tests

### Using the Test Script

The easiest way to run tests:

```bash
# Run all tests
./tests/run_cpp_tests.sh

# Run only CPU tests (recommended for quick feedback)
./tests/run_cpp_tests.sh --cpu-only

# Run only GPU tests (requires CUDA, expect 10+ minutes)
./tests/run_cpp_tests.sh --gpu-only

# Build tests without running
./tests/run_cpp_tests.sh --build-only
```

### Direct Execution

```bash
# Run all tests (GPU tests will be skipped by default)
./build/mad_escape_tests

# Run CPU tests only (fast - recommended for development)
./build/mad_escape_tests --gtest_filter="*CPU*"

# Run GPU tests in main suite (requires environment variable)
ALLOW_GPU_TESTS_IN_SUITE=1 ./build/mad_escape_tests --gtest_filter="*GPU*"

# List available tests
./build/mad_escape_tests --gtest_list_tests

# Run specific test
./build/mad_escape_tests --gtest_filter="CApiCPUTest.ManagerCreation"
```

### ⚠️ GPU Test Limitations

#### Performance Warning
**Each GPU test creates its own manager, triggering NVRTC compilation of GPU kernels. This compilation takes approximately 40-45 seconds per test.** With 9 GPU tests plus parameterized variants, the full GPU test suite can take **10+ minutes** to complete.

#### Critical Limitation: One GPU Manager Per Process
**Only one GPU manager can be created per process lifetime.** After destroying a GPU manager, attempting to create a second one in the same process will fail with:
```
Error at external/madrona/src/mw/cuda_exec.cpp:283 in void madrona::setCudaHeapSize()
invalid argument
```

This is a Madrona framework limitation where CUDA device state is not properly reset when a manager is destroyed. As a result:
- **GPU tests are skipped by default** in the main test suite
- They will show as `[SKIPPED]` when running `./build/mad_escape_tests`
- Use the isolation script or environment variable to run GPU tests

#### Workarounds

1. **Use the provided isolation script** (recommended):
```bash
# Automatically runs each GPU test in a separate process
./tests/run_gpu_tests_isolated.sh
```

2. **Run GPU tests individually**:
```bash
# Run a single GPU test
./build/mad_escape_tests --gtest_filter="CApiGPUTest.ManagerCreation"
```

3. **Skip GPU tests during development**:
```bash
# Run only CPU tests for quick iteration
./build/mad_escape_tests --gtest_filter="*CPU*"
```

#### Why This Happens
- GPU code is compiled at runtime using NVRTC
- CUDA heap size can only be set once per process
- The Madrona engine doesn't reset CUDA state on manager destruction
- Python tests have the same limitation but typically avoid it through process isolation

For faster iteration during development:
- Run CPU tests for quick feedback (< 1 second per test)
- Run specific GPU tests individually when testing GPU-specific functionality
- Consider running full GPU test suite only in CI/nightly builds with process isolation

### Using CMake/CTest

```bash
# Run all tests through CTest
cd build && ctest

# Verbose output
cd build && ctest -V

# Run specific test
cd build && ctest -R CPU
```

## Writing Tests

### Basic Test Structure

```cpp
#include <gtest/gtest.h>
#include "test_base.hpp"

// Simple test
TEST(TestSuiteName, TestName) {
    EXPECT_EQ(1 + 1, 2);
}

// Test with fixture
class MyTest : public MadronaTestBase {
protected:
    void SetUp() override {
        MadronaTestBase::SetUp();
        // Additional setup
    }
};

TEST_F(MyTest, TestName) {
    ASSERT_TRUE(CreateManager());
    // Test code
}
```

### Available Test Fixtures

#### C API Test Fixtures (Legacy)

**MadronaTestBase** - Base fixture for C API tests:
```cpp
class MyTest : public MadronaTestBase {
    // Provides:
    // - MER_ManagerHandle handle
    // - MER_ManagerConfig config
    // - CreateManager() helper
    // - GetTensor() helper
    // - ValidateTensorShape() helper
};
```

#### Direct C++ Test Fixtures (Recommended)

**MadronaCppTestBase** - Base fixture for direct C++ tests:
```cpp
class MyTest : public MadronaCppTestBase {
protected:
    std::unique_ptr<Manager> mgr;
    Manager::Config config;
    std::vector<std::optional<CompiledLevel>> testLevels;
    
    // Automatically creates test levels in SetUp()
    // Provides CreateManager() helper that handles level creation
    // Provides ValidateTensorShape() for tensor validation
};
```

**ViewerCoreTestBase** - Fixture for ViewerCore tests:
```cpp
class MyViewerTest : public ViewerCoreTestBase {
protected:
    std::unique_ptr<ViewerCore> viewer;
    ViewerCore::Config viewerConfig;
    
    // Inherits from MadronaCppTestBase
    // Provides CreateViewer() helper
};
```

**MadronaCppGPUTest** - GPU test fixture with mutex for process safety:
```cpp
class MyGPUTest : public MadronaCppGPUTest {
    // Automatically skips if ALLOW_GPU_TESTS_IN_SUITE != 1
    // Uses mutex to ensure sequential GPU test execution
};
```

**MadronaCppWorldCountTest** - Parameterized test for world counts:
```cpp
class MyWorldTest : public MadronaCppWorldCountTest {};

TEST_P(MyWorldTest, TestName) {
    // GetParam() returns the world count
    // config.numWorlds automatically set from parameter
}

INSTANTIATE_TEST_SUITE_P(
    WorldCounts,
    MyWorldTest,
    ::testing::Values(1, 2, 4, 8, 16)
);
```

### Test Assertions

GoogleTest provides two types of assertions:
- `ASSERT_*` - Fatal failure, stops test execution
- `EXPECT_*` - Non-fatal failure, continues test

Common assertions:
```cpp
ASSERT_TRUE(condition);
ASSERT_FALSE(condition);
ASSERT_EQ(expected, actual);
ASSERT_NE(val1, val2);
ASSERT_LT(val1, val2);  // Less than
ASSERT_GT(val1, val2);  // Greater than
ASSERT_FLOAT_EQ(expected, actual);
ASSERT_DOUBLE_EQ(expected, actual);
ASSERT_STREQ(str1, str2);  // C strings
ASSERT_NO_THROW(statement);
```

### GPU Test Patterns

```cpp
TEST_F(MadronaGPUTest, GPUSpecificTest) {
    // Test automatically skipped if no CUDA
    
    ASSERT_TRUE(CreateManager());
    
    MER_Tensor tensor;
    ASSERT_TRUE(GetTensor(tensor, mer_get_action_tensor));
    
    // GPU tensors have gpu_id >= 0
    EXPECT_GE(tensor.gpu_id, 0);
}
```

### Testing Error Conditions

```cpp
TEST(ErrorTest, NullPointer) {
    MER_Tensor tensor;
    MER_Result result = mer_get_action_tensor(nullptr, &tensor);
    EXPECT_NE(result, MER_SUCCESS);
}
```

## Test Naming Conventions

- Test suite names: `CamelCase` ending with `Test` (e.g., `CApiCPUTest`)
- Test names: `CamelCase` describing what is tested (e.g., `ManagerCreation`)
- Fixture classes: Same as test suite names
- Helper functions: `camelCase` starting with lowercase

## Debugging Tests

### Run Single Test
```bash
./build/tests/cpp/mad_escape_tests --gtest_filter="CApiCPUTest.ManagerCreation"
```

### Verbose Output
```bash
./build/tests/cpp/mad_escape_tests --gtest_print_time=1
```

### Debug with GDB
```bash
gdb ./build/tests/cpp/mad_escape_tests
(gdb) run --gtest_filter="CApiCPUTest.ManagerCreation"
```

### List Tests Without Running
```bash
./build/tests/cpp/mad_escape_tests --gtest_list_tests
```

## CI Integration

The test script returns appropriate exit codes:
- 0: All tests passed
- Non-zero: Test failures

This makes it easy to integrate with CI systems:
```bash
# In CI script
./tests/run_cpp_tests.sh --cpu-only || exit 1
```

## Test Architecture

### Design Philosophy

The test architecture supports two approaches:
1. **C API Testing**: Tests the C API wrapper (primarily for validation of the C API itself)
2. **Direct C++ Testing**: Tests C++ classes directly without wrapper overhead (recommended for most tests)

### Key Components

#### GoogleTest Integration
- Uses Madrona's pre-built GoogleTest from `external/madrona/external/googletest`
- Automatically linked with Madrona's custom toolchain (`madrona_libcxx`)
- Ensures ABI compatibility (avoids `std::__1` vs `std::__mad1` namespace conflicts)

#### Test Level Management
- All test fixtures automatically create simple 16x16 test levels
- Levels use `std::optional<CompiledLevel>` for per-world configuration
- Test levels are minimal (empty tiles) to reduce overhead

#### Tensor Shape Expectations
- Action tensor: 2D `[numWorlds * numAgents, actionDims]` where:
  - numAgents = 1 (from `consts.hpp`)
  - actionDims = 3 (moveAmount, moveAngle, rotate)
- Observation tensors: 3D `[numWorlds, numAgents, features]`
- Reward/Done tensors: 3D `[numWorlds, numAgents, 1]`

### Migration from C API to Direct C++

When migrating tests from C API to direct C++:

1. **Change base class**:
   ```cpp
   // Before
   class MyTest : public MadronaTestBase { ... };
   
   // After  
   class MyTest : public MadronaCppTestBase { ... };
   ```

2. **Update setup**:
   ```cpp
   // Before (C API)
   MER_ManagerHandle handle = MER_CreateManager(&config);
   
   // After (Direct C++)
   ASSERT_TRUE(CreateManager());  // Uses RAII via unique_ptr
   ```

3. **Access members directly**:
   ```cpp
   // Before
   MER_GetNumWorlds(handle);
   
   // After
   mgr->numWorlds();
   ```

## Common Issues

### Tests Not Building
- Ensure `-DBUILD_TESTS=ON` is set when configuring CMake
- This automatically sets `MADRONA_ENABLE_TESTS=ON` for GoogleTest
- Check that GoogleTest is available in `external/madrona/external/googletest`

### GPU Tests Failing
- GPU tests automatically skip if CUDA is not available
- Check CUDA installation with `nvidia-smi`
- Ensure CUDA libraries are in LD_LIBRARY_PATH
- Set `ALLOW_GPU_TESTS_IN_SUITE=1` to run GPU tests in main suite

### Linking Errors
- Tests must be compiled with same flags as main project (`-fno-rtti -fno-exceptions`)
- This is handled automatically by using Madrona's GoogleTest build
- ABI compatibility is ensured by linking with `madrona_libcxx`

### Namespace Conflicts
- If you see errors about `std::__1` vs `std::__mad1`, ensure:
  - You're not building GoogleTest separately
  - `MADRONA_ENABLE_TESTS` is set before including external directories
  - You're using the gtest_main target from Madrona's build

## Adding New Tests

### For Direct C++ Tests (Recommended)

1. Create new test file in appropriate directory:
   - `tests/cpp/unit/` for unit tests
   - `tests/cpp/integration/` for integration tests
   - `tests/cpp/e2e/` for end-to-end tests

2. Include the appropriate base fixture:
```cpp
// tests/cpp/unit/test_new_feature.cpp
#include "cpp_test_base.hpp"
#include "mgr.hpp"  // Include C++ headers directly

using namespace madEscape;

class NewFeatureTest : public MadronaCppTestBase {};

TEST_F(NewFeatureTest, BasicFunctionality) {
    ASSERT_TRUE(CreateManager());
    
    // Direct access to C++ objects
    auto actionTensor = mgr->actionTensor();
    EXPECT_EQ(actionTensor.numDims(), 2);
    
    // Test implementation
}
```

3. Add the file to `TEST_SOURCES` in `tests/cpp/CMakeLists.txt`

4. Rebuild and run:
```bash
make -C build mad_escape_tests -j8
./build/mad_escape_tests --gtest_filter="NewFeatureTest.*"
```

### For C API Tests (When Testing C API Itself)

```cpp
// tests/cpp/unit/test_c_api_feature.cpp
#include "test_base.hpp"
#include "madrona_escape_room_c_api.h"

TEST_F(MadronaTestBase, CApiFeature) {
    ASSERT_TRUE(CreateManager());
    // Test C API functionality
}
```

## Performance Considerations

### CPU Tests
- Keep individual tests fast (< 1 second)
- Use smaller world counts for basic functionality tests
- Reserve large world counts for specific stress tests

### GPU Tests
- Each test creates a new manager, triggering ~45 second NVRTC compilation
- Total GPU test suite runtime: 10+ minutes
- Only one GPU manager can exist at a time (tests run sequentially via mutex)
- Consider running GPU tests only when:
  - Testing GPU-specific functionality
  - Before committing GPU-related changes
  - In CI/nightly builds