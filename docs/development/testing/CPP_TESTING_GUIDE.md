# C++ Testing Guide

This guide covers writing and running C++ unit tests for the Madrona Escape Room project using GoogleTest.

## Overview

The project uses GoogleTest (gtest) for C++ unit testing. Tests are located in `tests/cpp/` and are built separately from the main project to keep the build clean.

## Project Constraints

The Madrona project has specific constraints that affect testing:
- **No exceptions** (`-fno-exceptions`)
- **No RTTI** (`-fno-rtti`)
- **GPU code requires CUDA**

These constraints are automatically applied to all test builds.

## Test Structure

```
tests/cpp/
├── CMakeLists.txt          # Test build configuration
├── fixtures/               # Shared test fixtures
│   ├── test_base.hpp      # Base test fixture classes
│   └── test_levels.hpp    # Test level data and helpers
└── unit/                   # Unit test files
    ├── test_c_api_cpu.cpp # CPU API tests
    └── test_c_api_gpu.cpp # GPU API tests
```

## Building Tests

Tests are not built by default. To enable test building:

```bash
# Configure with tests enabled
cmake -B build -DBUILD_TESTS=ON

# Build the tests
make -C build mad_escape_tests -j8
```

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
# Run all tests
./build/mad_escape_tests

# Run CPU tests only (fast - recommended for development)
./build/mad_escape_tests --gtest_filter="*CPU*"

# Run GPU tests only (slow - ~45 seconds per test due to NVRTC compilation)
./build/mad_escape_tests --gtest_filter="*GPU*"

# List available tests
./build/mad_escape_tests --gtest_list_tests

# Run specific test
./build/mad_escape_tests --gtest_filter="CApiCPUTest.ManagerCreation"
```

### ⚠️ GPU Test Performance Warning

**Each GPU test creates its own manager, triggering NVRTC compilation of GPU kernels. This compilation takes approximately 40-45 seconds per test.** With 9 GPU tests plus parameterized variants, the full GPU test suite can take **10+ minutes** to complete.

This is expected behavior due to the runtime compilation nature of the Madrona engine. The compilation happens for each manager creation because:
- GPU code is compiled at runtime using NVRTC
- Compiled kernels are not cached between manager instances
- Each test needs a fresh manager for proper test isolation

For faster iteration during development:
- Run CPU tests for quick feedback (< 1 second per test)
- Run specific GPU tests when testing GPU-specific functionality
- Consider running full GPU test suite only in CI/nightly builds

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

#### MadronaTestBase
Base fixture providing manager creation and cleanup:
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

#### MadronaGPUTest
Fixture for GPU tests with automatic skip if CUDA unavailable:
```cpp
class MyGPUTest : public MadronaGPUTest {
    // Automatically skips test if CUDA not available
};
```

#### MadronaWorldCountTest
Parameterized test for testing with different world counts:
```cpp
class MyWorldTest : public MadronaWorldCountTest {};

TEST_P(MyWorldTest, TestName) {
    // GetParam() returns the world count
    config.num_worlds = GetParam();
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

## Common Issues

### Tests Not Building
- Ensure `-DBUILD_TESTS=ON` is set when configuring CMake
- Check that GoogleTest is available in `external/madrona/external/googletest`

### GPU Tests Failing
- GPU tests automatically skip if CUDA is not available
- Check CUDA installation with `nvidia-smi`
- Ensure CUDA libraries are in LD_LIBRARY_PATH

### Linking Errors
- Tests must be compiled with same flags as main project (`-fno-rtti -fno-exceptions`)
- This is handled automatically by the CMakeLists.txt

## Adding New Tests

1. Create new test file in `tests/cpp/unit/`
2. Include necessary headers and fixtures
3. Write tests using GoogleTest macros
4. Add the file to `TEST_SOURCES` in `tests/cpp/CMakeLists.txt`
5. Rebuild and run tests

Example:
```cpp
// tests/cpp/unit/test_new_feature.cpp
#include <gtest/gtest.h>
#include "test_base.hpp"

TEST_F(MadronaTestBase, NewFeature) {
    ASSERT_TRUE(CreateManager());
    // Test implementation
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