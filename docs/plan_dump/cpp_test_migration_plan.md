# C++ Test Migration Plan: Removing C API Dependency

## Executive Summary

This plan outlines the migration of C++ tests from using the C API wrapper to directly using Manager and ViewerCore C++ classes. The GoogleTest ABI compatibility issue has been resolved by using Madrona's pre-configured GoogleTest build.

## Problem Statement

### Current Issues
1. **Unnecessary Indirection**: C++ tests use C API (`madrona_escape_room_c_api.h`) instead of direct C++ classes
2. **Limited Testing Capability**: Cannot access internal state or protected methods through C API
3. **Architectural Debt**: C API was designed for Python bindings, not internal C++ testing
4. **Maintenance Overhead**: Changes to C++ classes require updating both C API and tests

### Root Cause (Resolved)
- ✅ **ABI Incompatibility**: Fixed by using Madrona's GoogleTest build with correct toolchain linking
- ✅ **Build Configuration**: Updated CMakeLists.txt to use Madrona's test infrastructure

## Solution Overview

### Phase 1: Infrastructure Setup ✅ COMPLETED
1. **Use Madrona's GoogleTest**
   - Set `MADRONA_ENABLE_TESTS=ON` when building tests
   - Remove local GoogleTest build configuration
   - Link against Madrona's `gtest_main` target

2. **CMakeLists.txt Changes**
   ```cmake
   # Main CMakeLists.txt
   option(BUILD_TESTS "Build C++ unit tests" OFF)
   if(BUILD_TESTS)
       set(MADRONA_ENABLE_TESTS ON CACHE BOOL "" FORCE)
   endif()
   
   # tests/cpp/CMakeLists.txt
   # Remove GoogleTest subdirectory - use Madrona's version
   ```

### Phase 2: Test Classification

#### Tests to Keep Using C API (Legitimate Use)
These tests validate the C API itself and should remain unchanged:
- `test_c_api_cpu.cpp` - Tests C API CPU functionality
- `test_c_api_gpu.cpp` - Tests C API GPU functionality

#### Tests to Migrate to Direct C++
These tests should directly instantiate and test C++ classes:

| Test File | Current State | Migration Priority | Complexity |
|-----------|--------------|-------------------|------------|
| `test_viewer_core.cpp` | Uses C API wrapper | HIGH | Medium |
| `test_viewercore_trajectory.cpp` | Partial C++ usage | HIGH | Low |
| `test_viewer_integration.cpp` | Mixed usage | MEDIUM | High |
| `test_viewer_input.cpp` | Uses C API | MEDIUM | Medium |
| `test_viewer_errors.cpp` | Uses C API | LOW | Medium |
| `test_viewer_workflows.cpp` | Uses C API | LOW | High |
| `test_direct_cpp.cpp` | Already direct C++ | DONE | N/A |

#### Utility Tests (No Migration Needed)
These already use direct C++ and don't need migration:
- `test_option_parser.cpp`
- `test_level_utilities.cpp`
- `test_recording_utilities.cpp`

### Phase 3: Migration Pattern

#### Before (Using C API)
```cpp
TEST_F(ViewerCoreTest, TrajectoryTracking) {
    // Create manager through C API
    MER_ManagerHandle handle = MER_CreateManager(&config);
    
    // Create viewer through C API wrapper
    MER_ViewerHandle viewer = MER_CreateViewer(handle, &viewerConfig);
    
    // Limited access - can't test internal state
    MER_ToggleTrajectoryTracking(viewer, 0);
    
    // Cleanup through C API
    MER_DestroyViewer(viewer);
    MER_DestroyManager(handle);
}
```

#### After (Direct C++)
```cpp
TEST_F(ViewerCoreDirectTest, TrajectoryTracking) {
    // Direct Manager instantiation
    Manager::Config config{
        .execMode = madrona::ExecMode::CPU,
        .numWorlds = 4,
        // ... other config
    };
    Manager mgr(config);
    
    // Direct ViewerCore instantiation
    ViewerCore::Config viewerConfig{/* ... */};
    ViewerCore viewer(viewerConfig, &mgr);
    
    // Full access to public AND protected methods
    viewer.toggleTrajectoryTracking(0);
    EXPECT_TRUE(viewer.isTrackingTrajectory(0));
    
    // Can access internal state for validation
    auto& trajectories = viewer.getTrajectories();
    EXPECT_EQ(trajectories[0].points.size(), 1);
    
    // RAII handles cleanup automatically
}
```

### Phase 4: Implementation Steps

#### Step 1: Create New Test Base Classes
```cpp
// tests/cpp/fixtures/cpp_test_base.hpp
namespace madEscape {

class MadronaCppTestBase : public ::testing::Test {
protected:
    std::unique_ptr<Manager> mgr;
    Manager::Config config;
    
    void SetUp() override {
        config = {
            .execMode = madrona::ExecMode::CPU,
            .numWorlds = 1,
            // Default config
        };
    }
    
    ::testing::AssertionResult CreateManager() {
        try {
            mgr = std::make_unique<Manager>(config);
            return ::testing::AssertionSuccess();
        } catch (const std::exception& e) {
            return ::testing::AssertionFailure() 
                << "Failed to create Manager: " << e.what();
        }
    }
};

class ViewerCoreTestBase : public MadronaCppTestBase {
protected:
    std::unique_ptr<ViewerCore> viewer;
    ViewerCore::Config viewerConfig;
    
    ::testing::AssertionResult CreateViewer() {
        if (!mgr) {
            return ::testing::AssertionFailure() 
                << "Manager must be created first";
        }
        
        try {
            viewer = std::make_unique<ViewerCore>(viewerConfig, mgr.get());
            return ::testing::AssertionSuccess();
        } catch (const std::exception& e) {
            return ::testing::AssertionFailure() 
                << "Failed to create ViewerCore: " << e.what();
        }
    }
};

}
```

#### Step 2: Migrate Tests Incrementally

1. **Start with Simple Tests** (test_viewercore_trajectory.cpp)
   - Already partially uses direct C++
   - Good proof of concept
   
2. **Move to Core Tests** (test_viewer_core.cpp)
   - Central functionality
   - Many other tests depend on patterns here
   
3. **Handle Integration Tests** (test_viewer_integration.cpp)
   - More complex setup
   - May require mock objects
   
4. **Complete E2E Tests** (test_viewer_workflows.cpp)
   - Most complex scenarios
   - Do these last

#### Step 3: Update Each Test File

For each test file:
1. Change includes from C API to C++ headers
2. Replace test fixture base class
3. Update setup/teardown to use direct instantiation
4. Modify test assertions to use C++ methods
5. Add new tests for previously untestable internal state

### Phase 5: Validation

#### Success Criteria
- [ ] All tests compile without C API dependency (except legitimate C API tests)
- [ ] Tests run successfully with Madrona's GoogleTest
- [ ] No ABI compatibility errors
- [ ] Can test internal/protected methods
- [ ] Improved test coverage of C++ implementation

#### Testing the Migration
```bash
# Build with tests
cmake -B build -DBUILD_TESTS=ON
make -C build mad_escape_tests -j8

# Run all tests
./build/mad_escape_tests

# Run specific test suites
./build/mad_escape_tests --gtest_filter="ViewerCoreDirectTest.*"
```

## Benefits of Migration

1. **Direct Testing**: Test C++ classes without wrapper overhead
2. **Better Coverage**: Access to protected/internal methods
3. **Faster Tests**: No C API marshalling overhead
4. **Cleaner Architecture**: Tests match implementation language
5. **Easier Debugging**: Direct stack traces without C API layer
6. **Type Safety**: Full C++ type checking instead of void* handles

## Risk Mitigation

### Potential Issues & Solutions

| Risk | Mitigation |
|------|------------|
| Breaking existing tests | Keep C API tests separate, migrate incrementally |
| Missing test coverage | Add new tests for internal state during migration |
| Build complexity | Document build requirements clearly |
| Toolchain issues | Use Madrona's proven GoogleTest setup |

## Timeline Estimate

- **Week 1**: Migrate high-priority tests (viewer_core, trajectory)
- **Week 2**: Migrate medium-priority tests (integration, input)
- **Week 3**: Complete low-priority tests and validation
- **Week 4**: Documentation and cleanup

## Conclusion

The migration from C API to direct C++ testing is now feasible with the resolved GoogleTest compatibility. This will result in cleaner, more maintainable, and more comprehensive tests that directly validate the C++ implementation without unnecessary indirection.

## Appendix: Example Migrations

### Example 1: Simple State Test

**Before (C API)**:
```cpp
TEST(CApiTest, GetNumWorlds) {
    MER_Config config = {.num_worlds = 10};
    MER_ManagerHandle mgr = MER_CreateManager(&config);
    EXPECT_EQ(MER_GetNumWorlds(mgr), 10);
    MER_DestroyManager(mgr);
}
```

**After (Direct C++)**:
```cpp
TEST(ManagerTest, GetNumWorlds) {
    Manager::Config config{.numWorlds = 10};
    Manager mgr(config);
    EXPECT_EQ(mgr.numWorlds(), 10);
    // Automatic cleanup via RAII
}
```

### Example 2: Complex Interaction Test

**Before (C API)**:
```cpp
TEST(ViewerTest, RecordAndReplay) {
    // Complex setup through C API
    MER_ManagerHandle mgr = MER_CreateManager(&config);
    MER_ViewerHandle viewer = MER_CreateViewer(mgr, &viewerConfig);
    
    MER_StartRecording(viewer, "test.bin");
    MER_Step(mgr);
    MER_StopRecording(viewer);
    
    // Can't verify internal recording state
    
    MER_DestroyViewer(viewer);
    MER_DestroyManager(mgr);
}
```

**After (Direct C++)**:
```cpp
TEST_F(ViewerCoreTestBase, RecordAndReplay) {
    ASSERT_TRUE(CreateManager());
    ASSERT_TRUE(CreateViewer());
    
    viewer->startRecording("test.bin");
    EXPECT_TRUE(viewer->isRecording());  // Can check internal state!
    
    mgr->step();
    
    auto recordedFrames = viewer->getRecordedFrameCount();  // Direct access
    EXPECT_EQ(recordedFrames, 1);
    
    viewer->stopRecording();
    EXPECT_FALSE(viewer->isRecording());
    
    // Verify file was created
    EXPECT_TRUE(std::filesystem::exists("test.bin"));
}
```

### Example 3: Testing Protected Methods

**Before (C API)**: Not possible to test protected methods

**After (Direct C++)**:
```cpp
// Create test-specific friend class or use FRIEND_TEST macro
class ViewerCoreTestAccess : public ViewerCore {
public:
    using ViewerCore::processInternalState;  // Expose protected method
    using ViewerCore::validateConfiguration;
};

TEST(ViewerCoreInternals, ValidateConfiguration) {
    ViewerCore::Config config{/* invalid config */};
    ViewerCoreTestAccess viewer;
    
    EXPECT_FALSE(viewer.validateConfiguration(config));
    // Direct testing of protected validation logic
}
```