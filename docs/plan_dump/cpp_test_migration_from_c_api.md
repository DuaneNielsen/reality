# Plan: Migrate C++ Tests from C API to Direct Manager Usage - Single Test Pilot

## Context and Background

### Why This Change Is Needed

The Madrona Escape Room project is undergoing a significant architectural shift in how Python bindings are generated and maintained:

1. **Previous Approach**: Manual maintenance of duplicate struct definitions
   - C++ had `madEscape::CompiledLevel` in types.hpp
   - C API had `MER_CompiledLevel` in madrona_escape_room_c_api.h
   - Python had manual ctypes definitions
   - This led to synchronization issues and maintenance burden

2. **New Approach**: Automatic generation from compiled binaries
   - Single source of truth: C++ `madEscape::CompiledLevel` in types.hpp
   - Python structs auto-generated using `pahole` to extract exact memory layouts
   - Constants auto-generated using `libclang` AST parsing
   - No more manual `MER_CompiledLevel` definition in C API header

3. **The Problem**: C++ tests currently depend on `MER_CompiledLevel`
   - Tests were written assuming MER_CompiledLevel exists in C API
   - With the new codegen approach, MER_CompiledLevel is no longer defined
   - This causes compilation errors in all C++ test files

## Pre-Reading List

Before implementing this plan, review these files to understand the current state:

### Current Branch (feature/revenge_of_codegen)
1. **include/madrona_escape_room_c_api.h** - Note line 77-79: CompiledLevel no longer defined here
2. **src/types.hpp** - Line 144: `struct CompiledLevel` is defined in madEscape namespace
3. **tests/cpp/unit/test_recording_utilities.cpp** - Our pilot test file (uses MER_CompiledLevel extensively)
4. **tests/cpp/fixtures/test_base.hpp** - Base class that all tests inherit from

## The Plan: Single Test Pilot

### Pilot Test Selection

We'll use **test_recording_utilities.cpp** as our pilot because:
- It heavily uses MER_CompiledLevel (7+ occurrences)
- It's a unit test (simpler than integration tests)
- It tests a specific feature (recording) that's easy to verify
- It has clear pass/fail criteria

### Implementation Approach

#### Step 1: Create a Minimal Compatibility Header

Create a new file: `tests/cpp/fixtures/compiled_level_compat.hpp`
```cpp
#pragma once

#include "types.hpp" // Get madEscape::CompiledLevel

// Create type alias for backward compatibility
// This allows existing test code using MER_CompiledLevel to work unchanged
using MER_CompiledLevel = madEscape::CompiledLevel;
```

#### Step 2: Update Only test_recording_utilities.cpp

Add at the top of the file:
```cpp
#include "fixtures/compiled_level_compat.hpp"
```

This single include should:
- Make MER_CompiledLevel available as an alias
- Allow the test to compile without any other changes

#### Step 3: Build and Test

```bash
# Build just the recording utilities test
cd build
make mad_escape_test_recording_utilities

# Run the test
./tests/cpp/mad_escape_test_recording_utilities
```

### Expected Issues to Flush Out

1. **Include Path Issues**
   - Can test files find types.hpp?
   - Do we need to update CMakeLists.txt include directories?

2. **Size/Alignment Issues**
   - Does sizeof(MER_CompiledLevel) match expectations?
   - Are there any padding differences?

3. **C API Function Compatibility**
   - Does mer_create_manager accept the C++ struct via void*?
   - Does mer_write_compiled_level work correctly?

4. **Namespace Issues**
   - Are all madEscape types accessible?
   - Do we need additional using statements?

5. **Binary Compatibility**
   - Can we read/write recording files correctly?
   - Does the embedded level extraction still work?

### Success Criteria for Pilot

1. ✅ test_recording_utilities.cpp compiles without errors
2. ✅ All tests in the file pass
3. ✅ No runtime crashes or memory issues
4. ✅ Recording files can be created and read correctly
5. ✅ Embedded level extraction works

### If Pilot Succeeds

Once the pilot test works, we can:
1. Move the compatibility header to a common location
2. Include it in test_base.hpp for all tests to use
3. Apply the same fix to remaining test files

### If Pilot Fails

Depending on the failure mode:

1. **Include path issues**: Update CMakeLists.txt
2. **Size mismatches**: Investigate struct packing/alignment
3. **API incompatibility**: May need wrapper functions
4. **Namespace issues**: Add more using statements or explicit namespaces

### Files to Modify for Pilot

1. **Create**: `tests/cpp/fixtures/compiled_level_compat.hpp` (new file)
2. **Modify**: `tests/cpp/unit/test_recording_utilities.cpp` (add include)
3. **Possibly modify**: `tests/cpp/CMakeLists.txt` (if include paths needed)

### Estimated Time

- Implementation: 5-10 minutes
- Build & Test: 5 minutes  
- Debugging (if issues): 10-20 minutes
- Total: 10-35 minutes

### Risk Assessment

**Low Risk** - This is a minimal change that:
- Only affects one test file
- Uses a simple type alias
- Doesn't change any logic
- Can be easily reverted

### Next Steps After Pilot

If successful:
1. Document any issues found and solutions
2. Apply fix to test_base.hpp for all tests
3. Verify all C++ tests compile and pass
4. Consider if any test-specific adjustments are needed

This pilot approach allows us to validate the solution with minimal risk before applying it broadly.