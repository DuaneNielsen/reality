---
argument-hint: [cpp test file path or 'all' for all test files]
description: review C++ test files to ensure compliance with GoogleTest standards and project conventions
---
@.claude/include/substitutions.md

# algorithm

test_file_path = $ARGUMENT
violations_found = []

1. analyze test file structure
2. check fixture usage compliance  
3. verify test organization
4. validate googletest patterns
5. check gpu test constraints
6. clean test output
7. fix standard violations
8. rebuild and verify
9. generate compliance report

## analyze test file structure

Read the target test file to understand current implementation:

```tool
Read(file_path="$ARGUMENT")
```

If analyzing all tests, use Glob to find test files:
```tool
Glob(pattern="tests/cpp/**/*test*.cpp")
```

Examine the test structure for:
- Test file location (unit/, integration/, e2e/)
- Include statements and dependencies
- Test fixture usage
- GoogleTest macro usage
- GPU test patterns

Compare against C++ testing standards from tests/cpp/README.md:
- Proper fixture inheritance
- Test naming conventions
- GPU test constraints
- Direct C++ vs C API usage

## check fixture usage compliance

Identify violations in fixture usage:

**Standard Fixture Classes:**
- `MadronaCppTestBase` - Base for direct C++ tests (recommended)
- `ViewerCoreTestBase` - For ViewerCore tests
- `ReusableGPUManagerTest` - Fast GPU tests with shared manager
- `CustomGPUManagerTest` - Stress tests with custom managers
- `MadronaTestBase` - C API tests (legacy, avoid for new tests)

**Common Violations:**
- Using C API fixtures for new tests (should use direct C++)
- Not inheriting from appropriate base class
- Creating managers manually instead of using fixture helpers
- Missing SetUp()/TearDown() overrides
- Wrong fixture for test type (e.g., CustomGPUManagerTest in fast suite)

Check for proper fixture inheritance:
```tool
Grep(pattern="class.*Test.*:.*public", path="$ARGUMENT", output_mode="content", -n=true)
```

Verify CreateManager() usage:
```tool
Grep(pattern="CreateManager\\(\\)|MER_CreateManager", path="$ARGUMENT", output_mode="content", -n=true)
```

## verify test organization

Check test file organization and placement:

**Directory Structure:**
- `tests/cpp/unit/` - Unit tests for individual components
- `tests/cpp/integration/` - Integration tests
- `tests/cpp/e2e/` - End-to-end tests
- `tests/cpp/fixtures/` - Shared fixtures and utilities

**Test Executable Assignment:**
- `mad_escape_tests` - CPU tests (fast, ~30 seconds)
- `mad_escape_gpu_tests` - Fast GPU tests (shared manager, ~1-2 minutes)
- `mad_escape_gpu_stress_tests` - Stress tests (individual managers, ~8+ minutes)

**Violations to Check:**
- Test in wrong directory for its type
- GPU test in CPU executable
- Stress test in fast GPU suite
- Missing from CMakeLists.txt

Check CMakeLists.txt for test inclusion:
```tool
Grep(pattern="$ARGUMENT|TEST_SOURCES|GPU_TEST_SOURCES|STRESS_TEST_SOURCES", path="tests/cpp/CMakeLists.txt", output_mode="content", -n=true)
```

## validate googletest patterns

Review GoogleTest usage patterns:

**Required Patterns:**
- Test names: `TEST(SuiteName, TestName)` or `TEST_F(FixtureName, TestName)`
- Suite names ending with `Test`
- CamelCase for test and suite names
- Proper assertion usage (ASSERT_* vs EXPECT_*)

**Common Issues:**
- Using `ASSERT_*` when `EXPECT_*` would be better
- Missing `ASSERT_TRUE(CreateManager())` before using manager
- Not checking return values
- Improper parameterized test setup

Check test naming:
```tool
Grep(pattern="TEST(_F)?\\(\\w+,\\s*\\w+\\)", path="$ARGUMENT", output_mode="content", -n=true)
```

Check for proper assertions:
```tool
Grep(pattern="ASSERT_|EXPECT_|GTEST_SKIP", path="$ARGUMENT", output_mode="content", -n=true, -B=1, -A=1)
```

## check gpu test constraints

Validate GPU test compliance with constraints:

**Critical GPU Constraints:**
- Only one GPU manager per process lifetime
- NVRTC compilation takes ~40-45 seconds per test
- Must check CUDA availability
- Must set ALLOW_GPU_TESTS_IN_SUITE=1 environment variable

**Required Patterns for GPU Tests:**
```cpp
// Check CUDA availability
if (!CheckCudaAvailable()) {
    GTEST_SKIP() << "CUDA not available";
}

// Check environment variable
const char* allow_gpu = std::getenv("ALLOW_GPU_TESTS_IN_SUITE");
if (!allow_gpu || std::string(allow_gpu) != "1") {
    GTEST_SKIP() << "GPU tests disabled";
}
```

**Fast GPU Test Pattern (ReusableGPUManagerTest):**
- Use shared manager from SetUp()
- Call ResetAllWorlds() between tests
- Never call CreateManager() in test body

**Stress GPU Test Pattern (CustomGPUManagerTest):**
- Can create custom manager with CreateManager()
- Expect long compilation time
- Should be in mad_escape_gpu_stress_tests executable

Check for GPU test guards:
```tool
Grep(pattern="CheckCudaAvailable|ALLOW_GPU_TESTS_IN_SUITE|GTEST_SKIP", path="$ARGUMENT", output_mode="content", -n=true)
```

## clean test output

Check for and remove extraneous output that pollutes GoogleTest results:

**Clean GoogleTest Output Patterns:**
- `[==========]` test run headers/footers
- `[----------]` test suite separators
- `[ RUN      ]` individual test start
- `[       OK ]` / `[  FAILED  ]` test results
- `[  PASSED  ]` / `[  SKIPPED ]` summaries

**Common Dirty Output Sources:**
- `std::cout` / `printf` statements in tests
- `Recording to:` / `Recording complete:` messages
- Debug output from manager creation
- Custom GTEST_SKIP messages

Check for output statements:
```tool
Grep(pattern="std::cout|printf|fprintf|std::cerr|DEBUG|Recording", path="$ARGUMENT", output_mode="content", -n=true)
```

**Output Capture Pattern for Tests:**
```cpp
// Add GoogleTest output capture
#include <gtest/gtest-internal-inl.h>
using testing::internal::CaptureStdout;
using testing::internal::GetCapturedStdout;

TEST_F(TestFixture, TestName) {
    CaptureStdout();  // Capture output
    // Test code that produces output
    std::string output = GetCapturedStdout();
    // Optionally verify output
}
```

## fix standard violations

Apply fixes using the Edit tool for each violation found:

**Fix Fixture Inheritance:**
```tool
Edit(file_path="$ARGUMENT",
     old_string="class MyTest : public MadronaTestBase",
     new_string="class MyTest : public MadronaCppTestBase")
```

**Add GPU Test Guards:**
```tool
Edit(file_path="$ARGUMENT",
     old_string="TEST_F(MyGPUTest, TestName) {",
     new_string="TEST_F(MyGPUTest, TestName) {\n    if (!CheckCudaAvailable()) {\n        GTEST_SKIP() << \"CUDA not available\";\n    }")
```

**Remove Debug Output:**
```tool
Edit(file_path="$ARGUMENT",
     old_string="std::cout << \"Debug: \" << value << std::endl;",
     new_string="// Debug output removed for clean test output")
```

**Add Output Capture:**
```tool
Edit(file_path="$ARGUMENT",
     old_string="TEST_F(TestFixture, TestName) {\n    // Test with output",
     new_string="TEST_F(TestFixture, TestName) {\n    CaptureStdout();  // Capture output for clean test results\n    // Test with output")
```

**Fix Test Naming:**
```tool
Edit(file_path="$ARGUMENT",
     old_string="TEST(testSuite, test_name)",
     new_string="TEST(TestSuite, TestName)")
```

**Update Manager Creation:**
```tool
Edit(file_path="$ARGUMENT",
     old_string="MER_ManagerHandle handle = MER_CreateManager(&config);",
     new_string="ASSERT_TRUE(CreateManager());  // Use fixture helper")
```

## rebuild and verify

Rebuild the test executables after applying fixes:

Use the Task tool with project-builder agent:
```tool
Task(subagent_type="project-builder", description="Rebuild C++ tests",
     prompt="Rebuild the C++ test executables after code changes. Run: make -C build mad_escape_tests mad_escape_gpu_tests -j8")
```

Or use direct build commands:
```bash
# Rebuild test executables
make -C build mad_escape_tests -j8
make -C build mad_escape_gpu_tests -j8
make -C build mad_escape_gpu_stress_tests -j8
```

Verify tests still pass:
```bash
# Run CPU tests
./build/mad_escape_tests --gtest_brief=1

# Check for clean output
CLEAN_GTEST='^\\[==========\\]|^\\[----------\\]|^\\[ RUN      \\]|^\\[       OK \\]|^\\[  FAILED  \\]|^\\[  PASSED  \\]|^\\[  SKIPPED \\]|^[[:space:]]*$|^Running main|^Note:'
./build/mad_escape_tests 2>&1 | grep -v -E "$CLEAN_GTEST" | wc -l
# Should output 0 for clean tests
```

## generate compliance report

Analyze the results and generate a comprehensive compliance report:

**Compliance Checks:**
1. Fixture usage matches recommendations
2. Test organization follows directory structure
3. GoogleTest patterns are properly used
4. GPU tests have proper constraints
5. Test output is clean
6. Tests pass after modifications

**Generate Summary:**
- Count total violations found
- List fixes applied
- Note any remaining issues
- Provide recommendations

## output format

**C++ Test Standards Review Complete** \\n\\n

**File:** `$ARGUMENT` \\n\\n

**Compliance Summary:** \\n\\n

**✅ Passed Checks:** \\n\\n
- Fixture Usage: [PASS/FAIL] \\n\\n
- Test Organization: [PASS/FAIL] \\n\\n
- GoogleTest Patterns: [PASS/FAIL] \\n\\n
- GPU Constraints: [PASS/FAIL/N/A] \\n\\n
- Clean Output: [PASS/FAIL] \\n\\n

**Violations Found:** [VIOLATION_COUNT] \\n\\n

**Fixes Applied:** \\n\\n
1. **Fixture Migration:** [DESCRIPTION] \\n\\n
2. **GPU Guards Added:** [DESCRIPTION] \\n\\n
3. **Output Cleanup:** [DESCRIPTION] \\n\\n
4. **Test Naming:** [DESCRIPTION] \\n\\n
5. **Manager Creation:** [DESCRIPTION] \\n\\n

**Test Results:** \\n\\n
```bash
# All tests passing
[==========] X tests from Y test suites ran.
[  PASSED  ] X tests.
```

**Recommendations:** \\n\\n
- [RECOMMENDATION_1] \\n\\n
- [RECOMMENDATION_2] \\n\\n
- [RECOMMENDATION_3] \\n\\n

**Next Steps:** \\n\\n
```bash
# Run full test suite
./tests/run_cpp_tests.sh

# Run specific test
./build/mad_escape_tests --gtest_filter="TestSuite.*"

# Run GPU tests (if applicable)
ALLOW_GPU_TESTS_IN_SUITE=1 ./build/mad_escape_gpu_tests
```