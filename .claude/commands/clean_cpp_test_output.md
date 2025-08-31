---
argument-hint: [test executable path or 'all' for complete cleanup]
description: Remove extraneous print statements from C++ unit tests to achieve clean GoogleTest output
---
@.claude/include/substitutions.md

# algorithm

test_target = $ARGUMENT
cleanup_iterations = 0

**Systematic Approach:** The key is to be methodical - identify source, determine stream (stdout/stderr), choose cleanup method (removal vs capture), verify results.

**Criteria for Clean Output:** The only acceptable output should be standard GoogleTest framework messages:
- `[==========]` test run headers/footers
- `[----------]` test suite separators  
- `[ RUN      ]` individual test start
- `[       OK ]` / `[  FAILED  ]` test results
- `[  PASSED  ]` / `[  SKIPPED ]` summaries
- Empty lines and timing info

while (test output has dirty messages) todo:
   1. run tests and filter for dirty output
   2. locate source of dirty messages
   3. remove extraneous output calls
   4. rebuild test executables

report cleanup success

## test output has dirty messages

Run the test executable and filter output to detect non-GoogleTest messages:

```bash
# Run tests and filter out clean GoogleTest output to reveal dirty messages
# Comprehensive GoogleTest filter covers all standard patterns
CLEAN_GTEST='^\[==========\]|^\[----------\]|^\[ RUN      \]|^\[       OK \]|^\[  FAILED  \]|^\[  PASSED  \]|^\[  SKIPPED \]|^[[:space:]]*$|^Running main|^Note:'
$WORKING_DIR/build/mad_escape_tests 2>&1 | grep -v -E "$CLEAN_GTEST" | head -20
```

```bash
# Enhanced dirty pattern detection - comprehensive coverage
DIRTY_PATTERNS='Recording to:|Recording complete:|printf|std::cout|fprintf|DEBUG|\[DEBUG\]|Creating manager|Destroying manager|Loaded.*assets|Signal handler|malloc|free|cuda|CUDA|ERROR:|WARNING:|std::'
$WORKING_DIR/build/mad_escape_tests 2>&1 | grep -E "$DIRTY_PATTERNS" | head -10
```

Continue loop if any output appears (dirty messages found).
Stop loop if no output appears (clean GoogleTest format achieved).

Use the Bash tool to execute and check:
```tool
Bash(command="CLEAN_GTEST='^\[==========\]|^\[----------\]|^\[ RUN      \]|^\[       OK \]|^\[  FAILED  \]|^\[  PASSED  \]|^\[  SKIPPED \]|^[[:space:]]*$|^Running main|^Note:'; $WORKING_DIR/build/mad_escape_tests 2>&1 | grep -v -E \"$CLEAN_GTEST\" | head -5", 
     description="Filter test output to detect dirty messages with comprehensive GoogleTest filter")
```

If command returns empty output, tests are clean. If it returns lines, dirty messages remain.

## run tests and filter for dirty output

Execute the target test executable and use bash filtering to isolate problematic output:

**For specific test executable:**
```bash
# Run and capture only dirty output with comprehensive filters
CLEAN_GTEST='^\[==========\]|^\[----------\]|^\[ RUN      \]|^\[       OK \]|^\[  FAILED  \]|^\[  PASSED  \]|^\[  SKIPPED \]|^[[:space:]]*$|^Running main|^Note:'
DIRTY_PATTERNS='Recording to:|Recording complete:|printf|std::cout|fprintf|DEBUG|\[DEBUG\]|Creating manager|Destroying manager|Loaded.*assets|Signal handler|malloc|free|cuda|CUDA|ERROR:|WARNING:|std::'
$WORKING_DIR/build/$test_target 2>&1 | grep -v -E "$CLEAN_GTEST" | grep -E "$DIRTY_PATTERNS" | head -10
```

**For all test executables:**
```bash
# Check all test executables with enhanced pattern detection
DIRTY_PATTERNS='Recording to:|Recording complete:|printf|std::cout|fprintf|DEBUG|\[DEBUG\]|Creating manager|Destroying manager|Loaded.*assets|Signal handler|malloc|free|cuda|CUDA|ERROR:|WARNING:|std::'
for test_exe in $WORKING_DIR/build/*tests; do
  echo "=== $test_exe ===" 
  $test_exe 2>&1 | grep -E "$DIRTY_PATTERNS" | head -5
done
```

Focus on patterns that commonly pollute test output:
- `Recording to:` / `Recording complete:`
- `[DEBUG]` prefixed messages  
- `Creating manager` / `Destroying manager`
- Raw `printf` / `std::cout` statements not from GoogleTest

Use grep with line numbers to help locate source:
```bash
# Get dirty output with context for source identification
DIRTY_PATTERNS='Recording to:|Recording complete:|printf|std::cout|fprintf|DEBUG|\[DEBUG\]|Creating manager|Destroying manager|Loaded.*assets|Signal handler|malloc|free|cuda|CUDA|ERROR:|WARNING:|std::'
$WORKING_DIR/build/mad_escape_tests 2>&1 | grep -n -E "$DIRTY_PATTERNS"
```

## locate source of dirty messages

Find the source of problematic output using systematic identification:

### Step 1: Identify which specific test produces the output

Use binary search with --gtest_filter to narrow down:
```bash
# Test specific test suite
$WORKING_DIR/build/$test_target --gtest_filter="CApiCPUTest.*" 2>&1 | grep -v -E "$CLEAN_GTEST"

# Narrow to specific test
$WORKING_DIR/build/$test_target --gtest_filter="CApiCPUTest.ManagerCreation" 2>&1 | grep -v -E "$CLEAN_GTEST"
```

### Step 2: Determine output stream (stdout vs stderr)

```bash
# Test if output goes to stderr (disappears with 2>/dev/null)
$WORKING_DIR/build/$test_target --gtest_filter="TestSuite.TestName" 2>/dev/null | grep -v -E "$CLEAN_GTEST"

# Test if output goes to stdout (disappears with 1>/dev/null) 
$WORKING_DIR/build/$test_target --gtest_filter="TestSuite.TestName" 1>/dev/null | grep -v -E "$CLEAN_GTEST"
```

### Step 3: Find test source files

Use the Grep tool to search source code for the exact dirty messages and test definitions:
```tool
Grep(pattern="Recording to:|Recording complete:", glob="src/**/*.cpp", output_mode="content", -n=true)
Grep(pattern="TEST_F.*TestName|TEST.*TestName", glob="src/**/*.cpp", output_mode="content", -n=true) 
Grep(pattern="std::cout.*Recording|printf.*Recording", glob="src/**/*.cpp", output_mode="content", -n=true)
```

Search for the specific test producing output:
```bash
# Find test definition
grep -r "TEST.*ManagerCreation" src/ --include="*.cpp" -n
```

### Step 4: Check for GoogleTest capture setup

Check if test files already have capture includes:
```bash
# Look for existing capture setup
grep -r "CaptureStdout\|CaptureStderr" src/ --include="*.cpp" -n
```

Identify the specific files, line numbers, and output streams that need cleanup.

## remove extraneous output calls

**Option 1: Remove debug/logging statements (for non-test output)**

Use the Edit tool to remove problematic statements from source code:
```tool
Edit(file_path="src/mgr.cpp", 
     old_string="std::cout << \"Recording to: \" << filepath << \" (with embedded level data)\\n\";", 
     new_string="")
```

**Option 2: Add GoogleTest output capture (for test-specific output)**

For output originating from within tests, use GoogleTest capture instead of removal:

### Add capture includes if missing:
```tool
Edit(file_path="src/test_file.cpp",
     old_string="#include <gtest/gtest.h>",
     new_string="#include <gtest/gtest.h>\n#include <gtest/gtest-internal-inl.h>\n\nusing testing::internal::CaptureStdout;\nusing testing::internal::GetCapturedStdout;\nusing testing::internal::CaptureStderr;\nusing testing::internal::GetCapturedStderr;")
```

### Wrap test body with capture:
```tool
Edit(file_path="src/test_file.cpp",
     old_string="TEST_F(TestSuite, TestName) {\n    // existing test code",
     new_string="TEST_F(TestSuite, TestName) {\n    CaptureStdout();  // or CaptureStderr() based on stream detection\n    // existing test code")
```

### Add output verification at test end:
```tool
Edit(file_path="src/test_file.cpp",
     old_string="    // end of test\n}",
     new_string="    // end of test\n    std::string output = GetCapturedStdout();\n    EXPECT_TRUE(output.find(\"expected message\") != std::string::npos);\n}")
```

### Fix GTEST_SKIP with custom messages:
```tool
Edit(file_path="src/test_file.cpp",
     old_string="GTEST_SKIP() << \"custom message\";",
     new_string="GTEST_SKIP();")
```

**Target common patterns:**
- Manager lifecycle logging in tests
- Recording/replay status messages  
- Debug output from test utilities
- Custom GTEST_SKIP messages
- printf/cout statements in test code

**Preserve essential output:**
- GoogleTest framework messages
- Critical error reporting that affects test results
- User-facing functionality that tests verify

## rebuild test executables

Recompile the project to ensure removed debug statements take effect:

Use the Task tool with project-builder agent:
```tool
Task(subagent_type="project-builder", description="Rebuild after debug statement removal", 
     prompt="Rebuild the C++ project to compile the updated source code without debug statements")
```

Or use direct build commands:
```bash
# Rebuild project
cd $WORKING_DIR && cmake --build build --parallel 8
```

Verify build success:
```bash
echo "Build exit code: $?" 
# Exit code 0 = success, non-zero = build failure
```

If build fails due to syntax errors from cleanup, fix compilation issues before proceeding.

Increment cleanup iteration counter for progress tracking.

### Verification Steps:

**Step 1: Run the comprehensive filter to verify cleanup:**
```bash
CLEAN_GTEST='^\[==========\]|^\[----------\]|^\[ RUN      \]|^\[       OK \]|^\[  FAILED  \]|^\[  PASSED  \]|^\[  SKIPPED \]|^[[:space:]]*$|^Running main|^Note:'
$WORKING_DIR/build/$test_target 2>&1 | grep -v -E "$CLEAN_GTEST" | wc -l
# Should return 0
```

**Step 2: If still non-zero, examine remaining output:**
```bash
# Check for new framework patterns
$WORKING_DIR/build/$test_target 2>&1 | grep -v -E "$CLEAN_GTEST" | head -10
```

**Step 3: Update filters if new GoogleTest patterns found:**
```bash
# Document and add new patterns to CLEAN_GTEST if they're legitimate framework output\n# Example: if GoogleTest adds new output format like \"[  INFO    ]\", add to filter
```

## output format

if tests are clean

    ✅ **C++ Test Output Cleanup Successful** \n\n
    
    **Target:** $ARGUMENT \n\n
    
    **Cleanup Iterations:** [ITERATION_COUNT] \n\n
    
    **Final Status:** Clean GoogleTest output achieved \n\n
    
    **Verification:** \n\n
    ```bash
    # No dirty output detected - comprehensive GoogleTest filter
    CLEAN_GTEST='^\[==========\]|^\[----------\]|^\[ RUN      \]|^\[       OK \]|^\[  FAILED  \]|^\[  PASSED  \]|^\[  SKIPPED \]|^[[:space:]]*$|^Running main|^Note:'
    $WORKING_DIR/build/mad_escape_tests 2>&1 | grep -v -E "$CLEAN_GTEST" | wc -l
    # Output: 0 (clean)
    ```
    
    **Ready for Failed Test Filtering:** \n\n
    ```bash
    # Show only failed tests and summaries (comprehensive filter)
    PASSED_PATTERNS='^\[ RUN      \]|^\[       OK \]'
    $WORKING_DIR/build/mad_escape_tests 2>&1 | grep -v -E "$PASSED_PATTERNS"
    ```

else

    ❌ **Cleanup Incomplete After Maximum Iterations** \n\n
    
    **Target:** $ARGUMENT \n\n
    
    **Iterations Attempted:** [MAX_ITERATIONS] \n\n
    
    **Remaining Dirty Output:** \n\n
    ```
    [SAMPLE_DIRTY_MESSAGES]
    ```
    
    **Next Steps:** Manual investigation required for persistent debug output \n\n
    
    **Debug Command:** \n\n
    ```bash
    # Enhanced dirty pattern detection for debugging
    DIRTY_PATTERNS='Recording to:|Recording complete:|printf|std::cout|fprintf|DEBUG|\[DEBUG\]|Creating manager|Destroying manager|Loaded.*assets|Signal handler|malloc|free|cuda|CUDA|ERROR:|WARNING:|std::'
    $WORKING_DIR/build/mad_escape_tests 2>&1 | grep -E "$DIRTY_PATTERNS"
    ```