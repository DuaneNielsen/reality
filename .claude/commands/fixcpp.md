# C++ Test Fixing Process

The cpp test `$ARGUMENTS` is failing and we need to fix it.

Use the following process to fix the test, leveraging the traceability system when available:

## 1. Run the test to verify the error message

**For C++ tests with detailed output:**

For cpu tests (with debug output visible):

```bash
./build/mad_escape_tests --gtest_filter="CApiCPUTest.ManagerCreation" --disable-capture --gtest_print_time=1 --gtest_output=json -v
```

For gpu tests (with debug output visible):

```bash
./build/mad_escape_gpu_tests --gtest_filter="CApiGPUTest.ManagerCreationWithEmbeddedLevels" --disable-capture --gtest_print_time=1 --gtest_output=json -v
```

**Note:** The `--disable-capture` flag disables GoogleTest's stdout capture, `--gtest_print_time=1` shows execution time, `--gtest_output=json` provides structured output, and `-v` enables verbose output. These flags are essential for debugging failing tests.

**If the C++ test is called from Python (pytest wrapper):**

Run the Python test wrapper with verbose mode to get specification context:

```bash
# Run Python test that calls C++ code with spec display
uv run --group dev pytest tests/python/[TEST_FILE].py::[TEST_NAME] -vs --tb=long
```

**Benefits of Python pytest verbose mode:**
- Automatically displays relevant specification sections when tests fail
- Shows @pytest.mark.spec() markers and their documentation
- Provides immediate context about expected behavior according to specs
- Links C++ test failures to requirement specifications


## 2. Read the test code and create a summary

### Test Summary Format

#### `<testname>` Summary

**What is being tested:** `<description of the intent of the test>`

**The test process:**
1. CompiledLevel is populated with an agent spawn point
2. The test initializes a cpu_manager
3. The initial state is read
4. ...

**Assertions:**
1. The agent pos x is inside the level_x_min, level_x_max
2. The Agent pos is at x=-17, y=-19

**Reason for failure:**
- Assert 2. failed, the agent was not at x=-17, y=-19

## 3. Validate Test Premise

**Decision Point:** Is the premise of the test correct by checking against specifications?

**Step 1: Check for Python test wrapper with traceability**

Many C++ tests are called through Python pytest wrappers that may have @pytest.mark.spec() markers:

```tool
Glob(pattern="tests/python/*[RELATED_TEST_NAME]*.py")
```

If found, examine the Python wrapper for specification links:
```tool
Read(file_path="tests/python/[WRAPPER_FILE].py")
```

Look for @pytest.mark.spec() decorators that provide specification context:
```python
@pytest.mark.spec("docs/specs/mgr.md", "Manager")
def test_cpp_manager_functionality():
    # Calls C++ test executable
```

**Step 2: If specification markers found - Use them**

**Extract specification reference:**
- Spec file: `[SPEC_FILE_PATH]` from marker
- Section: `[SECTION_NAME]` from marker

**Read the referenced specification:**
```tool
Read(file_path="[SPEC_FILE_PATH]")
```

**Step 3: If no Python wrapper or markers - Legacy validation**

**Direct C++ test validation:**
- Examine the C++ test code to understand what system/component is being tested
- Look for related specification documentation in `docs/specs/`
- Check if the test matches documented behavior requirements

**Step 4: Validate test assertions against specification**

**Validation checklist:**
- Does the C++ test verify behavior documented in specifications?
- Are the test assertions consistent with requirement specifications?
- Does the test follow GoogleTest patterns and best practices?
- Is the test setup appropriate for what's being tested per the spec?

**Step 5: Handle validation results**

**If test-spec alignment confirmed:**
- Test premise is valid, proceed to step 4 (hypothesis formulation)
- C++ implementation should match documented specifications

**If test-spec mismatch found:**
- Work with user to determine if test or specification needs updating
- Consider adding @pytest.mark.spec() markers to Python wrappers for better traceability

**If no specification found:**
- Prompt the user to validate if the test logic and assertions are correct
- May need to create specification documentation for the tested component
- Proceed carefully with assumption that test logic represents intended behavior

## 4. Formulate Working Hypothesis

Read the code and think. When sufficient information is gathered, formulate a hypothesis about what is causing the test assertions to fail.

### Hypothesis Format

**Working hypothesis:** `<hypothesis and reasoning>`

**Assertion:** When `[methodcall]` in `[filename]:[lineno]` is called, `[variable]` is expected to be `[value]`

### Example:

**Hypothesis:** The test is failing because the Agent Pos in the CompiledLevel has been set to the wrong values by the level compiler.

**Assertion:** When `resetAgentPhysics()` in `level_gen.cpp:182` is called, `level.spawn_x[0]` should not be -17 and `level.spawn_y[0]` should not be -19

## 5. Test Hypothesis

Assume the hypothesis is false until the assertion is found to support it.

## 6. Debug Loop

While hypothesis is false, repeat the following steps:

### a. Start the test in the debugger

For C++ tests (direct executable):
```gdb_tool
gdb_load (MCP)(sessionId: "<session_id>", program: "./build/mad_escape_tests", arguments: ["--gtest_filter=CApiCPUTest.ManagerCreation", "--disable-capture", "--gtest_print_time=1", "--gtest_output=json", "-v"])
```

For Python tests calling C++ code (requires different approach):
```gdb_tool
# First find the Python executable
# uv run which python -> /home/duane/madrona_escape_room/.venv/bin/python

gdb_start()
gdb_load(
    sessionId="<session_id>", 
    program="/home/duane/madrona_escape_room/.venv/bin/python",
    arguments=["-m", "pytest", "tests/python/test_file.py::test_function", "-v", "-s"]
)

# CRUCIAL: Enable pending breakpoints before setting them
gdb_command(sessionId="<session_id>", command="set breakpoint pending on")
```

**Note:** For Python tests, C++ symbols aren't loaded until the shared library loads during test execution, requiring pending breakpoints.

### b. Set the breakpoint

```gdb_tool
gdb_set_breakpoint (MCP)(sessionId: "<session_id>", location: "level_gen:182")
```

### c. Run the program

```gdb_tool
gdb_command (MCP)(sessionId: "<session_id>", command: "run")
```

### d. Wait for breakpoint hit

```gdb_tool
gdb_command (MCP)(sessionId: "<session_id>", command: "info threads")
```

### e. Verify the assertion

```gdb_tool
gdb_print (MCP)(sessionId: "<session_id>", expression: "level.level_name")
```

### f. Decision Point

- **If hypothesis assertion is TRUE:** Create a plan for a fix and propose it to the user
- **If hypothesis assertion is FALSE:** Reformulate a new working hypothesis

### g. Reformulate Working Hypothesis (if needed)

Read the code and think. When sufficient information is gathered, formulate a new hypothesis:

**Working hypothesis:** `<hypothesis and reasoning>`

**Assertion:** When `[methodcall]` in `[filename]:[lineno]` is called, `[variable]` is expected to be `[value]`

### h. return to step a and repeat until working hypothesis is correct

## Additional Python Test Debugging Notes

### Common Mistakes to Avoid
- ❌ Loading shared library directly: `gdb build/libmadrona_escape_room_c_api.so`
- ❌ Not enabling pending breakpoints for Python tests
- ❌ Using system Python instead of venv Python
- ❌ Setting breakpoints after program already ran

### Execution Flow for Python Tests
1. Python starts and loads pytest
2. Python imports your modules  
3. C++ shared library loads (breakpoints become active)
4. Test executes and hits your C++ breakpoints

### Real Example: Shader Path Bug Discovery
When debugging Python test calling C++ rendering code:
```gdb
Thread 1 "python" hit Breakpoint 1, madrona::render::makeDrawShaders (
    depth_only=false  # <-- KEY: Using RGB shader, not depth shader!
) at batch_renderer.cpp:184
```
This revealed tests were using `batch_draw_rgb.hlsl` instead of `batch_draw_depth.hlsl`.