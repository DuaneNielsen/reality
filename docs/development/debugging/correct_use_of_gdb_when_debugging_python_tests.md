# Correct Use of GDB When Debugging Python Tests

This document provides the correct process for using GDB to debug C++ code that is called from Python tests in the Madrona Escape Room project.

## The Problem

When debugging C++ code that's executed through Python tests (e.g., pytest running tests that call into the C++ shared library), it's common to make these mistakes:

1. **Running the shared library directly in GDB** - This doesn't work because shared libraries need to be loaded by an executable
2. **Trying to debug the test runner (uv/pytest) directly** - This is overly complex and doesn't load the right symbols
3. **Not setting up pending breakpoints** - The C++ symbols aren't loaded until the shared library is dynamically loaded

## The Correct Process

### Step 1: Find the Python Executable Used by Your Environment

```bash
# For uv-managed projects
uv run which python
# Output: /home/duane/madrona_escape_room/.venv/bin/python
```

### Step 2: Start GDB and Load the Python Executable

```bash
gdb /home/duane/madrona_escape_room/.venv/bin/python
```

Or using the MCP GDB tool:

```python
mcp__gdb__gdb_start()
mcp__gdb__gdb_load(
    sessionId="your_session_id", 
    program="/home/duane/madrona_escape_room/.venv/bin/python",
    arguments=["-m", "pytest", "tests/python/test_file.py::test_function", "-v", "-s"]
)
```

### Step 3: Enable Pending Breakpoints

This is CRUCIAL because the C++ shared library isn't loaded yet:

```gdb
(gdb) set breakpoint pending on
```

### Step 4: Set Your C++ Breakpoints

```gdb
(gdb) break batch_renderer.cpp:178
# This will show as "pending" until the shared library loads
```

### Step 5: Run the Test

```gdb
(gdb) run
```

The program will:
1. Start Python
2. Load pytest
3. Import your Python modules
4. Load the C++ shared library (this is when breakpoints become active)
5. Execute the test code
6. Hit your breakpoint when the C++ code is called

## Real Example: Debugging Shader Path Resolution

This example shows how we discovered a critical bug in shader selection:

```gdb
$ gdb /home/duane/madrona_escape_room/.venv/bin/python
(gdb) set args -m pytest tests/python/test_horizontal_lidar.py::TestHorizontalLidar::test_64x64_depth_configuration_debug -v -s
(gdb) set breakpoint pending on
(gdb) break batch_renderer.cpp:178
(gdb) run

# Program loads many libraries, then hits:
Thread 1 "python" hit Breakpoint 1, madrona::render::makeDrawShaders (
    dev=..., 
    repeat_sampler=0xb2ba420, 
    clamp_sampler=0xb2ba420, 
    depth_only=false  # <-- KEY INSIGHT: We're using RGB shader, not depth shader!
) at /home/duane/madrona_escape_room/external/madrona/src/render/batch_renderer.cpp:184
```

This revealed that our test was using `batch_draw_rgb.hlsl` instead of `batch_draw_depth.hlsl`, explaining why our depth shader modifications weren't taking effect.

## Key Success Factors

1. **Use the correct Python executable** - The one from your virtual environment
2. **Always enable pending breakpoints first** - `set breakpoint pending on`
3. **Set breakpoints before running** - The shared library loading happens during test execution
4. **Be patient** - Python and pytest take time to start up and load modules
5. **Check function arguments** - They often reveal critical information about execution flow

## Common Mistakes to Avoid

- ❌ Loading the shared library directly: `gdb build/libmadrona_escape_room_c_api.so`
- ❌ Trying to run complex command chains in GDB
- ❌ Not enabling pending breakpoints
- ❌ Setting breakpoints after the program has already run
- ❌ Using the wrong Python executable (system Python vs venv Python)

## When This Approach Works Best

- Debugging C++ code called from Python tests
- Understanding execution flow in hybrid Python/C++ codebases
- Investigating why C++ modifications aren't taking effect
- Analyzing function parameters and call stacks across language boundaries

## Alternative Approaches

If GDB is still problematic, consider:
- Adding debug prints to C++ code and rebuilding
- Using logging frameworks in the C++ code
- Creating standalone C++ test programs that reproduce the issue
- Using Python's `pdb` debugger to understand the Python side before diving into C++