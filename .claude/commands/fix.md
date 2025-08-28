# Python Test Fixing Process

The pytest `$ARGUMENTS` is failing and we need to fix it.

**IMPORTANT: Test Selection Strategy**
- Always start by working on the actual failing unit test
- If the failing test doesn't adequately cover the scenario, you have options:
  1. **Enhance the existing test** - Add assertions or steps to make it more comprehensive
  2. **Add a new test function** - Create a properly structured test in the same file
  3. **Create a debug test** - For quick debugging only, write a minimal throwaway test that you can reliably execute in one shot

**Key Principle:** Production tests can be complex and thorough. Debug/throwaway tests must be simple and reliable - if you can't write it correctly on the first try, it's too complex for a debug test.

Use the following process to fix the test:

## 0. Add the following task list
* Run the test to verify the error message
* Read the test code and create a summary
* Validate Test Premise
* Formulate Working Hypothesis
* Start Debug Loop
* Decision Point - Reject/Accept Working Hypothesis

## 1. Run the test to verify the error message

```bash
# Run specific test with verbose output
uv run --group dev pytest tests/python/$ARGUMENTS -v --tb=short

# Or run with more detail
uv run --group dev pytest tests/python/$ARGUMENTS -vv --tb=long
```

## 2. Read the test code and create a summary

### Test Summary Format

#### `<testname>` Summary

**What is being tested:** `<description of the intent of the test>`

**The test process:**
1. SimManager is initialized with CPU/GPU mode
2. Initial observations are captured
3. Actions are applied to agents
4. Step() is called
5. ...

**Assertions:**
1. Agent position should be within level boundaries
2. Reward should be positive for forward movement
3. Done flag should be True when episode ends

**Reason for failure:**
- Assert 2. failed, reward was 0.0 instead of expected positive value

## 3. Validate Test Premise

**Decision Point:** Is the premise of the test correct?

- Prompt the user to validate if the test is valid and the assertions are correct
- Check if the test uses proper fixtures (cpu_manager, gpu_manager)
- Verify the test follows testing patterns from TESTING_GUIDE.md
- If valid, proceed to step 4
- If not, work with the user to redefine the test

## 4. Formulate Working Hypothesis

Read the code and think. When sufficient information is gathered, formulate a hypothesis about what is causing the test assertions to fail.

### Hypothesis Format

**Working hypothesis:** `<hypothesis and reasoning>`

**Assertion:** When `[method]` in `[filename]:[lineno]` is called, `[variable]` is expected to be `[value]`

### Example:

**Hypothesis:** The test is failing because the reward calculation in the step() method is not accounting for the agent's forward movement correctly.

**Assertion:** When `mgr.step()` in `madrona_escape_room/__init__.py:245` is called, `rewards[0,0]` should be > 0.0 after forward movement

Verify the working hypothesis by looking for evidence of the assertion using the debug loop.

Add the debug loop task list
* Start the test and set the breakpoint
* Run to the breakpoint and inspect variables to verify the assertion as true or false
* Reject or accept the hypothesis

## 5. Debug Loop

Add the following to the task list

While hypothesis is false, repeat the following steps:

### Choose Debug Method Based on Code Location

**For Python variables:** Use Python debugger (PDB)
**For C++ variables:** Use GDB debugger

---

### Option A: Python Debugging (for Python variables)

#### a. Start the test in the Python debugger

```tool
mcp__python-debugger-mcp__start_debug(file_path="tests/python/test_reward_system.py", use_pytest=true, args="--pdb -x test_reward_system.py::test_forward_movement_reward")
```

Alternative for single test function:
```tool
mcp__python-debugger-mcp__start_debug(file_path="tests/python/test_reward_system.py::test_forward_movement_reward", use_pytest=true)
```

#### b. Set the breakpoint

```tool
mcp__python-debugger-mcp__set_breakpoint(file_path="madrona_escape_room/__init__.py", line_number=245)
```

Or use relative path from project root:
```tool
mcp__python-debugger-mcp__set_breakpoint(file_path="tests/python/test_reward_system.py", line_number=50)
```

#### c. Run to the breakpoint

```tool
mcp__python-debugger-mcp__send_pdb_command(command="c")
```

#### d. Examine variables at breakpoint

```tool
mcp__python-debugger-mcp__examine_variable(variable_name="rewards")
mcp__python-debugger-mcp__examine_variable(variable_name="self.numWorlds")
mcp__python-debugger-mcp__examine_variable(variable_name="obs[0, 0, :3]")
```

#### e. Step through code

```tool
# Next line (step over)
mcp__python-debugger-mcp__send_pdb_command(command="n")

# Step into function
mcp__python-debugger-mcp__send_pdb_command(command="s")

# Return from current function
mcp__python-debugger-mcp__send_pdb_command(command="r")

# List current code context
mcp__python-debugger-mcp__send_pdb_command(command="l")

# List longer code context
mcp__python-debugger-mcp__send_pdb_command(command="ll")

# Print arguments of current function
mcp__python-debugger-mcp__send_pdb_command(command="a")

# Print stack trace
mcp__python-debugger-mcp__send_pdb_command(command="w")
```

---

### Option B: C++ Debugging with GDB (for C++ variables)

#### a. Start GDB session

```tool
mcp__gdb__gdb_start(gdbPath="gdb", workingDir="/home/duane/madrona_escape_room")
```

#### b. Configure GDB for Python test execution

```tool
# Enable pending breakpoints (for shared libraries loaded later)
mcp__gdb__gdb_command(sessionId="<session_id>", command="set breakpoint pending on")

# Load Python executable
mcp__gdb__gdb_command(sessionId="<session_id>", command="file /home/duane/madrona_escape_room/.venv/bin/python")

# Set arguments for pytest
mcp__gdb__gdb_command(sessionId="<session_id>", command="set args -m pytest tests/python/$ARGUMENTS -v --tb=short")
```

#### c. Set C++ breakpoints

```tool
# Set breakpoint by function name
mcp__gdb__gdb_set_breakpoint(sessionId="<session_id>", location="resetAgentPhysics")

# Or by file:line
mcp__gdb__gdb_set_breakpoint(sessionId="<session_id>", location="level_gen.cpp:118")

# Or by class::method
mcp__gdb__gdb_set_breakpoint(sessionId="<session_id>", location="madEscape::Sim::Sim")
```

#### d. Run the test

```tool
mcp__gdb__gdb_command(sessionId="<session_id>", command="run")
```

#### e. Examine C++ variables at breakpoint

```tool
# Print variable values
mcp__gdb__gdb_print(sessionId="<session_id>", expression="level.num_spawns")
mcp__gdb__gdb_print(sessionId="<session_id>", expression="level.spawn_x[0]")
mcp__gdb__gdb_print(sessionId="<session_id>", expression="pos.x")

# Print complex structures (may need to increase max-value-size)
mcp__gdb__gdb_command(sessionId="<session_id>", command="set max-value-size unlimited")
mcp__gdb__gdb_print(sessionId="<session_id>", expression="ctx.singleton<CompiledLevel>()")

# Show local variables
mcp__gdb__gdb_command(sessionId="<session_id>", command="info locals")

# Show function arguments
mcp__gdb__gdb_command(sessionId="<session_id>", command="info args")
```

#### f. Step through C++ code

```tool
# Step to next line (step over)
mcp__gdb__gdb_next(sessionId="<session_id>")

# Step into function
mcp__gdb__gdb_step(sessionId="<session_id>")

# Continue to next breakpoint
mcp__gdb__gdb_continue(sessionId="<session_id>")

# Finish current function
mcp__gdb__gdb_finish(sessionId="<session_id>")

# Show current code
mcp__gdb__gdb_command(sessionId="<session_id>", command="list")

# Show call stack
mcp__gdb__gdb_backtrace(sessionId="<session_id>")

# Show current location
mcp__gdb__gdb_command(sessionId="<session_id>", command="where")
```

#### g. Managing GDB session

```tool
# Check all breakpoints
mcp__gdb__gdb_command(sessionId="<session_id>", command="info breakpoints")

# Disable/enable breakpoints
mcp__gdb__gdb_command(sessionId="<session_id>", command="disable 1")
mcp__gdb__gdb_command(sessionId="<session_id>", command="enable 1")

# Delete breakpoints
mcp__gdb__gdb_command(sessionId="<session_id>", command="delete 1")

# Check thread state
mcp__gdb__gdb_command(sessionId="<session_id>", command="info threads")

# Terminate GDB session when done
mcp__gdb__gdb_terminate(sessionId="<session_id>")
```

### Common GDB Tips for Python Tests

1. **Optimized variables**: If you see `<optimized out>`, try:
   - Rebuild with debug symbols: `cmake -DCMAKE_BUILD_TYPE=Debug ..`
   - Step forward/backward to where the variable is actively used
   - Print the memory address directly

2. **Finding the right breakpoint during initialization vs reset**:
   - Many functions are called during both world creation and reset
   - You may need to disable early breakpoints to reach the actual test execution
   - Use conditional breakpoints: `break level_gen.cpp:118 if i == 0`

3. **Navigating Python/C++ boundary**:
   - The call stack will show Python frames mixed with C++ frames
   - Focus on the C++ frames for debugging C++ variables
   - Use `frame <n>` to switch between stack frames

### f. Decision Point - validate assertions

- **If hypothesis assertion is TRUE:** Create a plan for a fix and propose it to the user
- **If hypothesis assertion is FALSE:** Add the following task list

* Formulate Working Hypothesis
* Start Debug Loop
* Decision Point - Reject/Accept Working Hypothesis

### Reformulate

Read the code and think. When sufficient information is gathered, formulate a new hypothesis:

**Working hypothesis:** `<hypothesis and reasoning>`

**Assertion:** When `[method]` in `[filename]:[lineno]` is called, `[variable]` is expected to be `[value]`

### h. iterate in the Debug Loop until working hypothesis is supported

## 9. Fix Verification

After implementing the fix:

```bash
# Run the specific test
uv run --group dev pytest tests/python/$ARGUMENTS -v

# Run related tests to ensure no regression
uv run --group dev pytest tests/python/ -v --no-gpu

# If GPU test was fixed, verify GPU tests
uv run --group dev pytest tests/python/ -v -k "gpu"
```


## Additional Debug Techniques for Python Tests

### Managing Debug Sessions

```tool
# Check current debug status
mcp__python-debugger-mcp__get_debug_status()

# List all breakpoints
mcp__python-debugger-mcp__list_breakpoints()

# Clear a specific breakpoint
mcp__python-debugger-mcp__clear_breakpoint(file_path="tests/python/test_reward_system.py", line_number=50)

# Restart debugging session with same parameters
mcp__python-debugger-mcp__restart_debug()

# End the debugging session
mcp__python-debugger-mcp__end_debug()
```

### Using Debug Flags

```bash
# Record actions for replay debugging
uv run --group dev pytest tests/python/$ARGUMENTS --record-actions

# Trace agent trajectories
uv run --group dev pytest tests/python/$ARGUMENTS --trace-trajectories

# Auto-launch viewer after test
uv run --group dev pytest tests/python/$ARGUMENTS --record-actions --visualize
```

### Interactive Debugging with pytest -s

```bash
# Allow print statements and pdb.set_trace()
uv run --group dev pytest tests/python/$ARGUMENTS -s

# In test code, add:
import pdb; pdb.set_trace()
```

### Using Madrona REPL for Exploration

```tool
# Use mcp__madrona_repl__execute_python to explore simulation state
mcp__madrona_repl__execute_python(code="""
mgr = SimManager(exec_mode=madrona.ExecMode.CPU, num_worlds=1)
obs = mgr.self_observation_tensor().to_numpy()
print(f"Initial position: {obs[0, 0, :3]}")
print(f"Observation shape: {obs.shape}")
""")

# List variables in REPL session
mcp__madrona_repl__list_variables()

# Reset REPL session if needed
mcp__madrona_repl__execute_python(code="", reset=true)
```

## Common Python Test Issues

### GPU Manager Constraint
- Only one GPU manager per process
- Use gpu_manager fixture for GPU tests
- Never create SimManager(exec_mode=ExecMode.CUDA) directly in tests

### Custom Level Issues
- Ensure @pytest.mark.custom_level() decorator is used correctly
- Check that spawn points ('S') are placed in valid locations
- Verify level compilation with scale=2.5

### Tensor Shape Mismatches
- Action tensor: [numWorlds * numAgents, 3]
- Observation tensors: [numWorlds, numAgents, features]
- Always verify shapes match expected dimensions

### Import Errors
```bash
# Rebuild if C API changes
make -C build -j8 -s

# Reinstall Python package
uv pip install -e .
```
