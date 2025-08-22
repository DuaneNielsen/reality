# Python Test Fixing Process

The pytest `$ARGUMENTS` is failing and we need to fix it.

Use the following process to fix the test:

## 0. Add the following task list
* Run the test to verify the error message
* Read the test code and create a summary
* Validate Test Premise
* Formulate Working Hypothesis
* Test Hypothesis
* Start Debug Loop

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

## 5. Test Hypothesis

Assume the hypothesis is false until the assertion is found to support it.

## 6. Debug Loop

### a. Add the debug loop task list
* Start the test and set the breakpoint
* Run to the breakpoint and inspect variables
* Reject or accept the hypothesis

Add the following to the task list

While hypothesis is false, repeat the following steps:

### a. Start the test in the Python debugger

```tool
mcp__python-debugger-mcp__start_debug(file_path="tests/python/test_reward_system.py", use_pytest=true, args="--pdb -x test_reward_system.py::test_forward_movement_reward")
```

Alternative for single test function:
```tool
mcp__python-debugger-mcp__start_debug(file_path="tests/python/test_reward_system.py::test_forward_movement_reward", use_pytest=true)
```

### b. Set the breakpoint

```tool
mcp__python-debugger-mcp__set_breakpoint(file_path="madrona_escape_room/__init__.py", line_number=245)
```

Or use relative path from project root:
```tool
mcp__python-debugger-mcp__set_breakpoint(file_path="tests/python/test_reward_system.py", line_number=50)
```

### c. Run to the breakpoint

```tool
mcp__python-debugger-mcp__send_pdb_command(command="c")
```

### d. Examine variables at breakpoint

```tool
mcp__python-debugger-mcp__examine_variable(variable_name="rewards")
mcp__python-debugger-mcp__examine_variable(variable_name="self.numWorlds")
mcp__python-debugger-mcp__examine_variable(variable_name="obs[0, 0, :3]")
```

### e. Step through code

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

### f. Decision Point

- **If hypothesis assertion is TRUE:** Create a plan for a fix and propose it to the user
- **If hypothesis assertion is FALSE:** Reformulate a new working hypothesis

### g. Reformulate Working Hypothesis (if needed)

Read the code and think. When sufficient information is gathered, formulate a new hypothesis:

**Working hypothesis:** `<hypothesis and reasoning>`

**Assertion:** When `[method]` in `[filename]:[lineno]` is called, `[variable]` is expected to be `[value]`

### h. iterate in the Debug Loop until working hypothesis is supported

## 7. Additional Debug Techniques for Python Tests

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

## 8. Common Python Test Issues

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