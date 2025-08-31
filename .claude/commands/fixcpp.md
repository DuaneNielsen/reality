# C++ Test Fixing Process

The cpp test `$ARGUMENTS` is failing and we need to fix it.

Use the following process to fix the test:

## 1. Run the test to verify the error message

for cpu tests...

```bash
./build/mad_escape_tests --gtest_filter="CApiCPUTest.ManagerCreation"
```

for gpu tests

```bash
./build/mad_escape_gpu_tests --gtest_filter="CApiGPUTest.ManagerCreationWithEmbeddedLevels"
```


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

**Decision Point:** Is the premise of the test correct?

- Prompt the user to validate if the test is valid and the assertions are correct
- If valid, proceed to step 4
- If not, work with the user to redefine the test

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

```gdb_tool
gdb_load (MCP)(sessionId: "<session_id>", program: "./build/mad_escape_tests", arguments: ["--gtest_filter=CApiCPUTest.ManagerCreation"])
```

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