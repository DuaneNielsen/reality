---
argument-hint: [pytest test path]
description: iteratively debug and fix failing Python tests using structured debugging process
---
@.claude/include/substitutions.md

# algorithm

failing_test = $ARGUMENTS
hypothesis_verified = false
fixes_applied = []

1. run test to verify error message
2. read test code and create summary
3. validate test premise

while (hypothesis not verified) todo:
   1. formulate working hypothesis
   2. choose debug method based on code location
   3. debug and validate hypothesis
   4. accept or reject hypothesis

5. implement fix based on verified hypothesis
6. verify fix with test execution

## run test to verify error message

Execute the specific failing test with verbose output to capture the exact error and specification context:

**For detailed error analysis with spec display:**
```bash
# Run with verbose mode to display specifications on failure
uv run --group dev pytest tests/python/$ARGUMENTS -vs --tb=long --log-cli-level=DEBUG
```

**Key benefits of verbose mode (-v):**
- Automatically displays relevant specification sections when tests fail
- Shows @pytest.mark.spec() markers and their documentation
- Provides immediate context about expected behavior according to specs

Use the Bash tool to execute these commands and capture:
1. Exact error messages and stack traces
2. Specification sections displayed for failed tests (if marked with @pytest.mark.spec)
3. Expected vs actual values from assertions
4. Any traceability information linking test to requirements

Document the specific assertion that failed, the expected vs actual values, and any specification context provided.

## read test code and create summary

Use the Read tool to examine the failing test file and understand its structure:

```tool
Read(file_path="tests/python/[TEST_FILE].py")
```

Create a structured summary following this format:

### Test Summary Format

**Test Name:** `[test_function_name]`

**What is being tested:** Brief description of the test's intent and purpose

**The test process:**
1. SimManager initialization (CPU/GPU mode specification)
2. Initial state capture (observations, tensors)
3. Action application to agents
4. Simulation step execution
5. State verification and assertions

**Assertions being made:**
1. [Assertion 1 description] - Expected outcome
2. [Assertion 2 description] - Expected outcome  
3. [Assertion 3 description] - Expected outcome

**Specific failure point:**
- [Assertion X] failed: [actual_value] vs [expected_value]
- Error message: [exact_error_text]

## validate test premise

**Decision Point:** Determine if the test premise and assertions are correct by leveraging the traceability system.

**Step 1: Check for existing @pytest.mark.spec() markers**

Examine the test file for traceability markers:
```tool
Read(file_path="tests/python/[TEST_FILE].py")
```

Look for @pytest.mark.spec() decorators that link tests to specifications:
```python
@pytest.mark.spec("docs/specs/sim.md", "rewardSystem")
def test_step_zero_reward_is_zero(cpu_manager):
```

**Step 2: If @pytest.mark.spec() markers found - Use them directly**

**Extract specification reference:**
- Spec file: `[SPEC_FILE_PATH]` from marker
- Section: `[SECTION_NAME]` from marker

**Read the referenced specification:**
```tool
Read(file_path="[SPEC_FILE_PATH]")
```

**Key advantage:** Test already has verified traceability to specification documentation

**Step 3: If NO @pytest.mark.spec() markers found - Legacy approach**

⚠️ **Missing Traceability Markers**

**Test:** $ARGUMENTS

**Issue:** Test lacks @pytest.mark.spec() markers for requirement traceability

**Fallback approach:**
- Extract test name from failing test path: `tests/python/[TEST_FILE].py`
- Look for corresponding spec at: `docs/specs/[TEST_FILE_WITHOUT_PREFIX].md`
- Example: `tests/python/test_reward_termination_system.py` → `docs/specs/reward_termination_system.md`

Use the Glob tool to check if spec exists:
```tool
Glob(pattern="docs/specs/[TEST_NAME_WITHOUT_TEST_PREFIX].md")
```

**Recommendation:** Add proper @pytest.mark.spec() markers to improve traceability

**Step 4: Validate test against specification**

**If specification was displayed in verbose test output:**
- Use the specification context already provided by the test failure
- Compare failed assertions against the displayed specification requirements
- Leverage the automatic spec display feature for validation

**If manual spec lookup required:**
Use the Read tool to examine the specification file:
```tool
Read(file_path="[SPEC_FILE_PATH]")
```

**Validation checklist:**
- Does the test use proper fixtures (cpu_manager, gpu_manager)?
- Do the test assertions match the SPEC requirements listed in the specification?
- Are the test scenarios covering the behaviors defined in the specification?
- Does the test follow patterns from tests/README.md TESTING_GUIDE?
- Is the test setup appropriate for what's being tested per the spec?
- Are @pytest.mark.spec() markers present and pointing to correct spec sections?

**Step 5: Compare test assertions to specification requirements**

For each test assertion, verify it matches a corresponding SPEC requirement:
- **SPEC 1**: [Requirement from spec] → **Test assertion**: [Corresponding assertion]
- **SPEC 2**: [Requirement from spec] → **Test assertion**: [Corresponding assertion]
- **SPEC N**: [Requirement from spec] → **Test assertion**: [Corresponding assertion]

**Step 6: Handle validation results**

**If test-spec mismatch found:**
- Work with user to align test with specification
- Consider updating @pytest.mark.spec() markers if incorrect
- May need to update either test or specification for consistency

**If test matches specification:**
- Proceed to hypothesis formulation
- Test premise is validated against documented requirements

**If specification context was auto-displayed:**
- Leverage the specification information already shown in test failure output
- Use this context to inform debugging approach

## hypothesis not verified

Continue the debug loop until a working hypothesis is supported by evidence.

Check if current hypothesis assertion has been validated through debugging:
- **TRUE**: Hypothesis is supported, proceed to fix implementation
- **FALSE**: Hypothesis is rejected, formulate new hypothesis

## formulate working hypothesis

Analyze the code and failure point to create a testable hypothesis:

**Working hypothesis:** [Detailed explanation of what is believed to be causing the failure, including the specific mechanism and reasoning]

**Assertion:** When `[specific_method]` in `[filename]:[line_number]` is called, `[variable_name]` is expected to be `[expected_value]`

**Example:**
- **Hypothesis:** The reward calculation system is not properly accounting for forward movement because the position delta calculation in the step() method uses incorrect coordinate system mapping
- **Assertion:** When `mgr.step()` in `madrona_escape_room/__init__.py:245` is called, `rewards[0,0]` should be > 0.0 after forward movement action

## choose debug method based on code location

Determine the appropriate debugging approach based on where the suspected issue lies:

**For Python variables and logic:**
- Use Python debugger (PDB) via MCP tools
- Suitable for: Python tensor operations, SimManager method calls, test logic

**For C++ simulation code:**  
- Use GDB debugger via MCP tools
- Suitable for: ECS system behavior, physics calculations, C++ simulation core

**Decision criteria:**
- If hypothesis involves Python bindings, tensor shapes, or manager operations → Use Python debugging
- If hypothesis involves simulation physics, ECS components, or core C++ logic → Use GDB debugging

## debug and validate hypothesis

### Python Debugging Path

**Start debugging session:**
```tool
mcp__python-debugger-mcp__start_debug(file_path="tests/python/[TEST_FILE].py", use_pytest=true, args="--pdb -x [TEST_NAME]")
```

**Set strategic breakpoints:**
```tool
mcp__python-debugger-mcp__set_breakpoint(file_path="[TARGET_FILE].py", line_number=[LINE_NUMBER])
```

**Execute to breakpoint:**
```tool
mcp__python-debugger-mcp__send_pdb_command(command="c")
```

**Examine variables at breakpoint:**
```tool
mcp__python-debugger-mcp__examine_variable(variable_name="[KEY_VARIABLE]")
mcp__python-debugger-mcp__examine_variable(variable_name="[HYPOTHESIS_VARIABLE]")
```

**Step through code execution:**
```tool
# Next line (step over)
mcp__python-debugger-mcp__send_pdb_command(command="n")

# Step into function
mcp__python-debugger-mcp__send_pdb_command(command="s")

# List current code context
mcp__python-debugger-mcp__send_pdb_command(command="l")
```

### C++ Debugging Path

**Start GDB session:**
```tool
mcp__gdb__gdb_start(gdbPath="gdb", workingDir="/home/duane/madrona_escape_room")
```

**Configure for Python test execution:**
```tool
mcp__gdb__gdb_command(sessionId="[SESSION_ID]", command="set breakpoint pending on")
mcp__gdb__gdb_command(sessionId="[SESSION_ID]", command="file /home/duane/madrona_escape_room/.venv/bin/python")
mcp__gdb__gdb_command(sessionId="[SESSION_ID]", command="set args -m pytest tests/python/$ARGUMENTS -v --tb=short")
```

**Set C++ breakpoints:**
```tool
mcp__gdb__gdb_set_breakpoint(sessionId="[SESSION_ID]", location="[FUNCTION_NAME]")
mcp__gdb__gdb_set_breakpoint(sessionId="[SESSION_ID]", location="[FILE]:[LINE]")
```

**Run test and examine variables:**
```tool
mcp__gdb__gdb_command(sessionId="[SESSION_ID]", command="run")
mcp__gdb__gdb_print(sessionId="[SESSION_ID]", expression="[VARIABLE_NAME]")
mcp__gdb__gdb_command(sessionId="[SESSION_ID]", command="info locals")
```

**Navigate through execution:**
```tool
mcp__gdb__gdb_next(sessionId="[SESSION_ID]")
mcp__gdb__gdb_step(sessionId="[SESSION_ID]")
mcp__gdb__gdb_continue(sessionId="[SESSION_ID]")
```

### Validation Process

Compare observed values with hypothesis assertion:
- Document actual variable values at key execution points
- Trace execution flow to identify deviations from expected behavior
- Capture evidence that supports or refutes the hypothesis

## accept or reject hypothesis

**If hypothesis assertion is TRUE (supported by debugging evidence):**
- Document the confirmed root cause
- Proceed to fix implementation
- Create plan for addressing the identified issue

**If hypothesis assertion is FALSE (contradicted by debugging evidence):**
- Document why hypothesis was incorrect
- Analyze new evidence gathered during debugging
- Return to "formulate working hypothesis" step with new information

## implement fix based on verified hypothesis

Based on the confirmed root cause, implement the appropriate fix:

**For Python code issues:**
```tool
Edit(file_path="[PYTHON_FILE].py", old_string="[PROBLEMATIC_CODE]", new_string="[CORRECTED_CODE]")
```

**For C++ code issues:**
```tool
Edit(file_path="src/[CPP_FILE].cpp", old_string="[PROBLEMATIC_CODE]", new_string="[CORRECTED_CODE]")
```

**For configuration issues:**
```tool
Edit(file_path="[CONFIG_FILE]", old_string="[OLD_CONFIG]", new_string="[NEW_CONFIG]")
```

Document the specific change made and the reasoning behind it.

**If C++ changes were made:**
```tool
Task(subagent_type="project-builder", description="build project after C++ changes", prompt="Build the project after making C++ code changes to ensure compilation succeeds")
```

## verify fix with test execution

Execute comprehensive verification to ensure fix works and doesn't introduce regressions:

**Run the specific fixed test:**
```bash
uv run --group dev pytest tests/python/$ARGUMENTS -v
```

**Run related tests for regression checking:**
```bash
# Run all Python tests (excluding GPU if not relevant)
uv run --group dev pytest tests/python/ -v --no-gpu

# Run GPU tests if the fix involved GPU code
uv run --group dev pytest tests/python/ -v -k "gpu"
```

**Verify test results:**
- Confirm the originally failing test now passes
- Ensure no new test failures were introduced
- Document any remaining issues or edge cases discovered

Use the Bash tool for test execution and capture results according to the standardized test output format from the testing guide.

## output format

if fix successful

    ✅ **Test Fix Successful** \n\n
    
    **Test:** $ARGUMENTS \n\n
    
    **Root Cause:** [VERIFIED_HYPOTHESIS] \n\n
    
    **Fix Applied:** [DESCRIPTION_OF_CHANGE] \n\n
    
    **Files Modified:** \n\n
    - `[FILE_1]:[LINE_RANGE]` - [CHANGE_DESCRIPTION] \n\n
    - `[FILE_2]:[LINE_RANGE]` - [CHANGE_DESCRIPTION] \n\n
    
    **Debug Method Used:** [PDB/GDB] debugging \n\n
    
    **Verification:** ✅ Target test passes, ✅ No regressions detected \n\n

else

    ❌ **Test Fix Incomplete** \n\n
    
    **Test:** $ARGUMENTS \n\n
    
    **Hypotheses Tested:** [COUNT] \n\n
    
    **Last Hypothesis:** [FINAL_HYPOTHESIS] \n\n
    
    **Evidence Gathered:** \n\n
    - [FINDING_1] \n\n
    - [FINDING_2] \n\n
    - [FINDING_3] \n\n
    
    **Remaining Issues:** [UNRESOLVED_PROBLEMS] \n\n
    
    **Recommended Next Steps:** [MANUAL_INVESTIGATION_NEEDED]