---
argument-hint: [test file path]
description: review a test to ensure it complies with the testing standards
---
@.claude/include/substitutions.md

# algorithm

test_file_path = $ARGUMENT
violations_found = []

1. analyze test file structure
2. check fixture usage compliance
3. verify level creation patterns
4. check test helper usage
5. validate agent indexing logic
6. fix standard violations
7. update test configuration
8. verify compliance

## analyze test file structure

Read the target test file to understand current implementation:

```tool
Read(file_path="$ARGUMENT")
```

Examine the test structure for:
- Custom fixture definitions
- Import statements and dependencies
- Test class organization
- Agent/world indexing patterns
- Level creation methods

Also read the standard testing configuration:

```tool
Read(file_path="$WORKING_DIR/tests/python/conftest.py")
```

Compare against project testing standards:
- Use of standard fixtures (cpu_manager, gpu_manager)
- Proper level decorators (@pytest.mark.json_level, @pytest.mark.ascii_level)
- Consistent agent/world indexing
- Proper tensor shape assertions

## check fixture usage compliance

Identify violations in fixture usage:

**Custom Fixture Issues:**
- Look for custom @pytest.fixture definitions that duplicate standard functionality
- Check if tests use custom fixtures instead of conftest.py fixtures
- Verify fixture scope and configuration matches standards
- Custom sim_manager fixtures should be replaced with cpu_manager
- Environment fixtures (cpu_env, small_env) should use standard wrappers

**Standard Fixture Requirements:**
- cpu_manager fixture should be used for CPU tests (function-scoped, 4 worlds, 1 agent)
- gpu_manager fixture should be used for GPU tests (session-scoped, shared)
- Module-scoped _test_manager for specialized cases only
- log_and_verify_replay_cpu_manager for replay verification tests
- test_manager_from_replay for replay factory pattern

**Common Fixture Violations:**
- Using SimManager() directly instead of fixtures
- Creating custom sim_manager fixtures
- Wrong fixture scope (function vs session vs module)
- Not using @pytest.mark.skipif for GPU tests

Document each violation found with specific line numbers and recommended fixes.

## verify level creation patterns

Check level creation compliance:

**Decorator Usage:**
- Tests should use @pytest.mark.json_level for JSON level definitions
- Tests should use @pytest.mark.ascii_level for ASCII level definitions  
- Avoid inline level creation functions
- Level decorators must be placed on test functions, not classes

**Level Definition Standards:**
- Level data should be defined in decorators, not inline functions
- Custom levels should follow tileset conventions
- Agent facing and spawn configuration should be consistent
- ASCII levels use 'S' for spawn points, '#' for walls, '.' for empty space

**Anti-patterns to Fix:**
- Custom level creation functions (replace with decorators)
- Inline SimManager creation with custom levels
- Hardcoded level parameters that should use decorators
- Using compile_ascii_level() directly in test functions
- Passing compiled_levels parameter to SimManager

## check test helper usage

Verify proper use of test helper utilities:

**AgentController Requirements:**
- MUST use AgentController from test_helpers for all movement
- Never manually set action tensor indices
- Always call controller.reset_actions() before setting movement
- Use helper methods: move_forward(), strafe_left(), rotate_only(), etc.

**ObservationReader Requirements:**
- Use ObservationReader for reading agent observations
- Provides clean API for position, rotation, done flags
- Handles world/agent indexing properly

**Common Violations:**
- Direct tensor manipulation: `actions[:, 0] = 3`
- Missing rotation component causing circular movement
- Not resetting actions before movement commands
- Manual observation tensor parsing

## validate agent indexing logic

Review agent and world indexing for consistency:

**Common Issues:**
- Inconsistent world vs agent indexing
- Hardcoded agent counts that don't match fixture configuration
- Looping through non-existent agents (e.g., `for agent_idx in range(3)` when only 1 agent)
- Wrong tensor shape assumptions
- Using try/except blocks to handle agent indexing errors

**Standard Patterns:**
- cpu_manager fixture uses 4 worlds by default
- Single agent per world (index 0) is standard
- Use observer.get_done_flag(world_idx) for single agent per world
- Use observer.get_done_flag(world_idx, agent_idx) only with multiple agents
- Action tensor shape: [4, 1, 3] for cpu_manager (4 worlds, 1 agent, 3 components)
- Observation tensor shape: [4, 1, N] where N is observation size

**Required Assertions:**
- `assert actions.shape[0] == 4` for cpu_manager (not 1)
- `assert obs.shape[0] == 4` for cpu_manager
- Never loop through agent indices unless multiple agents configured

Check tensor shape assertions match fixture configuration (4 worlds, 1 agent, 3 action components for cpu_manager).

## fix standard violations

Apply fixes using the Edit tool for each violation found:

**Replace Custom Fixtures:**
```tool
Edit(file_path="$ARGUMENT", 
     old_string="@pytest.fixture\ndef sim_manager():\n    # Custom fixture code",
     new_string="# Removed custom fixture - using standard cpu_manager from conftest.py")
```

**Add Level Decorators:**
```tool
Edit(file_path="$ARGUMENT",
     old_string="def test_with_level():",
     new_string="@pytest.mark.ascii_level(TEST_LEVEL)\ndef test_with_level():")
```

**Fix Method Parameters:**
```tool
Edit(file_path="$ARGUMENT",
     old_string="def test_method(self, sim_manager):",
     new_string="def test_method(self, cpu_manager):")
```

**Update Variable References:**
```tool
Edit(file_path="$ARGUMENT",
     old_string="mgr = sim_manager",
     new_string="mgr = cpu_manager")
```

**Add GPU Skip Markers:**
```tool
Edit(file_path="$ARGUMENT",
     old_string="def test_gpu_feature(gpu_manager):",
     new_string="@pytest.mark.skipif(not torch.cuda.is_available(), reason=\"CUDA not available\")\ndef test_gpu_feature(gpu_manager):")
```

## update test configuration

Fix configuration and assertion issues:

**Tensor Shape Assertions:**
```tool
Edit(file_path="$ARGUMENT",
     old_string="assert actions.shape[0] == 1, \"Should have 1 world\"",
     new_string="assert actions.shape[0] == 4, \"Should have 4 worlds (from cpu_manager fixture)\"")
```

**Agent Indexing:**
```tool
Edit(file_path="$ARGUMENT",
     old_string="for agent_idx in range(3):\n    try:\n        if observer.get_done_flag(0, agent_idx):",
     new_string="if observer.get_done_flag(0):  # Single agent in world 0")
```

**Remove Direct SimManager Creation:**
```tool
Edit(file_path="$ARGUMENT",
     old_string="mgr = SimManager(exec_mode=ExecMode.CPU, num_worlds=4, ...)",
     new_string="# Use cpu_manager fixture instead - see conftest.py")
```

**Remove Unused Imports:**
```tool
Edit(file_path="$ARGUMENT",
     old_string="from madrona_escape_room.level_compiler import compile_ascii_level",
     new_string="# Removed unused import - level compilation handled by fixture")
```

## verify compliance

After applying fixes, perform final verification:

```tool
Read(file_path="$ARGUMENT")
```

Check that the updated test file:
- Uses only standard fixtures from conftest.py
- Has proper level decorators on test classes
- Uses consistent agent/world indexing
- Has correct tensor shape assertions
- Follows project testing conventions
- Maintains test functionality while improving compliance

Document any remaining issues that require manual review.

## output format

**Test Standards Review Complete** \\n\\n

**File:** `$ARGUMENT` \\n\\n

**Violations Found:** [VIOLATION_COUNT] \\n\\n

**Fixes Applied:**
1. **Custom Fixture Removal:** Replaced custom fixtures with standard cpu_manager \\n\\n
2. **Level Decorators:** Added @pytest.mark.ascii_level or @pytest.mark.json_level decorators \\n\\n  
3. **Agent Indexing:** Fixed inconsistent agent indexing logic \\n\\n
4. **Tensor Assertions:** Updated shape assertions to match fixture configuration \\n\\n
5. **Import Cleanup:** Removed unused imports and dependencies \\n\\n
6. **GPU Test Markers:** Added @pytest.mark.skipif for CUDA availability \\n\\n
7. **AgentController Usage:** Updated to use test_helpers.AgentController for movement \\n\\n

**Standards Compliance:** ✅ **PASSED** \\n\\n

**Test Functionality:** Maintained - all test logic preserved \\n\\n

**Critical Reminders:**
- NEVER manually set action tensors - use AgentController
- ALWAYS use fixtures from conftest.py
- GPU tests MUST use session-scoped gpu_manager fixture
- Custom levels use decorators, not inline compilation

**Recommendations:**
- Run test suite to verify functionality: `uv run --group dev pytest $ARGUMENT -v`
- For debugging: `uv run --group dev pytest $ARGUMENT --record-actions --trace-trajectories`
- To visualize: `uv run --group dev pytest $ARGUMENT --record-actions --visualize`