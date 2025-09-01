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
4. validate agent indexing logic
5. fix standard violations
6. update test configuration
7. verify compliance

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

**Standard Fixture Requirements:**
- cpu_manager fixture should be used for CPU tests
- gpu_manager fixture should be used for GPU tests  
- Module-scoped _test_manager for specialized cases
- Proper fixture parameters and configuration

Document each violation found with specific line numbers and recommended fixes.

## verify level creation patterns

Check level creation compliance:

**Decorator Usage:**
- Tests should use @pytest.mark.json_level for JSON level definitions
- Tests should use @pytest.mark.ascii_level for ASCII level definitions
- Avoid inline level creation functions

**Level Definition Standards:**
- Level data should be defined in decorators, not inline functions
- Custom levels should follow tileset conventions
- Agent facing and spawn configuration should be consistent

**Anti-patterns to Fix:**
- Custom level creation functions (replace with decorators)
- Inline SimManager creation with custom levels
- Hardcoded level parameters that should use decorators

## validate agent indexing logic

Review agent and world indexing for consistency:

**Common Issues:**
- Inconsistent world vs agent indexing
- Hardcoded agent counts that don't match fixture configuration
- Looping through non-existent agents
- Wrong tensor shape assumptions

**Standard Patterns:**
- cpu_manager fixture uses 4 worlds by default
- Single agent per world (index 0) is standard
- Use observer.get_done_flag(world_idx) for single agent
- Use observer.get_done_flag(world_idx, agent_idx) for multiple agents

Check tensor shape assertions match fixture configuration (4 worlds, 1 agent, 3 action components for cpu_manager).

## fix standard violations

Apply fixes using the Edit tool for each violation found:

**Replace Custom Fixtures:**
```tool
Edit(file_path="$ARGUMENT", 
     old_string="@pytest.fixture\ndef custom_manager():\n    # Custom fixture code",
     new_string="# Removed custom fixture - using standard cpu_manager from conftest.py")
```

**Add Level Decorators:**
```tool
Edit(file_path="$ARGUMENT",
     old_string="class TestClass:",
     new_string="@pytest.mark.json_level({\n    \"ascii\": \"layout\",\n    \"tileset\": {...}\n})\nclass TestClass:")
```

**Fix Method Parameters:**
```tool
Edit(file_path="$ARGUMENT",
     old_string="def test_method(self, custom_manager):",
     new_string="def test_method(self, cpu_manager):")
```

**Update Variable References:**
```tool
Edit(file_path="$ARGUMENT",
     old_string="mgr = custom_manager",
     new_string="mgr = cpu_manager")
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

**Remove Unused Imports:**
```tool
Edit(file_path="$ARGUMENT",
     old_string="import madrona_escape_room\nfrom madrona_escape_room.level_compiler import compile_level",
     new_string="# Removed unused imports - using standard fixtures")
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
2. **Level Decorators:** Added @pytest.mark.json_level decorator for test class \\n\\n  
3. **Agent Indexing:** Fixed inconsistent agent indexing logic \\n\\n
4. **Tensor Assertions:** Updated shape assertions to match fixture configuration \\n\\n
5. **Import Cleanup:** Removed unused imports and dependencies \\n\\n

**Standards Compliance:** ✅ **PASSED** \\n\\n

**Test Functionality:** Maintained - all test logic preserved \\n\\n

**Recommendations:**
- Run test suite to verify functionality: `uv run --group dev pytest $ARGUMENT -v`
- Consider adding more specific error messages for failed assertions
- Ensure test coverage remains comprehensive after refactoring