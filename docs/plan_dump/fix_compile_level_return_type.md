# Fix compile_level() Return Type Consistency

**Date:** 2025-01-11
**Issue:** `compile_level()` returns inconsistent types - single `CompiledLevel` for single levels, `List[CompiledLevel]` for multi-level
**Impact:** Breaking existing code that expects single levels, causing errors like `'list' object has no attribute 'num_tiles'`

## Problem Analysis

The current implementation of `compile_level()` has inconsistent return types:
- Single-level JSON → returns `CompiledLevel` 
- Multi-level JSON → returns `List[CompiledLevel]`

This breaks backward compatibility and creates confusion. The function should have a consistent return type.

## Proposed Solution

**Always return `List[CompiledLevel]`** from `compile_level()` for consistency:
- Single-level JSON → returns `List[CompiledLevel]` with 1 element
- Multi-level JSON → returns `List[CompiledLevel]` with N elements

## Implementation Plan

### 1. Update `compile_level()` Function
- Change return type annotation from `Union[CompiledLevel, List[CompiledLevel]]` to `List[CompiledLevel]`
- Always return a list, even for single levels
- Update docstring to reflect consistent return type

### 2. Update All Calling Code
**Core Files to Fix:**
- `madrona_escape_room/level_compiler.py` - `main()` function
- Any other code that calls `compile_level()` and expects single result

**Pattern to Replace:**
```python
# OLD - assumes single level
compiled = compile_level(json_data)
validate_compiled_level(compiled)

# NEW - handle list consistently  
compiled_levels = compile_level(json_data)
for compiled in compiled_levels:
    validate_compiled_level(compiled)
```

### 3. Provide Convenience Functions
Add helper functions for common use cases:
```python
def compile_single_level(json_data) -> CompiledLevel:
    """Compile single level JSON, raising error if multi-level."""
    levels = compile_level(json_data)
    if len(levels) != 1:
        raise ValueError(f"Expected single level, got {len(levels)}")
    return levels[0]

def compile_level_auto(json_data) -> Union[CompiledLevel, List[CompiledLevel]]:
    """Legacy function for backward compatibility (deprecated)."""
    levels = compile_level(json_data)
    return levels[0] if len(levels) == 1 else levels
```

### 4. Update Tests
- Update all tests that call `compile_level()`
- Ensure they handle list return type correctly
- Add tests for the new consistency

### 5. Update Documentation
- Fix all examples in docstrings
- Update usage examples in markdown files
- Add migration guide for existing code

## Breaking Change Migration

This is a **breaking change** but necessary for consistency. 

### Migration Path for Users:
```python
# Before
compiled = compile_level(json_data)
mgr = SimManager(compiled_levels=compiled)  # ERROR if multi-level

# After  
compiled_levels = compile_level(json_data)
mgr = SimManager(compiled_levels=compiled_levels)  # Works for both
```

### For Single Level Use Cases:
```python
# If you know it's always single level
compiled_levels = compile_level(json_data)
assert len(compiled_levels) == 1
single_level = compiled_levels[0]

# Or use convenience function
single_level = compile_single_level(json_data)  # Raises if multi-level
```

## Implementation Steps

1. **Update `compile_level()` return type** - Always return list
2. **Fix `main()` function** - Handle list properly  
3. **Update tests** - Fix all test code
4. **Update documentation** - Fix examples
5. **Add convenience functions** - For single-level use cases
6. **Test thoroughly** - Ensure no regressions

## Files to Modify

### Core Implementation:
- `madrona_escape_room/level_compiler.py` - Update `compile_level()` and `main()`

### Tests:
- `tests/python/test_ascii_level_compiler.py` - Update test assertions
- `tests/python/test_multi_level_integration.py` - Verify still works
- Any other tests calling `compile_level()`

### Documentation:
- Update docstrings in `level_compiler.py`
- Update examples in markdown files that show `compile_level()` usage

## Expected Outcome

After this fix:
- ✅ `compile_level()` always returns `List[CompiledLevel]` (consistent)
- ✅ Single levels work: `compile_level(single_json)` → `[CompiledLevel]`
- ✅ Multi levels work: `compile_level(multi_json)` → `[CompiledLevel, CompiledLevel, ...]`
- ✅ All existing SimManager code works (it already accepts lists)
- ✅ Command-line level compiler works with both formats
- ✅ Clear migration path for existing code

This change makes the API consistent, predictable, and future-proof while maintaining the functionality that users need.