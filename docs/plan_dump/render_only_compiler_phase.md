# Phase 2: Level Compiler Updates for Render-Only Assets

## Prerequisites

**Phase 1 Must Be Complete Before Starting:**
- All C++ tests passing with `tile_render_only` field
- `CompiledLevel` and `MER_CompiledLevel` structs updated and working
- Entity creation logic using data-driven approach instead of hardcoded asset IDs
- C API properly handling `tile_render_only` array

**Verification:**
```bash
# These should all pass before starting Phase 2
./tests/run_cpp_tests.sh
# Specifically check:
# - test_persistence.cpp passes
# - test_c_api_cpu.cpp passes  
# - test_level_utilities.cpp passes
```

## Required Reading

Before starting this implementation, read the following files in order:

### Level Compilation System
1. **`madrona_escape_room/level_compiler.py`** - Complete level compilation pipeline:
   - `compile_level()` function (lines 79-240)
   - Binary save/load functions (lines 297-475)
   - Validation logic (lines 243-295)
2. **`tests/python/test_ascii_level_compiler.py`** - Existing test patterns and expected behavior
3. **`tests/python/test_level_compiler_c_api.py`** - C API integration test patterns

### Level Format Understanding
4. **`docs/development/LEVEL_FORMAT.md`** - Level format specification (if exists)
5. **Review Phase 1 changes** - Understand the C++ struct changes made in Phase 1

## Implementation Plan

### Step 1: Extend Python Level Compiler Core

#### 1.1 Update `compile_level()` function in `madrona_escape_room/level_compiler.py`

**Add render-only character mapping:**
```python
# Add to CHAR_MAP around line 66
CHAR_MAP = {
    ".": TILE_EMPTY,
    " ": TILE_EMPTY,  
    "#": TILE_WALL,
    "C": TILE_CUBE,
    "S": TILE_SPAWN,
    "R": TILE_RENDER_DECORATION,  # NEW: Render-only decorative element
    "X": TILE_RENDER_MARKER,      # NEW: Render-only axis marker
    # Future: Add more render-only types as needed
}
```

**Update tile processing loop around line 166:**
```python
# In the parsing loop, determine if tile should be render-only
if tile_type != TILE_EMPTY:
    # Determine if this tile should be render-only
    is_render_only = tile_type in [TILE_RENDER_DECORATION, TILE_RENDER_MARKER]
    
    tiles.append((world_x, world_y, tile_type, is_render_only))  # Add render-only flag
```

**Initialize render-only array around line 195:**
```python
# Add after tile_persistent array initialization
tile_render_only = [False] * MAX_TILES_C_API  # Default: all physics entities

# Fill arrays with actual tile data - update around line 198
for i, (world_x, world_y, tile_type, is_render_only) in enumerate(tiles):
    object_ids[i] = tile_type
    tile_x[i] = world_x
    tile_y[i] = world_y
    tile_render_only[i] = is_render_only  # NEW: Set render-only flag
```

**Update return dictionary around line 236:**
```python
return {
    # ... existing fields ...
    "tile_render_only": tile_render_only,  # NEW: Add render-only array
    # ... rest of fields ...
}
```

#### 1.2 Add JSON Support for Render-Only Assets

**Extend `compile_level_from_json()` around line 477:**
```python
# Add support for render-only tile specifications in JSON
def compile_level_from_json(json_data: Union[str, Dict]) -> Dict:
    # ... existing code ...
    
    # Extract render-only overrides if specified
    render_only_positions = data.get("render_only_positions", [])
    
    compiled = compile_level(ascii_str, scale, agent_facing, level_name)
    
    # Apply render-only overrides for specific positions
    for pos in render_only_positions:
        x, y = pos.get("x"), pos.get("y") 
        # Find tile at position and mark as render-only
        for i in range(compiled["num_tiles"]):
            if (abs(compiled["tile_x"][i] - x) < 0.1 and 
                abs(compiled["tile_y"][i] - y) < 0.1):
                compiled["tile_render_only"][i] = True
                break
    
    return compiled
```

### Step 2: Update Binary File Format

#### 2.1 Update `save_compiled_level_binary()` around line 369

**Add render-only array to binary output:**
```python
# After writing tile_persistent array around line 375
# tile_render_only array - pad with false if needed  
for i in range(MAX_TILES_C_API):
    if i < array_size and "tile_render_only" in compiled:
        # Write as bool (1 byte)
        f.write(struct.pack("<B", 1 if compiled["tile_render_only"][i] else 0))
    else:
        f.write(struct.pack("<B", 0))  # Default: physics entity
```

#### 2.2 Update `load_compiled_level_binary()` around line 436

**Add render-only array loading with backward compatibility:**
```python
# After loading tile_persistent array around line 443
# Read tile_render_only array (if present in file)
tile_render_only = []
try:
    for _ in range(MAX_TILES_C_API):
        tile_render_only.append(struct.unpack("<B", f.read(1))[0] != 0)
except struct.error:
    # Old file format without render-only flags - default to all physics
    tile_render_only = [False] * MAX_TILES_C_API

# Add to returned dictionary around line 464
compiled = {
    # ... existing fields ...
    "tile_render_only": tile_render_only,  # NEW: Add render-only flags
    # ... rest of fields ...
}
```

#### 2.3 Update `validate_compiled_level()` around line 288

**Add render-only array validation:**
```python
# Add to required_fields list
required_fields = [
    # ... existing fields ...
    "tile_render_only",  # NEW: Add to validation
]

# Add array length validation around line 289
for array_name in ["object_ids", "tile_x", "tile_y", "tile_persistent", "tile_render_only"]:
    if len(compiled[array_name]) != MAX_TILES_C_API:
        raise ValueError(
            f"Invalid {array_name} array length: {len(compiled[array_name])} "
            f"(must be {MAX_TILES_C_API} for C++ compatibility)"
        )
```

### Step 3: Update Python Tests

#### 3.1 Update `tests/python/test_ascii_level_compiler.py`

**Add render-only functionality tests:**
```python
class TestRenderOnlyAssets:
    """Test render-only asset functionality"""
    
    def test_render_only_characters(self):
        """Test ASCII level with render-only characters"""
        level = """#######
#S....#
#..R..#  
#..X..#
#######"""
        
        compiled = compile_level(level)
        
        # Should have walls (physics) and render-only decorations
        render_only_count = sum(compiled["tile_render_only"])
        physics_count = compiled["num_tiles"] - render_only_count
        
        assert render_only_count == 2  # R and X tiles
        assert physics_count > 0       # Wall tiles
        
    def test_mixed_physics_render_only(self):
        """Test level with mixed physics and render-only entities"""
        level = """########
#S.....#
#.RCX..#
########"""
        
        compiled = compile_level(level) 
        
        # Check specific tile types
        for i in range(compiled["num_tiles"]):
            if compiled["object_ids"][i] == TILE_CUBE:  # 'C'
                assert not compiled["tile_render_only"][i]  # Physics
            elif compiled["object_ids"][i] == TILE_RENDER_DECORATION:  # 'R'  
                assert compiled["tile_render_only"][i]  # Render-only
```

**Update existing tests:**
```python
def test_dict_to_struct_conversion(self):
    # ... existing code ...
    
    # NEW: Verify render-only arrays copied correctly
    for i in range(compiled["num_tiles"]):
        assert struct.tile_render_only[i] == compiled["tile_render_only"][i]
```

#### 3.2 Update `tests/python/test_level_compiler_c_api.py`

**Add render-only array tests:**
```python
def test_render_only_array_sizing(self):
    """Test render-only arrays are correctly sized"""
    level = """#####
#S.R#
#####"""
    compiled = compile_level(level)
    
    assert len(compiled["tile_render_only"]) == MAX_TILES_C_API
    
def test_binary_roundtrip_render_only(self):
    """Test render-only field survives binary save/load"""
    level = """######
#S..R#
######"""
    original = compile_level(level)
    
    with tempfile.NamedTemporaryFile(suffix=".lvl", delete=False) as f:
        try:
            save_compiled_level_binary(original, f.name)
            loaded = load_compiled_level_binary(f.name)
            
            # Check render-only flags match
            for i in range(original["num_tiles"]):
                assert original["tile_render_only"][i] == loaded["tile_render_only"][i]
                
        finally:
            os.unlink(f.name)
```

#### 3.3 Add Integration Tests

**New test file: `tests/python/test_render_only_integration.py`**
```python
"""Integration tests for render-only assets with SimManager"""

import pytest
from madrona_escape_room.level_compiler import compile_level

class TestRenderOnlyIntegration:
    """Test render-only assets work end-to-end with SimManager"""
    
    @pytest.mark.custom_level("""##########
#S.......#
#..RRR...#
#..XXX...#
##########""")
    def test_sim_manager_with_render_only_level(self, cpu_manager):
        """Test SimManager handles render-only entities correctly"""
        # cpu_manager fixture automatically uses the custom level from the marker
        
        # Should run without physics conflicts
        for _ in range(50):
            cpu_manager.step()
            
    def test_performance_render_only_vs_physics(self, cpu_manager):
        """Test performance difference between render-only and physics entities"""
        # Create two test scenarios using compiled levels
        physics_level = """########
#S.....#
#.CCC..#
########"""
        
        render_level = """########
#S.....#
#.RRR..#
########"""
        
        # Test first level (uses default level from cpu_manager)
        for _ in range(10):
            cpu_manager.step()
            
        # For testing different levels, we'd need to create separate test methods
        # with @pytest.mark.custom_level markers, or test level compilation directly
        
    @pytest.mark.custom_level("""########
#S.....#
#.CCC..#
########""")
    def test_physics_entities_level(self, cpu_manager):
        """Test level with physics entities runs correctly"""
        for _ in range(10):
            cpu_manager.step()
            
    @pytest.mark.custom_level("""########
#S.....#
#.RRR..#
########""")  
    def test_render_only_entities_level(self, cpu_manager):
        """Test level with render-only entities runs correctly"""
        for _ in range(10):
            cpu_manager.step()
```

### Step 4: Update ctypes Bindings

#### 4.1 Update `madrona_escape_room/ctypes_bindings.py`

**Add render-only field to ctypes struct conversion:**
```python
def dict_to_compiled_level(compiled_dict: Dict) -> MER_CompiledLevel:
    # ... existing code ...
    
    # Copy render-only array
    for i in range(MAX_TILES):
        level.tile_render_only[i] = compiled_dict["tile_render_only"][i]
    
    return level
```

**Update validation:**
```python
def validate_compiled_level_ctypes(level: MER_CompiledLevel) -> None:
    # ... existing validation ...
    
    # Validate render-only flags are boolean-compatible
    for i in range(level.num_tiles):
        render_only = level.tile_render_only[i]
        assert isinstance(render_only, (bool, int, type(c_bool()))), \
            f"tile_render_only[{i}] must be boolean-compatible, got {type(render_only)}"
```

### Step 5: Documentation and Examples

#### 5.1 Update Level Format Documentation

**Create/update `docs/development/LEVEL_FORMAT.md`:**
```markdown
## Render-Only Assets

Render-only assets are visual elements that don't participate in physics simulation:

### ASCII Characters
- `R` - Decorative render-only element
- `X` - Axis marker (render-only)

### JSON Format
```json
{
    "ascii": "level layout with R and X",
    "render_only_positions": [
        {"x": 2.5, "y": 0.0},  // Override position as render-only
    ]
}
```

### Benefits
- Better performance (no physics computation)
- Visual decoration without collision
- Persistent visual markers
```

#### 5.2 Add Example Levels

**Create example files in `tests/test_data/levels/`:**
- `render_only_demo.txt` - ASCII level showcasing render-only features
- `mixed_entities.json` - JSON level with both physics and render-only entities

## Validation Criteria

### Phase 2 Complete When:

1. **All Python tests pass:**
   ```bash
   cd tests/python
   python -m pytest test_ascii_level_compiler.py -v
   python -m pytest test_level_compiler_c_api.py -v
   python -m pytest test_render_only_integration.py -v
   ```

2. **Binary format compatibility:**
   - Old `.lvl` files load with render-only defaulting to `false`
   - New `.lvl` files save and load render-only flags correctly
   - File size validation accounts for new boolean array

3. **End-to-end functionality:**
   - ASCII levels with 'R' and 'X' create render-only entities
   - SimManager works with render-only levels
   - Performance improvement measurable with render-only vs physics entities

4. **Integration tests pass:**
   ```bash
   # Test full pipeline from ASCII to simulation
   python -m pytest tests/python/test_render_only_integration.py -v
   ```

## Implementation Notes

- **Backward Compatibility**: Old `.lvl` files must work (default render-only to `false`)
- **Performance**: Render-only entities should not participate in physics broadphase
- **Validation**: Always validate array sizes match `MAX_TILES_C_API`
- **Documentation**: Update all relevant docs with render-only character meanings

## Testing Strategy

1. Start with unit tests - ensure level compiler produces correct data structures
2. Test binary I/O thoroughly - this is where compatibility issues appear
3. Add integration tests last - these verify the complete pipeline works
4. Test backward compatibility extensively with existing level files

## Success Metrics

- Level designer can mark tiles as render-only using 'R' and 'X' characters
- Render-only entities don't participate in physics (performance improvement)
- Existing levels continue to work without modification
- New binary `.lvl` format is forward/backward compatible
- Complete pipeline from ASCII → Compiled → Binary → SimManager works