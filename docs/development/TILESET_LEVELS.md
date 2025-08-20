# Tileset-Based Level Creation

The level compiler now supports tilesets that map ASCII characters to asset names from the statically loaded asset registry. This provides a flexible way to create levels using any available assets without hardcoding tile types.

## Quick Start

### Using the Default Tileset

When your level contains special characters like 'O' (for cylinder), the compiler automatically uses the default tileset:

```python
from madrona_escape_room.level_compiler import compile_level_from_json

level_json = {
    "ascii": """
###O###
#S...C#
#######
    """.strip()
}

compiled = compile_level_from_json(level_json)
```

### Custom Tileset

Define your own character-to-asset mappings:

```python
level_json = {
    "ascii": """
############
#S.........#
#..O...O...#
#....C.....#
############
    """.strip(),
    
    "tileset": {
        "#": {"asset": "wall"},      # Wall blocks
        "C": {"asset": "cube"},      # Pushable cube
        "O": {"asset": "cylinder"},  # Cylinder obstacles
        "S": {"asset": "spawn"},     # Agent spawn point
        ".": {"asset": "empty"}      # Empty space
    },
    
    "scale": 3.0,
    "name": "obstacle_course"
}

compiled = compile_level_from_json(level_json)
```

## Available Assets

The following assets are available for use in tilesets:

### Physics Assets (have collision)
- `cube` - Pushable cube
- `wall` - Static wall block
- `agent` - Agent (usually not placed manually)
- `plane` - Infinite ground plane
- `cylinder` - Static cylinder obstacle

### Render-Only Assets (no collision)
- `axis_x` - Red axis marker
- `axis_y` - Green axis marker
- `axis_z` - Blue axis marker

### Special Values
- `spawn` - Agent spawn location (not a physical entity)
- `empty` - Empty space (no entity created)

## Tileset Format

Each tileset entry maps a single ASCII character to an asset definition:

```json
{
    "CHARACTER": {
        "asset": "ASSET_NAME"
    }
}
```

## Default Tileset

When no tileset is specified, these mappings are used:

```python
DEFAULT_TILESET = {
    "#": {"asset": "wall"},
    "C": {"asset": "cube"},
    "O": {"asset": "cylinder"},
    "S": {"asset": "spawn"},
    ".": {"asset": "empty"},
    " ": {"asset": "empty"}
}
```

## Backward Compatibility

Existing levels using only the legacy characters (#, C, S, .) continue to work without modification:

```python
# Legacy format still works
level = """
######
#S.C.#
######
"""
compiled = compile_level(level)  # Uses hardcoded CHAR_MAP
```

## Creative Character Mapping

You can use any characters for your tileset:

```python
level_json = {
    "ascii": """
++++++++++
+@.......+
+..***...+
++++++++++
    """.strip(),
    
    "tileset": {
        "+": {"asset": "wall"},   # Use + for walls
        "*": {"asset": "cube"},   # Use * for cubes
        "@": {"asset": "spawn"},  # Use @ for spawn
        ".": {"asset": "empty"}   # Keep . for empty
    }
}
```

## Using with SimManager

Pass the JSON directly to SimManager:

```python
import json
from madrona_escape_room import SimManager, madrona

level_json = {
    "ascii": "...",
    "tileset": {...}
}

manager = SimManager(
    exec_mode=madrona.ExecMode.CPU,
    gpu_id=0,
    num_worlds=1,
    rand_seed=42,
    auto_reset=True,
    level_ascii=json.dumps(level_json)
)
```

## Error Handling

The compiler validates asset names and raises clear errors:

```python
# Invalid asset name
tileset = {
    "#": {"asset": "mystery_box"}  # Will raise: "Unknown asset name: 'mystery_box'"
}

# Invalid tileset structure
tileset = {
    "##": {"asset": "wall"}  # Will raise: "Tileset key must be single character"
}
```

## Examples

See `examples/tileset_level_example.py` for complete working examples including:
- Simple tileset with basic assets
- Advanced tileset with cylinders
- Automatic tileset detection
- Creative character mappings

## Testing

Run the tileset tests to verify functionality:

```bash
uv run pytest tests/python/test_tileset_compiler.py -xvs
```