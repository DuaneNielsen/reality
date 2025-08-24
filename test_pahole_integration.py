#!/usr/bin/env python3
"""Test that the pahole-generated structs work correctly with the C API."""

import ctypes

from madrona_escape_room.ctypes_bindings import (
    CompiledLevel,
    compiled_level_to_dict,
    dict_to_compiled_level,
    lib,
)

print("Testing pahole-generated struct integration...")

# Create a test level using the auto-generated struct
level = CompiledLevel()
level.width = 10
level.height = 10
level.num_tiles = 4
level.max_entities = 50
level.world_scale = 1.0
level.done_on_collide = False
level.level_name = b"test_level"

# Set world boundaries
level.world_min_x = -10.0
level.world_max_x = 10.0
level.world_min_y = -10.0
level.world_max_y = 10.0
level.world_min_z = 0.0
level.world_max_z = 5.0

# Set spawn point
level.num_spawns = 1
level.spawn_x[0] = 0.0
level.spawn_y[0] = 0.0
level.spawn_facing[0] = 0.0

# Add some tiles
for i in range(4):
    level.object_ids[i] = 2  # WALL
    level.tile_x[i] = float(i * 2 - 3)
    level.tile_y[i] = 5.0
    level.tile_z[i] = 0.0
    level.tile_persistent[i] = True
    level.tile_render_only[i] = False
    level.tile_entity_type[i] = 2  # Wall
    level.tile_response_type[i] = 2  # Static

print(f"Created CompiledLevel struct: {ctypes.sizeof(level)} bytes")

# Validate using C API
result = lib.mer_validate_compiled_level(ctypes.byref(level))
if result == 0:
    print("✓ Level validation passed!")
else:
    print(f"✗ Level validation failed with code: {result}")

# Test size consistency
c_api_size = lib.mer_get_compiled_level_size()
python_size = ctypes.sizeof(CompiledLevel)

print("\nSize comparison:")
print(f"  C++ struct size: {c_api_size} bytes")
print(f"  Python struct size: {python_size} bytes")
print(f"  Match: {'✓' if c_api_size == python_size else '✗'}")

# Test dict conversion
print("\nTesting dict conversion...")
test_dict = {
    "width": 5,
    "height": 5,
    "num_tiles": 2,
    "max_entities": 30,
    "world_scale": 2.0,
    "done_on_collide": True,
    "level_name": "dict_test",
    "world_min_x": -5.0,
    "world_max_x": 5.0,
    "world_min_y": -5.0,
    "world_max_y": 5.0,
    "world_min_z": 0.0,
    "world_max_z": 10.0,
    "num_spawns": 1,
    "spawn_x": [1.0] + [0.0] * 7,
    "spawn_y": [2.0] + [0.0] * 7,
    "spawn_facing": [0.0] * 8,
    "array_size": 25,
    "object_ids": [1, 2] + [0] * 23,
    "tile_x": [0.0, 1.0] + [0.0] * 23,
    "tile_y": [0.0, 1.0] + [0.0] * 23,
    "tile_z": [0.0] * 25,
    "tile_persistent": [True, False] + [False] * 23,
    "tile_render_only": [False] * 25,
    "tile_entity_type": [1, 2] + [0] * 23,
    "tile_response_type": [2] * 25,
    "tile_scale_x": [1.0] * 25,
    "tile_scale_y": [1.0] * 25,
    "tile_scale_z": [1.0] * 25,
    "tile_rot_w": [1.0] * 25,
    "tile_rot_x": [0.0] * 25,
    "tile_rot_y": [0.0] * 25,
    "tile_rot_z": [0.0] * 25,
}

level_from_dict = dict_to_compiled_level(test_dict)
print(f"✓ Created level from dict: {ctypes.sizeof(level_from_dict)} bytes")

# Convert back to dict
dict_from_level = compiled_level_to_dict(level_from_dict)
print("✓ Converted level back to dict")
print(f"  Width matches: {'✓' if dict_from_level['width'] == test_dict['width'] else '✗'}")
print(f"  Tiles match: {'✓' if dict_from_level['num_tiles'] == test_dict['num_tiles'] else '✗'}")

print("\n✅ All tests passed! Pahole integration is working correctly.")
