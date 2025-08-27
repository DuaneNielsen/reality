#!/usr/bin/env python3
"""Test ALL fields in the structs for alignment and values"""
import ctypes
from madrona_escape_room.generated_dataclasses import CompiledLevel, ManagerConfig
from madrona_escape_room.default_level import create_default_level

print("=" * 60)
print("TESTING ALL FIELDS IN STRUCTS")
print("=" * 60)

# Create instances
level = create_default_level()
config = ManagerConfig()
config.exec_mode = 0
config.num_worlds = 1
config.rand_seed = 42
config.auto_reset = True

c_level = level.to_ctype()
c_config = config.to_ctype()

# Test ALL fields in CompiledLevel
print("\n1. ALL COMPILEDLEVEL FIELDS:")
print(f"{'Field Name':<25} {'Python Value':<20} {'CTypes Value':<20} {'Match'}")
print("-" * 75)

# Get all fields from the dataclass
import dataclasses
for field in dataclasses.fields(CompiledLevel):
    field_name = field.name
    if field_name.startswith('_pad'):
        continue  # Skip padding fields
    
    py_value = getattr(level, field_name)
    c_value = getattr(c_level, field_name)
    
    # Handle different types
    if isinstance(py_value, list):
        # For arrays, check first few elements and length
        if len(py_value) > 0:
            # Check array length
            if hasattr(c_value, '__len__'):
                len_match = len(py_value) == len(c_value)
                # Check first 3 elements
                vals_match = True
                for i in range(min(3, len(py_value))):
                    if isinstance(py_value[i], tuple):
                        # Handle tuple arrays (like quaternions)
                        if py_value[i] != tuple(c_value[i]):
                            vals_match = False
                            break
                    else:
                        if py_value[i] != c_value[i]:
                            vals_match = False
                            break
                
                match = "✓" if len_match and vals_match else "✗"
                py_str = f"list[{len(py_value)}] [{py_value[0]},...]"
                c_str = f"array[{len(c_value)}] [{c_value[0]},...]"
            else:
                match = "?"
                py_str = f"list[{len(py_value)}]"
                c_str = str(type(c_value))
        else:
            match = "✓"
            py_str = "[]"
            c_str = "[]"
    elif isinstance(py_value, bytes):
        # For bytes, check if they match
        match = "✓" if py_value == c_value else "✗"
        py_str = py_value[:20].decode('utf-8', errors='ignore')
        c_str = c_value[:20].decode('utf-8', errors='ignore')
    else:
        # Simple values
        match = "✓" if py_value == c_value else "✗"
        py_str = str(py_value)
        c_str = str(c_value)
    
    print(f"{field_name:<25} {py_str:<20} {c_str:<20} {match}")

# Test ALL fields in ManagerConfig
print("\n2. ALL MANAGERCONFIG FIELDS:")
print(f"{'Field Name':<25} {'Python Value':<20} {'CTypes Value':<20} {'Match'}")
print("-" * 75)

for field in dataclasses.fields(ManagerConfig):
    field_name = field.name
    if field_name.startswith('_pad'):
        continue  # Skip padding fields
    
    py_value = getattr(config, field_name)
    c_value = getattr(c_config, field_name)
    
    match = "✓" if py_value == c_value else "✗"
    print(f"{field_name:<25} {str(py_value):<20} {str(c_value):<20} {match}")

# Check specific array elements that should have non-zero values
print("\n3. CHECKING NON-ZERO ARRAY ELEMENTS:")
print("-" * 60)

# Check spawn arrays
for i in range(level.num_spawns):
    print(f"spawn_x[{i}]: Python={level.spawn_x[i]}, CTypes={c_level.spawn_x[i]}, Match={'✓' if level.spawn_x[i] == c_level.spawn_x[i] else '✗'}")
    print(f"spawn_y[{i}]: Python={level.spawn_y[i]}, CTypes={c_level.spawn_y[i]}, Match={'✓' if level.spawn_y[i] == c_level.spawn_y[i] else '✗'}")
    print(f"spawn_facing[{i}]: Python={level.spawn_facing[i]}, CTypes={c_level.spawn_facing[i]}, Match={'✓' if level.spawn_facing[i] == c_level.spawn_facing[i] else '✗'}")

# Check first few tiles
print(f"\nFirst 5 tiles (of {level.num_tiles}):")
for i in range(min(5, level.num_tiles)):
    print(f"  Tile {i}:")
    print(f"    object_ids[{i}]: Python={level.object_ids[i]}, CTypes={c_level.object_ids[i]}, Match={'✓' if level.object_ids[i] == c_level.object_ids[i] else '✗'}")
    print(f"    tile_x[{i}]: Python={level.tile_x[i]}, CTypes={c_level.tile_x[i]}, Match={'✓' if level.tile_x[i] == c_level.tile_x[i] else '✗'}")
    print(f"    tile_y[{i}]: Python={level.tile_y[i]}, CTypes={c_level.tile_y[i]}, Match={'✓' if level.tile_y[i] == c_level.tile_y[i] else '✗'}")
    print(f"    tile_entity_type[{i}]: Python={level.tile_entity_type[i]}, CTypes={c_level.tile_entity_type[i]}, Match={'✓' if level.tile_entity_type[i] == c_level.tile_entity_type[i] else '✗'}")

# Check quaternion rotation field specifically
print("\n4. CHECKING QUATERNION FIELD (tile_rotation):")
print("-" * 60)
for i in range(min(3, level.num_tiles)):
    py_quat = level.tile_rotation[i]
    c_quat = c_level.tile_rotation[i]
    if isinstance(c_quat, (list, tuple)):
        c_quat_tuple = tuple(c_quat)
    else:
        c_quat_tuple = c_quat
    match = "✓" if py_quat == c_quat_tuple else "✗"
    print(f"tile_rotation[{i}]: Python={py_quat}, CTypes={c_quat_tuple}, Match={match}")

print("\n" + "=" * 60)