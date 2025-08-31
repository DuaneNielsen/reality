#!/usr/bin/env python3
"""Test the quaternion field specifically"""
import ctypes
from madrona_escape_room.generated_dataclasses import CompiledLevel
from madrona_escape_room.default_level import create_default_level

print("=" * 60)
print("TESTING QUATERNION FIELD ALIGNMENT")
print("=" * 60)

# Create instance
level = create_default_level()
c_level = level.to_ctype()

print("\n1. QUATERNION FIELD TYPE:")
print(f"Python tile_rotation type: {type(level.tile_rotation)}")
print(f"Python tile_rotation[0] type: {type(level.tile_rotation[0])}")
print(f"CTypes tile_rotation type: {type(c_level.tile_rotation)}")
print(f"CTypes tile_rotation[0] type: {type(c_level.tile_rotation[0])}")

print("\n2. QUATERNION VALUES (first 5):")
for i in range(5):
    py_quat = level.tile_rotation[i]
    c_quat_raw = c_level.tile_rotation[i]
    
    # Try to extract values from the c_float_Array_4
    if hasattr(c_quat_raw, '__getitem__'):
        c_quat = tuple(c_quat_raw[j] for j in range(4))
    else:
        c_quat = "Cannot extract"
    
    match = "✓" if py_quat == c_quat else "✗"
    print(f"  [{i}] Python: {py_quat}")
    print(f"      CTypes: {c_quat} {match}")

print("\n3. RAW MEMORY CHECK:")
# Check the raw bytes where quaternions are stored
# tile_rotation should be at a specific offset
field_info = type(c_level).tile_rotation
print(f"tile_rotation offset: {field_info.offset}")
print(f"tile_rotation size: {field_info.size}")

# Get raw bytes for first quaternion (16 bytes = 4 floats)
import struct
raw_bytes = bytes(c_level)[field_info.offset:field_info.offset + 16]
floats = struct.unpack('4f', raw_bytes)
print(f"First quaternion raw bytes: {raw_bytes.hex()}")
print(f"First quaternion as floats: {floats}")
print(f"Expected: (1.0, 0.0, 0.0, 0.0)")

print("\n4. FIELD OFFSET COMPARISON:")
# Check offsets of fields around tile_rotation
fields_around = [
    'tile_scale_z', 
    'tile_rotation', 
    'tile_rand_x'
]
for field_name in fields_around:
    field = getattr(type(c_level), field_name)
    print(f"{field_name:20s}: offset={field.offset:6d}, size={field.size:6d}")

print("\n5. SIZE CALCULATION CHECK:")
# Calculate expected offset for tile_rotation
# Based on the fields before it
expected_offset = 0
expected_offset += 4  # num_tiles
expected_offset += 4  # max_entities
expected_offset += 4  # width
expected_offset += 4  # height
expected_offset += 4  # world_scale
expected_offset += 1  # done_on_collide
expected_offset += 64 # level_name
expected_offset += 3  # _pad_85
expected_offset += 6 * 4  # world_min/max x,y,z (6 floats)
expected_offset += 4  # num_spawns
expected_offset += 8 * 4 * 3  # spawn_x, spawn_y, spawn_facing (3 arrays of 8 floats each)
expected_offset += 1024 * 4  # object_ids
expected_offset += 1024 * 4 * 3  # tile_x, tile_y, tile_z
expected_offset += 1024 * 1 * 2  # tile_persistent, tile_render_only
expected_offset += 1024 * 4 * 2  # tile_entity_type, tile_response_type
expected_offset += 1024 * 4 * 3  # tile_scale_x, tile_scale_y, tile_scale_z

print(f"Expected offset for tile_rotation: {expected_offset}")
print(f"Actual offset for tile_rotation: {field_info.offset}")
print(f"Match: {'✓' if expected_offset == field_info.offset else '✗'}")

print("\n" + "=" * 60)