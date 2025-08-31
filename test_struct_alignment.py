#!/usr/bin/env python3
"""Test struct alignment and sizes between Python and C"""
import ctypes
import struct
from madrona_escape_room.generated_dataclasses import CompiledLevel, ManagerConfig
from madrona_escape_room.default_level import create_default_level

print("=" * 60)
print("TESTING STRUCT SIZES AND ALIGNMENT")
print("=" * 60)

# 1. Check dataclass sizes
print("\n1. DATACLASS SIZES:")
print(f"CompiledLevel size: {CompiledLevel.size()} bytes")
print(f"ManagerConfig size: {ManagerConfig.size()} bytes")

# 2. Check ctypes conversion
print("\n2. CTYPES CONVERSION:")
level = create_default_level()
config = ManagerConfig()
config.exec_mode = 0
config.num_worlds = 1

c_level = level.to_ctype()
c_config = config.to_ctype()

print(f"C CompiledLevel size: {ctypes.sizeof(c_level)} bytes")
print(f"C ManagerConfig size: {ctypes.sizeof(c_config)} bytes")

# 3. Check field offsets in CompiledLevel
print("\n3. COMPILEDLEVEL FIELD OFFSETS:")
fields_to_check = [
    'num_tiles', 'max_entities', 'width', 'height', 'world_scale',
    'level_name', 'world_min_x', 'spawn_x', 'object_ids', 'tile_x'
]

for field_name in fields_to_check:
    if hasattr(c_level, field_name):
        field = getattr(type(c_level), field_name)
        offset = field.offset
        size = field.size
        print(f"  {field_name:20s}: offset={offset:6d}, size={size:6d}")

# 4. Check ManagerConfig field offsets
print("\n4. MANAGERCONFIG FIELD OFFSETS:")
for field_name, field_type in c_config._fields_:
    field = getattr(type(c_config), field_name)
    offset = field.offset
    size = field.size
    print(f"  {field_name:25s}: offset={offset:3d}, size={size:3d}")

# 5. Check alignment requirements
print("\n5. ALIGNMENT:")
print(f"CompiledLevel alignment: {ctypes.alignment(type(c_level))}")
print(f"ManagerConfig alignment: {ctypes.alignment(type(c_config))}")

# 6. Memory layout inspection
print("\n6. MEMORY DUMP (first 256 bytes of CompiledLevel):")
# Get raw bytes
raw_bytes = bytes(c_level)[:256]
# Print hex dump
for i in range(0, len(raw_bytes), 16):
    hex_str = ' '.join(f'{b:02x}' for b in raw_bytes[i:i+16])
    ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in raw_bytes[i:i+16])
    print(f"  {i:04x}: {hex_str:48s} |{ascii_str}|")

# 7. Check for padding issues
print("\n7. CHECKING FOR PADDING ISSUES:")
# Look for _pad fields in dataclass
import inspect
source = inspect.getsource(CompiledLevel)
pad_fields = [line.strip() for line in source.split('\n') if '_pad' in line]
if pad_fields:
    print("Found padding fields in CompiledLevel:")
    for pad in pad_fields[:5]:  # Show first 5
        print(f"  {pad}")

# 8. Test that values are where we expect them
print("\n8. VALUE VERIFICATION:")
test_values = {
    'num_tiles': level.num_tiles,
    'width': level.width, 
    'height': level.height,
    'num_spawns': level.num_spawns,
    'spawn_x[0]': level.spawn_x[0],
    'spawn_y[0]': level.spawn_y[0]
}

for name, expected in test_values.items():
    if '[' in name:
        # Array access
        field_name, index = name.split('[')
        index = int(index.rstrip(']'))
        actual = getattr(c_level, field_name)[index]
    else:
        actual = getattr(c_level, name)
    match = "✓" if actual == expected else "✗"
    print(f"  {name:20s}: expected={expected:10} actual={actual:10} {match}")

print("\n" + "=" * 60)