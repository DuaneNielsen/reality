# ruff: noqa
#!/usr/bin/env python3
"""Generate ctypes structs from compiled binary using pahole."""

import re
import subprocess
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class FieldInfo:
    """Information about a struct field from pahole."""

    name: str
    type_str: str
    offset: int
    size: int
    is_array: bool = False
    array_size: Optional[int] = None


def parse_pahole_output(pahole_output: str) -> Dict[str, FieldInfo]:
    """Parse pahole output to extract field information."""
    fields = {}

    # Pattern to match field lines with offset and size
    field_pattern = re.compile(
        r"^\s+(.+?)\s+(\w+)(?:\[(\d+)\])?;\s*/\*\s*(\d+)\s+(\d+)\s*\*/", re.MULTILINE
    )

    for match in field_pattern.finditer(pahole_output):
        type_str = match.group(1).strip()
        field_name = match.group(2)
        array_size_str = match.group(3)
        offset = int(match.group(4))
        size = int(match.group(5))

        is_array = array_size_str is not None
        array_size = int(array_size_str) if is_array else None

        fields[field_name] = FieldInfo(
            name=field_name,
            type_str=type_str,
            offset=offset,
            size=size,
            is_array=is_array,
            array_size=array_size,
        )

    return fields


def get_struct_layout(library_path: str, struct_name: str) -> Dict[str, FieldInfo]:
    """Extract struct layout using pahole."""
    try:
        result = subprocess.run(
            ["pahole", "-C", struct_name, library_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # Suppress warnings
            text=True,
        )
        return parse_pahole_output(result.stdout)
    except Exception as e:
        print(f"Error running pahole: {e}")
        return {}


def map_c_to_ctypes(c_type: str, size: int) -> str:
    """Map C type string to ctypes equivalent."""
    c_type = c_type.strip()

    type_map = {
        "int32_t": "ctypes.c_int32",
        "uint32_t": "ctypes.c_uint32",
        "int64_t": "ctypes.c_int64",
        "uint64_t": "ctypes.c_uint64",
        "float": "ctypes.c_float",
        "double": "ctypes.c_double",
        "bool": "ctypes.c_bool",
        "char": "ctypes.c_byte",  # Use c_byte for char arrays
        "int": "ctypes.c_int",
        "unsigned int": "ctypes.c_uint",
        "long": "ctypes.c_long",
        "unsigned long": "ctypes.c_ulong",
    }

    if c_type in type_map:
        return type_map[c_type]

    # Fall back to byte array
    return "ctypes.c_byte"


def generate_ctypes_fields(fields: Dict[str, FieldInfo]) -> list:
    """Generate the _fields_ list for a ctypes.Structure."""
    sorted_fields = sorted(fields.values(), key=lambda f: f.offset)

    current_offset = 0
    field_defs = []

    for field in sorted_fields:
        # Add padding if there's a gap
        if field.offset > current_offset:
            padding_size = field.offset - current_offset
            field_defs.append(("_pad_{}".format(current_offset), ctypes.c_byte * padding_size))

        # Map C type to ctypes
        ctype_str = map_c_to_ctypes(field.type_str, field.size)

        # Convert string to actual ctypes type
        if ctype_str == "ctypes.c_int32":
            ctype = ctypes.c_int32
        elif ctype_str == "ctypes.c_float":
            ctype = ctypes.c_float
        elif ctype_str == "ctypes.c_bool":
            ctype = ctypes.c_bool
        elif ctype_str == "ctypes.c_byte":
            ctype = ctypes.c_byte
        else:
            ctype = ctypes.c_byte

        # Handle arrays
        if field.is_array:
            field_defs.append((field.name, ctype * field.array_size))
        else:
            field_defs.append((field.name, ctype))

        current_offset = field.offset + field.size

    return field_defs


# Generate all the structs we need
import ctypes

# Handle both running from project root and from scratch dir
import os

if os.path.exists("build/libmadrona_escape_room_c_api.so"):
    LIBRARY_PATH = "build/libmadrona_escape_room_c_api.so"
else:
    LIBRARY_PATH = "../build/libmadrona_escape_room_c_api.so"

# Extract and generate CompiledLevel
print("Extracting CompiledLevel...")
compiled_level_fields = get_struct_layout(LIBRARY_PATH, "CompiledLevel")


class CompiledLevel(ctypes.Structure):
    pass


# We need to set _fields_ after class definition to handle forward references
CompiledLevel._fields_ = generate_ctypes_fields(compiled_level_fields)

# Extract and generate Action
print("Extracting Action...")
action_fields = get_struct_layout(LIBRARY_PATH, "Action")


class Action(ctypes.Structure):
    pass


Action._fields_ = generate_ctypes_fields(action_fields)

# Extract and generate SelfObservation
print("Extracting SelfObservation...")
obs_fields = get_struct_layout(LIBRARY_PATH, "SelfObservation")


class SelfObservation(ctypes.Structure):
    pass


SelfObservation._fields_ = generate_ctypes_fields(obs_fields)

# Extract and generate Done
print("Extracting Done...")
done_fields = get_struct_layout(LIBRARY_PATH, "Done")


class Done(ctypes.Structure):
    pass


Done._fields_ = generate_ctypes_fields(done_fields)

# Extract and generate Reward
print("Extracting Reward...")
reward_fields = get_struct_layout(LIBRARY_PATH, "Reward")


class Reward(ctypes.Structure):
    pass


Reward._fields_ = generate_ctypes_fields(reward_fields)

print("\nAll structs generated successfully!")
print(f"CompiledLevel size: {ctypes.sizeof(CompiledLevel)} bytes")
print(f"Action size: {ctypes.sizeof(Action)} bytes")
print(f"SelfObservation size: {ctypes.sizeof(SelfObservation)} bytes")
