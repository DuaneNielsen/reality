#!/usr/bin/env python3
"""
Generate Python ctypes structures from compiled binary using pahole.
This script is run as a post-build step to automatically generate
Python bindings that match the exact memory layout of C++ structs.
"""

import os
import re
import subprocess
import sys
from typing import Dict, List, Optional, Tuple


class FieldInfo:
    """Information about a struct field from pahole."""

    def __init__(
        self,
        name: str,
        type_str: str,
        offset: int,
        size: int,
        is_array: bool = False,
        array_size: Optional[int] = None,
    ):
        self.name = name
        self.type_str = type_str
        self.offset = offset
        self.size = size
        self.is_array = is_array
        self.array_size = array_size


def run_pahole(library_path: str, struct_name: str) -> str:
    """Run pahole to extract struct layout."""
    try:
        result = subprocess.run(
            ["pahole", "-C", struct_name, library_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            check=False,
        )
        return result.stdout
    except Exception as e:
        print(f"Error running pahole for {struct_name}: {e}", file=sys.stderr)
        return ""


def parse_pahole_output(pahole_output: str) -> Dict[str, FieldInfo]:
    """Parse pahole output to extract field information."""
    fields = {}

    # Pattern to match field lines with offset and size
    # Example: "int32_t                    moveAmount;           /*     0     4 */"
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
        "char": "ctypes.c_char",
        "int": "ctypes.c_int",
        "unsigned int": "ctypes.c_uint",
        "long": "ctypes.c_long",
        "unsigned long": "ctypes.c_ulong",
        "enum madrona::ExecMode": "ctypes.c_int",  # Enums are stored as integers
        "enum ExecMode": "ctypes.c_int",  # Pahole sometimes strips namespace
    }

    if c_type in type_map:
        return type_map[c_type]

    # Fall back to byte array for unknown types
    return "ctypes.c_byte"


def generate_ctypes_struct(struct_name: str, fields: Dict[str, FieldInfo]) -> str:
    """Generate Python ctypes.Structure code from field info."""
    lines = []
    lines.append(f"class {struct_name}(ctypes.Structure):")
    lines.append("    _fields_ = [")

    # Sort fields by offset
    sorted_fields = sorted(fields.values(), key=lambda f: f.offset)

    current_offset = 0

    for field in sorted_fields:
        # Add padding if there's a gap
        if field.offset > current_offset:
            padding_size = field.offset - current_offset
            lines.append(f"        ('_pad_{current_offset}', ctypes.c_byte * {padding_size}),")

        # Map C type to ctypes
        ctype = map_c_to_ctypes(field.type_str, field.size)

        # Handle arrays
        if field.is_array:
            if field.type_str == "char" and field.array_size:
                # Special case for char arrays (strings)
                lines.append(f"        ('{field.name}', ctypes.c_char * {field.array_size}),")
            else:
                lines.append(f"        ('{field.name}', {ctype} * {field.array_size}),")
        else:
            lines.append(f"        ('{field.name}', {ctype}),")

        current_offset = field.offset + field.size

    lines.append("    ]")

    return "\n".join(lines)


def get_struct_size(library_path: str, struct_name: str) -> int:
    """Extract struct size from pahole output."""
    output = run_pahole(library_path, struct_name)
    # Look for size in format: /* size: 84180, cachelines: 1316, members: 39 */
    size_match = re.search(r"/\*\s*size:\s*(\d+)", output)
    if size_match:
        return int(size_match.group(1))
    return 0


def generate_python_bindings(library_path: str, output_path: str):
    """Generate Python bindings for all required structs."""

    # Structs to extract
    structs_to_extract = [
        "CompiledLevel",
        "Action",
        "SelfObservation",
        "Done",
        "Reward",
        "Progress",
        "StepsRemaining",
        "ReplayMetadata",
        "ManagerConfig",
    ]

    # Start building the output file
    output_lines = [
        '"""',
        "Auto-generated Python ctypes structures from compiled binary.",
        "Generated by pahole during build process.",
        "DO NOT EDIT - this file is automatically regenerated.",
        '"""',
        "",
        "import ctypes",
        "",
        "",
    ]

    # Process each struct
    for struct_name in structs_to_extract:
        print(f"Extracting {struct_name}...", file=sys.stderr)

        # Get pahole output
        pahole_output = run_pahole(library_path, struct_name)
        if not pahole_output:
            print(f"  Warning: No output for {struct_name}", file=sys.stderr)
            continue

        # Parse fields
        fields = parse_pahole_output(pahole_output)
        if not fields:
            print(f"  Warning: No fields found for {struct_name}", file=sys.stderr)
            continue

        # Generate ctypes struct
        struct_code = generate_ctypes_struct(struct_name, fields)
        output_lines.append(struct_code)
        output_lines.append("")

        # Add size verification comment
        size = get_struct_size(library_path, struct_name)
        if size > 0:
            output_lines.append(f"# {struct_name} size: {size} bytes")
            output_lines.append(f"assert ctypes.sizeof({struct_name}) == {size}, \\")
            output_lines.append(
                f'    f"{struct_name} size mismatch: {{ctypes.sizeof({struct_name})}} != {size}"'
            )
            output_lines.append("")

    # Add helper to get MAX_TILES constant
    output_lines.extend(
        [
            "",
            "# Extract MAX_TILES from CompiledLevel if it exists",
            "def get_max_tiles():",
            '    """Get MAX_TILES from CompiledLevel struct."""',
            '    if hasattr(CompiledLevel, "_fields_"):',
            "        for field_name, field_type in CompiledLevel._fields_:",
            '            if "tile_x" in field_name and hasattr(field_type, "_length_"):',
            "                return field_type._length_",
            "    return 1024  # fallback",
            "",
            "MAX_TILES = get_max_tiles()",
            "",
        ]
    )

    # Write the output file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(output_lines))

    print(f"Generated {output_path}", file=sys.stderr)


def main():
    if len(sys.argv) != 3:
        print("Usage: generate_python_structs.py <library_path> <output_path>", file=sys.stderr)
        sys.exit(1)

    library_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.exists(library_path):
        print(f"Error: Library not found: {library_path}", file=sys.stderr)
        sys.exit(1)

    # Check if pahole is available
    try:
        subprocess.run(
            ["pahole", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: pahole not found. Install with: sudo apt install dwarves", file=sys.stderr)
        sys.exit(1)

    generate_python_bindings(library_path, output_path)


if __name__ == "__main__":
    main()
