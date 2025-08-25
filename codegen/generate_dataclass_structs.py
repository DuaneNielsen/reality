#!/usr/bin/env python3
"""
Generate Python dataclass structures from compiled binary using pahole.
These dataclasses use cdataclass to maintain C compatibility while providing
Pythonic debugging and array handling.
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
    field_pattern = re.compile(
        r"^\s+(.+?)\s+(\w+)(?:\[(\d+)\])?\s*;\s*/\*\s*(\d+)\s+(\d+)\s*\*/", re.MULTILINE
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


def map_c_to_python_type(c_type: str, size: int, is_array: bool = False) -> str:
    """Map C type to Python type annotation for dataclass."""
    c_type = c_type.strip()

    type_map = {
        "int32_t": "int",
        "uint32_t": "int",
        "int64_t": "int",
        "uint64_t": "int",
        "float": "float",
        "double": "float",
        "bool": "bool",
        "char": "bytes",  # char arrays become bytes
        "int": "int",
        "unsigned int": "int",
        "long": "int",
        "unsigned long": "int",
        "enum madrona::ExecMode": "int",
        "enum ExecMode": "int",
    }

    python_type = type_map.get(c_type, "int")  # Default to int for unknown types

    # Special case for char arrays - they're bytes, not List[bytes]
    if c_type == "char" and is_array:
        return "bytes"

    # Other arrays become Lists
    if is_array and python_type != "bytes":
        return f"List[{python_type}]"

    return python_type


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
        "enum madrona::ExecMode": "ctypes.c_int",
        "enum ExecMode": "ctypes.c_int",
    }

    return type_map.get(c_type, "ctypes.c_byte")


def generate_dataclass_struct(
    struct_name: str, fields: Dict[str, FieldInfo], is_big_endian: bool = False
) -> str:
    """Generate Python dataclass code from field info."""
    lines = []

    # Determine base class
    base_class = "BigEndianCDataMixIn" if is_big_endian else "NativeEndianCDataMixIn"

    lines.append("@dataclass")
    lines.append(f"class {struct_name}({base_class}):")

    # Sort fields by offset
    sorted_fields = sorted(fields.values(), key=lambda f: f.offset)

    current_offset = 0
    field_lines = []
    factories_needed = set()  # Track which factory functions we need

    for field in sorted_fields:
        # Add padding if there's a gap
        if field.offset > current_offset:
            padding_size = field.offset - current_offset
            pad_name = f"_pad_{current_offset}"
            meta_str = f"meta(ctypes.c_byte * {padding_size})"
            default_str = f"b'\\x00' * {padding_size}"
            field_lines.append(
                f"    {pad_name}: bytes = field(metadata={meta_str}, default={default_str})"
            )

        # Get Python type for annotation
        python_type = map_c_to_python_type(field.type_str, field.size, field.is_array)

        # Get ctypes type for metadata
        ctype = map_c_to_ctypes(field.type_str, field.size)

        # Handle arrays
        if field.is_array:
            if field.type_str == "char" and field.array_size:
                # Char arrays are strings/bytes
                meta_str = f"meta(ctypes.c_char * {field.array_size})"
                field_lines.append(
                    f"    {field.name}: {python_type} = field(metadata={meta_str}, default=b'')"
                )
            else:
                # Regular arrays with pre-sized factory
                base_type = python_type.replace("List[", "").replace("]", "")
                if base_type == "bool":
                    factory_name = f"_make_bool_array_{field.array_size}"
                    factories_needed.add((factory_name, field.array_size, "bool"))
                elif base_type == "int":
                    factory_name = f"_make_int_array_{field.array_size}"
                    factories_needed.add((factory_name, field.array_size, "int"))
                else:  # float
                    factory_name = f"_make_float_array_{field.array_size}"
                    factories_needed.add((factory_name, field.array_size, "float"))

                meta_str = f"meta({ctype} * {field.array_size})"
                field_lines.append(
                    f"    {field.name}: {python_type} = field("
                    f"metadata={meta_str}, default_factory={factory_name})"
                )
        else:
            # Single values
            default_val = (
                "0"
                if python_type in ["int", "float"]
                else "False"
                if python_type == "bool"
                else "b''"
            )
            field_lines.append(
                f"    {field.name}: {python_type} = field("
                f"metadata=meta({ctype}), default={default_val})"
            )

        current_offset = field.offset + field.size

    # Add fields or pass for empty class
    if field_lines:
        lines.extend(field_lines)
    else:
        lines.append("    pass")

    # Return struct code and needed factories
    return "\n".join(lines), factories_needed


def get_struct_size(library_path: str, struct_name: str) -> int:
    """Extract struct size from pahole output."""
    output = run_pahole(library_path, struct_name)
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
        "Auto-generated Python dataclass structures from compiled binary.",
        "Uses cdataclass for C compatibility with Pythonic interface.",
        "DO NOT EDIT - this file is automatically regenerated.",
        '"""',
        "",
        "import ctypes",
        "from dataclasses import dataclass, field",
        "from typing import List",
        "",
        "from cdataclass import NativeEndianCDataMixIn, BigEndianCDataMixIn, meta",
        "",
    ]

    struct_sizes = {}
    all_factories = set()  # Collect all factory functions needed
    struct_codes = []  # Store struct code to output after factories

    # First pass: process each struct and collect factories
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

        # Generate dataclass
        # Note: Using little-endian (NativeEndianCDataMixIn) for x86/x64 architectures
        struct_code, factories = generate_dataclass_struct(struct_name, fields, is_big_endian=False)
        all_factories.update(factories)  # Collect factory functions
        struct_codes.append((struct_name, struct_code))  # Store for later

        # Store size for validation
        size = get_struct_size(library_path, struct_name)
        if size > 0:
            struct_sizes[struct_name] = size

    # Generate factory functions for arrays (must come before dataclasses)
    if all_factories:
        output_lines.append("")
        output_lines.append("# Factory functions for pre-sized arrays")

        # Sort factories by name for consistent output
        for factory_name, size, type_name in sorted(all_factories):
            if type_name == "bool":
                default_val = "False"
            elif type_name == "int":
                default_val = "0"
            else:  # float
                default_val = "0.0"

            output_lines.append(f"def {factory_name}():")
            output_lines.append(f'    """Factory for {size}-element {type_name} array"""')
            output_lines.append(f"    return [{default_val}] * {size}")
            output_lines.append("")

    # Now output the dataclass definitions
    output_lines.append("")
    output_lines.append("# Dataclass structures")
    for struct_name, struct_code in struct_codes:
        output_lines.append(struct_code)
        output_lines.append("")

    # Add size validation
    output_lines.append("")
    output_lines.append("# Size validation")
    for struct_name, size in struct_sizes.items():
        output_lines.append(f"assert {struct_name}.size() == {size}, \\")
        output_lines.append(
            f'    f"{struct_name} size mismatch: {{{struct_name}.size()}} != {size}"'
        )
        output_lines.append("")

    # Add helpers
    output_lines.extend(
        [
            "",
            "# Import proper constants from generated_constants",
            "from madrona_escape_room.generated_constants import limits",
            "",
            "# Use the constant defined in consts.hpp",
            "MAX_TILES = limits.maxTiles",
            "MAX_SPAWNS = limits.maxSpawns",
            "",
            "",
            "# Helper function to convert any dataclass to ctypes for C API",
            "def to_ctypes(obj):",
            '    """Convert dataclass to ctypes Structure for C API."""',
            "    if hasattr(obj, 'to_ctype'):",
            "        return obj.to_ctype()",
            "    return obj  # Already a ctypes structure",
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
        print("Usage: generate_dataclass_structs.py <library_path> <output_path>", file=sys.stderr)
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
