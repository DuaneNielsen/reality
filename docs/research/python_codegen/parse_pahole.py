# ruff: noqa
#!/usr/bin/env python3
"""Parse pahole output to extract struct memory layouts."""

import re
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


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
    """Parse pahole output to extract field information.

    Returns dict mapping field name to FieldInfo.
    """
    fields = {}

    # Pattern to match field lines with offset and size
    # Example: "int32_t                    moveAmount;           /*     0     4 */"
    field_pattern = re.compile(
        r"^\s+(.+?)\s+(\w+)(?:\[(\d+)\])?;\s*/\*\s*(\d+)\s+(\d+)\s*\*/", re.MULTILINE
    )

    # Pattern to match padding holes
    # Example: "/* XXX 3 bytes hole, try to pack */"
    hole_pattern = re.compile(r"/\*\s*XXX\s+(\d+)\s+bytes?\s+hole", re.MULTILINE)

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
    """Extract struct layout using pahole.

    Args:
        library_path: Path to compiled library (.so or .a)
        struct_name: Name of struct to extract

    Returns:
        Dict mapping field names to FieldInfo
    """
    try:
        result = subprocess.run(
            ["pahole", "-C", struct_name, library_path], capture_output=True, text=True, check=False
        )

        if result.returncode != 0:
            # Try to filter out the warnings
            clean_output = []
            for line in result.stdout.split("\n"):
                if not line.startswith("die__process"):
                    clean_output.append(line)
            output = "\n".join(clean_output)
        else:
            output = result.stdout

        return parse_pahole_output(output)

    except subprocess.CalledProcessError as e:
        print(f"Error running pahole: {e}")
        return {}


def generate_ctypes_struct(struct_name: str, fields: Dict[str, FieldInfo]) -> str:
    """Generate Python ctypes.Structure code from field info.

    Args:
        struct_name: Name for the generated class
        fields: Dict of field information from pahole

    Returns:
        Python code string defining the ctypes.Structure
    """
    lines = [f"class {struct_name}(ctypes.Structure):"]
    lines.append("    _fields_ = [")

    # Sort fields by offset
    sorted_fields = sorted(fields.values(), key=lambda f: f.offset)

    current_offset = 0
    field_defs = []

    for field in sorted_fields:
        # Add padding if there's a gap
        if field.offset > current_offset:
            padding_size = field.offset - current_offset
            field_defs.append(f"        ('_pad_{current_offset}', ctypes.c_byte * {padding_size}),")

        # Map C type to ctypes
        ctype = map_c_to_ctypes(field.type_str, field.size)

        # Handle arrays
        if field.is_array:
            field_def = f"        ('{field.name}', {ctype} * {field.array_size}),"
        else:
            field_def = f"        ('{field.name}', {ctype}),"

        field_defs.append(field_def)
        current_offset = field.offset + field.size

    lines.extend(field_defs)
    lines.append("    ]")

    return "\n".join(lines)


def map_c_to_ctypes(c_type: str, size: int) -> str:
    """Map C type string to ctypes equivalent.

    This is a simple mapping - extend as needed.
    """
    # Clean up the type string
    c_type = c_type.strip()

    # Common type mappings
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
    }

    if c_type in type_map:
        return type_map[c_type]

    # Fall back to size-based inference
    if "int" in c_type:
        if size == 1:
            return "ctypes.c_int8"
        elif size == 2:
            return "ctypes.c_int16"
        elif size == 4:
            return "ctypes.c_int32"
        elif size == 8:
            return "ctypes.c_int64"

    # Default fallback
    return f"ctypes.c_byte * {size}"


def main():
    """Example usage."""
    import sys

    if len(sys.argv) != 3:
        print("Usage: parse_pahole.py <library_path> <struct_name>")
        sys.exit(1)

    library_path = sys.argv[1]
    struct_name = sys.argv[2]

    print(f"Extracting {struct_name} from {library_path}...")
    fields = get_struct_layout(library_path, struct_name)

    if not fields:
        print(f"No fields found for {struct_name}")
        return

    print(f"\nFound {len(fields)} fields:")
    for name, info in sorted(fields.items(), key=lambda x: x[1].offset):
        array_str = f"[{info.array_size}]" if info.is_array else ""
        print(f"  {info.offset:4d}: {info.type_str:20s} {name}{array_str} ({info.size} bytes)")

    print("\nGenerated ctypes.Structure:")
    print("import ctypes")
    print()
    print(generate_ctypes_struct(struct_name, fields))


if __name__ == "__main__":
    main()
