# ruff: noqa
#!/usr/bin/env python3
"""Parse constants from consts.hpp."""

import os

import clang.cindex
import clang.native

# Set libclang path
libclang_path = os.path.join(os.path.dirname(clang.native.__file__), "libclang.so")
clang.cindex.Config.set_library_file(libclang_path)

# Create a wrapper that includes necessary headers
wrapper_code = """
#include <cstdint>
using int32_t = int;
using uint32_t = unsigned int;
namespace madrona { using CountT = int32_t; }
#include "/home/duane/madrona_escape_room/src/consts.hpp"
"""

# Parse with the wrapper
index = clang.cindex.Index.create()
tu = index.parse(
    "wrapper.cpp", ["-std=c++17", "-xc++"], unsaved_files=[("wrapper.cpp", wrapper_code)]
)

if tu.diagnostics:
    print("Parse warnings/errors:")
    for diag in tu.diagnostics:
        print(f"  {diag}")

# Generate Python constants
output = []
output.append("# Generated Python constants from consts.hpp")
output.append("")


def extract_value(cursor):
    """Try to extract the literal value from a cursor."""
    tokens = list(cursor.get_tokens())

    # Look for the assignment operator, then get everything after it
    found_assign = False
    value_tokens = []

    for token in tokens:
        if token.spelling == "=":
            found_assign = True
            continue
        if found_assign and token.spelling != ";":
            value_tokens.append(token.spelling)

    if value_tokens:
        # Join tokens to handle negative numbers and expressions
        value = "".join(value_tokens)
        # Clean up 'f' suffix for floats
        if value.endswith("f") or value.endswith("F"):
            value = value[:-1]
        return value

    return None


def process_var_decl(cursor, namespace_path=""):
    """Process a variable declaration (for constexpr values)."""
    var_name = cursor.spelling
    var_type = cursor.type

    # Try to get the value from tokens
    value = extract_value(cursor)

    if value:
        # Clean up the name - convert to Python constant style
        full_name = "_".join(namespace_path + [var_name])
        full_name = full_name.upper()

        # Handle float literals
        if "f" in value.lower():
            value = value.rstrip("fF")

        output.append(f"{full_name} = {value}")


def traverse_namespace(cursor, namespace_path=None):
    """Traverse namespace looking for constants."""
    if namespace_path is None:
        namespace_path = []

    for child in cursor.get_children():
        # Skip if not in our file
        if child.location.file and "consts.hpp" not in child.location.file.name:
            continue

        if child.kind == clang.cindex.CursorKind.NAMESPACE:
            # Enter nested namespace
            new_path = namespace_path + [child.spelling]
            print(f"Found namespace: {'.'.join(new_path)}")
            traverse_namespace(child, new_path)

        elif child.kind == clang.cindex.CursorKind.VAR_DECL:
            # Found a variable (likely constexpr)
            process_var_decl(child, namespace_path)

        else:
            # Continue traversing
            traverse_namespace(child, namespace_path)


# Find the madEscape namespace and process it
for cursor in tu.cursor.get_children():
    if cursor.kind == clang.cindex.CursorKind.NAMESPACE and cursor.spelling == "madEscape":
        print("Processing madEscape namespace...")
        traverse_namespace(cursor)
        break

# Write the output
output_file = "scratch/madrona_constants.py"
with open(output_file, "w") as f:
    f.write("\n".join(output))

print(f"\nGenerated {output_file}")
print("=" * 50)

# Show what we generated
with open(output_file, "r") as f:
    content = f.read()
    print(content)
