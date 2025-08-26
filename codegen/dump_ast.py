#!/usr/bin/env python3
"""
Dump the AST parse tree to a file for inspection.
"""

import os
import sys

import clang.cindex
import clang.native

# Set libclang path
libclang_path = os.path.join(os.path.dirname(clang.native.__file__), "libclang.so")
clang.cindex.Config.set_library_file(libclang_path)


def dump_cursor(cursor, indent=0, file=None, max_depth=10):
    """Recursively dump cursor information."""
    if indent > max_depth:
        return

    # Get location info
    if cursor.location.file:
        location = f"{cursor.location.file.name}:{cursor.location.line}:{cursor.location.column}"
    else:
        location = "no location"

    # Basic info
    line = "  " * indent
    line += f"{cursor.kind.name}"

    if cursor.spelling:
        line += f" '{cursor.spelling}'"

    # Add type info for certain kinds
    if cursor.type and cursor.type.spelling:
        line += f" [type: {cursor.type.spelling}]"

    # Add location
    line += f" @ {location}"

    # Add special info for certain kinds
    if cursor.kind == clang.cindex.CursorKind.ENUM_CONSTANT_DECL:
        if cursor.enum_value is not None:
            line += f" = {cursor.enum_value}"

    print(line, file=file)

    # Recurse
    for child in cursor.get_children():
        dump_cursor(child, indent + 1, file, max_depth)


def main():
    if len(sys.argv) != 4:
        print("Usage: dump_ast.py <consts.hpp> <types.hpp> <output.txt>")
        sys.exit(1)

    consts_path = sys.argv[1]
    types_path = sys.argv[2]
    output_path = sys.argv[3]

    # Create index
    index = clang.cindex.Index.create()

    # Get includes
    madrona_include = os.environ.get("MADRONA_INCLUDE_DIR", "")

    # Create wrapper
    wrapper_code = f"""
#include <cstdint>
#include <cstddef>

#include "{os.path.abspath(consts_path)}"
#include "{os.path.abspath(types_path)}"
"""

    # Build include args
    include_args = [
        "-std=c++20",
        "-xc++",
        f"-I{os.path.dirname(consts_path)}",
    ]

    if madrona_include:
        include_args.append(f"-I{madrona_include}")

    # Parse
    tu = index.parse("wrapper.cpp", include_args, unsaved_files=[("wrapper.cpp", wrapper_code)])

    # Dump to file
    with open(output_path, "w") as f:
        print(f"AST dump for {consts_path} and {types_path}\n", file=f)
        print("=" * 80, file=f)

        # Dump everything - don't filter
        dump_cursor(tu.cursor, 0, f, max_depth=4)

    print(f"AST dumped to {output_path}")


if __name__ == "__main__":
    main()
