#!/usr/bin/env python3
"""
Generate Python constants from C++ headers using libclang AST parsing.
This script extracts constants from consts.hpp and types.hpp and generates
a Python module with nested classes matching the C++ namespace structure.
"""

import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import clang.cindex
import clang.native

# Set libclang path (mirror POC approach)
libclang_path = os.path.join(os.path.dirname(clang.native.__file__), "libclang.so")
clang.cindex.Config.set_library_file(libclang_path)


class NamespaceNode:
    """Represents a namespace or class node in the constant hierarchy."""

    def __init__(self, name: str):
        self.name = name
        self.constants: Dict[str, Any] = {}
        self.children: Dict[str, "NamespaceNode"] = {}
        self.enums: Dict[str, Dict[str, int]] = {}

    def add_constant(self, name: str, value: Any):
        """Add a constant to this namespace."""
        self.constants[name] = value

    def add_child(self, name: str) -> "NamespaceNode":
        """Add or get a child namespace."""
        if name not in self.children:
            self.children[name] = NamespaceNode(name)
        return self.children[name]

    def add_enum(self, enum_name: str, values: Dict[str, int]):
        """Add an enum to this namespace."""
        self.enums[enum_name] = values


class ConstantExtractor:
    """Extracts constants from C++ headers using libclang."""

    def __init__(self):
        self.index = clang.cindex.Index.create()
        self.root = NamespaceNode("root")

    def extract_value(self, cursor) -> Optional[Any]:
        """Extract the literal value from a variable declaration."""
        tokens = list(cursor.get_tokens())

        # Look for the assignment operator and get everything after it
        found_assign = False
        value_tokens = []

        for token in tokens:
            if token.spelling == "=":
                found_assign = True
                continue
            if found_assign and token.spelling not in (";", "{", "}"):
                value_tokens.append(token.spelling)

        if value_tokens:
            # Join tokens and clean up
            value = "".join(value_tokens)

            # Handle float literals
            if value.endswith("f") or value.endswith("F"):
                value = value[:-1]

            # Try to evaluate the value
            try:
                # Handle common expressions
                if "-" in value and not value.startswith("-"):
                    # This is likely an expression like "numBuckets - 1"
                    # Try to resolve it from context if possible
                    parts = value.split("-")
                    if len(parts) == 2:
                        # For now, return the raw expression
                        return value

                # Try to convert to appropriate type
                if "." in value:
                    return float(value)
                else:
                    return int(value)
            except ValueError:
                # Return as string if can't evaluate
                return value

        return None

    def process_enum(self, cursor, namespace_path: List[str]):
        """Process an enum declaration."""
        enum_name = cursor.spelling
        if not enum_name:
            return

        # Remove 'class' from enum class declarations
        enum_name = enum_name.replace("class ", "")

        values = {}
        current_value = 0

        for child in cursor.get_children():
            if child.kind == clang.cindex.CursorKind.ENUM_CONSTANT_DECL:
                name = child.spelling
                # Try to get explicit value
                if child.enum_value is not None:
                    current_value = child.enum_value
                values[name] = current_value
                current_value += 1

        # Add enum to appropriate namespace
        node = self.root
        for ns in namespace_path:
            node = node.add_child(ns)
        node.add_enum(enum_name, values)

    def process_variable(self, cursor, namespace_path: List[str]):
        """Process a variable declaration (constexpr or const)."""
        var_name = cursor.spelling
        if not var_name:
            return

        # Skip certain variables
        if var_name.startswith("_") or "operator" in var_name:
            return

        value = self.extract_value(cursor)
        if value is not None:
            # Add to appropriate namespace
            node = self.root
            for ns in namespace_path:
                node = node.add_child(ns)
            node.add_constant(var_name, value)

    def process_struct_constants(self, cursor, namespace_path: List[str]):
        """Extract static constants from struct definitions."""
        struct_name = cursor.spelling
        if not struct_name:
            return

        # Look for static constexpr members
        for child in cursor.get_children():
            if child.kind == clang.cindex.CursorKind.VAR_DECL:
                # Check if it's static constexpr
                if child.type.is_const_qualified():
                    var_name = child.spelling
                    value = self.extract_value(child)
                    if value is not None:
                        # Add to struct's namespace
                        node = self.root
                        for ns in namespace_path:
                            node = node.add_child(ns)
                        struct_node = node.add_child(struct_name)
                        struct_node.add_constant(var_name, value)

    def traverse(self, cursor, namespace_path: List[str] = None, in_target_file: bool = True):
        """Traverse the AST and extract constants."""
        if namespace_path is None:
            namespace_path = []

        for child in cursor.get_children():
            # Check if we're still in the target file
            if child.location.file:
                filename = child.location.file.name
                child_in_target = in_target_file and (
                    "consts.hpp" in filename or "types.hpp" in filename
                )
            else:
                child_in_target = in_target_file

            if child.kind == clang.cindex.CursorKind.NAMESPACE:
                # Enter namespace
                new_path = namespace_path + [child.spelling]
                self.traverse(child, new_path, child_in_target)

            elif child.kind == clang.cindex.CursorKind.VAR_DECL and child_in_target:
                # Process variable declaration
                self.process_variable(child, namespace_path)

            elif child.kind == clang.cindex.CursorKind.ENUM_DECL and child_in_target:
                # Process enum
                self.process_enum(child, namespace_path)

            elif child.kind == clang.cindex.CursorKind.STRUCT_DECL and child_in_target:
                # Process struct for static constants
                self.process_struct_constants(child, namespace_path)

            else:
                # Continue traversing
                self.traverse(child, namespace_path, child_in_target)

    def parse_headers(self, consts_path: str, types_path: str):
        """Parse the C++ headers and extract constants."""
        # Get the Madrona include directory from environment
        madrona_include = os.environ.get("MADRONA_INCLUDE_DIR", "")

        # Create wrapper that includes both headers
        wrapper_code = f"""
#include <cstdint>
#include <cstddef>

// Define Madrona types that might be used
namespace madrona {{ using CountT = int32_t; }}

// Include the headers
#include "{os.path.abspath(consts_path)}"
#include "{os.path.abspath(types_path)}"
"""

        # Build include paths
        include_args = [
            "-std=c++20",  # Changed to C++20 to support concepts, bit_cast, etc.
            "-xc++",
            f"-I{os.path.dirname(consts_path)}",
        ]

        if madrona_include:
            include_args.append(f"-I{madrona_include}")

        # Parse with C++17 standard
        tu = self.index.parse(
            "wrapper.cpp", include_args, unsaved_files=[("wrapper.cpp", wrapper_code)]
        )

        # Check for parse errors
        if tu.diagnostics:
            for diag in tu.diagnostics:
                if diag.severity >= clang.cindex.Diagnostic.Error:
                    print(f"Parse error: {diag}", file=sys.stderr)

        # Traverse the AST
        self.traverse(tu.cursor)

        # Also extract CompiledLevel constants from types.hpp
        self.extract_compiled_level_constants(types_path)

    def extract_compiled_level_constants(self, types_path: str):
        """Extract MAX_TILES and other constants from CompiledLevel struct."""
        # For now, hardcode these as they're template parameters
        # In the future, could parse more carefully
        types_node = self.root.add_child("types")
        compiled_level = types_node.add_child("CompiledLevel")
        compiled_level.add_constant("MAX_TILES", 1024)
        compiled_level.add_constant("MAX_SPAWNS", 8)
        compiled_level.add_constant("MAX_LEVEL_NAME_LENGTH", 64)

        # Add observation sizes
        types_node.add_constant("SELF_OBSERVATION_SIZE", 5)
        types_node.add_constant("STEPS_REMAINING_SIZE", 1)
        types_node.add_constant("AGENT_ID_SIZE", 1)
        types_node.add_constant("TOTAL_OBSERVATION_SIZE", 7)


class PythonGenerator:
    """Generates Python code from the extracted constant tree."""

    def __init__(self, root: NamespaceNode):
        self.root = root
        self.lines: List[str] = []
        self.indent_level = 0

    def indent(self):
        """Get current indentation."""
        return "    " * self.indent_level

    def add_line(self, line: str = ""):
        """Add a line with proper indentation."""
        if line:
            self.lines.append(f"{self.indent()}{line}")
        else:
            self.lines.append("")

    def generate_node(self, node: NamespaceNode, class_name: str = None):
        """Generate Python code for a namespace node."""
        if class_name:
            self.add_line(f"class {class_name}:")
            self.indent_level += 1

            # Add docstring
            self.add_line(f'"""Constants from {class_name} namespace"""')
            self.add_line()

            # Add __slots__ to prevent attribute assignment
            self.add_line("__slots__ = ()")
            self.add_line()

        # Generate constants
        if node.constants:
            for name, value in sorted(node.constants.items()):
                if isinstance(value, str):
                    # Check if it's an expression we couldn't evaluate
                    if "-" in value and not value.startswith("-"):
                        # Skip complex expressions for now
                        self.add_line(f"# {name} = {value}  # TODO: Resolve expression")
                    else:
                        self.add_line(f'{name} = "{value}"')
                elif isinstance(value, float):
                    self.add_line(f"{name} = {value}")
                else:
                    self.add_line(f"{name} = {value}")

            if node.children or node.enums:
                self.add_line()

        # Generate enums
        for enum_name, values in sorted(node.enums.items()):
            self.add_line(f"class {enum_name}:")
            self.indent_level += 1
            self.add_line(f'"""Enum values for {enum_name}"""')
            self.add_line("__slots__ = ()")
            self.add_line()

            for name, value in values.items():
                # Convert enum names to UPPER_CASE
                python_name = name.upper() if name != "None" else "NONE"
                self.add_line(f"{python_name} = {value}")

            self.indent_level -= 1
            self.add_line()

        # Generate child namespaces
        for child_name, child_node in sorted(node.children.items()):
            if child_name and child_name != "madEscape":  # Skip root madEscape namespace
                self.generate_node(child_node, child_name)

        if class_name:
            self.indent_level -= 1
            self.add_line()

    def generate(self) -> str:
        """Generate the complete Python module."""
        # Add header
        self.lines = [
            '"""',
            "Auto-generated Python constants from C++ headers.",
            "Generated by libclang AST parsing during build process.",
            "DO NOT EDIT - this file is automatically regenerated.",
            '"""',
            "",
            "# ruff: noqa",
            "",
            "",
        ]

        # Process madEscape namespace
        if "madEscape" in self.root.children:
            mad_escape = self.root.children["madEscape"]

            # Generate consts namespace
            if "consts" in mad_escape.children:
                self.generate_node(mad_escape.children["consts"], "consts")

        # Generate types namespace
        if "types" in self.root.children:
            self.generate_node(self.root.children["types"], "types")
        elif (
            "madEscape" in self.root.children
            and "types" in self.root.children["madEscape"].children
        ):
            # Sometimes types might be under madEscape
            self.generate_node(self.root.children["madEscape"].children["types"], "types")

        # Add convenience aliases at module level
        self.add_line("# Convenience aliases for common use")
        self.add_line("action = consts.action")
        self.add_line("physics = consts.physics")
        self.add_line("rendering = consts.rendering")
        self.add_line("math = consts.math")
        self.add_line()

        # Add __all__ export
        self.add_line("__all__ = [")
        self.add_line('    "consts",')
        self.add_line('    "types",')
        self.add_line('    "action",')
        self.add_line('    "physics",')
        self.add_line('    "rendering",')
        self.add_line('    "math",')
        self.add_line("]")

        return "\n".join(self.lines)


def main():
    """Main entry point."""
    if len(sys.argv) != 4:
        print(
            "Usage: generate_python_constants.py <consts.hpp> <types.hpp> <output.py>",
            file=sys.stderr,
        )
        sys.exit(1)

    consts_path = sys.argv[1]
    types_path = sys.argv[2]
    output_path = sys.argv[3]

    # Check input files exist
    if not os.path.exists(consts_path):
        print(f"Error: {consts_path} not found", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(types_path):
        print(f"Error: {types_path} not found", file=sys.stderr)
        sys.exit(1)

    print(f"Parsing {consts_path} and {types_path}...", file=sys.stderr)

    # Extract constants
    extractor = ConstantExtractor()
    extractor.parse_headers(consts_path, types_path)

    # Generate Python code
    generator = PythonGenerator(extractor.root)
    python_code = generator.generate()

    # Write output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(python_code)

    print(f"Generated {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
