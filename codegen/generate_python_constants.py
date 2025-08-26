#!/usr/bin/env python3
"""
Generate Python constants from C++ headers using libclang AST parsing.
This script extracts constants from consts.hpp and types.hpp and generates
a Python module with nested classes matching the C++ namespace structure.
"""

import os
import sys
from typing import Any, Dict, List, Optional

import clang.cindex
import clang.native

# Set libclang path
libclang_path = os.path.join(os.path.dirname(clang.native.__file__), "libclang.so")
clang.cindex.Config.set_library_file(libclang_path)

# Generation configuration - specifies what to generate and how
GENERATION_CONFIG = {
    # Namespace classes to generate as Python classes
    "namespace_classes": [
        {
            "path": ["madEscape", "consts"],  # Path through namespace tree
            "class_name": "consts",  # Name of Python class to generate
        },
        {
            "path": ["madEscape", "consts", "limits"],  # Structural limits and bounds
            "class_name": "limits",  # Generate as limits class
        },
        {"path": ["types"], "class_name": "types"},
    ],
    # Convenience aliases to create at module level
    "aliases": [
        {
            "name": "action",  # Alias name
            "target": "consts.action",  # What it points to
            "condition_path": ["madEscape", "consts", "action"],  # Only create if this path exists
        }
    ],
}


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

    def __init__(self, verbose: bool = False):
        self.index = clang.cindex.Index.create()
        self.root = NamespaceNode("root")
        self.verbose = verbose

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
            if self.verbose:
                print(f"DEBUG: Skipping unnamed enum in {namespace_path}", file=sys.stderr)
            return

        if self.verbose:
            print(
                f"DEBUG: Processing enum '{enum_name}' type: {cursor.type.spelling}",
                file=sys.stderr,
            )

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

        if not values and self.verbose:
            print(f"DEBUG: Enum {enum_name} has no values!", file=sys.stderr)

        # Add enum to appropriate namespace (no conversion)
        node = self.root
        for ns in namespace_path:
            node = node.add_child(ns)
        node.add_enum(enum_name, values)
        print(
            f"Generated enum {'.'.join(namespace_path)}.{enum_name} with {len(values)} values",
            file=sys.stderr,
        )

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
            print(
                f"Generated const {'.'.join(namespace_path)}.{var_name} = {value}", file=sys.stderr
            )

    def process_struct_constants(self, cursor, namespace_path: List[str]):
        """Extract static constants from struct definitions."""
        struct_name = cursor.spelling
        if not struct_name:
            return

        if self.verbose:
            print(
                f"DEBUG: Processing struct {struct_name} with "
                f"{len(list(cursor.get_children()))} children",
                file=sys.stderr,
            )

        # Look for static constexpr members
        for child in cursor.get_children():
            if self.verbose and child.kind == clang.cindex.CursorKind.VAR_DECL:
                print(
                    f"DEBUG: Found VAR_DECL '{child.spelling}' in struct {struct_name}, "
                    f"const={child.type.is_const_qualified()}",
                    file=sys.stderr,
                )

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
                        print(
                            f"Generated struct const "
                            f"{'.'.join(namespace_path)}.{struct_name}.{var_name} = {value}",
                            file=sys.stderr,
                        )

    def traverse(self, cursor, namespace_path: List[str] = None):
        """Traverse the AST and extract constants."""
        if namespace_path is None:
            namespace_path = []

        for child in cursor.get_children():
            # Verbose output
            if self.verbose and child.kind == clang.cindex.CursorKind.NAMESPACE:
                print(
                    f"DEBUG: Found namespace '{child.spelling}' at path {namespace_path}",
                    file=sys.stderr,
                )

            # Determine if we should process this node
            should_process = False
            if child.location.file:
                filename = child.location.file.name
                # Process if in our target files OR in madrona namespace (could be under std)
                should_process = (
                    "consts.hpp" in filename
                    or "types.hpp" in filename
                    or (namespace_path and namespace_path[0] == "madrona")
                    or (
                        len(namespace_path) >= 2
                        and namespace_path[0] == "std"
                        and namespace_path[1] == "madrona"
                    )
                )
            else:
                # If no file location, inherit from parent context
                should_process = (namespace_path and namespace_path[0] == "madrona") or (
                    len(namespace_path) >= 2
                    and namespace_path[0] == "std"
                    and namespace_path[1] == "madrona"
                )

            if child.kind == clang.cindex.CursorKind.NAMESPACE:
                # Enter namespace
                new_path = namespace_path + [child.spelling]
                if self.verbose and child.spelling == "madrona":
                    print(
                        f"DEBUG: Entering madrona namespace from path {namespace_path}",
                        file=sys.stderr,
                    )
                    print(f"DEBUG: New path will be {new_path}", file=sys.stderr)
                self.traverse(child, new_path)

            elif should_process:
                if self.verbose:
                    print(
                        f"DEBUG: Processing {child.kind.name} '{child.spelling}' "
                        f"in namespace {namespace_path}",
                        file=sys.stderr,
                    )

                if child.kind == clang.cindex.CursorKind.VAR_DECL:
                    # Process variable declaration
                    self.process_variable(child, namespace_path)

                elif child.kind == clang.cindex.CursorKind.ENUM_DECL:
                    # Process enum
                    self.process_enum(child, namespace_path)

                elif child.kind == clang.cindex.CursorKind.STRUCT_DECL:
                    # Process struct for static constants
                    self.process_struct_constants(child, namespace_path)

                # Note: USING_DECLARATION doesn't work reliably with libclang
                # We now include the Madrona headers directly instead
            elif child.kind == clang.cindex.CursorKind.ENUM_DECL:
                # Log all enums we see even if not processing
                if self.verbose:
                    print(
                        f"DEBUG: Saw enum '{child.spelling}' in namespace "
                        f"{namespace_path} (not processing)",
                        file=sys.stderr,
                    )
            else:
                # Continue traversing
                self.traverse(child, namespace_path)

    def parse_headers(self, consts_path: str, types_path: str):
        """Parse the C++ headers and extract constants."""
        # Get the Madrona include directory from environment
        madrona_include = os.environ.get("MADRONA_INCLUDE_DIR", "")

        # Create wrapper that includes headers directly
        # Include Madrona headers directly to avoid parse errors from types.hpp
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        wrapper_code = f"""
#include <cstdint>
#include <cstddef>

// Include Madrona headers directly for better parsing
#include "{os.path.join(project_root, "external/madrona/include/madrona/exec_mode.hpp")}"
#include "{os.path.join(project_root, "external/madrona/include/madrona/py/utils.hpp")}"

// Include our project headers
#include "{os.path.abspath(consts_path)}"
#include "{os.path.abspath(types_path)}"
"""

        if self.verbose:
            print(f"DEBUG: Wrapper code:\n{wrapper_code}", file=sys.stderr)

        # Build include paths
        include_args = [
            "-std=c++20",
            "-xc++",
            f"-I{os.path.dirname(consts_path)}",
        ]

        if madrona_include:
            include_args.append(f"-I{madrona_include}")

        # Parse
        tu = self.index.parse(
            "wrapper.cpp", include_args, unsaved_files=[("wrapper.cpp", wrapper_code)]
        )

        # Check for parse errors (only show if verbose flag is set)
        if tu.diagnostics and self.verbose:
            for diag in tu.diagnostics:
                if diag.severity >= clang.cindex.Diagnostic.Error:
                    print(f"Parse error: {diag}", file=sys.stderr)

        # Traverse the AST
        self.traverse(tu.cursor)

        # Also extract CompiledLevel constants from types.hpp
        self.extract_compiled_level_constants()

    def extract_compiled_level_constants(self):
        """CompiledLevel constants extracted from types.hpp via process_struct_constants."""
        # This method is now deprecated - constants are extracted directly from the AST
        pass


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

        # Generate enums as classes
        for enum_name, values in sorted(node.enums.items()):
            # Skip empty enums
            if not values:
                continue

            self.add_line(f"class {enum_name}:")
            self.indent_level += 1
            self.add_line(f'"""Enum values for {enum_name}"""')
            self.add_line("__slots__ = ()")
            self.add_line()

            for name, value in values.items():
                self.add_line(f"{name} = {value}")

            self.indent_level -= 1
            self.add_line()

        # Generate child namespaces
        for child_name, child_node in sorted(node.children.items()):
            if child_name:
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

        # Check what we have and generate appropriately

        # Generate ALL enums found, recursively through namespace tree
        def generate_all_enums(node, path=""):
            """Recursively generate all enums as top-level classes."""
            # Generate enums at this level
            for enum_name, enum_values in node.enums.items():
                self.add_line(f"class {enum_name}:")
                self.indent_level += 1
                self.add_line(f'"""Enum from {path if path else "root"}"""')
                self.add_line()
                self.add_line("__slots__ = ()")
                self.add_line()
                for name, value in enum_values.items():
                    # Handle Python keywords
                    if name in (
                        "None",
                        "True",
                        "False",
                        "and",
                        "or",
                        "not",
                        "if",
                        "else",
                        "elif",
                        "while",
                        "for",
                        "break",
                        "continue",
                        "pass",
                        "def",
                        "class",
                        "return",
                        "yield",
                        "import",
                        "from",
                        "as",
                        "with",
                        "try",
                        "except",
                        "finally",
                        "raise",
                        "assert",
                        "del",
                        "lambda",
                        "global",
                        "nonlocal",
                        "is",
                        "in",
                        "await",
                        "async",
                    ):
                        # Append underscore to avoid keyword conflict
                        safe_name = name + "_"
                        self.add_line(
                            f"{safe_name} = {value}  # Renamed from '{name}' (Python keyword)"
                        )
                    else:
                        self.add_line(f"{name} = {value}")
                self.indent_level -= 1
                self.add_line()

            # Recurse into child namespaces
            for child_name, child_node in node.children.items():
                child_path = f"{path}.{child_name}" if path else child_name
                generate_all_enums(child_node, child_path)

        # Generate all enums from all namespaces
        generate_all_enums(self.root)

        # Generate namespace classes based on configuration
        for ns_config in GENERATION_CONFIG["namespace_classes"]:
            # Navigate to the specified node
            node = self.root
            for part in ns_config["path"]:
                if part in node.children:
                    node = node.children[part]
                else:
                    node = None
                    break

            # Generate if found
            if node:
                self.generate_node(node, ns_config["class_name"])

        # Add convenience aliases based on configuration
        aliases_added = False
        for alias_config in GENERATION_CONFIG["aliases"]:
            # Check if condition path exists
            node = self.root
            for part in alias_config["condition_path"]:
                if part in node.children:
                    node = node.children[part]
                else:
                    node = None
                    break

            # Add alias if condition met
            if node:
                if not aliases_added:
                    self.add_line("# Convenience aliases for common use")
                    aliases_added = True
                self.add_line(f"{alias_config['name']} = {alias_config['target']}")

        if aliases_added:
            self.add_line()

        # Build __all__ export list dynamically
        all_exports = []

        # Collect all enum names that were generated
        def collect_enum_names(node):
            names = []
            for enum_name in node.enums.keys():
                names.append(enum_name)
            for child in node.children.values():
                names.extend(collect_enum_names(child))
            return names

        all_exports.extend(collect_enum_names(self.root))

        # Add namespace classes from configuration
        for ns_config in GENERATION_CONFIG["namespace_classes"]:
            # Check if the namespace exists
            node = self.root
            for part in ns_config["path"]:
                if part in node.children:
                    node = node.children[part]
                else:
                    node = None
                    break
            if node:
                all_exports.append(ns_config["class_name"])

        # Add aliases from configuration
        for alias_config in GENERATION_CONFIG["aliases"]:
            # Check if condition path exists
            node = self.root
            for part in alias_config["condition_path"]:
                if part in node.children:
                    node = node.children[part]
                else:
                    node = None
                    break
            if node:
                all_exports.append(alias_config["name"])

        self.add_line("__all__ = [")
        for export in all_exports:
            self.add_line(f'    "{export}",')
        self.add_line("]")

        return "\n".join(self.lines)


def main():
    """Main entry point."""
    # Check for verbose flag
    verbose = "--verbose" in sys.argv
    if verbose:
        sys.argv.remove("--verbose")

    if len(sys.argv) != 4:
        print(
            "Usage: generate_python_constants.py [--verbose] <consts.hpp> <types.hpp> <output.py>",
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
    extractor = ConstantExtractor(verbose=verbose)
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
