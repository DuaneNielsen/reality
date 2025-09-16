"""
Pytest plugin for specification traceability.
Shows relevant specification content when tests fail.
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

import pytest


class MarkdownHierarchyParser:
    """Simple markdown hierarchy parser focused on headings and content."""

    def __init__(self, content: str):
        self.lines = content.split("\n")
        self.tree = self._build_tree()

    def _build_tree(self) -> Dict:
        """Build a hierarchical tree from markdown headings."""
        root = {"title": "root", "level": 0, "children": [], "content": [], "line_start": 0}
        stack = [root]

        i = 0
        while i < len(self.lines):
            line = self.lines[i]

            # Check if it's a heading
            heading_match = re.match(r"^(#{1,6})\s+(.+)", line)
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()

                # Pop stack until we find the right parent
                while len(stack) > level:
                    stack.pop()

                # Create new node
                node = {
                    "title": title,
                    "level": level,
                    "children": [],
                    "content": [],
                    "line_start": i,
                    "line_end": None,
                }

                # Add to parent's children
                stack[-1]["children"].append(node)
                stack.append(node)
            elif line.strip() and len(stack) > 1:
                # Add non-empty lines to current section's content
                stack[-1]["content"].append(line)

            i += 1

        # Set line_end for all nodes
        self._set_line_ends(root)
        return root

    def _set_line_ends(self, node: Dict):
        """Recursively set line_end for all nodes."""
        for i, child in enumerate(node["children"]):
            if i + 1 < len(node["children"]):
                child["line_end"] = node["children"][i + 1]["line_start"] - 1
            else:
                child["line_end"] = len(self.lines) - 1
            self._set_line_ends(child)

    def find_section(self, *path: str) -> Optional[Dict]:
        """Find a section by path, e.g., find_section('Core Systems', 'rewardSystem')"""
        current = self.tree

        for part in path:
            found = False
            for child in current["children"]:
                # Clean both for comparison (remove numbering, asterisks, etc.)
                clean_title = re.sub(r"^\d+\.\s*|\*\*(.+?)\*\*", r"\1", child["title"])
                if clean_title == part or part in clean_title:
                    current = child
                    found = True
                    break
            if not found:
                return None

        return current

    def find_node_by_name(self, target_name: str) -> Optional[Dict]:
        """Find any node in the tree that matches the target name."""

        def search_recursive(node: Dict) -> Optional[Dict]:
            # Check if current node title matches (case-insensitive)
            if node["title"].lower() == target_name.lower():
                return node

            # Search in children
            for child in node["children"]:
                result = search_recursive(child)
                if result:
                    return result

            return None

        return search_recursive(self.tree)

    def extract_full_section(self, node: Dict) -> str:
        """Extract complete section content including all nested subsections."""
        if not node:
            return ""

        result = []

        # Add the section title
        if node["title"] != "root":
            level_marker = "#" * node["level"]
            result.append(f"{level_marker} {node['title']}")
            result.append("")

        # Add the section content
        for line in node["content"]:
            result.append(line)

        # Recursively add all child sections
        for child in node["children"]:
            child_content = self.extract_full_section(child)
            if child_content:
                result.append("")
                result.append(child_content)

        return "\n".join(result)

    def get_specifications(self, node: Dict) -> List[str]:
        """Extract specification bullet points from a section."""
        specs = []
        in_specs = False

        for line in node["content"]:
            if "Specifications" in line:
                in_specs = True
                continue

            if in_specs:
                # Look for bullet points
                if line.strip().startswith("-"):
                    specs.append(line.strip())
                elif line.strip() and not line.strip().startswith(" "):
                    # Stop at next non-indented content
                    if not line.strip().startswith("-"):
                        break

        return specs


def get_spec_content(doc_path: str, section_name: str) -> Optional[str]:
    """Get specification content for a given test marker using intelligent search."""

    # Read the spec file
    spec_file = Path(doc_path)
    if not spec_file.exists():
        spec_file = Path("tests") / ".." / doc_path  # Try relative to tests dir

    if not spec_file.exists():
        return None

    content = spec_file.read_text()
    parser = MarkdownHierarchyParser(content)

    # Use intelligent search to find the section by name
    section = parser.find_node_by_name(section_name)

    if not section:
        return f"Section '{section_name}' not found in {doc_path}"

    # Extract the complete section content
    full_content = parser.extract_full_section(section)

    if not full_content.strip():
        return f"Section '{section_name}' found but no content available."

    # Format the content nicely for display
    result = [f"\n📋 Specification: {section['title']}\n"]
    result.append("─" * 60)
    result.append(full_content)
    result.append("─" * 60)

    return "\n".join(result)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to add spec content to test failure reports."""
    outcome = yield
    report = outcome.get_result()

    # Only process failures
    if report.when == "call" and report.failed:
        # Check if verbose mode is enabled (-v or --verbose)
        config = item.config
        verbose_level = config.getoption("verbose", 0)

        # Only show specs if verbose is explicitly enabled
        if verbose_level <= 0:
            return  # Skip spec display in non-verbose mode

        # Look for spec marker
        spec_marker = None
        for marker in item.iter_markers(name="spec"):
            spec_marker = marker
            break

        if spec_marker and len(spec_marker.args) >= 2:
            doc_path = spec_marker.args[0]
            section = spec_marker.args[1]

            spec_content = get_spec_content(doc_path, section)
            if spec_content:
                # Add spec content as a section in the report
                report.sections.append(("📋 Specification", spec_content))


def pytest_configure(config):
    """Register the spec marker."""
    config.addinivalue_line(
        "markers", "spec(doc_path, section): Link test to specification document"
    )
