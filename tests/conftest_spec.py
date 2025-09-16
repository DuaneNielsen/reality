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


def get_spec_content(doc_path: str, section_path: str) -> Optional[str]:
    """Get specification content for a given test marker."""

    # Handle different path formats
    if "/" in section_path:
        # New format: "Implementation/Core Systems/rewardSystem"
        path_parts = section_path.split("/")
    else:
        # Old format: just "rewardSystem" - try to find it
        path_parts = [section_path]

    # Read the spec file
    spec_file = Path(doc_path)
    if not spec_file.exists():
        spec_file = Path("tests") / ".." / doc_path  # Try relative to tests dir

    if not spec_file.exists():
        return None

    content = spec_file.read_text()
    parser = MarkdownHierarchyParser(content)

    # Try different path combinations to find the section
    # First try the exact path
    section = None

    # For new full path format
    if len(path_parts) > 1:
        # Try: Simulation Core Specification -> Implementation -> Core Systems -> systemName
        section = parser.find_section(
            "Simulation Core Specification",
            "Implementation",
            "Core Systems (Task Graph Order)",
            path_parts[-1],
        )

    # For old format or if full path didn't work
    if not section and len(path_parts) == 1:
        # Try to find the system directly under Core Systems
        section = parser.find_section(
            "Simulation Core Specification",
            "Implementation",
            "Core Systems (Task Graph Order)",
            path_parts[0],
        )

    if not section:
        return None

    # Format the content nicely
    specs = parser.get_specifications(section)
    if not specs:
        return f"Section '{section['title']}' found but no specifications listed."

    result = [f"\n📋 Specification: {section['title']}\n"]
    result.append("─" * 60)
    for spec in specs[:10]:  # Limit to first 10 specs to avoid clutter
        result.append(spec)
    if len(specs) > 10:
        result.append(f"  ... and {len(specs) - 10} more specifications")
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
