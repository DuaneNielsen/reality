#!/usr/bin/env python3
"""
Simple requirement traceability report generator for pytest tests.
Links tests to specification documents and sections.
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Tuple


def extract_spec_markers(test_file: Path) -> List[Dict]:
    """Extract @pytest.mark.spec markers from a test file."""
    tests = []

    with open(test_file, "r") as f:
        content = f.read()

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return tests

    current_class = None

    for node in ast.walk(tree):
        # Track current class context
        if isinstance(node, ast.ClassDef):
            current_class = node.name

        # Find function definitions
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
            test_info = {
                "file": test_file.name,
                "function": node.name,
                "class": current_class,
                "spec_doc": None,
                "spec_section": None,
                "docstring": ast.get_docstring(node),
                "line": node.lineno,
            }

            # Look for @pytest.mark.spec decorators
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Attribute):
                    if (
                        hasattr(decorator.value, "attr")
                        and decorator.value.attr == "mark"
                        and decorator.attr == "spec"
                    ):
                        # This is a @pytest.mark.spec without arguments
                        continue
                elif isinstance(decorator, ast.Call):
                    # Check if this is pytest.mark.spec(...)
                    if isinstance(decorator.func, ast.Attribute):
                        if (
                            hasattr(decorator.func.value, "attr")
                            and decorator.func.value.attr == "mark"
                            and decorator.func.attr == "spec"
                        ):
                            # Extract arguments
                            if len(decorator.args) >= 1:
                                if isinstance(decorator.args[0], ast.Constant):
                                    test_info["spec_doc"] = decorator.args[0].value
                            if len(decorator.args) >= 2:
                                if isinstance(decorator.args[1], ast.Constant):
                                    test_info["spec_section"] = decorator.args[1].value

            # Only include tests with spec markers
            if test_info["spec_doc"] or test_info["spec_section"]:
                tests.append(test_info)

    return tests


def generate_traceability_report(test_dir: Path = Path("tests/python")) -> str:
    """Generate a traceability report for all tests with spec markers."""

    all_tests = []

    # Find all test files
    for test_file in test_dir.glob("test_*.py"):
        tests = extract_spec_markers(test_file)
        all_tests.extend(tests)

    # Group by specification document and section
    by_spec = {}
    for test in all_tests:
        spec_key = (test.get("spec_doc", "unspecified"), test.get("spec_section", "unspecified"))

        if spec_key not in by_spec:
            by_spec[spec_key] = []
        by_spec[spec_key].append(test)

    # Generate report
    report = []
    report.append("# Requirement Traceability Report")
    report.append("")
    report.append(f"Total tests with spec markers: {len(all_tests)}")
    report.append("")

    # Sort by document then section
    for doc, section in sorted(by_spec.keys()):
        report.append(f"## {doc} - {section}")
        report.append("")

        for test in sorted(by_spec[(doc, section)], key=lambda t: t["function"]):
            test_name = test["function"]
            if test["class"]:
                test_name = f"{test['class']}.{test_name}"

            report.append(f"- **{test_name}** ({test['file']}:{test['line']})")

            # Extract first line of docstring if available
            if test["docstring"]:
                first_line = test["docstring"].split("\n")[0].strip()
                if first_line:
                    report.append(f"  - {first_line}")

            report.append("")

    # List specification sections that may lack test coverage
    report.append("## Specification Coverage Summary")
    report.append("")

    # Count tests per section
    section_counts = {}
    for doc, section in by_spec.keys():
        if doc == "docs/specs/sim.md":
            if section not in section_counts:
                section_counts[section] = 0
            section_counts[section] = len(by_spec[(doc, section)])

    # Expected sections from sim.md
    expected_sections = [
        "movementSystem",
        "agentCollisionSystem",
        "agentZeroVelSystem",
        "stepTrackerSystem",
        "rewardSystem",
        "resetSystem",
        "initProgressAfterReset",
        "collectObservationsSystem",
        "compassSystem",
        "lidarSystem",
    ]

    report.append("### docs/specs/sim.md Coverage:")
    report.append("")

    for section in expected_sections:
        count = section_counts.get(section, 0)
        status = "✅" if count > 0 else "❌"
        report.append(f"- {status} **{section}**: {count} test(s)")

    report.append("")

    return "\n".join(report)


def main():
    """Generate and print the traceability report."""
    report = generate_traceability_report()

    # Save report
    report_file = Path("tests/traceability_report.md")
    with open(report_file, "w") as f:
        f.write(report)

    print(f"Traceability report generated: {report_file}")
    print()
    print(report)


if __name__ == "__main__":
    main()
