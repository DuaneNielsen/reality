#!/usr/bin/env python3
"""
Convert level JSON files from string format to array format for better editor display.

This script converts "ascii" fields from:
  "ascii": "##\\n#S\\n##"
to:
  "ascii": ["##", "#S", "##"]
"""

import json
import sys
from pathlib import Path


def convert_ascii_format(data):
    """Convert ASCII field from string to array format if needed."""
    changed = False

    # Convert ASCII format
    if "ascii" in data and isinstance(data["ascii"], str):
        # Split by newlines and convert to array
        lines = data["ascii"].split("\n")
        data["ascii"] = lines
        changed = True

    # Add default tileset if missing
    if "tileset" not in data:
        data["tileset"] = {
            "#": {"asset": "wall", "done_on_collision": False},
            "C": {"asset": "cube", "done_on_collision": True},
            "O": {"asset": "cylinder", "done_on_collision": True},
            "S": {"asset": "spawn"},
            ".": {"asset": "empty"},
            " ": {"asset": "empty"},
        }
        changed = True

    return changed


def convert_file(input_path, output_path=None):
    """Convert a JSON level file to use array format."""
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path
    else:
        output_path = Path(output_path)

    # Read the JSON file
    with open(input_path, "r") as f:
        data = json.load(f)

    # Convert format
    changed = convert_ascii_format(data)

    if changed:
        # Write back with pretty formatting
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"✓ Converted {input_path}")
        return True
    else:
        print(f"- No conversion needed for {input_path}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python convert_level_format.py <file1.json> [file2.json] ...")
        print("   or: python convert_level_format.py levels/*.json")
        sys.exit(1)

    converted_count = 0
    for file_path in sys.argv[1:]:
        try:
            if convert_file(file_path):
                converted_count += 1
        except Exception as e:
            print(f"✗ Error converting {file_path}: {e}")

    print(f"\nConverted {converted_count} files")


if __name__ == "__main__":
    main()
