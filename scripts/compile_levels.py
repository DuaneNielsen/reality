#!/usr/bin/env python3
"""
Level Compilation Script for Madrona Escape Room

Compiles all ASCII level files in the levels directory to binary .lvl format.
Automatically detects .txt and .json level files and compiles them.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def compile_level_file(input_path, output_path, scale=2.5, verbose=False):
    """Compile a single level file to binary format"""
    try:
        cmd = [
            "uv",
            "run",
            "python",
            "-m",
            "madrona_escape_room.level_compiler",
            str(input_path),
            str(output_path),
        ]

        if not str(input_path).endswith(".json"):
            cmd.extend(["--scale", str(scale)])

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ Compiled {input_path.name} -> {output_path.name}")

        if verbose:
            print(f"  Command: {' '.join(cmd)}")
            if result.stdout:
                print(f"  Output: {result.stdout.strip()}")

        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to compile {input_path.name}: {e.stderr}")
        if verbose:
            print(f"  Command: {' '.join(cmd)}")
            print(f"  Return code: {e.returncode}")
        return False


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Compile ASCII/JSON level files to binary .lvl format for Madrona Escape Room",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compile all levels in default directory
  python3 compile_levels.py

  # Compile levels with custom scale for .txt files
  python3 compile_levels.py --scale 3.0

  # Force recompilation of all levels
  python3 compile_levels.py --force

  # Compile levels from a different directory
  python3 compile_levels.py --input-dir /path/to/levels

  # Compile specific file patterns only
  python3 compile_levels.py --pattern "maze_*.txt"

  # Verbose output with compilation details
  python3 compile_levels.py --verbose

  # Dry run - show what would be compiled
  python3 compile_levels.py --dry-run
        """,
    )

    parser.add_argument(
        "--input-dir",
        "-i",
        type=Path,
        help="Input directory containing level files (default: levels/)",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        help="Output directory for compiled .lvl files (default: same as input)",
    )

    parser.add_argument(
        "--scale", "-s", type=float, default=2.5, help="Default scale for .txt files (default: 2.5)"
    )

    parser.add_argument(
        "--pattern",
        "-p",
        type=str,
        help="File pattern to match (e.g., 'maze_*.txt', default: all .txt and .json files)",
    )

    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force recompilation even if output files are newer",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show detailed compilation output"
    )

    parser.add_argument(
        "--dry-run",
        "-n",
        action="store_true",
        help="Show what files would be compiled without actually compiling",
    )

    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all level files and their compilation status",
    )

    return parser.parse_args()


def main():
    """Main compilation script"""
    args = parse_arguments()

    # Determine input directory
    if args.input_dir:
        levels_dir = args.input_dir
    else:
        project_root = Path(__file__).parent.parent
        levels_dir = project_root / "levels"

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = levels_dir

    if not levels_dir.exists():
        print(f"Error: Input directory not found at {levels_dir}")
        sys.exit(1)

    print(f"Input directory: {levels_dir}")
    if output_dir != levels_dir:
        print(f"Output directory: {output_dir}")

    if args.dry_run:
        print("DRY RUN MODE - No files will be compiled")

    print("=" * 60)

    compiled_count = 0
    failed_count = 0
    skipped_count = 0

    # Determine file patterns to process
    if args.pattern:
        patterns = [args.pattern]
    else:
        patterns = ["*.txt", "*.json"]

    # Find all matching files
    all_files = []
    for pattern in patterns:
        all_files.extend(levels_dir.glob(pattern))

    if not all_files:
        print(f"No files found matching patterns: {', '.join(patterns)}")
        return

    # Process each file
    for input_file in sorted(all_files):
        output_file = output_dir / input_file.with_suffix(".lvl").name

        # Check if we should skip this file
        should_compile = True
        status_reason = ""

        if not args.force and output_file.exists():
            if output_file.stat().st_mtime > input_file.stat().st_mtime:
                should_compile = False
                status_reason = "up to date"

        if args.list:
            status = "Would compile" if should_compile else f"Would skip ({status_reason})"
            print(f"📄 {input_file.name} -> {output_file.name} ({status})")
            continue

        if not should_compile:
            print(f"⚡ Skipping {input_file.name} ({status_reason})")
            skipped_count += 1
            continue

        if args.dry_run:
            print(f"📝 Would compile {input_file.name} -> {output_file.name}")
            compiled_count += 1
            continue

        # Actually compile the file
        if compile_level_file(input_file, output_file, args.scale, args.verbose):
            compiled_count += 1
        else:
            failed_count += 1

    # Print summary
    print("=" * 60)

    if args.list:
        print(f"Found {len(all_files)} level files in {levels_dir}")
        return

    if args.dry_run:
        print("Dry run complete:")
        print(f"  📝 {compiled_count} files would be compiled")
        print(f"  ⚡ {skipped_count} files would be skipped")
    else:
        print("Compilation complete:")
        print(f"  ✓ {compiled_count} files compiled successfully")
        print(f"  ⚡ {skipped_count} files skipped (up to date)")

        if failed_count > 0:
            print(f"  ✗ {failed_count} files failed to compile")
            sys.exit(1)
        elif compiled_count > 0:
            print("  🎉 All compilations successful!")
        else:
            print("  ✨ All files already up to date!")


if __name__ == "__main__":
    main()
