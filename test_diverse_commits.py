#!/usr/bin/env python3
"""Test performance system with diverse commits from different branches."""

import subprocess
import sys
from pathlib import Path

# Test commits from different branches
test_commits = [
    ("21f6749", "Remove warnings as errors (main)"),
    ("c439a05", "Update madrona (main)"),
    ("1d98927", "Merge .claude directory changes from minimal branch (test-driven-levels)"),
    ("64f72c1", "Fix GPU compilation error in madrona BVH debug output (test-driven-levels)"),
]


def test_commit(commit_hash, description):
    """Test a specific commit."""
    print(f"\n{'='*60}")
    print(f"Testing: {commit_hash} - {description}")
    print(f"{'='*60}")

    # Checkout commit
    result = subprocess.run(
        f"git checkout {commit_hash}", shell=True, capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"❌ Failed to checkout {commit_hash}")
        print(result.stderr)
        return False

    # Build
    print("Building...")
    result = subprocess.run("make -C build -j8 -s", shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Build failed for {commit_hash}")
        print("Build errors:")
        print(result.stderr)
        return False

    # Quick performance test
    print("Running performance test...")
    result = subprocess.run(
        "uv run python scripts/sim_bench.py --num-worlds 512 --num-steps 500",
        shell=True,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"❌ Performance test failed for {commit_hash}")
        print("Test errors:")
        print(result.stderr)
        return False

    # Extract FPS
    fps = None
    for line in result.stdout.split("\n"):
        if "│ FPS              │" in line:
            try:
                fps_str = line.split("│")[2].strip().replace(",", "")
                fps = int(fps_str)
                break
            except ValueError:
                continue

    if fps:
        print(f"✅ {commit_hash}: {fps:,} FPS")
    else:
        print(f"⚠️  {commit_hash}: Test ran but couldn't parse FPS")

    return True


def main():
    """Test diverse commits."""
    print("Testing performance system with commits from different branches...")

    # Save current state
    current_branch = subprocess.run(
        "git branch --show-current", shell=True, capture_output=True, text=True
    ).stdout.strip()

    try:
        success_count = 0
        for commit_hash, description in test_commits:
            if test_commit(commit_hash, description):
                success_count += 1

        print(f"\n{'='*60}")
        print(f"SUMMARY: {success_count}/{len(test_commits)} commits tested successfully")
        print(f"{'='*60}")

    finally:
        # Restore original branch
        print(f"\nRestoring branch: {current_branch}")
        subprocess.run(f"git checkout {current_branch}", shell=True)


if __name__ == "__main__":
    main()
