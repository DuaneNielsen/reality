#!/usr/bin/env python3
"""
Capture a screenshot of the initial level state using the viewer.
This tool is for development verification - launches viewer, captures frame 0,
and returns the image path.
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def capture_screenshot(num_worlds=1, exec_mode="cpu", gpu_id=0, output_path="level_screenshot.png"):
    """
    Launch viewer with automatic screenshot capture of frame 0.

    Args:
        num_worlds: Number of worlds to simulate
        exec_mode: "cpu" or "cuda"
        gpu_id: GPU device ID (only used if exec_mode="cuda")
        output_path: Path to save the screenshot

    Returns:
        Path to the saved screenshot
    """
    # Build viewer command
    viewer_path = Path(__file__).parent.parent / "build" / "viewer"
    if not viewer_path.exists():
        print(f"Error: Viewer not found at {viewer_path}")
        print("Please build the project first: make -C build -j$(nproc)")
        return None

    # Set environment variable for automatic screenshot
    env = os.environ.copy()
    env["SCREENSHOT_PATH"] = output_path

    # Build command arguments
    cmd = [str(viewer_path), "-n", str(num_worlds), "--hide-menu"]
    if exec_mode.lower() == "cuda":
        cmd.extend(["--cuda", str(gpu_id)])

    print("Launching viewer to capture screenshot...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Screenshot will be saved to: {output_path}")

    try:
        # Launch viewer with timeout (it will capture frame 0 and we'll kill it)
        # The viewer will automatically take a screenshot on frame 0 when SCREENSHOT_PATH is set
        process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Give it a moment to initialize and capture
        time.sleep(2)

        # Terminate the viewer
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()

        # Check if screenshot was created
        if Path(output_path).exists():
            print(f"✓ Screenshot captured successfully: {output_path}")
            return output_path
        else:
            print(f"✗ Screenshot was not created at {output_path}")
            return None

    except Exception as e:
        print(f"Error launching viewer: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Capture level screenshot for development verification"
    )
    parser.add_argument("-n", "--num-worlds", type=int, default=1, help="Number of worlds")
    parser.add_argument("--cuda", type=int, metavar="GPU_ID", help="Use CUDA on specified GPU")
    parser.add_argument(
        "-o", "--output", default="level_screenshot.png", help="Output file path (.png or .bmp)"
    )

    args = parser.parse_args()

    exec_mode = "cuda" if args.cuda is not None else "cpu"
    gpu_id = args.cuda if args.cuda is not None else 0

    result = capture_screenshot(
        num_worlds=args.num_worlds, exec_mode=exec_mode, gpu_id=gpu_id, output_path=args.output
    )

    if result:
        print(f"\nYou can now view the screenshot at: {result}")
        print("Use an image viewer or the Read tool to examine the level layout.")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
