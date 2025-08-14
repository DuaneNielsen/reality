#!/usr/bin/env python3
"""Test if the replay fix works by recording and replaying."""

import os
import subprocess
import time

# Clean up old files
for f in ["test_record.rec", "recording_trace.csv", "replay_trace.csv"]:
    if os.path.exists(f):
        os.remove(f)

print("Step 1: Recording with trajectory tracking...")
# Record with tracking
proc = subprocess.Popen(
    [
        "./build/viewer",
        "--load",
        "levels/quick_test.lvl",
        "-n",
        "1",
        "--record",
        "test_record.rec",
        "--track",
        "--track-file",
        "recording_trace.csv",
    ]
)

# Wait a bit for it to start
time.sleep(2)

# Kill it to stop recording (simulates user input)
proc.terminate()
proc.wait()

print("Step 2: Replaying with trajectory tracking...")
# Replay with tracking
proc = subprocess.Popen(
    [
        "./build/viewer",
        "-n",
        "1",
        "--replay",
        "test_record.rec",
        "--track",
        "--track-file",
        "replay_trace.csv",
    ]
)

# Let it run for a bit
time.sleep(3)

# Kill it
proc.terminate()
proc.wait()

print("\nStep 3: Comparing trajectories...")
# Read and compare first few lines
if os.path.exists("recording_trace.csv") and os.path.exists("replay_trace.csv"):
    with open("recording_trace.csv", "r") as f:
        rec_lines = f.readlines()[:10]
    with open("replay_trace.csv", "r") as f:
        rep_lines = f.readlines()[:10]

    print("Recording (first 10 lines):")
    for line in rec_lines:
        print(f"  {line.rstrip()}")

    print("\nReplay (first 10 lines):")
    for line in rep_lines:
        print(f"  {line.rstrip()}")

    # Check if they match
    if rec_lines == rep_lines:
        print("\n✓ Trajectories MATCH!")
    else:
        print("\n✗ Trajectories DIFFER!")
        for i, (r1, r2) in enumerate(zip(rec_lines, rep_lines)):
            if r1 != r2:
                print(f"  First difference at line {i}:")
                print(f"    Recording: {r1.rstrip()}")
                print(f"    Replay:    {r2.rstrip()}")
                break
else:
    print("ERROR: Could not find trace files")
