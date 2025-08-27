#!/usr/bin/env python3
"""
Script to trigger the intermittent segfault for debugging
"""
import sys
import subprocess
import os

# Enable core dumps
os.system("ulimit -c unlimited")

print("Running tests in a loop to trigger segfault...")
print("Core dumps will be generated when segfault occurs")

for i in range(50):
    print(f"\n=== Run {i+1} ===")
    result = subprocess.run(
        ["python", "-m", "pytest", "tests/python/test_ascii_level_compiler.py", "-q", "--tb=no"],
        capture_output=False
    )
    
    if result.returncode != 0:
        print(f"\nTest failed with return code {result.returncode}")
        if result.returncode == -11:  # SIGSEGV
            print("Segmentation fault detected!")
            break

print("\nCheck for core dump files in current directory or /var/lib/apport/coredump/")