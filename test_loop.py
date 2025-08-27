#!/usr/bin/env python3
"""Run test in a loop to trigger segfault"""
import subprocess
import sys

for i in range(20):
    print(f"Run {i+1}", flush=True)
    result = subprocess.run([
        "uv", "run", "pytest", 
        "tests/python/test_ascii_level_compiler.py", 
        "-q", "--tb=no"
    ])
    if result.returncode != 0:
        print(f"Failed with return code: {result.returncode}")
        sys.exit(result.returncode)