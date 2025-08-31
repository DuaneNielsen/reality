#!/bin/bash
# Run both C++ and Python tests quickly

# Run C++ tests (brief mode - only show failures)
./build/mad_escape_tests --gtest_brief=1

# Run Python tests (quiet mode - only show failures)
uv run pytest tests/python --tb=no -q -m "not slow" | grep -v -E "(PASSED|SKIPPED)"