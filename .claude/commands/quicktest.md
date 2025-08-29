!uv run pytest --tb=no -q -m "not slow" | grep -v -E "(PASSED|SKIPPED)"

Output the tests results in the below format

@.claude/include/test_formatting.md