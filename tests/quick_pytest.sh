uv run pytest --tb=no -q -m "not slow" | grep -v -E "(PASSED|SKIPPED)"
