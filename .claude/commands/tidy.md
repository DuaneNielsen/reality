# Run clangd-tidy

This command runs clangd-tidy on the C++ source files to check for code quality issues and fix them.

## Instructions for Claude

When the user runs `/tidy`, follow these steps:

1. **First, run clangd-tidy to identify issues:**
   ```bash
   uv run clangd-tidy --fail-on-severity=warning --compact -p build src/*.cpp src/*.hpp
   ```

2. **Review the output** and categorize the warnings/errors:
   - Auto-fixable issues (can be fixed with `--fix` flag)
   - Manual fixes needed (require code changes)
   - False positives or intentional code patterns

3. **Apply automatic fixes for safe issues:**
   ```bash
   uv run clangd-tidy --fix --fail-on-severity=error --compact -p build src/*.cpp src/*.hpp
   ```

4. **For remaining warnings that need manual fixes:**
   - Read the relevant code sections
   - Understand the context and intent
   - Apply appropriate fixes using the Edit tool
   - Common issues to fix:
     - Magic numbers → Named constants
     - Missing const correctness
     - Unnecessary includes
     - Variable naming conventions
     - Potential null pointer dereferences

5. **Re-run clangd-tidy to verify all issues are resolved:**
   ```bash
   uv run clangd-tidy --fail-on-severity=warning --compact -p build src/*.cpp src/*.hpp
   ```

6. **Build the project** to ensure changes compile:
   ```bash
   make -C build -j8
   ```

7. **Report the results** to the user with a summary of:
   - Issues automatically fixed
   - Issues manually fixed
   - Any remaining issues that couldn't be fixed (with explanations)

## Check for issues (without fixing)

```bash
uv run clangd-tidy --fail-on-severity=error --compact -p build src/*.cpp src/*.hpp
```

## Fix issues automatically

```bash
uv run clangd-tidy --fix --fail-on-severity=error --compact -p build src/*.cpp src/*.hpp
```

## Check specific files

```bash
# Check a single file
uv run clangd-tidy --fail-on-severity=error --compact -p build src/sim.cpp

# Check multiple specific files
uv run clangd-tidy --fail-on-severity=error --compact -p build src/sim.cpp src/mgr.cpp
```

## Run with different severity levels

```bash
# Only fail on errors (default)
uv run clangd-tidy --fail-on-severity=error --compact -p build src/*.cpp

# Fail on warnings and errors
uv run clangd-tidy --fail-on-severity=warning --compact -p build src/*.cpp

# Show all issues but don't fail
uv run clangd-tidy --compact -p build src/*.cpp
```

## Exclude certain files

The pre-commit hook excluded these files, which you may also want to exclude:
- `external/*` - External dependencies
- `tests/*` - Test files
- `src/dlpack_extension.cpp` - DLPack extension
- `src/test_c_wrapper_*.cpp` - C wrapper test files

To check all source files except these:

```bash
# Check all C++ files except excluded ones
find src -name "*.cpp" -o -name "*.hpp" | \
  grep -v "dlpack_extension.cpp" | \
  grep -v "test_c_wrapper_" | \
  xargs uv run clangd-tidy --fail-on-severity=error --compact -p build
```

## Common options

- `--fix` - Automatically fix issues where possible
- `--fail-on-severity=<level>` - Set failure threshold (error, warning, info)
- `--compact` - Use compact output format
- `-p build` - Use compile commands from the build directory
- `--quiet` - Suppress output except for errors

## Fix all issues in the main source directory

```bash
# This is the most common command you'll want to run
uv run clangd-tidy --fix --fail-on-severity=error --compact -p build src/*.cpp src/*.hpp
```