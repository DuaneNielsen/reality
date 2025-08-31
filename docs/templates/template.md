# Command Template Structure

This template provides the structure for creating new Claude commands based on different algorithmic patterns.

## Basic Template Structure

```markdown
---
argument-hint: [ARGUMENT_DESCRIPTION]
description: [COMMAND_DESCRIPTION]
---
@.claude/include/substitutions.md

# algorithm

[ALGORITHM_STRUCTURE]

[STEP_DEFINITIONS]

[CONDITION_DEFINITIONS]

## output format

[OUTPUT_FORMAT_SPECIFICATION]
```

## Pattern Types

### Sequential Pattern
Use for simple step-by-step processes without loops.

```markdown
# algorithm

1. [STEP_1]
2. [STEP_2]
3. [STEP_3]

## [STEP_1]
[STEP_1_DESCRIPTION]

## [STEP_2]
[STEP_2_DESCRIPTION]

## [STEP_3]  
[STEP_3_DESCRIPTION]
```

### For Loop Pattern
Use when processing a collection or list of items.

```markdown
# algorithm

[COLLECTION_NAME] = $ARGUMENT
[OUTPUT_COLLECTION] = [...]

for each [ITEM] in [COLLECTION_NAME]:
  1. [STEP_1]
  2. [STEP_2]
  3. [STEP_3]
  4. add [RESULT] to [OUTPUT_COLLECTION]

## [STEP_1]
[STEP_1_DESCRIPTION]

## [STEP_2]
[STEP_2_DESCRIPTION]

## [STEP_3]
[STEP_3_DESCRIPTION]
```

### While Loop Pattern
Use for repetitive processes with a continuation condition.

```markdown
# algorithm

[INITIALIZATION]

while ([CONDITION_NAME]) todo:
   1. [STEP_1]
   2. [STEP_2]
   3. [STEP_3]

[FINAL_ACTION]

## [CONDITION_NAME]
[CONDITION_DESCRIPTION]

## [STEP_1]
[STEP_1_DESCRIPTION]

## [STEP_2]
[STEP_2_DESCRIPTION]

## [STEP_3]
[STEP_3_DESCRIPTION]
```


## Output Format Guidelines

### Simple Text Output
```markdown
## output format

[RESULT_DESCRIPTION]
```

### Structured Output
```markdown
## output format

for each [ITEM] produce output in Markdown as per below

not the double line breaks \\n\\n these are required due to the way claude-code handles markdown

  =� [ITEM_NAME] \\n\\n

  [FIELD_1] \\n\\n

  [FIELD_2] \\n\\n

  [FIELD_3] \\n\\n
```

### Conditional Output
```markdown
## output format

if [SUCCESS_CONDITION]

    [SUCCESS_MESSAGE]
 
else

    [FAILURE_MESSAGE]
```

## Examples

### Sequential Pattern Example
**Use Case**: "First analyze the code, then run tests, finally create a summary report"

```markdown
---
argument-hint: [project directory]
description: analyze code quality and generate report
---
@.claude/include/substitutions.md

# algorithm

project_path = $ARGUMENT

1. analyze code structure
2. run test suite  
3. generate summary report

## analyze code structure

Review the codebase for patterns, complexity, and potential issues using multiple tools:

```bash
# Use grep to find code patterns
grep -r "TODO\|FIXME\|XXX" $project_path --include="*.py" --include="*.cpp"
```

Use the Glob tool to find relevant files:
```tool
Glob(pattern="**/*.py", path="$project_path")
Glob(pattern="**/*.cpp", path="$project_path") 
```

Read key files using the Read tool to understand structure and identify issues.

## run test suite

Execute the project's test suite and capture results:

```bash
# Run Python tests with detailed output
uv run --group dev pytest tests/python/ -v --tb=short

# Run C++ tests if available
./build/mad_escape_tests --gtest_output=xml
```

Use the Bash tool to execute commands and capture test results.
Parse output for pass/fail counts and specific failure messages.

## generate summary report

Create a comprehensive markdown report using the Write tool:

```tool
Write(file_path="$project_path/analysis_report.md", content="[GENERATED_REPORT]")
```

Include:
- Code metrics and quality indicators
- Test coverage and results summary
- Specific issues found with file locations
- Actionable recommendations with priority levels

## output format

**Code Analysis Complete** \\n\\n

**Project:** $ARGUMENT \\n\\n

**Structure Analysis:** [FINDINGS_SUMMARY] \\n\\n

**Test Results:** [PASS_COUNT] passed, [FAIL_COUNT] failed \\n\\n

**Issues Found:** [ISSUE_COUNT] \\n\\n

**Report Location:** `[PROJECT_PATH]/analysis_report.md` \\n\\n

**Top Recommendations:**
1. [HIGH_PRIORITY_ITEM]
2. [MEDIUM_PRIORITY_ITEM]
3. [LOW_PRIORITY_ITEM]
```

### For Loop Pattern Example
**Use Case**: "Process each file in the directory and extract metadata"

```markdown
---
argument-hint: [file pattern or directory]
description: extract metadata from multiple files
---
@.claude/include/substitutions.md

# algorithm

file_pattern = $ARGUMENT
metadata_results = [...]

for each file in file_pattern:
  1. read file contents
  2. parse metadata headers
  3. extract key information
  4. add metadata to metadata_results

## read file contents

Use the Glob tool to find matching files:
```tool
Glob(pattern="$ARGUMENT")
```

For each file found, use the Read tool to load contents:
```tool
Read(file_path="[FILE_PATH]")
```

Handle different file types:
```bash
# Check if file is binary
file "[FILE_PATH]" | grep -q "text" && echo "processable" || echo "skip binary"
```

Skip files that cannot be processed (binaries, large files > 10MB).

## parse metadata headers

Look for standard metadata formats in the file content:

**YAML Frontmatter** (for .md files):
```python
# Extract YAML between --- markers
import re
yaml_match = re.search(r'^---\\n(.*?)\\n---', content, re.MULTILINE | re.DOTALL)
```

**Comment headers** (for code files):
```bash
# Extract header comments
head -20 "[FILE_PATH]" | grep -E "^(#|//|\*)" | grep -i -E "(author|date|version|description)"
```

**File attributes**:
```bash
# Get file statistics
stat -c "%Y %s" "[FILE_PATH]"  # modification time, size
```

## extract key information

Calculate metrics based on file type:

```bash
# Line count for text files
wc -l "[FILE_PATH]"

# Word count for documentation
wc -w "[FILE_PATH]"

# Function count for Python files
grep -c "^def " "[FILE_PATH]"

# Class count for Python files  
grep -c "^class " "[FILE_PATH]"
```

Use the Grep tool for pattern matching:
```tool
Grep(pattern="^def ", path="[FILE_PATH]", output_mode="count")
Grep(pattern="TODO|FIXME", path="[FILE_PATH]", output_mode="content", -n=true)
```

Store results in structured format with file path, size, dates, and extracted metadata.

## output format

for each file produce output in Markdown as per below

note the double line breaks \\n\\n these are required due to the way claude-code handles markdown

  📁 **[FILENAME]** \\n\\n

  **Path:** `[FULL_PATH]` \\n\\n

  **Size:** [FILE_SIZE] bytes \\n\\n

  **Modified:** [MODIFICATION_DATE] \\n\\n

  **Type:** [FILE_TYPE] \\n\\n

  **Lines:** [LINE_COUNT] \\n\\n

  **Metadata:** [KEY_VALUE_PAIRS] \\n\\n

  **Issues:** [TODO_COUNT] TODOs, [FIXME_COUNT] FIXMEs \\n\\n
```

### While Loop Pattern Example
**Use Case**: "Keep fixing compilation errors until the build succeeds"

```markdown
---
argument-hint: [project to build]
description: iteratively fix build errors until success
---
@.claude/include/substitutions.md

# algorithm

project_path = $ARGUMENT
fixes_applied = []

while (build has errors) todo:
   1. attempt build
   2. analyze error output
   3. fix identified issues
   4. verify fixes

report build success

## build has errors

Run the build command and check exit code using the Bash tool:

```bash
# Try building the project
cd "$project_path"
make -j8 2>&1 | tee build_output.log
echo "Exit code: $?"
```

```tool
Bash(command="cd $project_path && make -j8", description="Build project and capture output")
```

Continue loop if exit code is non-zero, stop if zero (success).
Save build output for error analysis.

## attempt build

Execute the project's build command based on build system:

**CMake projects:**
```bash
# Build with cmake
cmake --build build --parallel 8 2>&1
```

**Make projects:**
```bash
# Traditional make build
make -j8 2>&1
```

**Python projects:**  
```bash
# Install/build Python package
uv pip install -e . 2>&1
```

Use appropriate Bash tool commands and capture both stdout and stderr.

## analyze error output

Parse error messages using Grep and Read tools:

```tool
# Find compilation errors
Grep(pattern="error:", path="build_output.log", output_mode="content", -n=true)

# Find missing includes
Grep(pattern="fatal error:.*No such file", path="build_output.log", output_mode="content")

# Find undefined references  
Grep(pattern="undefined reference", path="build_output.log", output_mode="content")
```

Categorize errors by type:
- **Syntax errors**: Missing semicolons, brackets, typos
- **Missing dependencies**: Include files, libraries  
- **Configuration issues**: CMake, compiler flags
- **Linker errors**: Missing symbols, library paths

## fix identified issues

Apply fixes based on error analysis using Edit tool:

**For missing includes:**
```tool
Edit(file_path="src/problematic_file.cpp", 
     old_string="#include <iostream>", 
     new_string="#include <iostream>\n#include <missing_header.h>")
```

**For syntax errors:**
```tool  
Edit(file_path="src/file.cpp",
     old_string="int x = 5", 
     new_string="int x = 5;")
```

**For CMake issues:**
```tool
Edit(file_path="CMakeLists.txt",
     old_string="find_package(SomeLib)",
     new_string="find_package(SomeLib REQUIRED)")
```

Document each fix applied for reporting.

## verify fixes

Run quick validation after each fix:

```bash
# Check syntax without full compilation
g++ -fsyntax-only problematic_file.cpp

# Test specific file compilation
g++ -c problematic_file.cpp -o /tmp/test.o
```

```tool
Bash(command="g++ -fsyntax-only src/fixed_file.cpp", description="Validate syntax fix")
```

Ensure fixes don't introduce new errors before proceeding to next iteration.

## output format

if build succeeds

    ✅ **Build Successful** \\n\\n
    
    **Project:** $ARGUMENT \\n\\n
    
    **Iterations:** [COUNT] build attempts \\n\\n
    
    **Total Fixes Applied:** [FIX_COUNT] \\n\\n
    
    **Fixes Applied:** \\n\\n
    1. [FIX_DESCRIPTION_1] \\n\\n
    2. [FIX_DESCRIPTION_2] \\n\\n
    3. [FIX_DESCRIPTION_3] \\n\\n
    
    **Final Command:** `make -j8` ✅
 
else

    ❌ **Build Failed After Maximum Attempts** \\n\\n
    
    **Project:** $ARGUMENT \\n\\n
    
    **Remaining Errors:** [ERROR_COUNT] \\n\\n
    
    **Last Error:** \\n\\n
    ```
    [ERROR_MESSAGE]
    ```
    
    **Fixes Applied:** [FIX_COUNT] \\n\\n
    
    **Recommendation:** [MANUAL_INTERVENTION_NEEDED]
```


## Template Variables

- `[ARGUMENT_DESCRIPTION]` - Description of command arguments
- `[COMMAND_DESCRIPTION]` - What the command does
- `[ALGORITHM_STRUCTURE]` - Core logic flow
- `[STEP_N]` - Individual step names
- `[STEP_N_DESCRIPTION]` - Detailed step descriptions
- `[CONDITION_NAME]` - Name of loop/exit condition
- `[CONDITION_DESCRIPTION]` - How to evaluate the condition
- `[OUTPUT_FORMAT_SPECIFICATION]` - How results should be presented
- `[COLLECTION_NAME]` - Name of input collection
- `[ITEM]` - Individual item being processed
- `[RESULT]` - Output from processing an item