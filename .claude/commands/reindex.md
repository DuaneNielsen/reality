---
argument-hint: [no arguments]
description: reindex all source files to FILEINDEX.md
---

IMPRTANT CONTEXT  BY DEFAULT YOU WILL WANT TO BATCH TOOL CALLS.  THIS WILL MAKE YOU WANT TO IGNORE THE ITERATIVE 
PROCEEDURE DEFINED IN THE ALGORITHM SECTION.  OVERRIDE YOUR INSTINCT TO BATCH CALLS AND MAKE SURE YOU READ
ENOUGH OF EACH FILE TO GENERATE A GOOD ENOUGH SUMMARY.  IF YOU ARE NOT SURE THEN READ MORE.

# algorithm

get the file list

!cd ~/madrona_escape_room && ls src/*.cpp && ls src/*.hpp && ls madrona_escape_room/*.py && ls codegen/*.py && ls docs/specs && ls tests/README.md && ls tests/cpp/README.md && ls/tests/python/README.md && train_src/madrona_escape_room_learn/

# Initialize FILEINDEX.md with header
Write("# File Index\n\n")

for each file in file_list:
  1. read significant portion of file
  2. analyze file contents
  3. write index entry for this file
  4. immediately append this entry to FILEINDEX.md (don't batch)

## read significant portion of file

Use the Read tool to read at least 200 lines of each file:

```tool
Read(file_path="[FILE_PATH]", limit=200)
```

For smaller files, read the entire file. For larger files, read at least 200 lines to understand the core functionality.

## analyze file contents

Examine the file to understand:
- Primary purpose and functionality
- Key classes, functions, or data structures
- Dependencies and relationships
- Role in the overall system

Look for:
- Function definitions and class declarations
- Import statements and includes
- Comments and docstrings explaining purpose
- Main entry points or exported functionality

## write index entry

Create a single line entry in the format:

```
[FILE_PATH] - [BRIEF_DESCRIPTION]
```

Where:
- FILE_PATH uses backslashes (e.g., src\file.cpp)
- BRIEF_DESCRIPTION is 3-8 words explaining what the file does

## output format

Each file produces one line in FILEINDEX.md following this format:

src\consts.hpp - game constants and configuration values
src\types.hpp - ECS component definitions and data structures
src\mgr.cpp - simulation manager and orchestration
madrona_escape_room\manager.py - python interface to C++ simulation
codegen\generate_dataclass_structs.py - generates python dataclasses from C++ structs


