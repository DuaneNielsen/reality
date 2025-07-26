# Generate Python Stubs

Regenerates Python stub files for the madrona_escape_room C++ extension module to provide IDE support.

## Usage
```
/generate-stubs
```

## What it does
1. Runs `uv run nanobind-stubgen madrona_escape_room` to generate .pyi stub files
2. Creates stubs in `madrona_escape_room/` directory:
   - `__init__.pyi` - Main module stubs with exported constants and SimManager class
   - `madrona.pyi` - Madrona submodule stubs

## When to use
- After adding new exports in `src/bindings.cpp`
- After modifying exported constants in `src/types.hpp`
- When IDE autocomplete is missing for C++ exports

## Prerequisites
- Project must be built (`make -j` in build directory)
- `nanobind-stubgen` must be installed (automatically installed with `uv pip install -e .`)