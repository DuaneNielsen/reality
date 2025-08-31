# Python Code Generation Research

This folder contains research and proof-of-concept implementations for automatically generating Python bindings from C++ code without manual struct definitions.

## Key Achievement

Successfully demonstrated using **pahole** to extract exact memory layouts from compiled binaries, eliminating all guesswork about struct padding and alignment. Combined with **libclang** for constants extraction and **DLPack** for tensor access, this provides a complete, reliable solution for Python bindings.

## Documentation

- **[binary_struct_extraction_research.md](binary_struct_extraction_research.md)** - Complete research on extracting struct layouts from binaries
- **[poc_struct_passing_results.md](poc_struct_passing_results.md)** - Detailed POC results and validation

## Implementation Files

### Core Tools
- **parse_pahole.py** - Parser for pahole output to extract struct definitions with exact offsets
- **generate_structs_from_binary.py** - Automatically generates ctypes structures from compiled binaries
- **parse_constants.py** - Extracts compile-time constants from C++ headers using libclang

### Generated Output
- **madrona_constants.py** - 130+ constants extracted from consts.hpp

### Test Programs
- **test_minimal_level.py** - Minimal test proving the concept works
- **create_default_level.py** - Full 16x16 room creation using generated structs
- **test_pahole_structs.py** - Validation tests for pahole-generated structures

## Key Results

- **100% Accuracy**: Exact memory layouts from compiler, no guessing
- **Zero Manual Definitions**: All structs auto-generated from binary
- **Platform Specific**: Correct for the exact compiler/architecture used
- **Successfully Tested**: Created SimManager, ran simulation steps

## Usage

1. Install pahole: `sudo apt install dwarves`
2. Compile with debug symbols: `-g` flag
3. Extract layout: `pahole -C StructName library.so`
4. Run generator: `python generate_structs_from_binary.py`
5. Use generated structs in Python with ctypes

## Technologies Used

- **pahole** (dwarves package) - Extracts struct layouts from DWARF debug info
- **libclang** - Parses C++ headers for constants via AST
- **ctypes** - Python's built-in C interface
- **DLPack** - Tensor interoperability standard