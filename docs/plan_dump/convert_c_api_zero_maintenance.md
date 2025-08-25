# Plan: Convert C API to Reduced/Zero-Maintenance Using Auto-Generation

## Pre-Reading List
1. **docs/research/python_codegen/README.md** - Overview of the auto-generation approach
2. **docs/research/python_codegen/binary_struct_extraction_research.md** - Technical details on pahole and struct extraction
3. **codegen/generate_python_constants.py** - How constants are currently auto-generated
4. **codegen/generate_python_structs.py** - How structs are currently auto-generated
5. **src/types.hpp** - Source of truth for all structs
6. **src/consts.hpp** - Source of truth for all constants
7. **include/madrona_escape_room_c_api.h** - Current C API header to be simplified
8. **include/madrona_escape_room_c_constants.h** - Header to be removed
9. **madrona_escape_room/__init__.py** - Python bindings that will use generated code

## Phase 1: Remove Redundant C Constants Header
1. **Delete include/madrona_escape_room_c_constants.h**
   - All constants are auto-generated from consts.hpp

2. **Update include/madrona_escape_room_c_api.h**
   - Remove `#include "madrona_escape_room_c_constants.h"` (line 22)
   - Remove all redundant #define constants (lines 93-127):
     - MER_SELF_OBSERVATION_SIZE through MER_TOTAL_OBSERVATION_SIZE
     - MER_NUM_AGENTS through MER_EPISODE_LENGTH  
     - MER_MOVE_* constants
     - MER_ROTATE_* constants
   - Keep only C-specific enums and structs

3. **Update CMakeLists.txt**
   - Remove references to madrona_escape_room_c_constants.h from install targets

## Phase 2: Update Python Bindings to Use Generated Constants
1. **Update madrona_escape_room/__init__.py**
   - Import from generated_constants instead of hardcoded values
   - Replace all MER_* constant references with generated equivalents:
     ```python
     from .generated_constants import consts, types
     # Use consts.action.move_amount.STOP instead of MER_MOVE_STOP
     ```

2. **Update madrona_escape_room/ctypes_bindings.py**
   - Remove any hardcoded constant definitions
   - Use generated constants for all values

## Phase 3: Simplify C API to Pure Function Interface
1. **Update src/madrona_escape_room_c_api.cpp**
   - Remove unnecessary includes if types.hpp is only needed for CompiledLevel
   - Keep using void* for CompiledLevel (already done)
   - Document that Python passes exact C++ structs via ctypes

2. **Clean up include/madrona_escape_room_c_api.h**
   - Keep only:
     - Export macros (platform-specific)
     - C-specific error codes (MER_Result)
     - Handle typedef (opaque pointer)
     - C-specific wrapper structs (MER_Tensor, MER_ManagerConfig, MER_ReplayMetadata)
     - Function declarations
   - Remove all game-specific constants

## Phase 4: Enhance Code Generation
1. **Update codegen/generate_python_constants.py**
   - Ensure it captures all constants from consts.hpp namespaces
   - Add any missing constant groups

2. **Verify codegen/generate_python_structs.py**
   - Confirm CompiledLevel is correctly generated with all fields
   - Add assertions for struct sizes

## Phase 5: Documentation Updates
1. **Update docs/deployment/PYTHON_BINDINGS_GUIDE.md**
   - Document the auto-generation approach
   - Explain how constants and structs are generated

2. **Update codegen/README.md**
   - Add section on maintenance-free approach
   - Document what is auto-generated vs what is C API specific

## Benefits After Implementation
- **Zero maintenance** for constants - automatically synchronized with C++
- **Zero maintenance** for struct layouts - extracted from compiled binary
- **Minimal C API surface** - only C-specific abstractions remain
- **Single source of truth** - C++ headers define everything
- **Build-time validation** - struct sizes verified during generation
- **No manual synchronization** - changes in C++ automatically propagate

## Files to Modify
- DELETE: include/madrona_escape_room_c_constants.h
- MODIFY: include/madrona_escape_room_c_api.h (remove ~40 lines)
- MODIFY: src/CMakeLists.txt (remove constants header from install)
- MODIFY: madrona_escape_room/__init__.py (use generated constants)
- MODIFY: madrona_escape_room/ctypes_bindings.py (use generated constants)

## Success Criteria
- All Python tests pass with generated constants
- No hardcoded constants in Python code
- C API header contains only C-specific definitions
- Build automatically generates correct bindings
- Changes to C++ constants/structs automatically propagate to Python

## Testing Strategy
- Break-fix approach: Make changes and fix issues as they arise
- Run existing tests to identify broken references
- Update test code as needed when constants are renamed