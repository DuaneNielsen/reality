# Fix GPU Compilation by Creating `host_types.hpp`

## Reading List

Before implementing this plan, review these key files and sections:

### Core Problem Files
- `src/types.hpp:240-270` - Current ReplayMetadata struct with STL usage
- `src/types.hpp:9` - `#include <cstring>` that needs to be moved
- GPU compilation error output showing `std::strncpy` and `std::memset` failures

### Files That Use ReplayMetadata (need host_types.hpp)
- `src/mgr.hpp:74` - ReplayMetadata member in Manager
- `src/mgr.cpp:136,1043,1048,1110,1119,1160,1172,1261` - ReplayMetadata usage
- `src/replay_loader.hpp:15,23,29,48` - ReplayMetadata loading functions
- `src/madrona_escape_room_c_api.cpp:484,491,492` - C API ReplayMetadata handling
- `src/headless.cpp:222` - Replay metadata reading
- `src/viewer.cpp:260` - Replay metadata reading
- `src/file_inspector.cpp:57,83` - File inspection with ReplayMetadata

### GPU Compilation System
- `src/CMakeLists.txt` - SIMULATOR_SRCS list (GPU-compiled files)
- `docs/development/CPP_CODING_STANDARDS.md` - GPU vs CPU code split rules
- NVRTC compilation constraints (no STL in GPU code)

### Codegen System
- `codegen/generate_dataclass_structs.py:68` - Current ReplayMetadata extraction
- `codegen/generate_dataclass_structs.py:60-70` - DEFAULT_STRUCTS_TO_EXTRACT list
- Python dataclass generation process

---

## Problem
GPU compilation fails because `ReplayMetadata::createDefault()` uses STL functions (`std::strncpy`, `std::memset`) that aren't available in NVRTC compilation environment. The `ReplayMetadata` struct was moved to `types.hpp` (GPU-compiled) but contains host-only code.

## Solution
Create `src/host_types.hpp` for host-only types and separate them from GPU-compiled types.

## Implementation Steps

### 1. Create `src/host_types.hpp`
- Move `ReplayMetadata` struct from `src/types.hpp` to new `src/host_types.hpp`
- Include all necessary headers (`<cstring>`, `<cstdint>`, etc.)
- Keep the STL-using `createDefault()` method as-is (works fine on host)

### 2. Update `src/types.hpp`
- Remove `ReplayMetadata` struct and its includes
- Remove `#include <cstring>` (was only needed for ReplayMetadata)
- Keep all other GPU-safe types

### 3. Update Files That Include `types.hpp`
**Host-only files that need `ReplayMetadata` (add `#include "host_types.hpp"`):**
- `src/mgr.hpp` - uses ReplayMetadata in Manager class
- `src/mgr.cpp` - implements ReplayMetadata functionality
- `src/replay_loader.hpp` - loads replay metadata
- `src/madrona_escape_room_c_api.cpp` - C API for replay metadata
- `src/headless.cpp` - reads replay metadata
- `src/viewer.cpp` - reads replay metadata  
- `src/file_inspector.cpp` - inspects replay files

**GPU-compiled files (no changes needed):**
- `src/sim.cpp`, `src/sim.hpp` - don't use ReplayMetadata
- `src/level_gen.cpp` - doesn't use ReplayMetadata

### 4. Update Codegen Configuration
- Add `"src/host_types.hpp"` to `codegen/generate_dataclass_structs.py`
- Update `DEFAULT_STRUCTS_TO_EXTRACT` to include structs from both files
- Ensure Python still gets `ReplayMetadata` dataclass

### 5. Update Build System
- Verify `src/host_types.hpp` is NOT included in `SIMULATOR_SRCS` in `src/CMakeLists.txt`
- Ensure it's only compiled for host code

## Files to Modify

1. **Create:** `src/host_types.hpp` - Host-only types with STL usage
2. **Edit:** `src/types.hpp` - Remove ReplayMetadata, keep GPU-safe types only
3. **Edit:** 7 host files - Add `#include "host_types.hpp"`
4. **Edit:** `codegen/generate_dataclass_structs.py` - Add host_types.hpp to extraction

## Testing
- Run `./build/headless --cuda 0 -n 100 -s 100 --rand-actions` to verify GPU compilation works
- Run Python tests to ensure ReplayMetadata dataclass still generates correctly
- Verify no regression in CPU functionality

## Expected Outcome
- GPU compilation succeeds (no more NVRTC STL errors)
- Python codegen continues to work
- All existing functionality preserved
- Clean separation between host-only and GPU-compatible types