# C++ Coding Standards

## Critical Constraints
- **NO EXCEPTIONS or RTTI** - Both are disabled in project settings
- **NO STL IN GPU CODE** - NVRTC cannot compile STL headers
- **ECS ONLY** - All simulation code must use Entity Component System

## GPU vs CPU Code Split

### NVRTC-Compiled (GPU) Files
Files in `SIMULATOR_SRCS` are compiled by NVRTC for GPU execution:
- `sim.cpp`, `sim.hpp`, `sim.inl` - Core ECS simulation
- `level_gen.cpp`, `level_gen.hpp` - Level generation
- `types.hpp` - ECS component definitions

**Restrictions**: No STL, exceptions, or RTTI. Use Madrona containers only.

### Regular C++ Files  
Manager and host code compiled normally:
- `mgr.cpp`, `mgr.hpp` - Manager layer (CPU-only)
- `viewer.cpp`, `headless.cpp` - Host applications
- `madrona_escape_room_c_api.cpp` - C API wrapper

**Allowed**: STL, standard C++ features (but still no exceptions/RTTI per project settings)

### How to Check
```bash
# Files compiled by NVRTC
grep -A 10 "SIMULATOR_SRCS" src/CMakeLists.txt

# GPU code markers
grep "__device__\|__global__\|__host__" src/filename.cpp
```

## Madrona-Specific Requirements

### STL Replacement Rules (All Code)
```cpp
// Use Madrona toolchain everywhere - avoid dual code paths

// std::string → Fixed-size char arrays
char name[64];                    // Instead of std::string name;

// std::vector → Madrona containers  
madrona::HeapArray<T>            // Fixed-size (like std::array)
madrona::DynArray<T>             // Growable (like std::vector)

// std::unordered_map → Linear search or custom hash
// For small datasets (<100 items), use linear search:
const Item* findItem(const HeapArray<Item>& items, uint32_t id) {
    for (const auto& item : items) {
        if (item.id == id) return &item;
    }
    return nullptr;
}

// std::filesystem → C functions or compile-time generation
// Prefer compile-time asset lists over runtime filesystem scanning
```

### ECS Components
```cpp
// Components MUST be POD types
struct Position { float x, y, z; };  // Good
struct Bad { std::string name; };    // Will not compile on GPU

// Use archetypes to group components
struct Agent : madrona::Archetype<Position, Velocity, Action> {};
```

### Avoid Dual Code Paths
```cpp
// BAD - Maintains two implementations
#ifdef MADRONA_GPU_MODE
    using Container = madrona::HeapArray<Entity>;
#else
    using Container = std::vector<Entity>;
#endif

// GOOD - Single implementation works everywhere
using Container = madrona::HeapArray<Entity>;
```

## Project Conventions

### Member Variables
```cpp
// Trailing underscore for member variables (not universal C++ style)
class Manager {
    uint32_t numWorlds_;
    HeapArray<Entity> entities_;
};
```

### Code Classification Markers
```cpp
// [BOILERPLATE] - Core Madrona framework code, never modify
// [REQUIRED_INTERFACE] - Must implement for any Madrona environment  
// [GAME_SPECIFIC] - Unique to this game, safe to modify
```

### Error Handling
```cpp
// Return codes or assert only (no exceptions available)
bool loadAsset(const char* path) {
    if (!path) return false;  // Runtime error
    assert(path[0]);          // Debug-only check
}
```