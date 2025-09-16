# [Feature/System Name] Specification

## Overview
[High-level description of the feature/system and its purpose in the Madrona Escape Room simulation]

## Key Files

### Source Code
Primary implementation files:

- `src/[file1].cpp` - [Purpose/responsibility]
- `src/[file2].hpp` - [Purpose/responsibility]
- `src/[file3].cpp` - [Purpose/responsibility]

### Test Files

#### C++ Tests
- `tests/cpp/test_[feature].cpp` - [What aspects are tested]
- `tests/cpp/test_[feature2].cpp` - [What aspects are tested]

#### Python Tests
- `tests/python/test_[feature].py` - [What aspects are tested]
- `tests/python/test_[feature2].py` - [What aspects are tested]

## Architecture

### System Integration
[How this feature integrates with existing Madrona systems]

### GPU/CPU Code Separation
- **GPU (NVRTC) Code**: [Files that run on GPU - sim.cpp, level_gen.cpp]
- **CPU-Only Code**: [Manager layer files - mgr.cpp]
- **Shared Headers**: [Common definitions - types.hpp]

## Implementation

### Data Structures

#### Entity-Component Matrix
[Overview of which components belong to which entity types]

| Component | [Entity1] | [Entity2] | [Entity3] | [Entity4] | Description |
|-----------|-----------|-----------|-----------|-----------|-------------|
| [Component1] | ✓ | ✓ | | | [What this component represents] |
| [Component2] | ✓ | | ✓ | | [What this component represents] |
| [Component3] | | ✓ | ✓ | | [What this component represents] |
| [Component4] | ✓ | ✓ | ✓ | ✓ | [What this component represents] |
| [NewComponent] | | | | ✓ | [What this component represents] |

#### Archetype Definitions
```cpp
// Archetype definitions based on the matrix above
struct [Entity1] : madrona::Archetype<
    [Component1],
    [Component2],
    [Component4]
> {};

struct [Entity2] : madrona::Archetype<
    [Component1],
    [Component3],
    [Component4]
> {};
```

#### Component Definitions
```cpp
// New component structures
struct [ComponentName] {
    float field1;  // [Description]
    int32_t field2;  // [Description]
};
```

### Core Systems

1. **[System Name]**
   - **Purpose**: [What this ECS system does]
   - **Components Used**: [List of components read/written]
   - **Task Graph Dependencies**: [Systems this depends on or feeds into]
   - **Specifications**:
     - [Specific behavior or requirement]
     - [Algorithm or calculation performed]
     - [Edge cases handled]
     - [Performance constraints]

2. **[System Name]**
   - **Purpose**: [What this ECS system does]
   - **Components Used**: [List of components read/written]
   - **Task Graph Dependencies**: [Systems this depends on or feeds into]
   - **Specifications**:
     - [Specific behavior or requirement]
     - [Algorithm or calculation performed]
     - [Edge cases handled]
     - [Performance constraints]

## Performance Considerations

### GPU Optimization
- [Kernel optimization strategies]
- [Memory access patterns]
- [Warp efficiency considerations]
- [Shared memory usage]
- [Coalesced memory access patterns]

## CompiledLevel Changes
[If applicable, describe changes to the level format]
```cpp
struct CompiledLevel {
    // Existing fields...

    // New fields for this feature
    float newField;
};
```

## Build Configuration

### CMake Changes
```cmake
# If new source files need to be added to SIMULATOR_SRCS or other targets
```

## Migration Notes
[If modifying existing functionality, describe migration path]

## Future Enhancements
[Potential improvements and extensibility points]