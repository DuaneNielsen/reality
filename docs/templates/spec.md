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

## Data Structures
[Brief description of the data structures and their purpose]

### Primary Structure

#### [Structure Name]
```cpp
struct [StructureName] {
    // [Category 1]
    [type] [field_name];              // [Purpose]
    [type] [field_name];              // [Purpose]

    // [Category 2]
    [type] [array_name][SIZE];        // [Purpose]
    [type] [array_name][SIZE];        // [Purpose]

    // ... additional fields
};
```

**Key Points:**
- [Important characteristic or constraint]
- [Important characteristic or constraint]

### Supporting Structures

#### [Structure Name]
```cpp
struct [StructureName] {
    [type] [field_name];  // [Purpose]
    [type] [field_name];  // [Purpose]
};
```

**Purpose:** [What this structure is for]

#### Invariants
- [Constraint that must always be true]
- [Constraint that must always be true]

## Module Interface
[Description of the public interface this module exposes to consumers]

### [Namespace/Class Name]

#### [methodName]

**Purpose:** [What this method accomplishes]

**Parameters:**
- `[param1]`: [Type, constraints, ownership]
- `[param2]`: [Type, constraints, ownership]

**Returns:** [Return type and semantics]

**Preconditions:**
- [What must be true before calling]
- [Required state or initialization]

**Specs:**
- [Specific behavior or requirement]
- [Algorithm or calculation performed]
- [Edge cases handled]
- [Performance constraints]

**Error Handling:**
- **Invalid Parameters:** [How invalid inputs are handled]
- **State Violations:** [What happens when preconditions aren't met]
- **Resource Failures:** [How resource allocation failures are handled]

#### [anotherMethodName]

**Purpose:** [Core functionality this method provides]

**Parameters:**
- `[param1]`: [Type and valid ranges]
- `[param2]`: [Optional/required, default values]
- `[param3]`: [Ownership semantics if pointer/reference]

**Returns:** [Type, nullptr conditions, error codes]

**Preconditions:**
- [System state requirements]
- [Dependencies that must be initialized]
- [Valid parameter ranges]

**Specs:**
- [Exact computation or transformation performed]
- [Order of operations if complex]
- [Special cases and their handling]
- [Performance guarantees or limits]

**Error Handling:**
- **Null Pointers:** [How nullptr inputs are handled]
- **Out of Bounds:** [Range violation behavior]
- **Resource Exhaustion:** [Memory/resource limit handling]

### [AnotherNamespace/Class]

#### [initializeFunction]

**Purpose:** [Initialization role and what it sets up]

**Parameters:**
- `[configParam]`: [Configuration structure or parameters]
- `[resourcePath]`: [Resource location requirements]

**Returns:** [Success/failure indication]

**Preconditions:**
- [Must be called before any other methods]
- [Required system resources]

**Specs:**
- [Initialization sequence]
- [Resource allocation strategy]
- [Default values applied]
- [One-time setup vs re-initialization]

**Error Handling:**
- **Already Initialized:** [Behavior if called twice]
- **Missing Resources:** [File/resource not found handling]
- **Invalid Configuration:** [Bad config parameter response]

#### [queryFunction]

**Purpose:** [Information retrieval without side effects]

**Parameters:**
- `[queryKey]`: [Identifier or search parameter]
- `[outputBuffer]`: [Buffer for results if applicable]

**Returns:** [Query result or status]

**Preconditions:**
- [System must be initialized]
- [Valid query parameters]

**Specs:**
- [Query algorithm complexity]
- [Caching behavior if any]
- [Consistency guarantees]
- [Thread-safety requirements]

**Error Handling:**
- **Not Found:** [Key doesn't exist behavior]
- **Buffer Too Small:** [Insufficient output space]
- **Invalid Query:** [Malformed query handling]

## Configuration

### Build Configuration
```cmake
# If new source files need to be added to targets
```

### Runtime Configuration
[Configuration options and their effects]