# Nanobind Architecture Guide for Madrona

## Table of Contents
1. [Overview](#overview)
2. [RTTI Constraints and Requirements](#rtti-constraints-and-requirements)
3. [Architecture Overview](#architecture-overview)
4. [Addressing RTTI Constraints](#addressing-rtti-constraints)
5. [Code Patterns for Python Integration](#code-patterns-for-python-integration)
6. [PyTorch and JAX Integration](#pytorch-and-jax-integration)
7. [Coding Standards](#coding-standards)
8. [Data Layout Requirements](#data-layout-requirements)

## Overview

Nanobind is a lightweight Python binding library that enables efficient interoperability between C++ and Python. In the Madrona engine context, nanobind presents unique challenges due to RTTI (Run-Time Type Information) requirements that conflict with GPU compilation constraints. This guide documents the architecture, patterns, and standards for using nanobind effectively in this environment.

## RTTI Constraints and Requirements

### Why Nanobind Requires RTTI

Nanobind fundamentally requires RTTI for its type registration and lookup system:

```cpp
// Nanobind internally uses typeid for type registration
template <typename T>
class class_ {
    d.type = &typeid(T);           // Type identification
    d.base = &typeid(Base);         // Inheritance tracking
};

// Type mapping table (simplified)
std::unordered_map<const std::type_info*, PyTypeObject*> type_registry;
```

RTTI is essential for:
1. **Type Registration**: Mapping C++ types to Python type objects
2. **Type Lookup**: Finding the correct Python type for a C++ object at runtime
3. **Inheritance Tracking**: Managing class hierarchies
4. **Type Safety**: Ensuring correct conversions between C++ and Python

### The GPU Compilation Constraint

The Madrona engine faces a fundamental limitation:

```cpp
// GPU code compiled by NVRTC
// CANNOT use: RTTI, exceptions, STL
struct SimulationComponent {  // Must be POD
    float x, y, z;
};

// CPU binding code compiled normally
// CAN use: RTTI (required by nanobind)
class PythonBinding {
    virtual ~PythonBinding() = default;  // Virtual functions OK here
};
```

**The Problem**: 
- GPU simulation code **must** compile with `-fno-rtti`
- Nanobind bindings **require** RTTI enabled
- These requirements appear mutually exclusive

## Architecture Overview

The Madrona architecture solves this through careful layering:

```
┌─────────────────────────────────────────────────────────────┐
│                     Python/PyTorch/JAX                       │
│                  (User Training Code)                        │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ DLPack Protocol
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Nanobind Binding Layer                    │
│                   (RTTI Enabled)                             │
│  - Type registration (typeid)                                │
│  - Python object creation                                    │
│  - Method dispatch                                           │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ POD Structs + Raw Pointers
                              │
┌─────────────────────────────────────────────────────────────┐
│                      Manager Layer                           │
│                   (No RTTI Required)                         │
│  - Tensor metadata (dimensions, types)                       │
│  - Memory management                                         │
│  - Execution control                                         │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ POD Components
                              │
┌─────────────────────────────────────────────────────────────┐
│                  GPU/CPU Simulation Core                     │
│                    (No RTTI, No STL)                         │
│  - ECS components (POD only)                                 │
│  - Physics simulation                                        │
│  - Game logic                                                │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Principles

1. **Selective RTTI Compilation**: Only Python binding files have RTTI enabled
2. **POD Data Transfer**: All shared data uses Plain Old Data structures
3. **Raw Pointer Passing**: Memory addresses cross boundaries without type information
4. **Metadata Wrapping**: Type information travels separately from data

## Addressing RTTI Constraints

### Solution 1: Selective Compilation Flags

The build system selectively enables RTTI:

```cmake
# CMakeLists.txt patterns

# Simulation core - NO RTTI
add_library(simulation_core STATIC
    sim.cpp
    physics.cpp
)
target_compile_options(simulation_core PRIVATE -fno-rtti -fno-exceptions)

# Manager layer - NO RTTI (communicates with simulation)
add_library(manager STATIC
    mgr.cpp
)
target_compile_options(manager PRIVATE -fno-rtti -fno-exceptions)

# Python bindings - RTTI ENABLED (nanobind requirement)
madrona_python_module(my_simulator
    bindings.cpp
)
# Note: madrona_python_module does NOT add -fno-rtti flag
```

### Solution 2: Interface Segregation

Keep RTTI-dependent code isolated:

```cpp
// types.hpp - Shared POD types (NO RTTI)
struct Position {
    float x, y, z;
};

struct Action {
    int32_t moveAmount;
    int32_t moveAngle;
    int32_t rotate;
};

// mgr.hpp - Manager interface (NO RTTI)
class Manager {
public:
    // No virtual functions in the interface
    void step();
    Tensor actionTensor() const;
    Tensor observationTensor() const;
    
private:
    struct Impl;  // PIMPL pattern for implementation hiding
    Impl* impl_;
};

// bindings.cpp - Python bindings (RTTI ENABLED)
NB_MODULE(simulator, m) {
    nb::class_<Manager>(m, "SimManager")  // RTTI used here
        .def("step", &Manager::step)
        .def("action_tensor", &Manager::actionTensor);
}
```

### Solution 3: Tensor Wrapper Pattern

The Tensor class acts as a bridge without requiring RTTI in the core:

```cpp
// Tensor class - can be used with or without RTTI
class Tensor {
    void* dev_ptr_;           // Raw pointer (no type info)
    TensorElementType type_;  // Enum (not RTTI)
    int64_t dims_[16];        // POD array
    int32_t num_dims_;        // Simple integer
    Optional<int> gpu_id_;    // Simple optional
    
public:
    // No virtual functions - POD-like interface
    void* devicePtr() const { return dev_ptr_; }
    TensorElementType type() const { return type_; }
};
```

## Code Patterns for Python Integration

### Pattern 1: Basic Class Binding

```cpp
// bindings.cpp
#include <nanobind/nanobind.h>
namespace nb = nanobind;

NB_MODULE(my_module, m) {
    // Module docstring
    m.doc() = "Madrona simulation Python bindings";
    
    // Bind a class
    nb::class_<SimManager>(m, "SimManager")
        // Constructor with multiple arguments
        .def("__init__", [](SimManager* self,
                           int64_t num_worlds,
                           bool gpu_mode) {
            new (self) SimManager(SimManager::Config{
                .numWorlds = (uint32_t)num_worlds,
                .gpuMode = gpu_mode,
            });
        }, nb::arg("num_worlds"), nb::arg("gpu_mode") = false)
        
        // Simple method
        .def("step", &SimManager::step)
        
        // Method returning custom type
        .def("get_tensor", &SimManager::getTensor,
             nb::rv_policy::reference_internal);
}
```

### Pattern 2: Enum Binding

```cpp
// Bind enums for Python access
nb::enum_<ActionType>(m, "ActionType")
    .value("MOVE_FORWARD", ActionType::MoveForward)
    .value("MOVE_BACKWARD", ActionType::MoveBackward)
    .value("TURN_LEFT", ActionType::TurnLeft)
    .value("TURN_RIGHT", ActionType::TurnRight);
```

### Pattern 3: Property Access

```cpp
nb::class_<Config>(m, "Config")
    // Read-write property
    .def_rw("num_worlds", &Config::numWorlds)
    
    // Read-only property
    .def_ro("max_steps", &Config::maxSteps)
    
    // Property with getter/setter lambdas
    .def_prop_rw("gpu_id",
        [](const Config& c) { return c.gpuID; },
        [](Config& c, int id) { 
            if (id < 0) throw std::invalid_argument("Invalid GPU ID");
            c.gpuID = id;
        });
```

### Pattern 4: Buffer Protocol (NumPy Arrays)

```cpp
// Expose C++ array as NumPy array
nb::class_<DataBuffer>(m, "DataBuffer")
    .def_buffer([](DataBuffer& buf) -> nb::buffer_info {
        return nb::buffer_info{
            buf.data(),                          // Pointer to data
            sizeof(float),                       // Size of one element
            nb::format_descriptor<float>::format(), // Format string
            2,                                    // Number of dimensions
            {buf.rows(), buf.cols()},            // Shape
            {sizeof(float) * buf.cols(),         // Strides
             sizeof(float)}
        };
    });
```

## PyTorch and JAX Integration

### Zero-Copy Tensor Transfer via DLPack

The DLPack protocol enables zero-copy tensor sharing between frameworks:

```cpp
// Pattern for PyTorch tensor export
auto tensor_to_pytorch(const Tensor& tensor) {
    nb::dlpack::dtype dtype = convert_to_dlpack_type(tensor.type());
    
    return nb::ndarray<nb::pytorch, void>{
        tensor.devicePtr(),      // Raw data pointer (zero-copy!)
        tensor.numDims(),        // Number of dimensions
        tensor.dims(),           // Dimension sizes
        nb::handle(),            // Owner (none - we own it)
        nullptr,                 // No strides (C-contiguous)
        dtype,                   // Data type
        tensor.isOnGPU() ? nb::device::cuda::value 
                         : nb::device::cpu::value,
        tensor.gpuID()           // Device ID
    };
}

// Pattern for JAX tensor export
auto tensor_to_jax(const Tensor& tensor) {
    return nb::ndarray<nb::jax, void>{
        tensor.devicePtr(),
        tensor.numDims(),
        tensor.dims(),
        nb::handle(),
        nullptr,
        convert_to_dlpack_type(tensor.type()),
        tensor.isOnGPU() ? nb::device::cuda::value 
                         : nb::device::cpu::value,
        tensor.gpuID()
    };
}
```

### Importing PyTorch Tensors

```cpp
// Accept PyTorch tensor as input
.def("set_actions", [](SimManager& mgr, nb::ndarray<> torch_tensor) {
    // Validate tensor properties
    if (torch_tensor.ndim() != 3) {
        throw std::invalid_argument("Expected 3D tensor");
    }
    
    // Get device location
    bool is_cuda = (torch_tensor.device_type() == nb::device::cuda::value);
    int gpu_id = is_cuda ? torch_tensor.device_id() : -1;
    
    // Access raw data pointer (zero-copy)
    void* data_ptr = torch_tensor.data();
    
    // Use the data directly
    mgr.setActionBuffer(data_ptr, gpu_id);
});
```

### CUDA Stream Synchronization

```cpp
// Pattern for CUDA-aware operations
nb::class_<CudaSync>(m, "CudaSync")
    .def("wait", [](CudaSync& sync) {
        #ifdef MADRONA_CUDA_SUPPORT
        cudaStreamSynchronize(sync.stream);
        #endif
    });
```

## Coding Standards

### Standard 1: File Organization

```
project/
├── src/
│   ├── sim.cpp           # Simulation core (NO RTTI)
│   ├── sim.hpp           
│   ├── mgr.cpp           # Manager layer (NO RTTI)
│   ├── mgr.hpp
│   ├── types.hpp         # Shared POD types (NO RTTI)
│   └── bindings.cpp      # Python bindings (RTTI ENABLED)
├── CMakeLists.txt        # Build configuration
└── python/
    └── __init__.py       # Python wrapper
```

### Standard 2: Type Definitions

```cpp
// types.hpp - Shared types must be POD
// GOOD: POD struct
struct Observation {
    float position[3];
    float velocity[3];
    int32_t health;
    uint8_t flags;
};

// BAD: Non-POD types
struct BadObservation {
    std::vector<float> data;     // STL container - NOT POD
    virtual void process();      // Virtual function - NOT POD
    std::string name;            // Dynamic allocation - NOT POD
};
```

### Standard 3: Manager Interface Design

```cpp
// mgr.hpp - Manager interface pattern
class Manager {
public:
    // Configuration should be POD
    struct Config {
        uint32_t numWorlds;
        int32_t gpuID;
        bool enableViz;
    };
    
    // Constructor takes POD config
    explicit Manager(const Config& cfg);
    
    // Methods return Tensor wrappers, not raw pointers
    Tensor observationTensor() const;
    Tensor actionTensor() const;
    
    // No virtual functions in public interface
    void step();
    
private:
    // Implementation details hidden
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
```

### Standard 4: Binding Patterns

```cpp
// bindings.cpp - Consistent binding patterns
NB_MODULE(module_name, m) {
    // 1. Always set up Madrona submodule first
    madrona::py::setupMadronaSubmodule(m);
    
    // 2. Bind configuration structures
    nb::class_<Manager::Config>(m, "Config")
        .def(nb::init<>())
        .def_rw("num_worlds", &Manager::Config::numWorlds)
        .def_rw("gpu_id", &Manager::Config::gpuID);
    
    // 3. Bind main manager class
    nb::class_<Manager>(m, "Manager")
        // Use lambda for constructor to handle type conversions
        .def("__init__", [](Manager* self, const Manager::Config& cfg) {
            new (self) Manager(cfg);
        })
        
        // Tensor methods use reference_internal policy
        .def("observation_tensor", &Manager::observationTensor,
             nb::rv_policy::reference_internal)
        
        // Action methods take numpy/torch arrays
        .def("set_actions", [](Manager& mgr, nb::ndarray<> actions) {
            // Validate and apply actions
        });
}
```

### Standard 5: Error Handling

```cpp
// Use exceptions in Python bindings (they're caught by nanobind)
.def("load_level", [](Manager& mgr, const std::string& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Level file not found: " + path);
    }
    mgr.loadLevel(path);
});

// But NOT in simulation core (no exceptions)
// sim.cpp - Use error codes or assertions
bool loadPhysicsAssets(const char* path) {
    if (!path) return false;  // Error code
    assert(strlen(path) > 0);  // Debug assertion
    // ...
}
```

## Data Layout Requirements

### Layout 1: ECS Components (Simulation Core)

```cpp
// All ECS components must be POD
// Aligned for GPU access efficiency

struct Position {
    float x, y, z;
    float padding;  // Align to 16 bytes
};

struct Velocity {
    float dx, dy, dz;
    float padding;
};

// Component arrays are tightly packed
struct AgentData {
    Position positions[MAX_AGENTS];
    Velocity velocities[MAX_AGENTS];
    Action actions[MAX_AGENTS];
};
```

### Layout 2: Tensor Memory (Manager Layer)

```cpp
// Tensors must be contiguous for Python interop
class TensorStorage {
    // Observations: [num_worlds, num_agents, observation_size]
    float* observations_;
    
    // Actions: [num_worlds, num_agents, action_size]  
    int32_t* actions_;
    
    // Ensure C-contiguous layout (row-major)
    size_t getOffset(size_t world, size_t agent, size_t element) {
        return (world * num_agents_ * element_size_) +
               (agent * element_size_) +
               element;
    }
};
```

### Layout 3: Python Tensor Interface

```python
# Python side expects specific layouts
class SimManager:
    def __init__(self, num_worlds: int, num_agents: int):
        self.mgr = native.Manager(num_worlds)
        
    def get_observations(self) -> torch.Tensor:
        # Returns view of C++ memory (zero-copy)
        tensor = self.mgr.observation_tensor().to_torch()
        # Shape: [num_worlds, num_agents, obs_dim]
        assert tensor.is_contiguous()
        return tensor
        
    def set_actions(self, actions: torch.Tensor):
        # Must be contiguous
        if not actions.is_contiguous():
            actions = actions.contiguous()
        # Direct memory transfer (zero-copy)
        self.mgr.action_tensor().copy_(actions)
```

### Layout 4: GPU Memory Alignment

```cpp
// GPU tensors must respect alignment requirements
struct GPUTensorLayout {
    // Align to 256 bytes for coalesced access
    static constexpr size_t ALIGNMENT = 256;
    
    void* allocate(size_t size) {
        void* ptr;
        #ifdef MADRONA_CUDA_SUPPORT
        cudaMalloc(&ptr, align_up(size, ALIGNMENT));
        #else
        posix_memalign(&ptr, ALIGNMENT, size);
        #endif
        return ptr;
    }
};
```

## Best Practices Summary

1. **Always compile simulation core with `-fno-rtti`**
2. **Only enable RTTI for binding files**
3. **Use POD types for all shared data structures**
4. **Employ PIMPL pattern to hide implementation details**
5. **Leverage DLPack for zero-copy tensor sharing**
6. **Validate tensor layouts and continuity**
7. **Handle GPU/CPU memory transparently**
8. **Use clear error messages in Python bindings**
9. **Document memory ownership clearly**
10. **Test with both CPU and GPU backends**

## Conclusion

The nanobind architecture in Madrona demonstrates that RTTI constraints can be successfully managed through careful architectural design. By segregating RTTI-dependent code to the binding layer and using POD types for data exchange, the system achieves both high performance and clean Python integration. The zero-copy tensor sharing via DLPack ensures that the overhead of the Python binding is minimal, making this architecture suitable for high-performance simulation and reinforcement learning applications.