# Creating New ECS Tables from Scratch

This guide explains how to create new Entity Component System (ECS) tables in the Madrona engine based on lessons learned from implementing the ECS debug tracker system.

## Overview

Creating a new ECS table involves several steps that must be done in the correct order to avoid runtime assertions and compilation errors. This guide provides a step-by-step process with working examples.

## Prerequisites

- Understanding of Madrona's ECS architecture
- Familiarity with C++ templates and CRTP (Curiously Recurring Template Pattern)
- Access to a working Madrona project setup

## Step-by-Step Process

### 1. Define Your Components

First, create simple POD (Plain Old Data) structures for your components. Components must be POD types to work on both CPU and GPU.

```cpp
// Example: Simple component with just one value
struct TestComponent {
    float value;
};

// Example: More complex component
struct HealthComponent {
    float currentHealth;
    float maxHealth;
    bool isDead;
};
```

### 2. Define Your Archetypes

Archetypes group related components together. They inherit from `madrona::Archetype` and specify which components they contain.

```cpp
// Simple archetype with one component
struct SimpleEntity : public madrona::Archetype<TestComponent> {};

// Complex archetype with multiple components
struct HealthEntity : public madrona::Archetype<HealthComponent, Position, Velocity> {};
```

### 3. Create World Data Class

Your world data class must inherit from `madrona::WorldBase` and follow the required interface pattern:

```cpp
// Forward declare your engine/context class
class Engine;

class Sim : public madrona::WorldBase {
public:
    // Required static registration function
    static void registerTypes(madrona::ECSRegistry &registry, const Config &cfg);

    // Required static task setup function
    static void setupTasks(madrona::TaskGraphManager &mgr, const Config &cfg);

    // Constructor called for each world
    Sim(Engine &ctx, const Config &cfg, const WorldInit &init);

    // Your world-specific data
    bool someWorldState;
    uint32_t worldID;
};
```

### 4. Create Context/Engine Class

Your context class provides the interface for ECS operations and inherits from `madrona::CustomContext`:

```cpp
class Engine : public madrona::CustomContext<Engine, Sim> {
public:
    using CustomContext::CustomContext;

    // System functions that operate on components
    static void mySystem(Engine &ctx, TestComponent &comp) {
        comp.value += 1.0f;
    }
};
```

### 5. Implement Registration Function

**CRITICAL**: Components must be registered BEFORE archetypes that use them, or you'll get assertion failures.

```cpp
void Sim::registerTypes(madrona::ECSRegistry &registry, const Config &cfg) {
    (void)cfg; // Suppress unused parameter warning

    // FIRST: Register all component types
    registry.registerComponent<TestComponent>();
    registry.registerComponent<HealthComponent>();
    registry.registerComponent<Position>();
    registry.registerComponent<Velocity>();

    // THEN: Register archetypes that use those components
    registry.registerArchetype<SimpleEntity>();
    registry.registerArchetype<HealthEntity>();

    // Optional: Register singletons
    registry.registerSingleton<SomeGlobalState>();
}
```

### 6. Implement Task Setup Function

Define your ECS systems using the task graph builder:

```cpp
// Define task graph IDs
enum class TaskGraphID : uint32_t {
    Step,
    NumTaskGraphs,
};

void Sim::setupTasks(madrona::TaskGraphManager &mgr, const Config &cfg) {
    (void)cfg; // Suppress unused parameter warning

    madrona::TaskGraphBuilder &builder = mgr.init(TaskGraphID::Step);

    // Add systems that operate on specific components
    auto my_system = builder.addToGraph<madrona::ParallelForNode<Engine,
        Engine::mySystem, TestComponent>>({});

    // Systems can depend on other systems
    auto dependent_system = builder.addToGraph<madrona::ParallelForNode<Engine,
        Engine::otherSystem, HealthComponent>>({my_system});
}
```

### 7. Implement Constructor

The constructor is called for each world and is where you can create initial entities:

```cpp
Sim::Sim(Engine &ctx, const Config &cfg, const WorldInit &init)
    : madrona::WorldBase(ctx), someWorldState(true), worldID(init.worldID) {

    // Create initial entities - this triggers table allocation
    auto entity1 = ctx.makeEntity<SimpleEntity>();
    auto &comp1 = ctx.get<TestComponent>(entity1);
    comp1.value = 10.0f;

    auto entity2 = ctx.makeEntity<HealthEntity>();
    auto &health = ctx.get<HealthComponent>(entity2);
    health.currentHealth = 100.0f;
    health.maxHealth = 100.0f;
    health.isDead = false;
}
```

### 8. Create TaskGraphExecutor

Finally, create the executor that runs your ECS system:

```cpp
int main() {
    // Setup configuration and world initialization
    MyConfig cfg;
    cfg.enableDebug = true;

    MyWorldInit world_init;
    world_init.worldID = 0;

    // Create the task graph executor
    using MyTaskGraph = madrona::TaskGraphExecutor<
        Engine,       // Context type
        Sim,          // World data type
        MyConfig,     // Config type
        MyWorldInit   // World init type
    >;

    MyTaskGraph exec({
        .numWorlds = 1,          // Number of parallel worlds
        .numExportedBuffers = 0, // Number of exported data buffers
        .numWorkers = 1          // Number of worker threads
    }, cfg, &world_init, 1);     // Number of task graphs

    // Run the simulation
    exec.runTaskGraph(0);

    return 0;
}
```

## Common Pitfalls and Solutions

### 1. Assertion: `component_id != TypeTracker::unassignedTypeID`

**Problem**: Trying to register an archetype before registering its component types.

**Solution**: Always register components with `registerComponent<>()` before archetypes.

### 2. Compilation Error: `no matching constructor for initialization of 'Context'`

**Problem**: World data class doesn't inherit from `madrona::WorldBase` or constructor signature is wrong.

**Solution**: Ensure your world class inherits from `madrona::WorldBase` and constructor takes `Engine &ctx` as first parameter.

### 3. Template Deduction Issues with ParallelForNode

**Problem**: Not specifying component types that the system operates on.

**Solution**: Systems that iterate over entities must specify component types:
```cpp
// Wrong
builder.addToGraph<madrona::ParallelForNode<Engine, Engine::mySystem>>();

// Correct
builder.addToGraph<madrona::ParallelForNode<Engine, Engine::mySystem, TestComponent>>();
```

### 4. Empty std::array Template Error

**Problem**: System function doesn't match expected signature for component iteration.

**Solution**: System functions must take `Engine &ctx` and component references:
```cpp
// Correct signature
static void mySystem(Engine &ctx, TestComponent &comp) {
    comp.value += 1.0f;
}
```

## Build System Integration

### CMakeLists.txt Setup

Add your test executable to CMakeLists.txt:

```cmake
add_executable(my_ecs_test my_ecs_test.cpp)

target_include_directories(my_ecs_test PRIVATE
    ${CMAKE_SOURCE_DIR}/external/madrona/include
    ${CMAKE_SOURCE_DIR}/external/madrona/src
)

target_link_libraries(my_ecs_test
    PRIVATE
        madrona_mw_cpu
)

set_target_properties(my_ecs_test PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)
```

### Critical Build System Note

**NEVER attempt to compile Madrona ECS code manually** (e.g., with direct `g++` commands). You will encounter errors like:

```
error: static assertion failed: Unimplemented
STATIC_UNIMPLEMENTED();
```

This happens because functions like `rawAllocAligned` are intentionally marked as unimplemented in headers. The actual implementations are provided by Madrona libraries (`madrona_mw_cpu`, etc.) through the build system's linking process.

**Always use the project's CMake build system:**
```bash
make -C build my_ecs_test
# or
./build.sh
```

The build system handles complex linking requirements and provides the correct memory allocation implementations for your target platform (CPU vs GPU, different allocators, etc.).

### Required Includes

```cpp
#include <madrona/mw_cpu.hpp>
#include <madrona/taskgraph_builder.hpp>
#include <madrona/custom_context.hpp>
#include <madrona/ecs.hpp>
```

## Debugging Tips

1. **Enable Debug Flags**: Compile with `MADRONA_ECS_DEBUG_TRACKING` to enable ECS memory tracking
2. **Check Component Registration Order**: Components before archetypes, always
3. **Verify Task Dependencies**: Ensure task graph dependencies are correct
4. **Use Minimal Tests**: Start with simple components and gradually add complexity

## Performance Considerations

- Keep components small and POD-compatible
- Minimize the number of components per archetype for better cache performance
- Use task graph dependencies to ensure proper execution order
- Consider memory layout when designing component structures

## Complete Working Example

See `tests/cpp/test_simple_table.cpp` for a complete working example that demonstrates:
- Proper component and archetype registration
- Task graph setup
- Entity creation and table allocation
- System execution with component iteration

This example successfully creates ECS tables and can be used as a reference implementation.