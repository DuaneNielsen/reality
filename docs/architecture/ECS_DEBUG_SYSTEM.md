# ECS Debug System Architecture

## Overview

The ECS Debug System provides comprehensive debugging capabilities for Madrona Entity Component Systems, featuring automatic type name registration, component value formatting (Python-style `__repr__`), and reverse address lookup. This system enables developers to inspect ECS memory layout, component values, and relationships without manual configuration.

## Core Features

### 1. Automatic Type Name Registration
- **Zero Configuration**: Component and archetype names are automatically extracted using compiler intrinsics
- **Runtime Type Names**: Uses `__PRETTY_FUNCTION__` (Clang/GCC) to extract type names without RTTI
- **Namespace Preservation**: Full type names including namespaces (e.g., `TestECS::HealthComponent`)

### 2. Component Value Formatting (`__repr__` System)
- **Automatic Formatter Registration**: Formatters are registered alongside component types
- **Template Specialization**: Custom formatters via template specialization
- **Rich Context**: Specialized formatters show `Archetype(id)-Component(id) : value` format
- **Clean Defaults**: Default formatter shows `TypeName@address` without dependencies

### 3. Reverse Address Lookup
- **Memory Range Tracking**: Tracks ECS table memory allocations
- **Component Resolution**: Given any component address, provides full ECS context
- **Live Integration**: Works with runtime component queries and iteration

### 4. Tensor Name Reverse Lookup
- **Export Buffer Tracking**: Maps export buffer addresses to tensor names
- **Magic Enum Integration**: Automatic enum-to-string conversion using magic_enum library
- **Clean Debug Output**: Shows Component → TensorName (Address) format
- **Zero Configuration**: Automatic registration during export process

## Architecture Components

### Core C Implementation (`ecs_simple_tracker.c/h`)

```c
// Core data structures
typedef struct {
    uint32_t component_id;
    uint32_t archetype_id;
    uint32_t world_id;
    uint32_t column_idx;
    uint32_t row;
    uint32_t component_size;
    void* column_base;
    char component_name[64];
    char archetype_name[64];
    char formatted_value[256];  // Component __repr__ output
} address_info_t;

// Export buffer mapping for tensor name reverse lookup
typedef struct {
    void* buffer_address;
    char tensor_name[64];
} export_buffer_t;

// Formatter function pointer
typedef const char* (*component_formatter_t)(const void* component_ptr, const void* info);
```

**Key Functions:**
- `simple_tracker_lookup(void* address, address_info_t* info)` - Reverse address lookup
- `simple_tracker_register_component_formatter(uint32_t id, component_formatter_t formatter)` - Register formatters
- `simple_tracker_format_component_value(void* address)` - Format component value
- `simple_tracker_register_export_buffer(void* buffer_address, const char* tensor_name)` - Register export buffer
- `simple_tracker_lookup_export_tensor_name(void* buffer_address)` - Lookup tensor name from buffer address

### C++ Integration (`state.inl` and `registry.inl`)

**Tensor Name Export Registration:**
```cpp
// Automatic tensor name registration using magic_enum
template <typename ArchetypeT, typename ComponentT, EnumType EnumT>
void ECSRegistry::exportColumn(EnumT slot)
{
    void* buffer_ptr = state_mgr_->exportColumn<ArchetypeT, ComponentT>();
    export_ptrs_[static_cast<uint32_t>(slot)] = buffer_ptr;

#ifdef MADRONA_ECS_DEBUG_TRACKING
    // Use magic_enum to automatically get tensor name from enum value
    auto tensor_name = magic_enum::enum_name(slot);
    if (!tensor_name.empty()) {
        simple_tracker_register_export_buffer(buffer_ptr, std::string(tensor_name).c_str());
    }
#endif
}
```

**Automatic Type Name Extraction:**
```cpp
template<typename T>
const char* extract_type_name() {
#ifdef __clang__
    constexpr const char* sig = __PRETTY_FUNCTION__;
    const char* start = strstr(sig, "[T = ") + 5;
    const char* end = strchr(start, ']');
    static thread_local char buffer[64];
    size_t len = std::min(size_t(end - start), size_t(63));
    strncpy(buffer, start, len);
    buffer[len] = '\0';
    return buffer;
#endif
}
```

**Automatic Registration Hook:**
```cpp
template <typename ComponentT>
ComponentID StateManager::registerComponent(uint32_t num_bytes) {
    // ... existing registration logic ...

#ifdef MADRONA_ECS_DEBUG_TRACKING
    simple_tracker_register_component_type(
        id,
        extract_type_name<ComponentT>(),  // 🎯 AUTOMATIC TYPE NAME!
        sizeof(ComponentT),
        alignof(ComponentT)
    );
    register_component_formatter<ComponentT>();  // 🎯 AUTOMATIC FORMATTER REGISTRATION!
#endif

    return ComponentID { id };
}
```

## Formatter System Design

### Default Formatter Template
```cpp
template<typename ComponentT>
const char* format_component_default(const void* ptr, const void* info_ptr) {
    static thread_local char buffer[256];
    (void)info_ptr; // Keep simple to avoid header dependencies

    // Simple default format
    snprintf(buffer, sizeof(buffer), "%s@%p", extract_type_name<ComponentT>(), ptr);
    return buffer;
}
```

**Output:** `TestECS::Position@0x648368e040f0`

### Specialized Formatter (Example)
```cpp
// Must be defined outside namespaces
template<>
const char* format_component_default<TestECS::HealthComponent>(const void* ptr, const void* info_ptr) {
    static thread_local char buffer[128];
    const auto& health = *static_cast<const TestECS::HealthComponent*>(ptr);
    const auto* info = static_cast<const address_info_t*>(info_ptr);

    snprintf(buffer, sizeof(buffer), "%s(%u)-%s(%u) : HealthComponent{currentHealth=%.1f}",
             info->archetype_name, info->archetype_id,
             info->component_name, info->component_id, health.currentHealth);
    return buffer;
}
```

**Output:** `TestECS::GameEntity(0)-TestECS::HealthComponent(2) : HealthComponent{currentHealth=101.0}`

## Usage Examples

### Tensor Name Debug Output
The system automatically displays tensor names in export debug output:

```
ECS EXPORT: copying 11 components
  madrona::phys::PhysicsSystemState -> Action (0x70d0b8be8000): 4 rows, 12 bytes/row
  madrona::phys::xpbd::XPBDContactState -> Reward (0x70cfca535000): 4 rows, 4 bytes/row
  madrona::phys::xpbd::SolverState -> Done (0x70cedbe82000): 4 rows, 4 bytes/row
  madrona::render::Renderable -> TerminationReason (0x70cded7cf000): 4 rows, 4 bytes/row
  madrona::phys::ObjectData -> SelfObservation (0x70cd45652000): 4 rows, 20 bytes/row
  madrona::base::Position -> AgentPosition (0x70ca43ba2000): 4 rows, 12 bytes/row
  madrona::base::Position -> TargetPosition (0x70c97878a000): 4 rows, 12 bytes/row
```

**Format:** `ComponentType -> TensorName (BufferAddress): Details`

This provides complete data flow traceability from ECS components to their exported tensor names.

### Application Integration
```cpp
// Application code - uses enum values directly
enum class ExportID : uint32_t {
    Action, Reward, Done, SelfObservation, // etc...
};

// Export registration automatically registers tensor names
registry.exportColumn<Agent, Action>(ExportID::Action);  // magic_enum converts to "Action"
registry.exportColumn<Agent, Reward>(ExportID::Reward);  // magic_enum converts to "Reward"
```

### Basic Component Inspection
```cpp
// During ECS iteration
world_ctx.iterateQuery(query, [&](TestECS::HealthComponent &health) {
    void* comp_addr = &health;

    // Get formatted value
    const char* formatted = simple_tracker_format_component_value(comp_addr);
    std::cout << "Component: " << formatted << std::endl;

    // Get full debug info
    address_info_t info;
    if (simple_tracker_lookup(comp_addr, &info)) {
        std::cout << "Archetype: " << info.archetype_name << " (ID: " << info.archetype_id << ")" << std::endl;
        std::cout << "World: " << info.world_id << ", Row: " << info.row << std::endl;
    }
});
```

### Custom Formatter Implementation
```cpp
namespace MyGame {
    struct Transform { float x, y, z, rotation; };
}

// Specialized formatter (outside namespace)
template<>
const char* format_component_default<MyGame::Transform>(const void* ptr, const void* info_ptr) {
    static thread_local char buffer[256];
    const auto& transform = *static_cast<const MyGame::Transform*>(ptr);
    const auto* info = static_cast<const address_info_t*>(info_ptr);

    snprintf(buffer, sizeof(buffer), "%s(%u)-%s(%u) : Transform{pos=(%.2f,%.2f,%.2f), rot=%.2f}",
             info->archetype_name, info->archetype_id,
             info->component_name, info->component_id,
             transform.x, transform.y, transform.z, transform.rotation);
    return buffer;
}
```

## Memory Tracking Architecture

### Range-Based Tracking
The system tracks ECS table memory allocations as ranges:

```c
typedef struct {
    uintptr_t start;
    uintptr_t end;
    uint32_t component_id;
    uint32_t archetype_id;
    uint32_t world_id;
    uint32_t column_idx;
    uint32_t component_size;
    uint32_t num_rows;
    void* base_address;
} address_range_t;
```

### Lookup Algorithm
1. **Linear Search**: O(n) search through tracked ranges (acceptable for debug builds)
2. **Address Bounds Check**: `addr >= range->start && addr < range->end`
3. **Row Calculation**: `row = (addr - range->start) / range->component_size`
4. **Context Assembly**: Combines memory info with type metadata

## Integration Points

### ECS Registration Hooks
- `StateManager::registerComponent<T>()` - Component type registration
- `StateManager::registerArchetype<T>()` - Archetype type registration

### Madrona Integration
- Enabled with `MADRONA_ECS_DEBUG_TRACKING` compile flag
- Automatic initialization during ECS setup
- Thread-safe operation with simple locking

### Magic Enum Integration
- **Header-only library**: `external/madrona/include/madrona/magic_enum.hpp`
- **C++17 requirement**: Uses compile-time template metaprogramming
- **No RTTI dependency**: Works with RTTI disabled projects
- **Compile-time reflection**: Zero runtime overhead for enum-to-string conversion

### Build System
- Debug-only compilation (removed in release builds)
- No runtime overhead when disabled
- Optional dependency - graceful degradation

## Performance Characteristics

### Debug Build Impact
- **Registration**: O(1) per component/archetype type
- **Lookup**: O(n) linear search through ranges
- **Formatting**: O(1) template instantiation per type
- **Memory**: ~612 bytes for 4 component types (test case)

### Release Build
- Complete removal via `#ifdef MADRONA_ECS_DEBUG_TRACKING`
- Zero runtime overhead
- No binary size impact

## Testing Strategy

### Comprehensive Test Coverage
Located in `tests/cpp/test_simple_table.cpp`:

1. **Automatic Registration**: Verifies type names extracted correctly
2. **Specialized Formatters**: Tests rich `Archetype(id)-Component(id) : value` format
3. **Default Formatters**: Tests simple `TypeName@address` fallback
4. **Reverse Lookup**: Validates address-to-component resolution
5. **Multi-Component Archetypes**: Tests complex ECS layouts
6. **Error Handling**: Validates behavior with invalid addresses

### Test Output Examples
```
🔍 Testing component #1:
  Address: 0x648368e01450
  Health: 101
  Formatted Value: TestECS::GameEntity(0)-TestECS::HealthComponent(2) : HealthComponent{currentHealth=101.0}
  ✅ REVERSE LOOKUP SUCCESS:
    Component: TestECS::HealthComponent (ID: 2)
    Value: TestECS::GameEntity(0)-TestECS::HealthComponent(2) : HealthComponent{currentHealth=101.0}
    Archetype: TestECS::GameEntity (ID: 0)
```

## Design Principles

### 1. **Zero Configuration**
- Automatic registration during normal ECS usage
- No manual type registration required
- Works with existing Madrona ECS patterns

### 2. **Clean Separation**
- Default formatters: Simple, dependency-free
- Specialized formatters: Rich context when needed
- No header conflicts or brittle dependencies

### 3. **Template-Based Extensibility**
- Easy to add custom formatters via template specialization
- Type-safe formatter registration
- Compile-time formatter selection

### 4. **Debug-Only Impact**
- Complete removal in release builds
- Thread-local buffers for safety
- Graceful fallbacks for unsupported compilers

### 5. **Integration Friendly**
- Works with existing ECS queries and iteration
- Compatible with component address patterns
- Minimal API surface

### 6. **Tensor Name Traceability**
- Automatic enum-to-string conversion via magic_enum
- Zero-configuration export buffer registration
- Clear data flow visibility from components to tensors
- Framework-agnostic design using template specialization

## Future Enhancements

### Potential Improvements
1. **Namespace Cleaning**: Strip verbose namespaces for cleaner output
2. **MSVC Support**: Add Microsoft Visual C++ function signature parsing
3. **Custom Name Override**: Allow manual override for specific types
4. **Performance Optimization**: Hash table for faster address lookup
5. **Serialization Support**: Export debug info to external tools

### Extension Points
1. **Custom Formatter Interfaces**: Beyond template specialization
2. **Multi-World Debugging**: Enhanced support for multiple worlds
3. **Component Relationship Visualization**: Show archetype composition
4. **Memory Usage Analytics**: Track allocation patterns and growth

## Conclusion

The ECS Debug System provides a comprehensive, zero-configuration debugging solution for Madrona ECS applications. By leveraging compiler intrinsics, template metaprogramming, and careful architectural design, it delivers rich debugging capabilities without impacting production performance or requiring manual setup.

The system's design prioritizes developer experience while maintaining clean separation between debug and production code paths. The automatic type name registration and component formatting capabilities significantly improve the debugging experience for complex ECS applications.