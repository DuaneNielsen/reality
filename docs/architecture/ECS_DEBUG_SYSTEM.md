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

// Formatter function pointer
typedef const char* (*component_formatter_t)(const void* component_ptr, const void* info);
```

**Key Functions:**
- `simple_tracker_lookup(void* address, address_info_t* info)` - Reverse address lookup
- `simple_tracker_register_component_formatter(uint32_t id, component_formatter_t formatter)` - Register formatters
- `simple_tracker_format_component_value(void* address)` - Format component value

### C++ Integration (`state.inl`)

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