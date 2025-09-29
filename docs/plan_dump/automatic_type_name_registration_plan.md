# Plan: Simple Runtime Type Name Hook for ECS Debug Tracker

## Problem Statement
The current ECS debug tracker requires manual registration because line 84 in `state.inl` hardcodes:
```cpp
simple_tracker_register_component_type(
    id,
    "Component",  // Can't use typeid without RTTI
    sizeof(ComponentT),
    alignof(ComponentT)
);
```

The comment reveals the core issue: **Madrona disables RTTI**, so `typeid(ComponentT).name()` isn't available. However, this is misleading - C++ provides multiple runtime alternatives for type name extraction.

## Current Hook Point (Perfect!)
The registration is already hooked in `external/madrona/include/madrona/state.inl` at line 82-87:

```cpp
#ifdef MADRONA_ECS_DEBUG_TRACKING
    simple_tracker_register_component_type(
        id,
        "Component",  // <-- REPLACE THIS LINE!
        sizeof(ComponentT),
        alignof(ComponentT)
    );
#endif
```

## Solution: Direct Runtime Type Name Extraction

### 1. **Add Type Name Utility Function**
Add to `external/madrona/include/madrona/state.inl` before the template:

```cpp
#ifdef MADRONA_ECS_DEBUG_TRACKING
template<typename T>
const char* extract_type_name() {
#ifdef __clang__
    constexpr const char* sig = __PRETTY_FUNCTION__;
    // Extract from: "const char* extract_type_name() [T = MyType]"
    const char* start = strstr(sig, "[T = ") + 5;
    const char* end = strchr(start, ']');
    static thread_local char buffer[64];
    size_t len = std::min(size_t(end - start), size_t(63));
    strncpy(buffer, start, len);
    buffer[len] = '\0';
    return buffer;
#elif defined(__GNUC__)
    constexpr const char* sig = __PRETTY_FUNCTION__;
    // Extract from: "const char* extract_type_name() [with T = MyType]"
    const char* start = strstr(sig, "[with T = ") + 10;
    const char* end = strchr(start, ']');
    static thread_local char buffer[64];
    size_t len = std::min(size_t(end - start), size_t(63));
    strncpy(buffer, start, len);
    buffer[len] = '\0';
    return buffer;
#else
    return "Component"; // Fallback for unsupported compilers
#endif
}
#endif
```

### 2. **Replace the Hardcoded String**
Change line 84 from:
```cpp
"Component",  // Can't use typeid without RTTI
```

To:
```cpp
extract_type_name<ComponentT>(),  // 🎯 AUTOMATIC TYPE NAME!
```

### 3. **Add Archetype Registration Hook**
Add similar hook in the `registerArchetype` template (around line 170):

```cpp
#ifdef MADRONA_ECS_DEBUG_TRACKING
    simple_tracker_register_archetype_type(
        TypeTracker::typeID<ArchetypeT>(),
        extract_type_name<ArchetypeT>()
    );
#endif
```

### 4. **Remove Manual Registration from Test**
Delete the manual registration calls in `tests/cpp/test_simple_table.cpp`:

```cpp
// DELETE THESE LINES:
simple_tracker_register_component_type(0, "Entity", 8, 8);
simple_tracker_register_component_type(1, "UnknownComponent1", 4, 4);
simple_tracker_register_component_type(2, "HealthComponent", sizeof(float), alignof(float));
simple_tracker_register_archetype_type(0, "GameEntity");
```

## Technical Details

### C++ Runtime Type Name Capabilities (No RTTI Required)

1. **`__PRETTY_FUNCTION__` / `__FUNCSIG__` (Compiler Built-ins)**
   - Available at runtime during function execution
   - Contains full template parameter information
   - Works with GCC, Clang, MSVC

2. **Function Signature Parsing**
   - Extract type names from compiler-generated function signatures
   - Simple string parsing at runtime
   - No compile-time complexity required

3. **Thread-Local Storage**
   - Use `thread_local` buffer to store extracted names
   - Avoids memory allocation issues
   - Safe for concurrent access

## Benefits

1. **🎯 Zero Manual Work**: Names extracted automatically during ECS registration
2. **📍 Perfect Hook Point**: Uses existing registration call - no new hooks needed
3. **🚀 Simple Implementation**: Just replace one line + add utility function
4. **⚡ Runtime Available**: Function signatures available at runtime, no compile-time complexity
5. **🛡️ Graceful Fallback**: Falls back to "Component" on unsupported compilers
6. **🔧 Minimal Invasive**: Leverages existing debug infrastructure

## Expected Result

**Before (Manual Registration):**
```
Component: Component (ID: 2)      ← Manual override needed
Archetype: GameEntity (ID: 0)    ← Manual registration needed
```

**After (Automatic Registration):**
```
Component: HealthComponent (ID: 2)  ← 🎯 Automatic from registerComponent<HealthComponent>()!
Archetype: GameEntity (ID: 0)      ← 🎯 Automatic from registerArchetype<GameEntity>()!
```

## Files to Modify

1. `external/madrona/include/madrona/state.inl`
   - Add utility function + replace hardcoded string
   - Add archetype registration hook

2. `tests/cpp/test_simple_table.cpp`
   - Remove manual registration calls
   - Clean up debug output sections

## Implementation Notes

### Error Handling
- Graceful fallback to "Component" if parsing fails
- Bounds checking for buffer operations
- Null pointer protection

### Compiler Support
- **Clang**: `__PRETTY_FUNCTION__` with `[T = TypeName]` format
- **GCC**: `__PRETTY_FUNCTION__` with `[with T = TypeName]` format
- **MSVC**: `__FUNCSIG__` with different format (can be added later)
- **Others**: Fallback to generic "Component"

### Performance Impact
- Minimal runtime overhead (string parsing during registration only)
- No impact on hot paths (registration happens at startup)
- Thread-local storage avoids allocation overhead

## Future Enhancements

1. **Namespace Cleaning**: Strip namespaces (`TestECS::HealthComponent` → `HealthComponent`)
2. **Template Parameter Handling**: Clean up template types
3. **MSVC Support**: Add Microsoft Visual C++ function signature parsing
4. **Custom Name Override**: Allow manual override for specific types if needed

## Validation

Test with various type names:
- Simple types: `HealthComponent`
- Namespaced types: `TestECS::HealthComponent`
- Template types: `Optional<int>`
- Nested types: `Outer::Inner::Component`

**That's it!** This leverages the existing hook point perfectly and requires minimal code changes while providing automatic type name registration for much more readable ECS debugging.