# Plan: Register Export Buffer Addresses for Tensor Name Reverse Lookup

## Pre-Reading List

Before implementing this plan, review these key files to understand the architecture:

1. **`/home/duane/madroom/docs/architecture/ECS_DEBUG_SYSTEM.md`** - Complete ECS debug system architecture and API
2. **`/home/duane/madroom/external/madrona/include/madrona/registry.inl:47-50`** - Current export registration flow
3. **`/home/duane/madroom/src/sim.hpp:23-39`** - ExportID enum with tensor names
4. **`/home/duane/madroom/src/sim.cpp:94-117`** - Example export registrations
5. **`/home/duane/madroom/external/madrona/src/debug/ecs_simple_tracker.h`** - Current debug tracker API
6. **`/home/duane/madroom/external/madrona/src/core/state.cpp:587-618`** - Current simplified debug output

## Goal

Enable the ECS debug system to reverse lookup export buffer addresses (`export_job.mem.ptr()`) to display **tensor names** (Action, Reward, SelfObservation) instead of component type names.

## Current Problem

```
ECS EXPORT: copying 11 components
  madrona::phys::PhysicsSystemState: 2 rows, 12 bytes/row -> 0x7f5b9654b000
```

We see component types but not the **export/tensor names** these map to.

## Desired Output

```
ECS EXPORT: copying 11 components
  Action: 2 rows, 16 bytes/row -> 0x7f5b9654b000
  Reward: 2 rows, 4 bytes/row -> 0x7f5b9654b004
  SelfObservation: 2 rows, 60 bytes/row -> 0x7f5b9654b008
```

## Implementation Strategy: Option B - ECS Debug System Registration

Register export buffer addresses with meaningful tensor names using the ECS debug system.

### Step 1: Create Slot-to-Name Mapping Function

**File**: `external/madrona/include/madrona/registry.inl` or `registry.hpp`

Add a helper function to convert slot IDs to readable names:
```cpp
#ifdef MADRONA_ECS_DEBUG_TRACKING
const char* getExportSlotName(uint32_t slot) {
    // This could be a lookup table, enum-to-string conversion,
    // or application-specific mapping
    static char buffer[64];
    snprintf(buffer, sizeof(buffer), "ExportSlot_%u", slot);
    return buffer;
}
#endif
```

### Step 2: Extend ECS Debug System API

**File**: `external/madrona/src/debug/ecs_simple_tracker.h`

Add function to register export buffers:
```cpp
#ifdef MADRONA_ECS_DEBUG_TRACKING
void simple_tracker_register_export_buffer(
    void* buffer_address,
    const char* tensor_name,
    uint32_t buffer_size
);
#endif
```

**File**: `external/madrona/src/debug/ecs_simple_tracker.c`

Implement the registration function that stores buffer address with tensor name.

### Step 3: Register Export Buffers During Export

**File**: `external/madrona/include/madrona/registry.inl:47-50`

Modify `ECSRegistry::exportColumn(int32_t slot)`:
```cpp
template <typename ArchetypeT, typename ComponentT>
void ECSRegistry::exportColumn(int32_t slot)
{
    void* buffer_ptr = state_mgr_->exportColumn<ArchetypeT, ComponentT>();
    export_ptrs_[slot] = buffer_ptr;

#ifdef MADRONA_ECS_DEBUG_TRACKING
    // Register this buffer with its tensor name for reverse lookup
    const char* tensor_name = getExportSlotName(slot);
    simple_tracker_register_export_buffer(buffer_ptr, tensor_name, /* size info */);
#endif
}
```

### Step 4: Enhance Debug Output with Tensor Names

**File**: `external/madrona/src/core/state.cpp:594-618`

Modify the debug output in `copyOutExportedColumns()`:
```cpp
#ifdef MADRONA_ECS_DEBUG_TRACKING
// Try to get tensor name from export buffer registration
const char* tensor_name = nullptr;
address_info_t export_info;
if (simple_tracker_lookup(export_job.mem.ptr(), &export_info)) {
    tensor_name = export_info.component_name;  // Will be our tensor name
}

// Use tensor name if available, fallback to component name
const char* display_name = tensor_name ? tensor_name : component_name;

printf("  %s: %u rows, %u bytes/row -> %p\n",
       display_name, (uint32_t)total_rows, export_job.numBytesPerRow, export_job.mem.ptr());
#endif
```

## Alternative Name Resolution Options

### Option 1: Hardcoded Mapping
Create a compile-time mapping from common slot values to names.

### Option 2: Application-Provided Callback
Allow applications to register a slot-to-name conversion function.

### Option 3: Enum Reflection
If available, use compile-time enum reflection to convert `ExportID` values to strings.

## Files to Modify

1. **`external/madrona/src/debug/ecs_simple_tracker.h`** - Add export buffer registration API
2. **`external/madrona/src/debug/ecs_simple_tracker.c`** - Implement registration function
3. **`external/madrona/include/madrona/registry.inl`** - Register buffers during export
4. **`external/madrona/src/core/state.cpp`** - Use tensor names in debug output

## Expected Results

- Clear tensor name visibility in export debug output
- Complete data flow traceability: ECS Component → Export Buffer → Tensor Name
- Preserved fallback to component names when tensor names unavailable
- Zero runtime overhead when debug tracking disabled

## Testing Approach

1. Build with `MADRONA_ECS_DEBUG_TRACKING` enabled
2. Run Python simulation to trigger exports
3. Verify debug output shows tensor names like "Action", "Reward"
4. Confirm fallback behavior for unregistered components