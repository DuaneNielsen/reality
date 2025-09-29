/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#pragma once

#ifdef MADRONA_ECS_DEBUG_TRACKING

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

#define MAX_RANGES 4096
#define MAX_TYPE_NAME_LEN 64
#define MAX_COMPONENTS 256
#define MAX_ARCHETYPES 256
#define MAX_EXPORT_BUFFERS 64

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

typedef struct {
    uint32_t component_id;
    uint32_t archetype_id;
    uint32_t world_id;
    uint32_t column_idx;
    uint32_t row;
    uint32_t component_size;
    void* column_base;
    char component_name[MAX_TYPE_NAME_LEN];
    char archetype_name[MAX_TYPE_NAME_LEN];
    char formatted_value[256];  // Formatted component value
} address_info_t;

// Function pointer type for component value formatters
// Passes the complete address info for rich formatting context
typedef const char* (*component_formatter_t)(const void* component_ptr, const void* info);

typedef struct {
    char type_name[MAX_TYPE_NAME_LEN];
    uint32_t component_id;
    uint32_t size;
    uint32_t alignment;
    component_formatter_t formatter;  // Optional formatter function
} component_type_t;

typedef struct {
    char archetype_name[MAX_TYPE_NAME_LEN];
    uint32_t archetype_id;
} archetype_type_t;

// Core functions
int simple_tracker_lookup(void* address, address_info_t* info);
void simple_tracker_register_range(
    void* base_address, size_t total_bytes,
    uint32_t archetype_id, uint32_t world_id, uint32_t column_idx,
    uint32_t component_id, uint32_t component_size, uint32_t num_rows);
void simple_tracker_unregister_range(void* base_address);

// Type registration
void simple_tracker_register_component_type(
    uint32_t component_id, const char* type_name, uint32_t size, uint32_t alignment);
void simple_tracker_register_component_formatter(
    uint32_t component_id, component_formatter_t formatter);
void simple_tracker_register_archetype_type(
    uint32_t archetype_id, const char* archetype_name);

// Export buffer registration for tensor name reverse lookup
void simple_tracker_register_export_buffer(
    void* buffer_address,
    const char* tensor_name,
    uint32_t buffer_size);
const char* simple_tracker_lookup_export_tensor_name(void* buffer_address);

// Statistics and debugging
void simple_tracker_print_memory_map(void);
void simple_tracker_print_statistics(void);
uint32_t simple_tracker_get_range_count(void);
const char* simple_tracker_format_component_value(void* address);
void simple_tracker_print_component_value(void* address);

// Test functions for debugging the tracker itself
void simple_tracker_dump_ranges(void);
int simple_tracker_validate_integrity(void);

#ifdef __cplusplus
}
#endif

#endif // MADRONA_ECS_DEBUG_TRACKING