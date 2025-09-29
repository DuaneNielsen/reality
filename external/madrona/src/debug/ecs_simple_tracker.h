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
} address_info_t;

typedef struct {
    char type_name[MAX_TYPE_NAME_LEN];
    uint32_t component_id;
    uint32_t size;
    uint32_t alignment;
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
void simple_tracker_register_archetype_type(
    uint32_t archetype_id, const char* archetype_name);

// Statistics and debugging
void simple_tracker_print_memory_map(void);
void simple_tracker_print_statistics(void);
uint32_t simple_tracker_get_range_count(void);

// Test functions for debugging the tracker itself
void simple_tracker_dump_ranges(void);
int simple_tracker_validate_integrity(void);

#ifdef __cplusplus
}
#endif

#endif // MADRONA_ECS_DEBUG_TRACKING