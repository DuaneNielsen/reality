/*
 * Copyright 2021-2022 Brennan Shacklett and contributors
 *
 * Use of this source code is governed by an MIT-style
 * license that can be found in the LICENSE file or at
 * https://opensource.org/licenses/MIT.
 */
#include "ecs_simple_tracker.h"

#ifdef MADRONA_ECS_DEBUG_TRACKING

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// Export buffer mapping for tensor name reverse lookup
typedef struct {
    void* buffer_address;
    char tensor_name[MAX_TYPE_NAME_LEN];
    uint32_t buffer_size;
} export_buffer_t;

// Global storage - simple arrays for minimal overhead
static address_range_t g_ranges[MAX_RANGES];
static component_type_t g_components[MAX_COMPONENTS];
static archetype_type_t g_archetypes[MAX_ARCHETYPES];
static export_buffer_t g_export_buffers[MAX_EXPORT_BUFFERS];
static uint32_t g_range_count = 0;
static uint32_t g_component_count = 0;
static uint32_t g_archetype_count = 0;
static uint32_t g_export_buffer_count = 0;

// Simple mutex simulation - for now just use a flag
static volatile int g_lock = 0;

static void simple_lock(void) {
    while (__sync_lock_test_and_set(&g_lock, 1)) {
        // spin
    }
}

static void simple_unlock(void) {
    __sync_lock_release(&g_lock);
}

int simple_tracker_lookup(void* address, address_info_t* info) {
    if (!address || !info) return 0;

    simple_lock();

    uintptr_t addr = (uintptr_t)address;

    // Linear search through ranges (fine for debug builds)
    for (uint32_t i = 0; i < g_range_count; i++) {
        const address_range_t* range = &g_ranges[i];
        if (addr >= range->start && addr < range->end) {
            // Found it!
            info->component_id = range->component_id;
            info->archetype_id = range->archetype_id;
            info->world_id = range->world_id;
            info->column_idx = range->column_idx;
            info->component_size = range->component_size;
            info->column_base = range->base_address;

            // Calculate row
            uintptr_t offset = addr - range->start;
            info->row = (uint32_t)(offset / range->component_size);

            // Look up component type name and formatter
            const char* comp_name = "Unknown";
            component_formatter_t formatter = NULL;
            for (uint32_t j = 0; j < g_component_count; j++) {
                if (g_components[j].component_id == range->component_id) {
                    comp_name = g_components[j].type_name;
                    formatter = g_components[j].formatter;
                    break;
                }
            }
            strncpy(info->component_name, comp_name, MAX_TYPE_NAME_LEN - 1);
            info->component_name[MAX_TYPE_NAME_LEN - 1] = '\0';

            // Look up archetype type name first (needed for formatter)
            const char* archetype_name = "Unknown";
            for (uint32_t k = 0; k < g_archetype_count; k++) {
                if (g_archetypes[k].archetype_id == range->archetype_id) {
                    archetype_name = g_archetypes[k].archetype_name;
                    break;
                }
            }
            strncpy(info->archetype_name, archetype_name, MAX_TYPE_NAME_LEN - 1);
            info->archetype_name[MAX_TYPE_NAME_LEN - 1] = '\0';

            // Format component value if formatter available
            if (formatter) {
                const char* formatted = formatter(address, (const void*)info);
                strncpy(info->formatted_value, formatted ? formatted : "<formatter error>",
                       sizeof(info->formatted_value) - 1);
                info->formatted_value[sizeof(info->formatted_value) - 1] = '\0';
            } else {
                snprintf(info->formatted_value, sizeof(info->formatted_value),
                        "<%s at %p (no formatter)>", comp_name, address);
            }

            simple_unlock();
            return 1; // Found
        }
    }

    simple_unlock();
    return 0; // Not found
}

void simple_tracker_register_range(
    void* base_address, size_t total_bytes,
    uint32_t archetype_id, uint32_t world_id, uint32_t column_idx,
    uint32_t component_id, uint32_t component_size, uint32_t num_rows) {

    if (!base_address || g_range_count >= MAX_RANGES) return;

    simple_lock();

    address_range_t* range = &g_ranges[g_range_count];
    range->start = (uintptr_t)base_address;
    range->end = range->start + total_bytes;
    range->component_id = component_id;
    range->archetype_id = archetype_id;
    range->world_id = world_id;
    range->column_idx = column_idx;
    range->component_size = component_size;
    range->num_rows = num_rows;
    range->base_address = base_address;

    g_range_count++;

    simple_unlock();
}

void simple_tracker_unregister_range(void* base_address) {
    if (!base_address) return;

    simple_lock();

    uintptr_t addr = (uintptr_t)base_address;

    // Find and remove the range
    for (uint32_t i = 0; i < g_range_count; i++) {
        if (g_ranges[i].start == addr) {
            // Move the last range into this slot
            if (i < g_range_count - 1) {
                g_ranges[i] = g_ranges[g_range_count - 1];
            }
            g_range_count--;
            break;
        }
    }

    simple_unlock();
}

void simple_tracker_register_component_type(
    uint32_t component_id, const char* type_name, uint32_t size, uint32_t alignment) {

    if (!type_name || g_component_count >= MAX_COMPONENTS) return;

    simple_lock();

    // Check if already exists - if so, update the name if it's more specific
    for (uint32_t i = 0; i < g_component_count; i++) {
        if (g_components[i].component_id == component_id) {
            // Update if the new name is more specific than the existing generic name
            if (strcmp(g_components[i].type_name, "Component") == 0 &&
                strcmp(type_name, "Component") != 0) {
                strncpy(g_components[i].type_name, type_name, MAX_TYPE_NAME_LEN - 1);
                g_components[i].type_name[MAX_TYPE_NAME_LEN - 1] = '\0';
                g_components[i].size = size;
                g_components[i].alignment = alignment;
            }
            simple_unlock();
            return; // Already registered (possibly updated)
        }
    }

    component_type_t* comp = &g_components[g_component_count];
    comp->component_id = component_id;
    comp->size = size;
    comp->alignment = alignment;
    comp->formatter = NULL;  // Initialize formatter to NULL
    strncpy(comp->type_name, type_name, MAX_TYPE_NAME_LEN - 1);
    comp->type_name[MAX_TYPE_NAME_LEN - 1] = '\0';

    g_component_count++;

    simple_unlock();
}

void simple_tracker_register_archetype_type(
    uint32_t archetype_id, const char* archetype_name) {

    if (!archetype_name || g_archetype_count >= MAX_ARCHETYPES) return;

    simple_lock();

    // Check if already exists
    for (uint32_t i = 0; i < g_archetype_count; i++) {
        if (g_archetypes[i].archetype_id == archetype_id) {
            simple_unlock();
            return; // Already registered
        }
    }

    archetype_type_t* archetype = &g_archetypes[g_archetype_count];
    archetype->archetype_id = archetype_id;
    strncpy(archetype->archetype_name, archetype_name, MAX_TYPE_NAME_LEN - 1);
    archetype->archetype_name[MAX_TYPE_NAME_LEN - 1] = '\0';

    g_archetype_count++;

    simple_unlock();
}

void simple_tracker_register_component_formatter(
    uint32_t component_id, component_formatter_t formatter) {

    if (!formatter) return;

    simple_lock();

    // Find the component and add the formatter
    for (uint32_t i = 0; i < g_component_count; i++) {
        if (g_components[i].component_id == component_id) {
            g_components[i].formatter = formatter;
            simple_unlock();
            return;
        }
    }

    simple_unlock();
    // Component not found - this is OK, formatter may be registered before type
}

void simple_tracker_print_memory_map(void) {
    simple_lock();

    printf("=== Simple ECS Memory Map ===\n");
    printf("Tracked ranges: %u\n", g_range_count);
    printf("Registered components: %u\n", g_component_count);
    printf("\n");

    for (uint32_t i = 0; i < g_range_count; i++) {
        const address_range_t* range = &g_ranges[i];
        printf("Range %u: 0x%lx - 0x%lx (size: %lu)\n",
               i, range->start, range->end, range->end - range->start);
        printf("  Archetype: %u, World: %u, Column: %u, Component: %u\n",
               range->archetype_id, range->world_id, range->column_idx, range->component_id);
        printf("  Component size: %u, Rows: %u\n",
               range->component_size, range->num_rows);
        printf("\n");
    }

    simple_unlock();
}

void simple_tracker_print_statistics(void) {
    simple_lock();

    printf("=== Simple ECS Tracker Statistics ===\n");
    printf("Tracked ranges: %u / %u\n", g_range_count, MAX_RANGES);
    printf("Component types: %u / %u\n", g_component_count, MAX_COMPONENTS);
    printf("Archetype types: %u / %u\n", g_archetype_count, MAX_ARCHETYPES);

    size_t memory_usage = g_range_count * sizeof(address_range_t) +
                         g_component_count * sizeof(component_type_t) +
                         g_archetype_count * sizeof(archetype_type_t);
    printf("Memory usage: %zu bytes\n", memory_usage);

    simple_unlock();
}

uint32_t simple_tracker_get_range_count(void) {
    simple_lock();
    uint32_t count = g_range_count;
    simple_unlock();
    return count;
}

void simple_tracker_dump_ranges(void) {
    simple_lock();

    printf("=== Debug: All Ranges ===\n");
    for (uint32_t i = 0; i < g_range_count; i++) {
        const address_range_t* range = &g_ranges[i];
        printf("Range[%u]: start=0x%lx end=0x%lx comp=%u arch=%u world=%u\n",
               i, range->start, range->end, range->component_id,
               range->archetype_id, range->world_id);
    }

    simple_unlock();
}

int simple_tracker_validate_integrity(void) {
    simple_lock();

    int errors = 0;

    // Check for overlapping ranges
    for (uint32_t i = 0; i < g_range_count; i++) {
        for (uint32_t j = i + 1; j < g_range_count; j++) {
            const address_range_t* r1 = &g_ranges[i];
            const address_range_t* r2 = &g_ranges[j];

            if ((r1->start < r2->end && r1->end > r2->start)) {
                printf("ERROR: Overlapping ranges %u and %u\n", i, j);
                errors++;
            }
        }
    }

    printf("Integrity check: %d errors found\n", errors);

    simple_unlock();
    return errors;
}

const char* simple_tracker_format_component_value(void* address) {
    static char fallback_buffer[128];

    if (!address) return "NULL";

    address_info_t info;
    if (!simple_tracker_lookup(address, &info)) {
        snprintf(fallback_buffer, sizeof(fallback_buffer), "<unknown address: %p>", address);
        return fallback_buffer;
    }

    simple_lock();

    // Find the component type and use its formatter
    for (uint32_t i = 0; i < g_component_count; i++) {
        if (g_components[i].component_id == info.component_id) {
            if (g_components[i].formatter) {
                const char* result = g_components[i].formatter(address, (const void*)&info);
                simple_unlock();
                return result ? result : "<formatter error>";
            } else {
                snprintf(fallback_buffer, sizeof(fallback_buffer),
                    "<%s at %p (no formatter)>", g_components[i].type_name, address);
                simple_unlock();
                return fallback_buffer;
            }
        }
    }

    simple_unlock();
    snprintf(fallback_buffer, sizeof(fallback_buffer), "<unknown component type %u at %p>",
        info.component_id, address);
    return fallback_buffer;
}

void simple_tracker_print_component_value(void* address) {
    printf("Component value: %s\n", simple_tracker_format_component_value(address));
}

void simple_tracker_register_export_buffer(
    void* buffer_address,
    const char* tensor_name,
    uint32_t buffer_size) {

    if (!buffer_address || !tensor_name || g_export_buffer_count >= MAX_EXPORT_BUFFERS) return;

    simple_lock();

    // Check if already exists - update if so
    for (uint32_t i = 0; i < g_export_buffer_count; i++) {
        if (g_export_buffers[i].buffer_address == buffer_address) {
            strncpy(g_export_buffers[i].tensor_name, tensor_name, MAX_TYPE_NAME_LEN - 1);
            g_export_buffers[i].tensor_name[MAX_TYPE_NAME_LEN - 1] = '\0';
            g_export_buffers[i].buffer_size = buffer_size;
            simple_unlock();
            return;
        }
    }

    // Add new entry
    export_buffer_t* buffer_entry = &g_export_buffers[g_export_buffer_count];
    buffer_entry->buffer_address = buffer_address;
    buffer_entry->buffer_size = buffer_size;
    strncpy(buffer_entry->tensor_name, tensor_name, MAX_TYPE_NAME_LEN - 1);
    buffer_entry->tensor_name[MAX_TYPE_NAME_LEN - 1] = '\0';

    g_export_buffer_count++;

    simple_unlock();
}

const char* simple_tracker_lookup_export_tensor_name(void* buffer_address) {
    if (!buffer_address) return NULL;

    simple_lock();

    for (uint32_t i = 0; i < g_export_buffer_count; i++) {
        if (g_export_buffers[i].buffer_address == buffer_address) {
            const char* result = g_export_buffers[i].tensor_name;
            simple_unlock();
            return result;
        }
    }

    simple_unlock();
    return NULL; // Not found
}

#endif // MADRONA_ECS_DEBUG_TRACKING