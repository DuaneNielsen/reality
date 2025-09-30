/*
 * ECS Export Debug Output Instructions
 *
 * This file documents how to add debug output to the ECS tensor export system
 * to show component-to-tensor mapping with buffer addresses.
 *
 * The debug output shows the format:
 * ComponentType -> TensorName (BufferAddress): rows, bytes/row
 *
 * Example output:
 * ECS EXPORT: copying 11 components
 *   madEscape::Action -> Action (0x7db870be8000): 4 rows, 12 bytes/row
 *   madEscape::Reward -> Reward (0x7db782535000): 4 rows, 4 bytes/row
 *   madrona::base::Position -> AgentPosition (0x7db1fbba2000): 4 rows, 12 bytes/row
 */

// To enable this debug output, add the following code to:
// external/madrona/src/core/state.cpp in the copyOutExportedColumns() function

void StateManager::copyOutExportedColumns()
{
#ifdef MADRONA_MW_MODE

#ifdef MADRONA_ECS_DEBUG_TRACKING
    printf("ECS EXPORT: copying %u components\n", (uint32_t)export_jobs_.size());
#endif

    for (ExportJob &export_job : export_jobs_) {
        auto &archetype = *archetype_stores_[export_job.archetypeIdx];

#ifdef MADRONA_ECS_DEBUG_TRACKING
        // Get component name using ECS debug system
        const char* component_name = nullptr;

        if (archetype.tblStorage.tbls.size() > 0 && archetype.tblStorage.tbls[0].numRows() > 0) {
            void* sample_addr = archetype.tblStorage.tbls[0].data(export_job.columnIdx);
            address_info_t debug_info;
            if (simple_tracker_lookup(sample_addr, &debug_info)) {
                component_name = debug_info.component_name;
            }
        }

        if (!component_name) {
            component_name = "Unknown";
        }

        // Try to get tensor name from export buffer registration
        const char* tensor_name = simple_tracker_lookup_export_tensor_name(export_job.mem.ptr());

        // Count total rows across all tables
        CountT total_rows = 0;
        for (Table &tbl : archetype.tblStorage.tbls) {
            total_rows += tbl.numRows();
        }

        if (tensor_name) {
            printf("  %s -> %s (%p): %u rows, %u bytes/row\n",
                   component_name, tensor_name, export_job.mem.ptr(),
                   (uint32_t)total_rows, export_job.numBytesPerRow);
        } else {
            printf("  %s: %u rows, %u bytes/row -> %p\n",
                   component_name, (uint32_t)total_rows, export_job.numBytesPerRow, export_job.mem.ptr());
        }
#endif

        // ... rest of function continues with actual copying logic