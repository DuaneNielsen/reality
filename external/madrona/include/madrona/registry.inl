#pragma once

#ifdef MADRONA_ECS_DEBUG_TRACKING
#include "../src/debug/ecs_simple_tracker.h"
#endif

namespace madrona {

template <typename ComponentT>
void ECSRegistry::registerComponent(uint32_t num_bytes)
{
    state_mgr_->registerComponent<ComponentT>(num_bytes);
}

template <typename ArchetypeT>
void ECSRegistry::registerArchetype()
{
    state_mgr_->registerArchetype<ArchetypeT>(
        ComponentMetadataSelector {}, ArchetypeFlags::None, 0);
}

template <typename ArchetypeT, typename... MetadataComponentTs>
void ECSRegistry::registerArchetype(
        ComponentMetadataSelector<MetadataComponentTs...> component_metadatas,
        ArchetypeFlags archetype_flags,
        CountT max_num_entities_per_world)
{
    state_mgr_->registerArchetype<ArchetypeT>(
        component_metadatas, archetype_flags, max_num_entities_per_world);
}

template <typename BundleT>
void ECSRegistry::registerBundle()
{
    state_mgr_->registerBundle<BundleT>();
}

template <typename AliasT, typename BundleT>
void ECSRegistry::registerBundleAlias()
{
    state_mgr_->registerBundleAlias<AliasT, BundleT>();
}

template <typename SingletonT>
void ECSRegistry::registerSingleton()
{
    state_mgr_->registerSingleton<SingletonT>();
}

#ifdef MADRONA_ECS_DEBUG_TRACKING
// Helper function to convert export slot IDs to readable names
// This is application-specific and should be customized per project
inline const char* getExportSlotName(uint32_t slot) {
    // This mapping should match the ExportID enum in the application
    switch (slot) {
        case 0: return "Reset";
        case 1: return "LidarVisControl";
        case 2: return "Action";
        case 3: return "Reward";
        case 4: return "Done";
        case 5: return "TerminationReason";
        case 6: return "SelfObservation";
        case 7: return "CompassObservation";
        case 8: return "Lidar";
        case 9: return "StepsTaken";
        case 10: return "Progress";
        case 11: return "AgentPosition";
        case 12: return "TargetPosition";
        case 13: return "WorldChecksum";
        default: {
            static char buffer[64];
            snprintf(buffer, sizeof(buffer), "ExportSlot_%u", slot);
            return buffer;
        }
    }
}
#endif

template <typename ArchetypeT, typename ComponentT>
void ECSRegistry::exportColumn(int32_t slot)
{
    void* buffer_ptr = state_mgr_->exportColumn<ArchetypeT, ComponentT>();
    export_ptrs_[slot] = buffer_ptr;

#ifdef MADRONA_ECS_DEBUG_TRACKING
    // Register this buffer with its tensor name for reverse lookup
    const char* tensor_name = getExportSlotName(slot);
    simple_tracker_register_export_buffer(buffer_ptr, tensor_name, 0);
#endif
}

template <typename SingletonT>
void ECSRegistry::exportSingleton(int32_t slot)
{
    void* buffer_ptr = state_mgr_->exportSingleton<SingletonT>();
    export_ptrs_[slot] = buffer_ptr;

#ifdef MADRONA_ECS_DEBUG_TRACKING
    // Register this buffer with its tensor name for reverse lookup
    const char* tensor_name = getExportSlotName(slot);
    simple_tracker_register_export_buffer(buffer_ptr, tensor_name, 0);
#endif
}

template <typename ArchetypeT, typename ComponentT, EnumType EnumT>
void ECSRegistry::exportColumn(EnumT slot)
{
    exportColumn<ArchetypeT, ComponentT>(static_cast<uint32_t>(slot));
}

template <typename SingletonT, EnumType EnumT>
void ECSRegistry::exportSingleton(EnumT slot)
{
    exportSingleton<SingletonT>(static_cast<uint32_t>(slot));
}

}
