#pragma once

#ifdef MADRONA_ECS_DEBUG_TRACKING
#include "../src/debug/ecs_simple_tracker.h"
#include "magic_enum.hpp"
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

template <typename ArchetypeT, typename ComponentT>
void ECSRegistry::exportColumn(int32_t slot)
{
    export_ptrs_[slot] = state_mgr_->exportColumn<ArchetypeT, ComponentT>();
}

template <typename SingletonT>
void ECSRegistry::exportSingleton(int32_t slot)
{
    export_ptrs_[slot] = state_mgr_->exportSingleton<SingletonT>();
}

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

template <typename SingletonT, EnumType EnumT>
void ECSRegistry::exportSingleton(EnumT slot)
{
    void* buffer_ptr = state_mgr_->exportSingleton<SingletonT>();
    export_ptrs_[static_cast<uint32_t>(slot)] = buffer_ptr;

#ifdef MADRONA_ECS_DEBUG_TRACKING
    // Use magic_enum to automatically get tensor name from enum value
    auto tensor_name = magic_enum::enum_name(slot);
    if (!tensor_name.empty()) {
        simple_tracker_register_export_buffer(buffer_ptr, std::string(tensor_name).c_str());
    }
#endif
}

}
