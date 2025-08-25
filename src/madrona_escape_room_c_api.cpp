#include "types.hpp"
#include "madrona_escape_room_c_api.h"
#include "mgr.hpp"
#include "asset_registry.hpp"
#include "asset_ids.hpp"

#include <cstring>
#include <new>
#include <vector>
#include <cstdio>

using namespace madEscape;

// Define MER error codes as mappings to madEscape::Result
#define MER_SUCCESS static_cast<MER_Result>(Result::Success)
#define MER_ERROR_NULL_POINTER static_cast<MER_Result>(Result::ErrorNullPointer)
#define MER_ERROR_INVALID_PARAMETER static_cast<MER_Result>(Result::ErrorInvalidParameter)
#define MER_ERROR_ALLOCATION_FAILED static_cast<MER_Result>(Result::ErrorAllocationFailed)
#define MER_ERROR_NOT_INITIALIZED static_cast<MER_Result>(Result::ErrorNotInitialized)
#define MER_ERROR_CUDA_FAILURE static_cast<MER_Result>(Result::ErrorCudaFailure)
#define MER_ERROR_FILE_NOT_FOUND static_cast<MER_Result>(Result::ErrorFileNotFound)
#define MER_ERROR_INVALID_FILE static_cast<MER_Result>(Result::ErrorInvalidFile)
#define MER_ERROR_FILE_IO static_cast<MER_Result>(Result::ErrorFileIO)

// Helper function to convert madrona::py::Tensor to MER_Tensor
static void convertTensor(const madrona::py::Tensor& src, MER_Tensor* dst) {
    dst->data = src.devicePtr();
    dst->num_dimensions = static_cast<int32_t>(src.numDims());
    dst->gpu_id = src.isOnGPU() ? src.gpuID() : -1;
    
    // Copy dimensions
    for (int i = 0; i < dst->num_dimensions && i < 16; ++i) {
        dst->dimensions[i] = src.dims()[i];
    }
    
    // Convert element type - store as int32_t (matches madrona::py::TensorElementType values)
    dst->element_type = static_cast<int32_t>(src.type());
    
    // Calculate total bytes
    int64_t total_elements = 1;
    for (int i = 0; i < dst->num_dimensions; ++i) {
        total_elements *= dst->dimensions[i];
    }
    dst->num_bytes = total_elements * src.numBytesPerItem();
}

extern "C" {

MER_Result mer_create_manager(
    MER_ManagerHandle* out_handle,
    const void* config,
    const void* compiled_levels,
    uint32_t num_compiled_levels
) {
    if (!out_handle || !config) {
        return MER_ERROR_NULL_POINTER;
    }
    
    *out_handle = nullptr;
    
    // Direct cast config from void* - Python passes the exact C++ struct via ctypes
    const ManagerConfig* mgr_cfg = reinterpret_cast<const ManagerConfig*>(config);
    
    // Direct cast from void* - Python passes the exact C++ struct via ctypes
    using namespace madEscape;
    const CompiledLevel* cpp_levels = reinterpret_cast<const CompiledLevel*>(compiled_levels);
    
    // Convert array of compiled levels to C++ vector (if provided)
    std::vector<std::optional<CompiledLevel>> cpp_per_world_levels;
    if (cpp_levels != nullptr && num_compiled_levels > 0) {
        // Validate array size doesn't exceed number of worlds
        if (num_compiled_levels > mgr_cfg->num_worlds) {
            return MER_ERROR_INVALID_PARAMETER;
        }
        
        cpp_per_world_levels.reserve(mgr_cfg->num_worlds);
        
        for (uint32_t i = 0; i < num_compiled_levels; i++) {
            // Direct copy of the CompiledLevel struct
            cpp_per_world_levels.push_back(cpp_levels[i]);
        }
        
        // Fill remaining worlds with nullopt if array is smaller than num_worlds
        while (cpp_per_world_levels.size() < mgr_cfg->num_worlds) {
            cpp_per_world_levels.push_back(std::nullopt);
        }
    }
    
    // Convert to Manager::Config
    Manager::Config mgr_config {
        .execMode = mgr_cfg->exec_mode,
        .gpuID = mgr_cfg->gpu_id,
        .numWorlds = mgr_cfg->num_worlds,
        .randSeed = mgr_cfg->rand_seed,
        .autoReset = mgr_cfg->auto_reset,
        .enableBatchRenderer = mgr_cfg->enable_batch_renderer,
        .batchRenderViewWidth = mgr_cfg->batch_render_view_width ? 
            mgr_cfg->batch_render_view_width : consts::display::defaultBatchRenderSize,
        .batchRenderViewHeight = mgr_cfg->batch_render_view_height ? 
            mgr_cfg->batch_render_view_height : consts::display::defaultBatchRenderSize,
        .perWorldCompiledLevels = std::move(cpp_per_world_levels),  // Per-world compiled levels
    };
    
    // Allocate Manager - using placement new to avoid exceptions
    void* mgr_memory = ::operator new(sizeof(Manager), std::nothrow);
    if (!mgr_memory) {
        return MER_ERROR_ALLOCATION_FAILED;
    }
    
    // Construct Manager in allocated memory
    Manager* mgr = new (mgr_memory) Manager(mgr_config);
    
    *out_handle = reinterpret_cast<MER_ManagerHandle>(mgr);
    return MER_SUCCESS;
}

MER_Result mer_validate_compiled_level(const void* level) {
    if (!level) return MER_ERROR_NULL_POINTER;
    
    // Direct cast from void* - Python passes the exact C++ struct
    const CompiledLevel* compiled_level = reinterpret_cast<const CompiledLevel*>(level);
    
    // Calculate expected array size from dimensions
    int32_t expected_array_size = compiled_level->width * compiled_level->height;
    
    // Validate basic constraints
    if (compiled_level->num_tiles < 0 || compiled_level->num_tiles > expected_array_size) return MER_ERROR_INVALID_PARAMETER;
    if (compiled_level->max_entities < 0) return MER_ERROR_INVALID_PARAMETER;
    if (compiled_level->width <= 0 || compiled_level->height <= 0) return MER_ERROR_INVALID_PARAMETER;
    if (compiled_level->world_scale <= 0.0f) return MER_ERROR_INVALID_PARAMETER;
    
    // Validate array size doesn't exceed our fixed buffer limits
    if (expected_array_size > CompiledLevel::MAX_TILES) return MER_ERROR_INVALID_PARAMETER;
    
    return MER_SUCCESS;
}

MER_Result mer_destroy_manager(MER_ManagerHandle handle) {
    if (!handle) {
        return MER_ERROR_NULL_POINTER;
    }
    
    Manager* mgr = reinterpret_cast<Manager*>(handle);
    mgr->~Manager();
    ::operator delete(mgr);
    
    return MER_SUCCESS;
}

MER_Result mer_step(MER_ManagerHandle handle) {
    if (!handle) {
        return MER_ERROR_NULL_POINTER;
    }
    
    Manager* mgr = reinterpret_cast<Manager*>(handle);
    mgr->step();
    
    return MER_SUCCESS;
}

MER_Result mer_get_reset_tensor(MER_ManagerHandle handle, MER_Tensor* out_tensor) {
    if (!handle || !out_tensor) {
        return MER_ERROR_NULL_POINTER;
    }
    
    Manager* mgr = reinterpret_cast<Manager*>(handle);
    madrona::py::Tensor tensor = mgr->resetTensor();
    convertTensor(tensor, out_tensor);
    
    return MER_SUCCESS;
}

MER_Result mer_get_action_tensor(MER_ManagerHandle handle, MER_Tensor* out_tensor) {
    if (!handle || !out_tensor) {
        return MER_ERROR_NULL_POINTER;
    }
    
    Manager* mgr = reinterpret_cast<Manager*>(handle);
    madrona::py::Tensor tensor = mgr->actionTensor();
    convertTensor(tensor, out_tensor);
    
    return MER_SUCCESS;
}

MER_Result mer_get_reward_tensor(MER_ManagerHandle handle, MER_Tensor* out_tensor) {
    if (!handle || !out_tensor) {
        return MER_ERROR_NULL_POINTER;
    }
    
    Manager* mgr = reinterpret_cast<Manager*>(handle);
    madrona::py::Tensor tensor = mgr->rewardTensor();
    convertTensor(tensor, out_tensor);
    
    return MER_SUCCESS;
}

MER_Result mer_get_done_tensor(MER_ManagerHandle handle, MER_Tensor* out_tensor) {
    if (!handle || !out_tensor) {
        return MER_ERROR_NULL_POINTER;
    }
    
    Manager* mgr = reinterpret_cast<Manager*>(handle);
    madrona::py::Tensor tensor = mgr->doneTensor();
    convertTensor(tensor, out_tensor);
    
    return MER_SUCCESS;
}

MER_Result mer_get_self_observation_tensor(MER_ManagerHandle handle, MER_Tensor* out_tensor) {
    if (!handle || !out_tensor) {
        return MER_ERROR_NULL_POINTER;
    }
    
    Manager* mgr = reinterpret_cast<Manager*>(handle);
    madrona::py::Tensor tensor = mgr->selfObservationTensor();
    convertTensor(tensor, out_tensor);
    
    return MER_SUCCESS;
}

MER_Result mer_get_steps_remaining_tensor(MER_ManagerHandle handle, MER_Tensor* out_tensor) {
    if (!handle || !out_tensor) {
        return MER_ERROR_NULL_POINTER;
    }
    
    Manager* mgr = reinterpret_cast<Manager*>(handle);
    madrona::py::Tensor tensor = mgr->stepsRemainingTensor();
    convertTensor(tensor, out_tensor);
    
    return MER_SUCCESS;
}

MER_Result mer_get_progress_tensor(MER_ManagerHandle handle, MER_Tensor* out_tensor) {
    if (!handle || !out_tensor) {
        return MER_ERROR_NULL_POINTER;
    }
    
    Manager* mgr = reinterpret_cast<Manager*>(handle);
    madrona::py::Tensor tensor = mgr->progressTensor();
    convertTensor(tensor, out_tensor);
    
    return MER_SUCCESS;
}

MER_Result mer_get_rgb_tensor(MER_ManagerHandle handle, MER_Tensor* out_tensor) {
    if (!handle || !out_tensor) {
        return MER_ERROR_NULL_POINTER;
    }
    
    Manager* mgr = reinterpret_cast<Manager*>(handle);
    madrona::py::Tensor tensor = mgr->rgbTensor();
    convertTensor(tensor, out_tensor);
    
    return MER_SUCCESS;
}

MER_Result mer_get_depth_tensor(MER_ManagerHandle handle, MER_Tensor* out_tensor) {
    if (!handle || !out_tensor) {
        return MER_ERROR_NULL_POINTER;
    }
    
    Manager* mgr = reinterpret_cast<Manager*>(handle);
    madrona::py::Tensor tensor = mgr->depthTensor();
    convertTensor(tensor, out_tensor);
    
    return MER_SUCCESS;
}

MER_Result mer_trigger_reset(MER_ManagerHandle handle, int32_t world_idx) {
    if (!handle) {
        return MER_ERROR_NULL_POINTER;
    }
    
    Manager* mgr = reinterpret_cast<Manager*>(handle);
    mgr->triggerReset(world_idx);
    
    return MER_SUCCESS;
}

MER_Result mer_set_action(
    MER_ManagerHandle handle,
    int32_t world_idx,
    int32_t move_amount,
    int32_t move_angle,
    int32_t rotate
) {
    if (!handle) {
        return MER_ERROR_NULL_POINTER;
    }
    
    Manager* mgr = reinterpret_cast<Manager*>(handle);
    mgr->setAction(world_idx, move_amount, move_angle, rotate);
    
    return MER_SUCCESS;
}

MER_Result mer_enable_trajectory_logging(
    MER_ManagerHandle handle,
    int32_t world_idx,
    int32_t agent_idx,
    const char* filename
) {
    if (!handle) {
        return MER_ERROR_NULL_POINTER;
    }
    
    Manager* mgr = reinterpret_cast<Manager*>(handle);
    
    if (filename) {
        mgr->enableTrajectoryLogging(world_idx, agent_idx, filename);
    } else {
        mgr->enableTrajectoryLogging(world_idx, agent_idx, std::nullopt);
    }
    
    return MER_SUCCESS;
}

MER_Result mer_disable_trajectory_logging(MER_ManagerHandle handle) {
    if (!handle) {
        return MER_ERROR_NULL_POINTER;
    }
    
    Manager* mgr = reinterpret_cast<Manager*>(handle);
    mgr->disableTrajectoryLogging();
    
    return MER_SUCCESS;
}

MER_Result mer_start_recording(
    MER_ManagerHandle handle,
    const char* filepath,
    uint32_t seed
) {
    if (!handle || !filepath) {
        return MER_ERROR_NULL_POINTER;
    }
    
    Manager* mgr = reinterpret_cast<Manager*>(handle);
    mgr->startRecording(filepath, seed);
    
    return MER_SUCCESS;
}

MER_Result mer_stop_recording(MER_ManagerHandle handle) {
    if (!handle) {
        return MER_ERROR_NULL_POINTER;
    }
    
    Manager* mgr = reinterpret_cast<Manager*>(handle);
    mgr->stopRecording();
    
    return MER_SUCCESS;
}

MER_Result mer_is_recording(MER_ManagerHandle handle, bool* out_is_recording) {
    if (!handle || !out_is_recording) {
        return MER_ERROR_NULL_POINTER;
    }
    
    Manager* mgr = reinterpret_cast<Manager*>(handle);
    *out_is_recording = mgr->isRecording();
    
    return MER_SUCCESS;
}

MER_Result mer_load_replay(MER_ManagerHandle handle, const char* filepath) {
    if (!handle || !filepath) {
        return MER_ERROR_NULL_POINTER;
    }
    
    Manager* mgr = reinterpret_cast<Manager*>(handle);
    bool success = mgr->loadReplay(filepath);
    
    return success ? MER_SUCCESS : MER_ERROR_INVALID_FILE;
}

MER_Result mer_has_replay(MER_ManagerHandle handle, bool* out_has_replay) {
    if (!handle || !out_has_replay) {
        return MER_ERROR_NULL_POINTER;
    }
    
    Manager* mgr = reinterpret_cast<Manager*>(handle);
    *out_has_replay = mgr->hasReplay();
    
    return MER_SUCCESS;
}

MER_Result mer_replay_step(MER_ManagerHandle handle, bool* out_success) {
    if (!handle || !out_success) {
        return MER_ERROR_NULL_POINTER;
    }
    
    Manager* mgr = reinterpret_cast<Manager*>(handle);
    *out_success = mgr->replayStep();
    
    return MER_SUCCESS;
}

MER_Result mer_get_replay_step_count(
    MER_ManagerHandle handle,
    uint32_t* out_current,
    uint32_t* out_total
) {
    if (!handle || !out_current || !out_total) {
        return MER_ERROR_NULL_POINTER;
    }
    
    Manager* mgr = reinterpret_cast<Manager*>(handle);
    *out_current = mgr->getCurrentReplayStep();
    *out_total = mgr->getTotalReplaySteps();
    
    return MER_SUCCESS;
}

MER_Result mer_read_replay_metadata(
    const char* filepath,
    void* out_metadata
) {
    if (!filepath || !out_metadata) {
        return MER_ERROR_NULL_POINTER;
    }
    
    // Use Manager's static method to read metadata
    auto metadata_opt = Manager::readReplayMetadata(filepath);
    if (!metadata_opt.has_value()) {
        return MER_ERROR_INVALID_FILE;
    }
    
    const auto& metadata = metadata_opt.value();
    
    // Direct cast to ReplayMetadata* - Python passes the exact C++ struct via ctypes
    ReplayMetadata* replay_meta = reinterpret_cast<ReplayMetadata*>(out_metadata);
    
    // Copy metadata
    replay_meta->num_worlds = metadata.num_worlds;
    replay_meta->num_agents_per_world = metadata.num_agents_per_world;
    replay_meta->num_steps = metadata.num_steps;
    replay_meta->seed = metadata.seed;
    replay_meta->timestamp = metadata.timestamp;
    
    // Copy sim name safely
    std::strncpy(replay_meta->sim_name, metadata.sim_name, sizeof(replay_meta->sim_name) - 1);
    replay_meta->sim_name[sizeof(replay_meta->sim_name) - 1] = '\0';
    
    return MER_SUCCESS;
}

int32_t mer_get_max_tiles(void) {
    return CompiledLevel::MAX_TILES;
}

int32_t mer_get_max_spawns(void) {
    return CompiledLevel::MAX_SPAWNS;
}

size_t mer_get_compiled_level_size(void) {
    return sizeof(CompiledLevel);
}

const char* mer_result_to_string(MER_Result result) {
    Result res = static_cast<Result>(result);
    switch (res) {
        case Result::Success:
            return "Success";
        case Result::ErrorNullPointer:
            return "Null pointer error";
        case Result::ErrorInvalidParameter:
            return "Invalid parameter";
        case Result::ErrorAllocationFailed:
            return "Memory allocation failed";
        case Result::ErrorNotInitialized:
            return "Not initialized";
        case Result::ErrorCudaFailure:
            return "CUDA operation failed";
        case Result::ErrorFileNotFound:
            return "File not found";
        case Result::ErrorInvalidFile:
            return "Invalid file";
        case Result::ErrorFileIO:
            return "File I/O error";
        default:
            return "Unknown error";
    }
}

// Asset descriptor functions
int32_t mer_get_physics_assets_count(void) {
    return static_cast<int32_t>(madEscape::Assets::getPhysicsAssetCount());
}

int32_t mer_get_render_assets_count(void) {
    return static_cast<int32_t>(madEscape::Assets::getRenderAssetCount());
}

const char* mer_get_physics_asset_name(int32_t index) {
    // Collect physics assets from static table
    int32_t count = 0;
    for (uint32_t i = 0; i < madEscape::AssetIDs::MAX_ASSETS; ++i) {
        const auto* asset = madEscape::Assets::getAssetInfo(i);
        if (asset && asset->hasPhysics) {
            if (count == index) {
                return asset->name;
            }
            count++;
        }
    }
    return nullptr;
}

const char* mer_get_render_asset_name(int32_t index) {
    // Collect render assets from static table
    int32_t count = 0;
    for (uint32_t i = 0; i < madEscape::AssetIDs::MAX_ASSETS; ++i) {
        const auto* asset = madEscape::Assets::getAssetInfo(i);
        if (asset && asset->hasRender) {
            if (count == index) {
                return asset->name;
            }
            count++;
        }
    }
    return nullptr;
}

int32_t mer_get_physics_asset_object_id(const char* name) {
    if (name == nullptr) {
        return -1;
    }
    
    // Check if this asset has physics
    uint32_t id = madEscape::Assets::getAssetId(name);
    if (id != madEscape::AssetIDs::INVALID && madEscape::Assets::assetHasPhysics(id)) {
        return static_cast<int32_t>(id);
    }
    return -1;  // Not found or doesn't have physics
}

int32_t mer_get_render_asset_object_id(const char* name) {
    if (name == nullptr) {
        return -1;
    }
    
    // Check if this asset has render
    uint32_t id = madEscape::Assets::getAssetId(name);
    if (id != madEscape::AssetIDs::INVALID && madEscape::Assets::assetHasRender(id)) {
        return static_cast<int32_t>(id);
    }
    return -1;  // Not found or doesn't have render
}

MER_Result mer_write_compiled_level(
    const char* filepath, 
    const void* level
) {
    if (!filepath || !level) {
        return MER_ERROR_NULL_POINTER;
    }
    
    // Direct cast from void* - Python passes the exact C++ struct
    const CompiledLevel* compiled_level = reinterpret_cast<const CompiledLevel*>(level);
    
    FILE* f = fopen(filepath, "wb");
    if (!f) {
        return MER_ERROR_FILE_IO;
    }
    
    size_t written = fwrite(compiled_level, sizeof(CompiledLevel), 1, f);
    fclose(f);
    
    return (written == 1) ? MER_SUCCESS : MER_ERROR_FILE_IO;
}

MER_Result mer_read_compiled_level(
    const char* filepath, 
    void* level
) {
    if (!filepath || !level) {
        return MER_ERROR_NULL_POINTER;
    }
    
    // Direct cast from void* - Python passes the exact C++ struct
    CompiledLevel* compiled_level = reinterpret_cast<CompiledLevel*>(level);
    
    FILE* f = fopen(filepath, "rb");
    if (!f) {
        return MER_ERROR_FILE_IO;
    }
    
    size_t read = fread(compiled_level, sizeof(CompiledLevel), 1, f);
    fclose(f);
    
    return (read == 1) ? MER_SUCCESS : MER_ERROR_FILE_IO;
}

} // extern "C"