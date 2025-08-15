#include "madrona_escape_room_c_api.h"
#include "mgr.hpp"
#include "types.hpp"
#include "consts.hpp"

#include <cstring>
#include <new>

using namespace madEscape;

// Helper function to convert madrona::py::Tensor to MER_Tensor
static void convertTensor(const madrona::py::Tensor& src, MER_Tensor* dst) {
    dst->data = src.devicePtr();
    dst->num_dimensions = static_cast<int32_t>(src.numDims());
    dst->gpu_id = src.isOnGPU() ? src.gpuID() : -1;
    
    // Copy dimensions
    for (int i = 0; i < dst->num_dimensions && i < 16; ++i) {
        dst->dimensions[i] = src.dims()[i];
    }
    
    // Convert element type
    switch (src.type()) {
        case madrona::py::TensorElementType::UInt8:
            dst->element_type = MER_TENSOR_TYPE_UINT8;
            break;
        case madrona::py::TensorElementType::Int8:
            dst->element_type = MER_TENSOR_TYPE_INT8;
            break;
        case madrona::py::TensorElementType::Int16:
            dst->element_type = MER_TENSOR_TYPE_INT16;
            break;
        case madrona::py::TensorElementType::Int32:
            dst->element_type = MER_TENSOR_TYPE_INT32;
            break;
        case madrona::py::TensorElementType::Int64:
            dst->element_type = MER_TENSOR_TYPE_INT64;
            break;
        case madrona::py::TensorElementType::Float16:
            dst->element_type = MER_TENSOR_TYPE_FLOAT16;
            break;
        case madrona::py::TensorElementType::Float32:
            dst->element_type = MER_TENSOR_TYPE_FLOAT32;
            break;
    }
    
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
    const MER_ManagerConfig* config,
    const MER_CompiledLevel* compiled_levels,
    uint32_t num_compiled_levels
) {
    if (!out_handle || !config) {
        return MER_ERROR_NULL_POINTER;
    }
    
    *out_handle = nullptr;
    
    // Convert array of C compiled levels to C++ vector (if provided)
    std::vector<std::optional<CompiledLevel>> cpp_per_world_levels;
    if (compiled_levels != nullptr && num_compiled_levels > 0) {
        // Validate array size doesn't exceed number of worlds
        if (num_compiled_levels > config->num_worlds) {
            return MER_ERROR_INVALID_PARAMETER;
        }
        
        cpp_per_world_levels.reserve(config->num_worlds);
        
        for (uint32_t i = 0; i < num_compiled_levels; i++) {
            const MER_CompiledLevel* c_level = &compiled_levels[i];
            
            CompiledLevel cpp_level;
            cpp_level.num_tiles = c_level->num_tiles;
            cpp_level.max_entities = c_level->max_entities;
            cpp_level.width = c_level->width;
            cpp_level.height = c_level->height;
            cpp_level.scale = c_level->scale;
            
            // Copy spawn data
            cpp_level.num_spawns = c_level->num_spawns;
            std::memcpy(cpp_level.spawn_x, c_level->spawn_x, sizeof(float) * CompiledLevel::MAX_SPAWNS);
            std::memcpy(cpp_level.spawn_y, c_level->spawn_y, sizeof(float) * CompiledLevel::MAX_SPAWNS);
            
            // Calculate actual array size for this level
            int32_t array_size = c_level->width * c_level->height;
            
            // Copy arrays (full array size for proper GPU indexing)
            // The arrays contain meaningful data up to num_tiles, rest is zero-filled
            std::memcpy(cpp_level.tile_types, c_level->tile_types,
                       sizeof(int32_t) * array_size);
            std::memcpy(cpp_level.tile_x, c_level->tile_x,
                       sizeof(float) * array_size);
            std::memcpy(cpp_level.tile_y, c_level->tile_y,
                       sizeof(float) * array_size);
            
            cpp_per_world_levels.push_back(cpp_level);
        }
        
        // Fill remaining worlds with nullopt if array is smaller than num_worlds
        while (cpp_per_world_levels.size() < config->num_worlds) {
            cpp_per_world_levels.push_back(std::nullopt);
        }
    }
    
    // Convert C config to Manager::Config
    Manager::Config mgr_config {
        .execMode = (config->exec_mode == MER_EXEC_MODE_CUDA) ? 
            madrona::ExecMode::CUDA : madrona::ExecMode::CPU,
        .gpuID = config->gpu_id,
        .numWorlds = config->num_worlds,
        .randSeed = config->rand_seed,
        .autoReset = config->auto_reset,
        .enableBatchRenderer = config->enable_batch_renderer,
        .batchRenderViewWidth = config->batch_render_view_width ? 
            config->batch_render_view_width : 64,
        .batchRenderViewHeight = config->batch_render_view_height ? 
            config->batch_render_view_height : 64,
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

MER_Result mer_validate_compiled_level(const MER_CompiledLevel* level) {
    if (!level) return MER_ERROR_NULL_POINTER;
    
    // Calculate expected array size from dimensions
    int32_t expected_array_size = level->width * level->height;
    
    // Validate basic constraints
    if (level->num_tiles < 0 || level->num_tiles > expected_array_size) return MER_ERROR_INVALID_PARAMETER;
    if (level->max_entities < 0) return MER_ERROR_INVALID_PARAMETER;
    if (level->width <= 0 || level->height <= 0) return MER_ERROR_INVALID_PARAMETER;
    if (level->scale <= 0.0f) return MER_ERROR_INVALID_PARAMETER;
    
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
    MER_ReplayMetadata* out_metadata
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
    
    // Convert from C++ metadata to C metadata structure
    out_metadata->num_worlds = metadata.num_worlds;
    out_metadata->num_agents_per_world = metadata.num_agents_per_world;
    out_metadata->num_steps = metadata.num_steps;
    out_metadata->seed = metadata.seed;
    out_metadata->timestamp = metadata.timestamp;
    
    // Copy sim name safely
    std::strncpy(out_metadata->sim_name, metadata.sim_name, sizeof(out_metadata->sim_name) - 1);
    out_metadata->sim_name[sizeof(out_metadata->sim_name) - 1] = '\0';
    
    return MER_SUCCESS;
}

int32_t mer_get_max_tiles(void) {
    return CompiledLevel::MAX_TILES;
}

int32_t mer_get_max_spawns(void) {
    return CompiledLevel::MAX_SPAWNS;
}

const char* mer_result_to_string(MER_Result result) {
    switch (result) {
        case MER_SUCCESS:
            return "Success";
        case MER_ERROR_NULL_POINTER:
            return "Null pointer error";
        case MER_ERROR_INVALID_PARAMETER:
            return "Invalid parameter";
        case MER_ERROR_ALLOCATION_FAILED:
            return "Memory allocation failed";
        case MER_ERROR_NOT_INITIALIZED:
            return "Not initialized";
        case MER_ERROR_CUDA_FAILURE:
            return "CUDA operation failed";
        case MER_ERROR_FILE_NOT_FOUND:
            return "File not found";
        case MER_ERROR_INVALID_FILE:
            return "Invalid file";
        default:
            return "Unknown error";
    }
}

} // extern "C"