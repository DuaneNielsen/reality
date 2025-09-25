#include "types.hpp"
#include "madrona_escape_room_c_api.h"
#include "mgr.hpp"
#include "asset_registry.hpp"
#include "asset_ids.hpp"

#include <cstring>
#include <new>
#include <vector>
#include <cstdio>
#include <csignal>
#include <execinfo.h>
#include <unistd.h>
#include <atomic>

using namespace madEscape;

// Global counter to track manager creations for debugging
static std::atomic<uint32_t> g_manager_creation_count{0};
static std::atomic<uint32_t> g_manager_destruction_count{0};

// Signal handler for debugging segfaults
static void segfault_handler(int sig, siginfo_t *info, void *) {
    fprintf(stderr, "\n========== SEGFAULT TRAPPED ==========\n");
    fprintf(stderr, "Signal: %d\n", sig);
    fprintf(stderr, "Fault address: %p\n", info->si_addr);
    fprintf(stderr, "Manager creations: %u\n", g_manager_creation_count.load());
    fprintf(stderr, "Manager destructions: %u\n", g_manager_destruction_count.load());
    fprintf(stderr, "Currently alive managers: %u\n", 
            g_manager_creation_count.load() - g_manager_destruction_count.load());
    
    // Print backtrace
    void *buffer[100];
    int nptrs = backtrace(buffer, 100);
    fprintf(stderr, "\nBacktrace (%d frames):\n", nptrs);
    backtrace_symbols_fd(buffer, nptrs, STDERR_FILENO);
    fprintf(stderr, "=====================================\n\n");
    
    // Re-raise the signal to get default behavior
    signal(sig, SIG_DFL);
    raise(sig);
}

// Install signal handler on library load
static void install_signal_handler() {
    static bool installed = false;
    if (!installed) {
        struct sigaction sa;
        memset(&sa, 0, sizeof(sa));
        sa.sa_sigaction = segfault_handler;
        sa.sa_flags = SA_SIGINFO;
        sigemptyset(&sa.sa_mask);
        
        sigaction(SIGSEGV, &sa, nullptr);
        sigaction(SIGILL, &sa, nullptr);
        installed = true;
    }
}

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
    // Install signal handler on first manager creation
    install_signal_handler();
    
    g_manager_creation_count.fetch_add(1);
    
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
        // Reserve space for all worlds
        cpp_per_world_levels.reserve(mgr_cfg->num_worlds);
        
        // Distribute levels across worlds using round-robin for curriculum learning
        // This allows more levels than worlds (e.g., 20 levels across 4 worlds)
        for (uint32_t world_idx = 0; world_idx < mgr_cfg->num_worlds; world_idx++) {
            if (num_compiled_levels > 0) {
                // Round-robin distribution: world i gets level (i % num_levels)
                uint32_t level_idx = world_idx % num_compiled_levels;
                cpp_per_world_levels.push_back(cpp_levels[level_idx]);
            } else {
                // No levels provided - use default
                cpp_per_world_levels.push_back(std::nullopt);
            }
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
        .customVerticalFov = mgr_cfg->custom_vertical_fov,
        .renderMode = mgr_cfg->render_mode,
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

MER_Result mer_create_manager_from_replay(
    MER_ManagerHandle* out_handle,
    const char* filepath,
    int32_t exec_mode,
    int32_t gpu_id,
    bool enable_batch_renderer)
{
    // Install signal handler on first manager creation
    install_signal_handler();
    
    g_manager_creation_count.fetch_add(1);
    
    if (!out_handle || !filepath) {
        return MER_ERROR_NULL_POINTER;
    }
    
    *out_handle = nullptr;
    
    // Convert int32_t to ExecMode enum
    madrona::ExecMode execMode = static_cast<madrona::ExecMode>(exec_mode);
    
    // Use Manager::fromReplay static factory method
    auto mgr = Manager::fromReplay(std::string(filepath), execMode, gpu_id, enable_batch_renderer);
    if (!mgr) {
        return MER_ERROR_FILE_NOT_FOUND; // Or appropriate error
    }
    
    // Transfer ownership to C API
    Manager* mgr_ptr = mgr.release();
    *out_handle = reinterpret_cast<MER_ManagerHandle>(mgr_ptr);
    
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
    
    // Validate number of filled tiles doesn't exceed our fixed buffer limits
    if (compiled_level->num_tiles > CompiledLevel::MAX_TILES) return MER_ERROR_INVALID_PARAMETER;
    
    return MER_SUCCESS;
}

MER_Result mer_destroy_manager(MER_ManagerHandle handle) {
    if (!handle) {
        return MER_ERROR_NULL_POINTER;
    }
    
    g_manager_destruction_count.fetch_add(1);
    
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

MER_Result mer_get_termination_reason_tensor(MER_ManagerHandle handle, MER_Tensor* out_tensor) {
    if (!handle || !out_tensor) {
        return MER_ERROR_NULL_POINTER;
    }

    Manager* mgr = reinterpret_cast<Manager*>(handle);
    madrona::py::Tensor tensor = mgr->terminationReasonTensor();
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

MER_Result mer_get_compass_tensor(MER_ManagerHandle handle, MER_Tensor* out_tensor) {
    if (!handle || !out_tensor) {
        return MER_ERROR_NULL_POINTER;
    }
    
    Manager* mgr = reinterpret_cast<Manager*>(handle);
    madrona::py::Tensor tensor = mgr->compassTensor();
    convertTensor(tensor, out_tensor);
    
    return MER_SUCCESS;
}

MER_Result mer_get_lidar_tensor(MER_ManagerHandle handle, MER_Tensor* out_tensor) {
    if (!handle || !out_tensor) {
        return MER_ERROR_NULL_POINTER;
    }
    
    Manager* mgr = reinterpret_cast<Manager*>(handle);
    madrona::py::Tensor tensor = mgr->lidarTensor();
    convertTensor(tensor, out_tensor);
    
    return MER_SUCCESS;
}

MER_Result mer_get_steps_taken_tensor(MER_ManagerHandle handle, MER_Tensor* out_tensor) {
    if (!handle || !out_tensor) {
        return MER_ERROR_NULL_POINTER;
    }
    
    Manager* mgr = reinterpret_cast<Manager*>(handle);
    madrona::py::Tensor tensor = mgr->stepsTakenTensor();
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

MER_Result mer_get_target_position_tensor(MER_ManagerHandle handle, MER_Tensor* out_tensor) {
    if (!handle || !out_tensor) {
        return MER_ERROR_NULL_POINTER;
    }

    Manager* mgr = reinterpret_cast<Manager*>(handle);
    madrona::py::Tensor tensor = mgr->targetPositionTensor();
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
    const char* filepath
) {
    if (!handle || !filepath) {
        return MER_ERROR_NULL_POINTER;
    }
    
    Manager* mgr = reinterpret_cast<Manager*>(handle);
    Result result = mgr->startRecording(filepath);
    
    return static_cast<MER_Result>(result);
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
    
    // Copy ALL metadata fields including v3 additions
    replay_meta->magic = metadata.magic;
    replay_meta->version = metadata.version;
    replay_meta->num_worlds = metadata.num_worlds;
    replay_meta->num_agents_per_world = metadata.num_agents_per_world;
    replay_meta->num_steps = metadata.num_steps;
    replay_meta->actions_per_step = metadata.actions_per_step;
    replay_meta->timestamp = metadata.timestamp;
    replay_meta->seed = metadata.seed;
    
    std::memcpy(replay_meta->reserved, metadata.reserved, sizeof(replay_meta->reserved));
    
    // Copy string fields safely
    std::strncpy(replay_meta->sim_name, metadata.sim_name, sizeof(replay_meta->sim_name) - 1);
    replay_meta->sim_name[sizeof(replay_meta->sim_name) - 1] = '\0';
    
    std::strncpy(replay_meta->level_name, metadata.level_name, sizeof(replay_meta->level_name) - 1);
    replay_meta->level_name[sizeof(replay_meta->level_name) - 1] = '\0';
    
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
        case Result::ErrorRecordingAlreadyActive:
            return "Recording already in progress";
        case Result::ErrorRecordingNotAtStepZero:
            return "Recording can only be started from step zero of a fresh simulation";
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


MER_Result mer_write_compiled_levels(
    const char* filepath,
    const void* levels,
    uint32_t num_levels
) {
    // Validation
    if (!filepath || !levels || num_levels == 0) {
        return MER_ERROR_INVALID_PARAMETER;
    }
    
    FILE* f = fopen(filepath, "wb");
    if (!f) {
        return MER_ERROR_FILE_NOT_FOUND;
    }
    
    // Write unified format header
    const char magic[] = "LEVELS";  // Changed from "MLEVL"
    size_t magic_written = fwrite(magic, sizeof(char), 6, f);
    if (magic_written != 6) {
        fclose(f);
        return MER_ERROR_FILE_IO;
    }
    
    // Write count
    size_t count_written = fwrite(&num_levels, sizeof(uint32_t), 1, f);
    if (count_written != 1) {
        fclose(f);
        return MER_ERROR_FILE_IO;
    }
    
    // Write all levels
    size_t levels_size = sizeof(CompiledLevel) * num_levels;
    size_t levels_written = fwrite(levels, 1, levels_size, f);
    if (levels_written != levels_size) {
        fclose(f);
        return MER_ERROR_FILE_IO;
    }
    
    fclose(f);
    return MER_SUCCESS;
}

MER_Result mer_read_compiled_levels(
    const char* filepath,
    void* out_levels,
    uint32_t* out_num_levels,
    uint32_t max_levels
) {
    if (!filepath || !out_num_levels) {
        return MER_ERROR_INVALID_PARAMETER;
    }
    
    FILE* f = fopen(filepath, "rb");
    if (!f) {
        return MER_ERROR_FILE_NOT_FOUND;
    }
    
    // Read magic header
    char magic[7] = {0};
    size_t magic_read = fread(magic, sizeof(char), 6, f);
    
    if (magic_read != 6 || strcmp(magic, "LEVELS") != 0) {
        fclose(f);
        return MER_ERROR_INVALID_FILE;
    }
    
    // Read number of levels
    uint32_t num_levels;
    size_t count_read = fread(&num_levels, sizeof(uint32_t), 1, f);
    if (count_read != 1) {
        fclose(f);
        return MER_ERROR_FILE_IO;
    }
    
    *out_num_levels = num_levels;
    
    // Read levels if buffer provided
    if (out_levels && max_levels > 0) {
        uint32_t levels_to_read = (num_levels < max_levels) ? num_levels : max_levels;
        size_t levels_size = sizeof(CompiledLevel) * levels_to_read;
        size_t read = fread(out_levels, 1, levels_size, f);
        if (read != levels_size) {
            fclose(f);
            return MER_ERROR_FILE_IO;
        }
    }
    
    fclose(f);
    return MER_SUCCESS;
}

} // extern "C"