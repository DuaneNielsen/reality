#ifndef MADRONA_ESCAPE_ROOM_C_API_H
#define MADRONA_ESCAPE_ROOM_C_API_H

#ifdef __cplusplus
extern "C" {
#endif

// Export macros for shared library visibility
#ifdef _WIN32
    #ifdef BUILDING_MER_DLL
        #define MER_EXPORT __declspec(dllexport)
    #else
        #define MER_EXPORT __declspec(dllimport)
    #endif
#else
    #define MER_EXPORT __attribute__((visibility("default")))
#endif

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

// Error codes
typedef enum {
    MER_SUCCESS = 0,
    MER_ERROR_NULL_POINTER = -1,
    MER_ERROR_INVALID_PARAMETER = -2,
    MER_ERROR_ALLOCATION_FAILED = -3,
    MER_ERROR_NOT_INITIALIZED = -4,
    MER_ERROR_CUDA_FAILURE = -5,
    MER_ERROR_FILE_NOT_FOUND = -6,
    MER_ERROR_INVALID_FILE = -7,
    MER_ERROR_FILE_IO = -8,
} MER_Result;

// Opaque handle for the Manager
typedef struct MER_Manager* MER_ManagerHandle;

// Replay metadata structure
typedef struct {
    uint32_t num_worlds;
    uint32_t num_agents_per_world;
    uint32_t num_steps;
    uint32_t seed;
    char sim_name[64];  // Fixed size for C API
    uint64_t timestamp;
} MER_ReplayMetadata;

// Execution modes
typedef enum {
    MER_EXEC_MODE_CPU = 0,
    MER_EXEC_MODE_CUDA = 1,
} MER_ExecMode;

// Tensor element types (matching TensorElementType)
typedef enum {
    MER_TENSOR_TYPE_UINT8 = 0,
    MER_TENSOR_TYPE_INT8 = 1,
    MER_TENSOR_TYPE_INT16 = 2,
    MER_TENSOR_TYPE_INT32 = 3,
    MER_TENSOR_TYPE_INT64 = 4,
    MER_TENSOR_TYPE_FLOAT16 = 5,
    MER_TENSOR_TYPE_FLOAT32 = 6,
} MER_TensorElementType;

// Tensor descriptor
typedef struct {
    void* data;                    // Pointer to tensor data
    int64_t dimensions[16];        // Tensor dimensions
    int32_t num_dimensions;        // Number of dimensions
    MER_TensorElementType element_type;  // Data type
    int64_t num_bytes;            // Total size in bytes
    int32_t gpu_id;               // GPU ID (-1 for CPU tensors)
} MER_Tensor;

// CompiledLevel structure is no longer defined in C API
// Python passes the C++ CompiledLevel struct directly via ctypes
// The struct is auto-generated from the compiled binary using pahole

// Manager configuration
typedef struct {
    MER_ExecMode exec_mode;
    int gpu_id;
    uint32_t num_worlds;
    uint32_t rand_seed;
    bool auto_reset;
    bool enable_batch_renderer;
    uint32_t batch_render_view_width;   // Default: 64
    uint32_t batch_render_view_height;  // Default: 64
} MER_ManagerConfig;

// Manager lifecycle functions
MER_EXPORT MER_Result mer_create_manager(
    MER_ManagerHandle* out_handle,
    const MER_ManagerConfig* config,
    const void* compiled_levels,  // Direct CompiledLevel pointer from Python (NULL for default)
    uint32_t num_compiled_levels  // Length of compiled_levels array
);

MER_EXPORT MER_Result mer_destroy_manager(MER_ManagerHandle handle);

// Level validation functions
MER_EXPORT MER_Result mer_validate_compiled_level(const void* level);

// Simulation functions
MER_EXPORT MER_Result mer_step(MER_ManagerHandle handle);

// Tensor access functions
MER_EXPORT MER_Result mer_get_reset_tensor(MER_ManagerHandle handle, MER_Tensor* out_tensor);
MER_EXPORT MER_Result mer_get_action_tensor(MER_ManagerHandle handle, MER_Tensor* out_tensor);
MER_EXPORT MER_Result mer_get_reward_tensor(MER_ManagerHandle handle, MER_Tensor* out_tensor);
MER_EXPORT MER_Result mer_get_done_tensor(MER_ManagerHandle handle, MER_Tensor* out_tensor);
MER_EXPORT MER_Result mer_get_self_observation_tensor(MER_ManagerHandle handle, MER_Tensor* out_tensor);
MER_EXPORT MER_Result mer_get_steps_remaining_tensor(MER_ManagerHandle handle, MER_Tensor* out_tensor);
MER_EXPORT MER_Result mer_get_progress_tensor(MER_ManagerHandle handle, MER_Tensor* out_tensor);
MER_EXPORT MER_Result mer_get_rgb_tensor(MER_ManagerHandle handle, MER_Tensor* out_tensor);
MER_EXPORT MER_Result mer_get_depth_tensor(MER_ManagerHandle handle, MER_Tensor* out_tensor);

// Control functions (for viewer)
MER_EXPORT MER_Result mer_trigger_reset(MER_ManagerHandle handle, int32_t world_idx);
MER_EXPORT MER_Result mer_set_action(
    MER_ManagerHandle handle,
    int32_t world_idx,
    int32_t move_amount,
    int32_t move_angle,
    int32_t rotate
);

// Trajectory logging
MER_EXPORT MER_Result mer_enable_trajectory_logging(
    MER_ManagerHandle handle,
    int32_t world_idx,
    int32_t agent_idx,
    const char* filename  // NULL for stdout
);

MER_EXPORT MER_Result mer_disable_trajectory_logging(MER_ManagerHandle handle);

// Recording functionality
MER_EXPORT MER_Result mer_start_recording(
    MER_ManagerHandle handle,
    const char* filepath,
    uint32_t seed
);

MER_EXPORT MER_Result mer_stop_recording(MER_ManagerHandle handle);
MER_EXPORT MER_Result mer_is_recording(MER_ManagerHandle handle, bool* out_is_recording);

// Replay metadata reading (static function - no handle needed)
MER_EXPORT MER_Result mer_read_replay_metadata(
    const char* filepath,
    MER_ReplayMetadata* out_metadata
);

// Replay functionality
MER_EXPORT MER_Result mer_load_replay(MER_ManagerHandle handle, const char* filepath);
MER_EXPORT MER_Result mer_has_replay(MER_ManagerHandle handle, bool* out_has_replay);
MER_EXPORT MER_Result mer_replay_step(MER_ManagerHandle handle, bool* out_success);
MER_EXPORT MER_Result mer_get_replay_step_count(
    MER_ManagerHandle handle,
    uint32_t* out_current,
    uint32_t* out_total
);

// Utility functions
MER_EXPORT const char* mer_result_to_string(MER_Result result);

// Binary I/O functions for CompiledLevel
MER_EXPORT MER_Result mer_write_compiled_level(
    const char* filepath,
    const void* level  // Direct CompiledLevel pointer from Python
);

MER_EXPORT MER_Result mer_read_compiled_level(
    const char* filepath,
    void* level  // Direct CompiledLevel pointer from Python
);

// Get CompiledLevel constants
MER_EXPORT int32_t mer_get_max_tiles(void);
MER_EXPORT int32_t mer_get_max_spawns(void);
MER_EXPORT size_t mer_get_compiled_level_size(void);  // For validation

// Asset descriptor functions
// Get counts
MER_EXPORT int32_t mer_get_physics_assets_count(void);
MER_EXPORT int32_t mer_get_render_assets_count(void);

// Get asset names by index
MER_EXPORT const char* mer_get_physics_asset_name(int32_t index);
MER_EXPORT const char* mer_get_render_asset_name(int32_t index);

// Lookup object IDs by name
MER_EXPORT int32_t mer_get_physics_asset_object_id(const char* name);
MER_EXPORT int32_t mer_get_render_asset_object_id(const char* name);

#ifdef __cplusplus
}
#endif

#endif // MADRONA_ESCAPE_ROOM_C_API_H