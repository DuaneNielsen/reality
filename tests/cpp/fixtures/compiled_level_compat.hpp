#pragma once

#include "types.hpp" // Get madEscape types
#include "madrona_escape_room_c_api.h" // For MER_Result
#include <madrona/exec_mode.hpp> // For madrona::ExecMode

// Create type aliases for backward compatibility
// This allows existing test code using MER_ types to work unchanged
using MER_CompiledLevel = madEscape::CompiledLevel;
using MER_ManagerConfig = madEscape::ManagerConfig;
using MER_ReplayMetadata = madEscape::ReplayMetadata;

// Define constants for backward compatibility
constexpr madrona::ExecMode MER_EXEC_MODE_CPU = madrona::ExecMode::CPU;
constexpr madrona::ExecMode MER_EXEC_MODE_CUDA = madrona::ExecMode::CUDA;

// Error code constants for backward compatibility
#define MER_SUCCESS static_cast<MER_Result>(madEscape::Result::Success)
#define MER_ERROR_NULL_POINTER static_cast<MER_Result>(madEscape::Result::ErrorNullPointer)
#define MER_ERROR_INVALID_PARAMETER static_cast<MER_Result>(madEscape::Result::ErrorInvalidParameter)
#define MER_ERROR_ALLOCATION_FAILED static_cast<MER_Result>(madEscape::Result::ErrorAllocationFailed)
#define MER_ERROR_NOT_INITIALIZED static_cast<MER_Result>(madEscape::Result::ErrorNotInitialized)
#define MER_ERROR_CUDA_FAILURE static_cast<MER_Result>(madEscape::Result::ErrorCudaFailure)
#define MER_ERROR_FILE_NOT_FOUND static_cast<MER_Result>(madEscape::Result::ErrorFileNotFound)
#define MER_ERROR_INVALID_FILE static_cast<MER_Result>(madEscape::Result::ErrorInvalidFile)
#define MER_ERROR_FILE_IO static_cast<MER_Result>(madEscape::Result::ErrorFileIO)