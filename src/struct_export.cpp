// Dummy file to export API boundary structures for pahole extraction
// Only includes structs that cross the Python-C API boundary directly
// ECS components are accessed through tensor exports, not direct struct manipulation

#include "types.hpp"
#include "mgr.hpp"

using namespace madEscape;

// Force API boundary structures to be compiled into the binary
// by creating dummy instantiations that won't be optimized away
namespace {
    // Use volatile to prevent optimization
    volatile CompiledLevel dummy_compiled_level = {};
    volatile ReplayMetadata dummy_replay_metadata = {};
    volatile ManagerConfig dummy_manager_config = {};
    volatile SensorConfig dummy_sensor_config = {};
}

// Export functions to ensure symbols are kept
extern "C" {
    void* _export_CompiledLevel() { return (void*)&dummy_compiled_level; }
    void* _export_ReplayMetadata() { return (void*)&dummy_replay_metadata; }
    void* _export_ManagerConfig() { return (void*)&dummy_manager_config; }
    void* _export_SensorConfig() { return (void*)&dummy_sensor_config; }
}