// Dummy file to export internal structures for pahole extraction
// These structures are needed by Python bindings but aren't directly
// exposed through the C API

#include "types.hpp"

using namespace madEscape;

// Force these structures to be compiled into the binary
// by creating dummy instantiations that won't be optimized away
namespace {
    // Use volatile to prevent optimization
    volatile Action dummy_action = {};
    volatile SelfObservation dummy_obs = {};
    volatile Done dummy_done = {};
    volatile Reward dummy_reward = {};
    volatile Progress dummy_progress = {};
    volatile StepsRemaining dummy_steps = {};
}

// Export functions to ensure symbols are kept
extern "C" {
    void* _export_Action() { return (void*)&dummy_action; }
    void* _export_SelfObservation() { return (void*)&dummy_obs; }
    void* _export_Done() { return (void*)&dummy_done; }
    void* _export_Reward() { return (void*)&dummy_reward; }
    void* _export_Progress() { return (void*)&dummy_progress; }
    void* _export_StepsRemaining() { return (void*)&dummy_steps; }
}