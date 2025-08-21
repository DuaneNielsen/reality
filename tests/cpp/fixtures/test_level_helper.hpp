#pragma once

#include "types.hpp"
#include "madrona_escape_room_c_api.h"
#include <cstring>

// Helper class to provide the default level for tests
class DefaultLevelProvider {
public:
    // Get the default level as a CompiledLevel
    static madEscape::CompiledLevel GetDefaultLevel() {
        // Include the data here - only in this translation unit
        #include "default_level_data.h"
        madEscape::CompiledLevel level;
        static_assert(sizeof(madEscape::CompiledLevel) <= sizeof(default_level_lvl), 
                      "Level data too small");
        std::memcpy(&level, default_level_lvl, sizeof(madEscape::CompiledLevel));
        return level;
    }
    
    // Get the default level as MER_CompiledLevel (C API version)
    static MER_CompiledLevel GetDefaultLevelC() {
        // Just reinterpret the C++ type as the C type (they're the same)
        madEscape::CompiledLevel cpp_level = GetDefaultLevel();
        MER_CompiledLevel c_level;
        std::memcpy(&c_level, &cpp_level, sizeof(MER_CompiledLevel));
        return c_level;
    }
};