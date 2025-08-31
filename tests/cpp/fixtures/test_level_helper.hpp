#pragma once

#include "types.hpp"
#include "compiled_level_compat.hpp"
#include "madrona_escape_room_c_api.h"
#include <cstring>

// Helper class to provide the default level for tests
class DefaultLevelProvider {
private:
    // Include the data directly as a static array
    static const unsigned char* GetLevelData() {
        // Include the data outside of any struct to avoid array size issues
        static const
        #include "default_level_data.h"
        ;
        return default_level_lvl;
    }
    
public:
    // Get the default level as a CompiledLevel
    static madEscape::CompiledLevel GetDefaultLevel() {
        const unsigned char* data = GetLevelData();
        // Cast the byte array directly to the struct
        return *reinterpret_cast<const madEscape::CompiledLevel*>(data);
    }
    
    // Get the default level as MER_CompiledLevel (C API version)
    static MER_CompiledLevel GetDefaultLevelC() {
        const unsigned char* data = GetLevelData();
        // Cast the byte array directly to the C struct
        return *reinterpret_cast<const MER_CompiledLevel*>(data);
    }
};