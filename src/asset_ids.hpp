#pragma once

#include <cstdint>

namespace madEscape {

namespace AssetIDs {
    constexpr uint32_t INVALID = 0;  // 0 means no asset/empty
    constexpr uint32_t CUBE = 1;
    constexpr uint32_t WALL = 2;
    constexpr uint32_t AGENT = 3;
    constexpr uint32_t PLANE = 4;
    constexpr uint32_t AXIS_X = 5;
    constexpr uint32_t AXIS_Y = 6;
    constexpr uint32_t AXIS_Z = 7;
    
    constexpr uint32_t DYNAMIC_START = 32;
    constexpr uint32_t MAX_ASSETS = 256;
}

}