#pragma once

#include <madrona/types.hpp>

namespace madEscape {

namespace consts {
// Each random world is composed of a fixed number of rooms that the agents
// must solve in order to maximize their reward.
inline constexpr madrona::CountT numRooms = 1;

// Generated levels assume 1 agent (reduced from 2)
inline constexpr madrona::CountT numAgents = 1;

// NOTE: maxEntitiesPerRoom removed - now dynamically calculated in CompiledLevel.max_entities

// Various world / entity size parameters
inline constexpr float worldLength = 40.f;
inline constexpr float worldWidth = 20.f;
inline constexpr float wallWidth = 1.f;
inline constexpr float agentRadius = 1.f;

// Each unit of distance forward (+ y axis) rewards the agents by this amount
inline constexpr float rewardPerDist = 0.05f;
// Each step that the agents don't make additional progress they get a small
// penalty reward
inline constexpr float slackReward = -0.005f;

// Steps per episode
inline constexpr int32_t episodeLen = 200;

// How many discrete options for actions
inline constexpr madrona::CountT numMoveAmountBuckets = 4;
inline constexpr madrona::CountT numMoveAngleBuckets = 8;
inline constexpr madrona::CountT numTurnBuckets = 5;

// Number of action components per agent
inline constexpr madrona::CountT numActionComponents = 3;

// Action bucket max values (for random generation)
inline constexpr int32_t maxMoveAmountValue = numMoveAmountBuckets - 1;  // 3
inline constexpr int32_t maxMoveAngleValue = numMoveAngleBuckets - 1;    // 7
inline constexpr int32_t maxTurnValue = numTurnBuckets - 1;              // 4

// Action value constants for better code readability and Python bindings
namespace action {
    // Movement amount values (0-3)
    namespace move_amount {
        inline constexpr int32_t STOP = 0;
        inline constexpr int32_t SLOW = 1;
        inline constexpr int32_t MEDIUM = 2;
        inline constexpr int32_t FAST = 3;
    }
    
    // Movement angle values (0-7) - 8 directions
    namespace move_angle {
        inline constexpr int32_t FORWARD = 0;
        inline constexpr int32_t FORWARD_RIGHT = 1;
        inline constexpr int32_t RIGHT = 2;
        inline constexpr int32_t BACKWARD_RIGHT = 3;
        inline constexpr int32_t BACKWARD = 4;
        inline constexpr int32_t BACKWARD_LEFT = 5;
        inline constexpr int32_t LEFT = 6;
        inline constexpr int32_t FORWARD_LEFT = 7;
    }
    
    // Rotation values (0-4)
    // NOTE: Uses NON-STANDARD encoding (opposite of geometric convention)
    // Lower values = left/counterclockwise, Higher values = right/clockwise
    // This works with the physics formula: t_z = -160 * (action.rotate - 2)
    namespace rotate {
        inline constexpr int32_t FAST_LEFT = 0;   // Counterclockwise (non-standard: low value)
        inline constexpr int32_t SLOW_LEFT = 1;   // Counterclockwise (non-standard: low value)
        inline constexpr int32_t NONE = 2;        // Center value - no rotation
        inline constexpr int32_t SLOW_RIGHT = 3;  // Clockwise (non-standard: high value)
        inline constexpr int32_t FAST_RIGHT = 4;  // Clockwise (non-standard: high value)
    }
}

// Time (seconds) per step
inline constexpr float deltaT = 0.04f;

// Number of physics substeps
inline constexpr madrona::CountT numPhysicsSubsteps = 4.f;

// Physics constants
namespace physics {
    // Standard Earth gravity acceleration (m/s²)
    inline constexpr float gravityAcceleration = 9.8f;
}

// Rendering and color constants  
namespace rendering {
    // Material properties
    namespace material {
        inline constexpr float defaultRoughness = 0.8f;
        inline constexpr float defaultMetallic = 0.2f;
        inline constexpr float agentBodyMetallic = 1.0f;
        inline constexpr float floorAlpha = 0.0f;
    }
    
    // RGB color values (0-255)
    namespace colors {
        // Brown (cube)
        inline constexpr int32_t cubeRed = 191;
        inline constexpr int32_t cubeGreen = 108;
        inline constexpr int32_t cubeBlue = 10;
        
        // Light gray (agent parts)  
        inline constexpr int32_t agentGray = 230;
        
        // Red (door/X-axis)
        inline constexpr int32_t axisRed = 230;
        inline constexpr int32_t redGreen = 20;
        inline constexpr int32_t redBlue = 20;
        
        // Yellow (button)
        inline constexpr int32_t yellowRed = 230;
        inline constexpr int32_t yellowGreen = 230;
        inline constexpr int32_t yellowBlue = 20;
        
        // Green (Y-axis)
        inline constexpr int32_t greenRed = 20;
        inline constexpr int32_t axisGreen = 230;
        inline constexpr int32_t greenBlue = 20;
        
        // Blue (Z-axis)
        inline constexpr int32_t blueRed = 20;
        inline constexpr int32_t blueGreen = 20;
        inline constexpr int32_t axisBlue = 230;
    }
    
    // Normalized color values (0.0-1.0)
    namespace normalizedColors {
        // Gray colors
        inline constexpr float wallGray = 0.4f;
        inline constexpr float floorBrownRed = 0.5f;
        inline constexpr float floorBrownGreen = 0.3f;
        inline constexpr float floorBrownBlue = 0.3f;
        
        // White colors
        inline constexpr float white = 1.0f;
    }
    
    // Camera positioning for agents
    inline constexpr float cameraDistance = 100.f;
    inline constexpr float cameraOffsetZ = 0.001f;
    inline constexpr float agentHeightMultiplier = 1.5f;
}

// Display and UI constants
namespace display {
    // Default window resolution
    inline constexpr int32_t defaultWindowWidth = 1920;
    inline constexpr int32_t defaultWindowHeight = 1080;
    
    // Timing constants  
    inline constexpr int32_t defaultSimTickRate = 20;
    inline constexpr float defaultCameraDist = 10.0f;
    inline constexpr float cameraRotationAngle = 30.0f;
    inline constexpr float viewerRotationFactor = 0.4f;
    
    // UI spacing values
    inline constexpr int32_t defaultSpacing = 5;
    inline constexpr int32_t standardSpacing = 7;
    inline constexpr int32_t largeSpacing = 10;
}

// Performance and buffer constants
namespace performance {
    // Buffer sizes
    inline constexpr int32_t defaultBufferSize = 4096;
    inline constexpr int32_t maxProgressEntries = 1000;
    
    // Logging intervals
    inline constexpr int32_t frameLoggingInterval = 100;
    
    // Timing and limits
    inline constexpr int32_t defaultDecimalPlaces = 10;
}

// File format constants
namespace fileFormat {
    // Magic number lengths for file headers
    inline constexpr int32_t replayMagicLength = 7;
    inline constexpr int32_t levelMagicLength = 5;
    
    // Default values
    inline constexpr int32_t defaultSeed = 5;
}


// Math constants
namespace math {
    // Degrees to radians conversion
    inline constexpr float degreesInHalfCircle = 180.0f;
    
    // Camera and view constants
    inline constexpr float cameraFovYDegrees = 0.075f;
    inline constexpr float cameraZNear = 0.75f;
}

}

}
