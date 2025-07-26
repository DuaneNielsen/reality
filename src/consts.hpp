#pragma once

#include <madrona/types.hpp>

namespace madEscape {

namespace consts {
// Each random world is composed of a fixed number of rooms that the agents
// must solve in order to maximize their reward.
inline constexpr madrona::CountT numRooms = 1;

// Generated levels assume 1 agent (reduced from 2)
inline constexpr madrona::CountT numAgents = 1;

// Maximum number of interactive objects per challenge room. This is needed
// in order to setup the fixed-size learning tensors appropriately.
inline constexpr madrona::CountT maxEntitiesPerRoom = 6;

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
    namespace rotate {
        inline constexpr int32_t FAST_LEFT = 0;
        inline constexpr int32_t SLOW_LEFT = 1;
        inline constexpr int32_t NONE = 2;  // Center value - no rotation
        inline constexpr int32_t SLOW_RIGHT = 3;
        inline constexpr int32_t FAST_RIGHT = 4;
    }
}

// Time (seconds) per step
inline constexpr float deltaT = 0.04f;

// Number of physics substeps
inline constexpr madrona::CountT numPhysicsSubsteps = 4.f;

}

}
