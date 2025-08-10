#pragma once

#include <madrona/components.hpp>
#include <madrona/math.hpp>
#include <madrona/physics.hpp>
#include <madrona/rand.hpp>
#include <madrona/render/ecs.hpp>

#include "consts.hpp"

namespace madEscape {

    // Include several madrona types into the simulator namespace for convenience
    using madrona::CountT;
    using madrona::Entity;
    using madrona::RandKey;
    using madrona::base::ObjectID;
    using madrona::base::Position;
    using madrona::base::Rotation;
    using madrona::base::Scale;
    using madrona::phys::ExternalForce;
    using madrona::phys::ExternalTorque;
    using madrona::phys::ResponseType;
    using madrona::phys::RigidBody;
    using madrona::phys::Velocity;


    // [BOILERPLATE]
    // WorldReset is a per-world singleton component that causes the current
    // episode to be terminated and the world regenerated
    // (Singleton components like WorldReset can be accessed via Context::singleton
    // (eg ctx.singleton<WorldReset>().reset = 1)
    struct WorldReset {
        int32_t reset;
    };

    // [GAME_SPECIFIC]
    // Discrete action component for agent movement.
    // Movement is EGOCENTRIC - all directions are relative to agent's current facing.
    // This implements standard WASD+QE game controls.
    struct Action {
        int32_t moveAmount; // [0, 3] - 0=stop, 1=slow, 2=medium, 3=fast
        int32_t moveAngle;  // [0, 7] - 8 directions: 0=forward, 2=right, 4=back, 6=left
                           //          Intermediate values are diagonals (1=forward-right, etc.)
        int32_t rotate;     // [0, 4] - 0=fast left, 1=slow left, 2=none, 3=slow right, 4=fast right
                           //          Note: 2 is the center/default (no rotation)
    };

    //[BOILERPLATE]
    // Per-agent reward
    // Exported as an [N * A, 1] float tensor to training code
    struct Reward {
        float v;
    };

    // [BOILERPLATE]
    // Per-agent component that indicates that the agent's episode is finished
    // This is exported per-agent for simplicity in the training code
    // TODO: Done should be a world-level singleton, not per-agent. The entire
    // episode is done, not individual agents. This causes unnecessary tensor
    // dimensions [num_worlds, num_agents, 1] instead of [num_worlds, 1].
    struct Done {
        // Currently bool components are not supported due to
        // padding issues, so Done is an int32_t
        int32_t v;
    };

    // [GAME_SPECIFIC]
    // Observation state for the current agent.
    // Positions are rescaled to the bounds of the play area to assist training.
    struct SelfObservation {
        float globalX;
        float globalY;
        float globalZ;
        float maxY;
        float theta;
    };
    inline constexpr size_t SelfObservationFloatCount = sizeof(SelfObservation) / sizeof(float);

    // [GAME_SPECIFIC]
    // The state of the world is passed to each agent in terms of egocentric
    // polar coordinates. theta is degrees off agent forward.
    struct PolarObservation {
        float r;
        float theta;
    };


    // [GAME_SPECIFIC]
    // Removed RoomEntityObservations - no longer tracking room entities
    // since interaction mechanics have been removed

    // [BOILERPLATE]
    // Number of steps remaining in the episode. Allows non-recurrent policies
    // to track the progression of time.
    struct StepsRemaining {
        uint32_t t;
    };
    inline constexpr size_t StepsRemainingCount = 1;  // Single value

    //[GAME_SPECIFIC]
    // Tracks progress the agent has made through the challenge, used to add
    // reward when more progress has been made
    struct Progress {
        float maxY;
    };



    // [GAME_SPECIFIC]
    // This enum is used to track the type of each entity for the purposes of
    // classifying the objects hit by each lidar sample.
    enum class EntityType : uint32_t {
        None,
        Cube,
        Wall,
        Agent,
        NumTypes,
    };


    // [BOILERPLATE]
    // Generic archetype for entities that need physics but don't have custom
    // logic associated with them.
    struct PhysicsEntity : public madrona::Archetype<RigidBody, EntityType, madrona::render::Renderable> {};

    // [GAME_SPECIFIC]
    // Archetype for entities that only need rendering, no physics
    struct RenderOnlyEntity : public madrona::Archetype<
        Position, Rotation, Scale, ObjectID,
        madrona::render::Renderable
    > {};


    // [GAME_SPECIFIC]
    // Room itself is not a component but is used by the singleton
    // component "LevelState" (below) to represent the state of the full level
    struct Room {
        // These are entities the agent will interact with
        Entity entities[consts::maxEntitiesPerRoom];
    };

    //[GAME_SPECIFIC]
    // A singleton component storing the state of all the rooms in the current
    // randomly generated level
    struct LevelState {
        Room rooms[consts::numRooms];
    };

    // [GAME_SPECIFIC] Phase 2: Test-Driven Level System
    // Tile types for compiled level system
    enum TileType : int32_t {
        TILE_EMPTY = 0,
        TILE_WALL = 1,
        TILE_CUBE = 2,
        TILE_SPAWN = 3,
        TILE_DOOR = 4,    // Future
        TILE_BUTTON = 5,  // Future
        TILE_GOAL = 6,    // Future
    };

    // GPU-compatible compiled level data structure
    // Fixed-size arrays for GPU efficiency, no dynamic allocation
    struct CompiledLevel {
        static constexpr int32_t MAX_TILES = 256;  // 16x16 grid max
        
        // Tile data (packed for GPU efficiency)
        int32_t tile_types[MAX_TILES];    // Type enum for each tile
        float tile_x[MAX_TILES];          // World X position
        float tile_y[MAX_TILES];          // World Y position
        int32_t num_tiles;                // Actual tiles used
        
        // Agent spawn data
        float spawn_x[consts::numAgents];
        float spawn_y[consts::numAgents];
        float spawn_rot[consts::numAgents];
        
        // Level metadata
        int32_t width;                    // Grid width
        int32_t height;                   // Grid height
        float scale;                      // World scale factor
    };

    /* ECS Archetypes for the game */

    // [GAME_SPECIFIC]
    // There are 2 Agents in the environment trying to get to the destination
    struct Agent
        : public madrona::Archetype<
                  // RigidBody is a "bundle" component defined in physics.hpp in Madrona.
                  // This includes a number of components into the archetype, including
                  // Position, Rotation, Scale, Velocity, and a number of other components
                  // used internally by the physics.
                  RigidBody,

                  // Internal logic state.
                  Progress, EntityType,

                  // Input
                  Action,

                  // Observations
                  SelfObservation, StepsRemaining,

                  // Reward, episode termination
                  Reward, Done,

                  // Visualization: In addition to the fly camera, src/viewer.cpp can
                  // view the scene from the perspective of entities with this component
                  madrona::render::RenderCamera,
                  // All entities with the Renderable component will be drawn by the
                  // viewer and batch renderer
                  madrona::render::Renderable> {};

    // Additional observation dimensions added by the Python wrapper
    inline constexpr size_t AgentIDDimension = 1;  // Agent ID for multi-agent scenarios
    
    // Total observation size for the RL environment
    inline constexpr size_t TotalObservationSize = SelfObservationFloatCount + StepsRemainingCount + AgentIDDimension;

} // namespace madEscape
