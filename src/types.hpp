#pragma once

#include <madrona/components.hpp>
#include <madrona/math.hpp>
#include <madrona/physics.hpp>
#include <madrona/rand.hpp>
#include <madrona/render/ecs.hpp>
#include <madrona/exec_mode.hpp>

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
    using madrona::math::Quat;


    // [BOILERPLATE]
    // WorldReset is a per-world singleton component that causes the current
    // episode to be terminated and the world regenerated
    // (Singleton components like WorldReset can be accessed via Context::singleton
    // (eg ctx.singleton<WorldReset>().reset = 1)
    struct WorldReset {
        int32_t reset;
    };

    // [GAME_SPECIFIC]
    // LidarVisControl is a per-world singleton component that controls
    // whether lidar ray visualization is enabled
    struct LidarVisControl {
        int32_t enabled;  // 0=disabled, 1=enabled
    };

    // [GAME_SPECIFIC]
    // Discrete action component for agent movement.
    // Movement is EGOCENTRIC - all directions are relative to agent's current facing.
    // This implements standard WASD+QE game controls.
    struct Action {
        int32_t moveAmount; // [0, 3] - 0=stop, 1=slow, 2=medium, 3=fast
        int32_t moveAngle;  // [0, 7] - 8 directions: 0=forward, 2=right, 4=back, 6=left
                           //          Intermediate values are diagonals (1=forward-right, etc.)
        int32_t rotate;     // [0, 4] - NON-STANDARD: 0=fast left, 1=slow left, 2=none, 3=slow right, 4=fast right
                           //          NOTE: Lower values = left turn (opposite of geometric convention)
                           //          2 is the center/default (no rotation)
    };

    // [GAME_SPECIFIC]
    // Component to track whether collision with this entity should end the episode
    struct DoneOnCollide {
        bool value;
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
    // Compass observation - one-hot encoded agent facing direction in world frame
    // 128 buckets covering 360 degrees (2.8125 degrees per bucket)
    struct CompassObservation {
        float compass[128];  // One-hot encoded compass direction
    };
    inline constexpr size_t CompassObservationFloatCount = 128;

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

    // [GAME_SPECIFIC]  
    // Number of steps taken in the episode (counts up from 0)
    struct StepsTaken {
        uint32_t t;
    };
    inline constexpr size_t StepsTakenCount = 1;  // Single value

    // [GAME_SPECIFIC]
    struct LidarSample {
        float depth;
    };

    // [GAME_SPECIFIC]
    // Linear depth values and entity type in a circle around the agent
    struct Lidar {
        LidarSample samples[consts::numLidarSamples];
    };

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
        NoEntity,  // Renamed from None to avoid Python keyword conflict
        Cube,
        Wall,
        Agent,
        NumTypes,
    };


    // [BOILERPLATE]
    // Generic archetype for entities that need physics but don't have custom
    // logic associated with them.
    struct PhysicsEntity : public madrona::Archetype<RigidBody, EntityType, DoneOnCollide, madrona::render::Renderable> {};

    // [GAME_SPECIFIC]
    // Archetype for entities that only need rendering, no physics
    struct RenderOnlyEntity : public madrona::Archetype<
        Position, Rotation, Scale, ObjectID, DoneOnCollide,
        madrona::render::Renderable
    > {};

    // [GAME_SPECIFIC]
    // Archetype for lidar ray visualization
    struct LidarRayEntity : public madrona::Archetype<
        Position, Rotation, Scale, ObjectID,
        madrona::render::Renderable
    > {};


    // [GAME_SPECIFIC] Phase 2: Test-Driven Level System
    // Tile types removed - now using object_ids with AssetIDs
    // Object IDs directly reference assets from AssetIDs namespace

    // GPU-compatible compiled level data structure
    // Fixed-size arrays for GPU efficiency, no dynamic allocation
    // Unified CompiledLevel structure matching C API format
    // This replaces both the C++ CompiledLevel and C API MER_CompiledLevel  
    struct CompiledLevel {
        static constexpr int32_t MAX_TILES = consts::limits::maxTiles;  // From consts.hpp
        static constexpr int32_t MAX_SPAWNS = consts::limits::maxSpawns;  // From consts.hpp
        static constexpr int32_t MAX_LEVEL_NAME_LENGTH = consts::limits::maxLevelNameLength;  // From consts.hpp
        
        // Header fields (matching MER_CompiledLevel layout)
        int32_t num_tiles;                // Actual tiles used
        int32_t max_entities;             // BVH sizing - calculated during level compilation
        int32_t width;                    // Grid width
        int32_t height;                   // Grid height  
        float world_scale;                // World scale factor
        char level_name[MAX_LEVEL_NAME_LENGTH];  // Level name for identification
        
        // World boundaries in world units (calculated from grid dimensions * scale)
        float world_min_x;                // Minimum X boundary in world units
        float world_max_x;                // Maximum X boundary in world units
        float world_min_y;                // Minimum Y boundary in world units
        float world_max_y;                // Maximum Y boundary in world units
        float world_min_z;                // Minimum Z boundary in world units
        float world_max_z;                // Maximum Z boundary in world units
        
        // Spawn data
        int32_t num_spawns;               // Number of spawn points
        float spawn_x[MAX_SPAWNS];        // Spawn X positions
        float spawn_y[MAX_SPAWNS];        // Spawn Y positions
        float spawn_facing[MAX_SPAWNS];   // Spawn facing angles in radians
        
        // Tile data arrays (packed for GPU efficiency)
        int32_t object_ids[MAX_TILES];    // Asset ID for each tile (from AssetIDs namespace)
        float tile_x[MAX_TILES];          // World X position
        float tile_y[MAX_TILES];          // World Y position
        float tile_z[MAX_TILES];          // World Z position
        bool tile_persistent[MAX_TILES];  // Whether tile persists across episodes
        bool tile_render_only[MAX_TILES]; // Whether tile is render-only (no physics)
        bool tile_done_on_collide[MAX_TILES]; // Whether collision with this tile ends episode
        int32_t tile_entity_type[MAX_TILES]; // EntityType value for each tile (0=None, 1=Cube, 2=Wall, 3=Agent)
        int32_t tile_response_type[MAX_TILES]; // ResponseType value for each tile (0=Dynamic, 1=Kinematic, 2=Static)
        
        // Transform data arrays (per-tile scale and rotation)
        float tile_scale_x[MAX_TILES];    // Local X scale
        float tile_scale_y[MAX_TILES];    // Local Y scale
        float tile_scale_z[MAX_TILES];    // Local Z scale
        Quat tile_rotation[MAX_TILES];    // Rotation quaternion for each tile
        
        // Randomization arrays (per-tile random ranges)
        float tile_rand_x[MAX_TILES];     // Random X offset range (-range/2 to +range/2)
        float tile_rand_y[MAX_TILES];     // Random Y offset range (-range/2 to +range/2)
        float tile_rand_z[MAX_TILES];     // Random Z offset range (-range/2 to +range/2)
        float tile_rand_rot_z[MAX_TILES]; // Random Z-axis rotation range in radians
        float tile_rand_scale_x[MAX_TILES]; // Random X scale variation range
        float tile_rand_scale_y[MAX_TILES]; // Random Y scale variation range
        float tile_rand_scale_z[MAX_TILES]; // Random Z scale variation range
    };

    // [GAME_SPECIFIC]
    // Room itself is not a component but is used by the singleton
    // component "LevelState" (below) to represent the state of the full level
    struct Room {
        // These are entities the agent will interact with
        // Using CompiledLevel::MAX_TILES as upper bound for entity storage
        Entity entities[CompiledLevel::MAX_TILES];
    };

    //[GAME_SPECIFIC]
    // A singleton component storing the state of all the rooms in the current
    // randomly generated level
    struct LevelState {
        Room rooms[consts::numRooms];
    };

    // [BOILERPLATE] Error codes for C API
    enum class Result : int32_t {
        Success = 0,
        ErrorNullPointer = -1,
        ErrorInvalidParameter = -2,
        ErrorAllocationFailed = -3,
        ErrorNotInitialized = -4,
        ErrorCudaFailure = -5,
        ErrorFileNotFound = -6,
        ErrorInvalidFile = -7,
        ErrorFileIO = -8,
    };


    // [BOILERPLATE] Manager configuration
    struct ManagerConfig {
        madrona::ExecMode exec_mode;
        int gpu_id;
        uint32_t num_worlds;
        uint32_t rand_seed;
        bool auto_reset;
        bool enable_batch_renderer;
        uint32_t batch_render_view_width;   // Default: 64
        uint32_t batch_render_view_height;  // Default: 64
        float custom_vertical_fov;          // Custom vertical FOV in degrees (0 = use default)
        int32_t render_mode;                // Render mode: 0=RGBD, 1=Depth (default: 0)
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
                  SelfObservation, CompassObservation, Lidar, StepsTaken,

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
    inline constexpr size_t TotalObservationSize = SelfObservationFloatCount + StepsTakenCount + AgentIDDimension;

} // namespace madEscape
