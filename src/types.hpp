#pragma once

#include <madrona/components.hpp>
#include <madrona/math.hpp>
#include <madrona/physics.hpp>
#include <madrona/rand.hpp>
#include <madrona/render/ecs.hpp>
#include <madrona/exec_mode.hpp>
#include <cstring>

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
        int32_t rotate;     // [0, 4] - NON-STANDARD: 0=fast left, 1=slow left, 2=none, 3=slow right, 4=fast right
                           //          NOTE: Lower values = left turn (opposite of geometric convention)
                           //          2 is the center/default (no rotation)
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
        NoEntity,  // Renamed from None to avoid Python keyword conflict
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


    // [GAME_SPECIFIC] Phase 2: Test-Driven Level System
    // Tile types removed - now using object_ids with AssetIDs
    // Object IDs directly reference assets from AssetIDs namespace

    // GPU-compatible compiled level data structure
    // Fixed-size arrays for GPU efficiency, no dynamic allocation
    // Unified CompiledLevel structure matching C API format
    // This replaces both the C++ CompiledLevel and C API MER_CompiledLevel  
    struct CompiledLevel {
        static constexpr int32_t MAX_TILES = 1024;  // 32x32 grid max
        static constexpr int32_t MAX_SPAWNS = 8;    // Max spawn points
        static constexpr int32_t MAX_LEVEL_NAME_LENGTH = 64;  // Max level name length
        
        // Header fields (matching MER_CompiledLevel layout)
        int32_t num_tiles;                // Actual tiles used
        int32_t max_entities;             // BVH sizing - calculated during level compilation
        int32_t width;                    // Grid width
        int32_t height;                   // Grid height  
        float world_scale;                // World scale factor
        bool done_on_collide;             // Episode ends on collision
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
        int32_t tile_entity_type[MAX_TILES]; // EntityType value for each tile (0=None, 1=Cube, 2=Wall, 3=Agent)
        int32_t tile_response_type[MAX_TILES]; // ResponseType value for each tile (0=Dynamic, 1=Kinematic, 2=Static)
        
        // Transform data arrays (per-tile scale and rotation)
        float tile_scale_x[MAX_TILES];    // Local X scale
        float tile_scale_y[MAX_TILES];    // Local Y scale
        float tile_scale_z[MAX_TILES];    // Local Z scale
        float tile_rot_w[MAX_TILES];      // Quaternion W component
        float tile_rot_x[MAX_TILES];      // Quaternion X component
        float tile_rot_y[MAX_TILES];      // Quaternion Y component
        float tile_rot_z[MAX_TILES];      // Quaternion Z component
        
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

    // [BOILERPLATE] Replay metadata structure
    // Magic number to identify replay files
    static constexpr uint32_t REPLAY_MAGIC = 0x4D455352;  // "MESR" (Madrona Escape Room)
    
    // Version of the replay format
    static constexpr uint32_t REPLAY_VERSION = 2;
    
    // Maximum length for sim name string
    static constexpr uint32_t MAX_SIM_NAME_LENGTH = 64;
    
    struct ReplayMetadata {
        uint32_t magic;                         // Magic number for file identification
        uint32_t version;                       // Format version
        char sim_name[MAX_SIM_NAME_LENGTH];     // Name of the simulation
        char level_name[MAX_SIM_NAME_LENGTH];   // Name of the level being played
        uint32_t num_worlds;                    // Number of worlds recorded
        uint32_t num_agents_per_world;          // Number of agents per world
        uint32_t num_steps;                     // Total number of steps recorded
        uint32_t actions_per_step;              // Number of action components (3: move_amount, move_angle, rotate)
        uint64_t timestamp;                     // Unix timestamp when recording started
        uint32_t seed;                          // Random seed used for simulation
        uint32_t reserved[consts::fileFormat::replayMagicLength];  // Reserved for future use
        
        static ReplayMetadata createDefault() {
            ReplayMetadata meta;
            meta.magic = REPLAY_MAGIC;
            meta.version = REPLAY_VERSION;
            std::strncpy(meta.sim_name, "madrona_escape_room", sizeof(meta.sim_name) - 1);
            meta.sim_name[sizeof(meta.sim_name) - 1] = '\0';
            std::strncpy(meta.level_name, "unknown_level", sizeof(meta.level_name) - 1);
            meta.level_name[sizeof(meta.level_name) - 1] = '\0';
            meta.num_worlds = 1;
            meta.num_agents_per_world = 1;
            meta.num_steps = 0;
            meta.actions_per_step = consts::numActionComponents;
            meta.timestamp = 0;
            meta.seed = consts::fileFormat::defaultSeed;
            std::memset(meta.reserved, 0, sizeof(meta.reserved));
            return meta;
        }
        
        bool isValid() const {
            return magic == REPLAY_MAGIC && (version == 1 || version == 2);
        }
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
