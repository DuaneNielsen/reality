#include "level_gen.hpp"
#include "asset_ids.hpp"
#include <cmath>

namespace madEscape {

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

// Door-related constants removed - no longer needed

static inline float randInRangeCentered(RandKey key, float range)
{
    return rand::sampleUniform(key) * range - range / 2.f;
}

// Commented out - currently unused but may be useful in future
// static inline float randBetween(Engine &ctx, float min, float max)
// {
//     return ctx.data().rng.sampleUniform() * (max - min) + min;
// }

// Initialize the basic components needed for physics rigid body entities
static inline void setupRigidBodyEntity(
    Engine &ctx,
    Entity e,
    Vector3 pos,
    Quat rot,
    uint32_t object_id,
    EntityType entity_type,
    ResponseType response_type = ResponseType::Dynamic,
    Diag3x3 scale = {1, 1, 1})
{
    ObjectID obj_id { (int32_t)object_id };

    ctx.get<Position>(e) = pos;
    ctx.get<Rotation>(e) = rot;
    ctx.get<Scale>(e) = scale;
    ctx.get<ObjectID>(e) = obj_id;
    ctx.get<Velocity>(e) = {
        Vector3::zero(),
        Vector3::zero(),
    };
    ctx.get<ResponseType>(e) = response_type;
    ctx.get<ExternalForce>(e) = Vector3::zero();
    ctx.get<ExternalTorque>(e) = Vector3::zero();
    ctx.get<EntityType>(e) = entity_type;
}

// Register the entity with the broadphase system
// This is needed for every entity with all the physics components.
// Not registering an entity will cause a crash because the broadphase
// systems will still execute over entities with the physics components.
static void registerRigidBodyEntity(
    Engine &ctx,
    Entity e,
    uint32_t object_id)
{
    ObjectID obj_id { (int32_t)object_id };
    ctx.get<broadphase::LeafID>(e) =
        PhysicsSystem::registerEntity(ctx, e, obj_id);
}


/**
 * Creates agent entities with initial component setup.
 * Called once during persistent entity creation.
 */
static void createAgentEntities(Engine &ctx) {
    for (CountT i = 0; i < consts::numAgents; ++i) {
        Entity agent = ctx.data().agents[i] =
            ctx.makeRenderableEntity<Agent>();

        // Create a render view for the agent
        if (ctx.data().enableRender) {
            float fov = ctx.data().customVerticalFov > 0.0f ? 
                       ctx.data().customVerticalFov : 
                       consts::rendering::cameraFovYDegrees;
            render::RenderingSystem::attachEntityToView(ctx,
                    agent,
                    fov, consts::rendering::cameraZNear,
                    Vector3{0.0f, 1.0f, 1.0f}); // 1.0 units in front of agent (agent looks down +Y), at centerline
        }

        ctx.get<Scale>(agent) = Diag3x3 { 1, 1, 1 };
        ctx.get<ObjectID>(agent) = ObjectID { (int32_t)AssetIDs::AGENT };
        ctx.get<ResponseType>(agent) = ResponseType::Dynamic;
        ctx.get<EntityType>(agent) = EntityType::Agent;
    }
}

/**
 * Find a valid spawn position that avoids collisions with all entities.
 * Uses rejection sampling with configurable exclusion radius.
 * Uses deterministic sub-keys based on agent index for reproducible results.
 */
static inline Vector2 findValidSpawnPosition(Engine &ctx, float exclusion_radius, CountT agent_idx)
{
    CompiledLevel& level = ctx.singleton<CompiledLevel>();

    const float WALL_MARGIN = 2.0f;
    const int MAX_ATTEMPTS = 30;
    float exclusion_sq = exclusion_radius * exclusion_radius;

    // Create deterministic spawn base key for this agent using episode RNG key
    // Use the same pattern as in sim.cpp:144-145 to ensure identical base key
    RandKey episode_key = rand::split_i(ctx.data().initRandKey,
                                       ctx.data().curWorldEpisode - 1, // -1 because episode was already incremented
                                       (uint32_t)ctx.worldID().idx);
    RandKey spawn_base_key = rand::split_i(episode_key, 500u + agent_idx, 0u);

    for (int attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
        // Generate candidate position using deterministic sub-keys
        RandKey attempt_key = rand::split_i(spawn_base_key, attempt, 0u);
        Vector2 candidate = {
            rand::sampleUniform(attempt_key) *
                (level.world_max_x - level.world_min_x - 2*WALL_MARGIN) +
                level.world_min_x + WALL_MARGIN,
            rand::sampleUniform(rand::split_i(attempt_key, 0u, 1u)) *
                (level.world_max_y - level.world_min_y - 2*WALL_MARGIN) +
                level.world_min_y + WALL_MARGIN
        };

        bool valid = true;

        // Check all entities with Position and EntityType components using ECS query
        auto collision_query = ctx.query<Position, EntityType>();
        ctx.iterateQuery(collision_query, [&](Position &entity_pos, EntityType &entity_type) {
            // Skip floor entities (NoEntity type)
            if (entity_type == EntityType::NoEntity) {
                return;
            }

            // Skip agents (don't collide with self)
            if (entity_type == EntityType::Agent) {
                return;
            }

            float dx = candidate.x - entity_pos.x;
            float dy = candidate.y - entity_pos.y;
            float dist_sq = dx*dx + dy*dy;

            if (dist_sq < exclusion_sq) {
                valid = false;
            }
        });

        if (valid) {
            return candidate;
        }
    }

    // Fallback: Center position with small offset
    return Vector2{0.0f, 0.0f};
}

/**
 * Resets agent physics and positions for a new episode.
 * Called during each episode reset.
 */
static void resetAgentPhysics(Engine &ctx) {
    CompiledLevel& level = ctx.singleton<CompiledLevel>();
    
    for (CountT i = 0; i < consts::numAgents; i++) {
        Entity agent_entity = ctx.data().agents[i];
        registerRigidBodyEntity(ctx, agent_entity, AssetIDs::AGENT);

        // Use spawn positions from level if available, otherwise use defaults
        Vector3 pos;
        if (i < level.num_spawns) {
            // Use spawn position from level data
            pos = Vector3 {
                level.spawn_x[i],
                level.spawn_y[i],
                1.0f  // Above ground
            };
        } else {
            // Fallback: place agents in center with slight offset
            pos = Vector3 {
                i * consts::rendering::agentSpacing - 1.0f,  // Slight offset between agents
                0.0f,              // Center of room
                1.0f,              // Above ground
            };
        }

        ctx.get<Position>(agent_entity) = pos;

        // Use spawn facing from level data if available
        float facing_angle = 0.0f;
        if (i < level.num_spawns) {
            facing_angle = level.spawn_facing[i];
        }
        
        ctx.get<Rotation>(agent_entity) = Quat::angleAxis(
            facing_angle,  // Use facing angle from level data
            math::up);

        // Initialize Progress with sentinel values - reward system will set them after physics settles
        Progress &progress = ctx.get<Progress>(agent_entity);
        progress.maxY = -999999.0f;  // Sentinel value to indicate not initialized
        progress.initialY = -999999.0f;

        ctx.get<Velocity>(agent_entity) = {
            Vector3::zero(),
            Vector3::zero(),
        };
        ctx.get<ExternalForce>(agent_entity) = Vector3::zero();
        ctx.get<ExternalTorque>(agent_entity) = Vector3::zero();
        ctx.get<Action>(agent_entity) = Action {
            .moveAmount = 0,
            .moveAngle = 0,
            .rotate = consts::numTurnBuckets / 2,
        };

        ctx.get<StepsTaken>(agent_entity).t = 0;  // Reset to 0 for new episode
        ctx.get<Done>(agent_entity).v = 0;  // Reset done flag for new episode
        ctx.get<CollisionDeath>(agent_entity).died = 0;  // Reset collision death flag
        ctx.get<Reward>(agent_entity).v = 0.0f;  // Reset reward for new episode
        
        // Initialize compass observation to all zeros (will be updated by compassSystem)
        CompassObservation &compass_obs = ctx.get<CompassObservation>(agent_entity);
        for (int j = 0; j < 128; j++) {
            compass_obs.compass[j] = 0.0f;
        }
    }
}

/**
 * Sets up physics for a physics entity.
 * ONLY for physics entities, not render-only.
 */
static void setupEntityPhysics(Engine& ctx, Entity e, uint32_t objectId,
                              Vector3 pos, Quat rot, Diag3x3 scale, int32_t entityTypeValue, int32_t responseTypeValue, bool doneOnCollide) {
    EntityType entityType = static_cast<EntityType>(entityTypeValue);
    ResponseType responseType = static_cast<ResponseType>(responseTypeValue);
    
    setupRigidBodyEntity(ctx, e, pos, rot, objectId,
                       entityType, responseType, scale);
    registerRigidBodyEntity(ctx, e, objectId);
    ctx.get<DoneOnCollide>(e).value = doneOnCollide;
    
}

/**
 * Sets up render-only entity components.
 * ONLY for render-only entities.
 */
static void setupRenderOnlyEntity(Engine& ctx, Entity e, uint32_t objectId,
                                 Vector3 pos, Quat rot, Diag3x3 scale, bool doneOnCollide) {
    ctx.get<Position>(e) = pos;
    ctx.get<Rotation>(e) = rot;
    ctx.get<Scale>(e) = scale;
    ctx.get<ObjectID>(e) = ObjectID{(int32_t)objectId};
    ctx.get<DoneOnCollide>(e).value = doneOnCollide;
    
}

/**
 * Helper function to create the floor plane entity.
 * Called once from createPersistentEntities() during initialization.
 */
static void createFloorPlane(Engine &ctx)
{
    ctx.data().floorPlane = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        ctx.data().floorPlane,
        Vector3 { 0, 0, 0 },
        Quat { 1, 0, 0, 0 },
        AssetIDs::PLANE,
        EntityType::NoEntity, // Floor plane type should never be queried
        ResponseType::Static);
}

/**
 * Simple decoder for harmonic motion parameters from flattened array.
 * Converts target_params[target_idx * 8 + param_idx] to MotionParams structure.
 */
static void decodeHarmonicParams(const float* target_params, int32_t target_idx, MotionParams& params)
{
    int32_t base_idx = target_idx * 8;
    params.omega_x = target_params[base_idx + 0];    // omega_x
    params.omega_y = target_params[base_idx + 1];    // omega_y
    params.center_x = target_params[base_idx + 2];   // center_x
    params.center_y = target_params[base_idx + 3];   // center_y
    params.center_z = target_params[base_idx + 4];   // center_z
    params.mass = target_params[base_idx + 5];       // mass
    params.phase_x = target_params[base_idx + 6];    // phase_x
    params.phase_y = target_params[base_idx + 7];    // phase_y
}

/**
 * Simple decoder for circular motion parameters from flattened array.
 * Converts target_params[target_idx * 8 + param_idx] to MotionParams structure.
 */
static void decodeCircularParams(const float* target_params, int32_t target_idx, MotionParams& params)
{
    int32_t base_idx = target_idx * 8;
    params.omega_x = target_params[base_idx + 0];    // angular_velocity
    params.omega_y = target_params[base_idx + 1];    // randomize_flag
    params.center_x = target_params[base_idx + 2];   // center_x
    params.center_y = target_params[base_idx + 3];   // center_y
    params.center_z = target_params[base_idx + 4];   // center_z
    params.mass = target_params[base_idx + 5];       // direction (will be randomized if needed)
    params.phase_x = target_params[base_idx + 6];    // radius
    params.phase_y = target_params[base_idx + 7];    // initial_angle (will be randomized if needed)
}

/**
 * Helper function to create target entities from CompiledLevel configuration.
 * Creates configurable chase rabbit targets with custom motion equations.
 * Called once from createPersistentEntities() during initialization.
 */
static void createTargetEntity(Engine &ctx)
{
    CompiledLevel& level = ctx.singleton<CompiledLevel>();

    // Initialize target count
    ctx.data().numTargets = level.num_targets;

    // Create targets based on CompiledLevel configuration
    for (int32_t i = 0; i < level.num_targets && i < CompiledLevel::MAX_TARGETS; i++) {
        Entity target = ctx.data().targets[i] = ctx.makeRenderableEntity<TargetEntity>();

        // Set position from configuration
        ctx.get<Position>(target) = Vector3{
            level.target_x[i],
            level.target_y[i],
            level.target_z[i]
        };
        ctx.get<Rotation>(target) = Quat{1, 0, 0, 0};
        ctx.get<Scale>(target) = Diag3x3{0.3f, 0.3f, 0.3f}; // Small sphere
        ctx.get<ObjectID>(target) = ObjectID{(int32_t)AssetIDs::TARGET};

        // Initialize velocity to zero
        ctx.get<Velocity>(target) = {
            Vector3::zero(),
            Vector3::zero(),
        };

        // Set motion parameters from configuration
        MotionParams& params = ctx.get<MotionParams>(target);
        params.motion_type = level.target_motion_type[i];

        if (params.motion_type == 0) {
            // Static motion
            params.omega_x = 0.0f;
            params.omega_y = 0.0f;
            params.center_x = level.target_x[i];
            params.center_y = level.target_y[i];
            params.center_z = level.target_z[i];
            params.mass = 1.0f;
        } else if (params.motion_type == 1) {
            // Harmonic motion - decode parameters from flattened array
            decodeHarmonicParams(level.target_params, i, params);
        } else if (params.motion_type == 2) {
            // Circular motion - decode parameters from flattened array
            decodeCircularParams(level.target_params, i, params);
        } else {
            // Unknown motion type - default to static
            params.omega_x = 0.0f;
            params.omega_y = 0.0f;
            params.center_x = level.target_x[i];
            params.center_y = level.target_y[i];
            params.center_z = level.target_z[i];
            params.mass = 1.0f;
            params.motion_type = 0; // Force to static
        }

        // Set target identification
        TargetTag& tag = ctx.get<TargetTag>(target);
        tag.id = i; // Target index
    }
}

static void createOriginMarkerGizmo(Engine &ctx)
{
    using namespace madEscape::consts::rendering::gizmo;

    // Box 0: Red box along X axis
    ctx.data().originMarkerBoxes[0] = ctx.makeRenderableEntity<RenderOnlyEntity>();
    ctx.get<Position>(ctx.data().originMarkerBoxes[0]) = Vector3{axisMarkerOffset, 0, 0};  // Offset along X
    ctx.get<Rotation>(ctx.data().originMarkerBoxes[0]) = Quat{1, 0, 0, 0};
    ctx.get<Scale>(ctx.data().originMarkerBoxes[0]) = Diag3x3{axisMarkerLength, axisMarkerThickness, axisMarkerThickness};  // Elongated along X
    ctx.get<ObjectID>(ctx.data().originMarkerBoxes[0]) = ObjectID{(int32_t)AssetIDs::AXIS_X};
    
    // Box 1: Green box along Y axis  
    ctx.data().originMarkerBoxes[1] = ctx.makeRenderableEntity<RenderOnlyEntity>();
    ctx.get<Position>(ctx.data().originMarkerBoxes[1]) = Vector3{0, axisMarkerOffset, 0};  // Offset along Y
    ctx.get<Rotation>(ctx.data().originMarkerBoxes[1]) = Quat{1, 0, 0, 0};
    ctx.get<Scale>(ctx.data().originMarkerBoxes[1]) = Diag3x3{axisMarkerThickness, axisMarkerLength, axisMarkerThickness};  // Elongated along Y
    ctx.get<ObjectID>(ctx.data().originMarkerBoxes[1]) = ObjectID{(int32_t)AssetIDs::AXIS_Y};
    
    // Box 2: Blue box along Z axis
    ctx.data().originMarkerBoxes[2] = ctx.makeRenderableEntity<RenderOnlyEntity>();
    ctx.get<Position>(ctx.data().originMarkerBoxes[2]) = Vector3{0, 0, axisMarkerOffset};  // Offset along Z
    ctx.get<Rotation>(ctx.data().originMarkerBoxes[2]) = Quat{1, 0, 0, 0};
    ctx.get<Scale>(ctx.data().originMarkerBoxes[2]) = Diag3x3{axisMarkerThickness, axisMarkerThickness, axisMarkerLength};  // Elongated along Z
    ctx.get<ObjectID>(ctx.data().originMarkerBoxes[2]) = ObjectID{(int32_t)AssetIDs::AXIS_Z};
}

/**
 * Helper function to create lidar ray visualization entities.
 * Creates 128 ray entities for each agent to visualize lidar sensor data.
 * Entities are initially hidden and will be positioned/scaled by the lidar system.
 */
static void createLidarRayEntities(Engine &ctx)
{
    // Create lidar ray entities for visualization
    for (int32_t agent_idx = 0; agent_idx < consts::numAgents; agent_idx++) {
        for (int32_t ray_idx = 0; ray_idx < consts::numLidarSamples; ray_idx++) {
            Entity ray = ctx.makeRenderableEntity<LidarRayEntity>();
            ctx.data().lidarRays[agent_idx][ray_idx] = ray;
            
            // Initially hidden (scale to zero)
            ctx.get<Position>(ray) = Vector3{0, 0, -1000};  // Off-screen
            ctx.get<Rotation>(ray) = Quat{1, 0, 0, 0};
            ctx.get<Scale>(ray) = Diag3x3{0, 0, 0};  // Hidden
            ctx.get<ObjectID>(ray) = ObjectID{(int32_t)AssetIDs::LIDAR_RAY};
        }
    }
}

/**
 * Helper function to create compass indicator visualization entities.
 * Creates one compass indicator entity per agent to show direction to target.
 * Entities are initially hidden and will be positioned/oriented by the compass indicator system.
 */
static void createCompassIndicatorEntities(Engine &ctx)
{
    // Create compass indicator entities for visualization
    for (int32_t agent_idx = 0; agent_idx < consts::numAgents; agent_idx++) {
        Entity indicator = ctx.makeRenderableEntity<CompassIndicatorEntity>();
        ctx.data().compassIndicators[agent_idx] = indicator;

        // Initially hidden (scale to zero)
        ctx.get<Position>(indicator) = Vector3{0, 0, -1000};  // Off-screen
        ctx.get<Rotation>(indicator) = Quat{1, 0, 0, 0};
        ctx.get<Scale>(indicator) = Diag3x3{0, 0, 0};  // Hidden
        ctx.get<ObjectID>(indicator) = ObjectID{(int32_t)AssetIDs::COMPASS_INDICATOR};
    }
}

/**
 * Entry point #1: Called ONCE at simulation startup from Sim constructor.
 * Creates all entities that persist for the entire simulation lifetime.
 */
void createPersistentEntities(Engine &ctx)
{
    // Initialize persistent entity count - we'll track them but not create them yet
    ctx.data().numPersistentLevelEntities = 0;
    
    // Create the floor plane
    createFloorPlane(ctx);

    // Create origin marker gizmo
    createOriginMarkerGizmo(ctx);

    // Create lidar ray visualization entities
    createLidarRayEntities(ctx);

    // Create compass indicator visualization entities
    createCompassIndicatorEntities(ctx);

    // Create target entity for compass tracking
    createTargetEntity(ctx);

    // Phase 1.1: Old border walls removed

    // Create agent entities. Note that this leaves a lot of components
    // uninitialized, these will be set during world generation, which is
    // called for every episode.
    createAgentEntities(ctx);

    // Create persistent level tiles from compiled level data
    // These entities will NOT be registered with physics yet - that happens in resetPersistentEntities()
    CompiledLevel& level = ctx.singleton<CompiledLevel>();
    
    // Use the scale from the compiled level to size tiles properly
    // float tile_scale = level.world_scale;  // Currently unused
    
    // Create persistent tiles
    for (int32_t i = 0; i < level.num_tiles && i < CompiledLevel::MAX_TILES; i++) {
        if (level.tile_persistent[i]) {
            uint32_t objectId = level.object_ids[i];
            
            if (objectId != AssetIDs::INVALID && objectId != 0) {
                
                // Create persistent entity shell only - no physics setup
                bool isRenderOnly = level.tile_render_only[i];
                Entity entity = isRenderOnly ? 
                    ctx.makeRenderableEntity<RenderOnlyEntity>() : 
                    ctx.makeRenderableEntity<PhysicsEntity>();
                
                // Store in persistent array
                if (ctx.data().numPersistentLevelEntities < Sim::MAX_PERSISTENT_TILES) {
                    ctx.data().persistentLevelEntities[ctx.data().numPersistentLevelEntities++] = entity;
                }
            }
        }
    }
}

/**
 * Resets target entities with randomization for a new episode.
 * Called from resetPersistentEntities() each episode.
 * Applies deterministic randomization to circular motion targets if randomize flag is set.
 */
static void resetTargets(Engine &ctx) {
    CompiledLevel& level = ctx.singleton<CompiledLevel>();

    // Iterate through stored target entities instead of using ECS query
    for (CountT i = 0; i < ctx.data().numTargets; i++) {
        Entity target = ctx.data().targets[i];
        MotionParams& params = ctx.get<MotionParams>(target);
        TargetTag& tag = ctx.get<TargetTag>(target);

        // Only randomize circular motion targets with randomize flag set
        if (params.motion_type == 2 && params.omega_y > 0.0f) {
            // Create deterministic episode key using same pattern as spawn generation
            RandKey episode_key = rand::split_i(ctx.data().initRandKey,
                                               ctx.data().curWorldEpisode - 1, // -1 because episode was already incremented
                                               (uint32_t)ctx.worldID().idx);
            RandKey target_base_key = rand::split_i(episode_key, 3000u + tag.id, 0u);

            // Randomize initial angle (0 to 2π)
            float random_initial_angle = rand::sampleUniform(rand::split_i(target_base_key, 0u, 0u)) * 2.0f * math::pi;
            params.phase_y = random_initial_angle;

            // Randomize direction (clockwise vs counter-clockwise)
            float random_direction_val = rand::sampleUniform(rand::split_i(target_base_key, 1u, 0u));
            params.mass = (random_direction_val < 0.5f) ? -1.0f : 1.0f;  // -1 = clockwise, 1 = counter-clockwise
        }
    }
}

/**
 * Resets persistent entities for a new episode.
 * Called from generateWorld() at the start of each episode.
 * 
 * Although agents and walls persist between episodes, we still need to
 * re-register them with the broadphase system and, in the case of the agents,
 * reset their positions.
 */
static void resetPersistentEntities(Engine &ctx)
{
    registerRigidBodyEntity(ctx, ctx.data().floorPlane, AssetIDs::PLANE);

     // Get compiled level for persistent entity setup
     CompiledLevel& level = ctx.singleton<CompiledLevel>();
     // float tile_scale = level.world_scale;  // Currently unused
     
     // Set up physics for persistent level entities
     CountT persistent_idx = 0;
     for (int32_t i = 0; i < level.num_tiles && persistent_idx < ctx.data().numPersistentLevelEntities; i++) {
         if (level.tile_persistent[i]) {
             uint32_t objectId = level.object_ids[i];
             if (objectId != AssetIDs::INVALID && objectId != 0) {
                 Entity e = ctx.data().persistentLevelEntities[persistent_idx++];
                 
                 float x = level.tile_x[i];
                 float y = level.tile_y[i];
                 Diag3x3 scale = {level.tile_scale_x[i], level.tile_scale_y[i], level.tile_scale_z[i]};
                 float z = level.tile_z[i];
                 
                 bool isRenderOnly = level.tile_render_only[i];
                 Quat rotation = level.tile_rotation[i];
                 if (isRenderOnly) {
                     bool doneOnCollideValue = level.tile_done_on_collide[i];
                     setupRenderOnlyEntity(ctx, e, objectId, Vector3{x, y, z}, rotation, scale, doneOnCollideValue);
                 } else {
                     int32_t entityTypeValue = level.tile_entity_type[i];
                     int32_t responseTypeValue = level.tile_response_type[i];
                     bool doneOnCollideValue = level.tile_done_on_collide[i];
                     setupEntityPhysics(ctx, e, objectId, Vector3{x, y, z}, rotation, scale, entityTypeValue, responseTypeValue, doneOnCollideValue);
                 }
             }
         }
     }

     // Reset agent physics and positions for the new episode
     resetAgentPhysics(ctx);

     // Reset targets with randomization for the new episode
     resetTargets(ctx);
}




/**
 * Creates per-episode entities from compiled level data.
 * Called from generateLevel() each episode.
 * Iterates through tile arrays and creates entities for non-empty tiles.
 */
static void generateFromCompiled(Engine &ctx, CompiledLevel* level)
{
    LevelState &level_state = ctx.singleton<LevelState>();
    CountT entity_count = 0;
    
    // Use the scale from the compiled level to size tiles properly
    // float tile_scale = level->scale;  // Currently unused
    
    // Generate tiles from compiled data
    for (int32_t i = 0; i < level->num_tiles && entity_count < CompiledLevel::MAX_TILES; i++) {
        uint32_t objectId = level->object_ids[i];
        float x = level->tile_x[i];
        float y = level->tile_y[i];
        
        Entity entity = Entity::none();
        
        // Create entity based on object ID
        if (objectId != AssetIDs::INVALID && objectId != 0) {  // 0 means empty
            // Check if this entity is persistent and already exists
            bool isPersistent = level->tile_persistent[i];
            
            if (isPersistent) {
                // Persistent entity was already created in createPersistentEntities()
                // Find it by matching position
                for (CountT j = 0; j < ctx.data().numPersistentLevelEntities; j++) {
                    Entity e = ctx.data().persistentLevelEntities[j];
                    if (e != Entity::none()) {
                        Position pos = ctx.get<Position>(e);
                        if (std::abs(pos.x - x) < 0.01f && std::abs(pos.y - y) < 0.01f) {
                            entity = e;
                            break;
                        }
                    }
                }
            } else {
                // Non-persistent entity - create with full physics setup
                Diag3x3 scale = {level->tile_scale_x[i], level->tile_scale_y[i], level->tile_scale_z[i]};
                float z = level->tile_z[i];
                
                // Apply randomization using deterministic sub-keys based on tile index
                Vector3 position{x, y, z};
                // Create deterministic tile key based on tile index for consistent randomization
                // Use the same episode key pattern as spawn generation
                RandKey episode_key = rand::split_i(ctx.data().initRandKey,
                                                   ctx.data().curWorldEpisode - 1, // -1 because episode was already incremented
                                                   (uint32_t)ctx.worldID().idx);
                RandKey tile_base_key = rand::split_i(episode_key, 2000u + (uint32_t)i, 0u);

                if (level->tile_rand_x[i] > 0.0f) {
                    position.x += randInRangeCentered(rand::split_i(tile_base_key, 0u, 0u), level->tile_rand_x[i]);
                }
                if (level->tile_rand_y[i] > 0.0f) {
                    position.y += randInRangeCentered(rand::split_i(tile_base_key, 1u, 0u), level->tile_rand_y[i]);
                }
                if (level->tile_rand_z[i] > 0.0f) {
                    position.z += randInRangeCentered(rand::split_i(tile_base_key, 2u, 0u), level->tile_rand_z[i]);
                }

                // Apply rotation randomization if specified
                Quat rotation = level->tile_rotation[i];
                if (level->tile_rand_rot_z[i] > 0.0f) {
                    // Apply random Z-axis rotation
                    float randomAngle = randInRangeCentered(rand::split_i(tile_base_key, 3u, 0u), level->tile_rand_rot_z[i]);
                    Quat randomRot = Quat::angleAxis(randomAngle, math::up);
                    rotation = randomRot * rotation; // Combine rotations
                }

                // Apply scale randomization if specified (as percentage of base scale)
                if (level->tile_rand_scale_x[i] > 0.0f) {
                    float variation = randInRangeCentered(rand::split_i(tile_base_key, 4u, 0u), level->tile_rand_scale_x[i]);
                    scale.d0 *= (1.0f + variation);
                    // Ensure scale doesn't go negative or too small
                    if (scale.d0 < 0.1f) scale.d0 = 0.1f;
                }
                if (level->tile_rand_scale_y[i] > 0.0f) {
                    float variation = randInRangeCentered(rand::split_i(tile_base_key, 5u, 0u), level->tile_rand_scale_y[i]);
                    scale.d1 *= (1.0f + variation);
                    // Ensure scale doesn't go negative or too small
                    if (scale.d1 < 0.1f) scale.d1 = 0.1f;
                }
                if (level->tile_rand_scale_z[i] > 0.0f) {
                    float variation = randInRangeCentered(rand::split_i(tile_base_key, 6u, 0u), level->tile_rand_scale_z[i]);
                    scale.d2 *= (1.0f + variation);
                    // Ensure scale doesn't go negative or too small
                    if (scale.d2 < 0.1f) scale.d2 = 0.1f;
                }
                
                // Create entity and set up immediately
                bool isRenderOnly = level->tile_render_only[i];
                entity = isRenderOnly ?
                    ctx.makeRenderableEntity<RenderOnlyEntity>() :
                    ctx.makeRenderableEntity<PhysicsEntity>();
                
                if (isRenderOnly) {
                    bool doneOnCollideValue = level->tile_done_on_collide[i];
                    setupRenderOnlyEntity(ctx, entity, objectId, position, rotation, scale, doneOnCollideValue);
                } else {
                    int32_t entityTypeValue = level->tile_entity_type[i];
                    int32_t responseTypeValue = level->tile_response_type[i];
                    bool doneOnCollideValue = level->tile_done_on_collide[i];
                    setupEntityPhysics(ctx, entity, objectId, position, rotation, scale, entityTypeValue, responseTypeValue, doneOnCollideValue);
                }
            }
        }
        
        if (entity != Entity::none()) {
            level_state.rooms[0].entities[entity_count++] = entity;
        }
    }
    
    // Fill remaining slots with none
    for (CountT i = entity_count; i < CompiledLevel::MAX_TILES; i++) {
        level_state.rooms[0].entities[i] = Entity::none();
    }
}

/**
 * Apply random spawn positions after level generation is complete.
 * Called after generateLevel() so all entities exist for collision checking.
 */
static void applyRandomSpawnPositions(Engine &ctx) {
    CompiledLevel& level = ctx.singleton<CompiledLevel>();

    // Only apply random spawning if flag is set
    if (!level.spawn_random) {
        return;
    }


    for (CountT i = 0; i < consts::numAgents; i++) {
        Entity agent_entity = ctx.data().agents[i];

        // New: Random spawn with collision avoidance
        const float EXCLUSION_RADIUS = 3.0f;
        Vector2 spawn_2d = findValidSpawnPosition(ctx, EXCLUSION_RADIUS, i);
        Vector3 pos = Vector3{spawn_2d.x, spawn_2d.y, 1.0f};


        ctx.get<Position>(agent_entity) = pos;

        // Random facing using deterministic sub-key for agent i
        // Use the same episode key pattern as spawn generation
        RandKey episode_key = rand::split_i(ctx.data().initRandKey,
                                           ctx.data().curWorldEpisode - 1, // -1 because episode was already incremented
                                           (uint32_t)ctx.worldID().idx);
        RandKey facing_key = rand::split_i(episode_key, 1000u + i, 0u);
        float facing_angle = rand::sampleUniform(facing_key) * 2.0f * math::pi;

        ctx.get<Rotation>(agent_entity) = Quat::angleAxis(
            facing_angle,  // Random facing angle
            math::up);
    }
}

/**
 * Generates the level-specific entities for this episode.
 * Called from generateWorld() each episode.
 * Uses the compiled level singleton to create tile entities.
 */
static void generateLevel(Engine &ctx)
{
    // Always use compiled level - no fallback to hardcoded generation
    CompiledLevel& level = ctx.singleton<CompiledLevel>();
    if (level.num_tiles == 0) {
        // Fatal error - all managers must provide a level
        FATAL("No compiled level provided! All simulations must use compiled level data.\n"
              "Please provide a valid .lvl file or use the level compiler to create one.");
    }
    
    // Generate from compiled level
    generateFromCompiled(ctx, &level);
}

/**
 * Entry point #2: Called at the START of EACH episode.
 * - Called from initWorld() for the first episode
 * - Called from resetSystem() for subsequent episodes
 * 
 * Coordinates the per-episode world generation process.
 */
void generateWorld(Engine &ctx)
{
    resetPersistentEntities(ctx);
    generateLevel(ctx);
    applyRandomSpawnPositions(ctx);
}

}
