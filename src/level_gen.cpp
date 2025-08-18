#include "level_gen.hpp"
#include "asset_ids.hpp"

namespace madEscape {

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

// Door-related constants removed - no longer needed

// static inline float randInRangeCentered(Engine &ctx, float range)
// {
//     return ctx.data().rng.sampleUniform() * range - range / 2.f;
// }

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

// Creates floor, outer walls, and agent entities.
// All these entities persist across all episodes.
void createPersistentEntities(Engine &ctx)
{
    // Create the floor entity, just a simple static plane.
    ctx.data().floorPlane = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        ctx.data().floorPlane,
        Vector3 { 0, 0, 0 },
        Quat { 1, 0, 0, 0 },
        AssetIDs::PLANE,
        EntityType::None, // Floor plane type should never be queried
        ResponseType::Static);

    // Phase 1.1: Old border walls removed - using hardcoded 16x16 room instead

    // Create agent entities. Note that this leaves a lot of components
    // uninitialized, these will be set during world generation, which is
    // called for every episode.
    for (CountT i = 0; i < consts::numAgents; ++i) {
        Entity agent = ctx.data().agents[i] =
            ctx.makeRenderableEntity<Agent>();

        // Create a render view for the agent
        if (ctx.data().enableRender) {
            render::RenderingSystem::attachEntityToView(ctx,
                    agent,
                    consts::rendering::cameraDistance, consts::rendering::cameraOffsetZ,
                    consts::rendering::agentHeightMultiplier * math::up);
        }

        ctx.get<Scale>(agent) = Diag3x3 { 1, 1, 1 };
        ctx.get<ObjectID>(agent) = ObjectID { (int32_t)AssetIDs::AGENT };
        ctx.get<ResponseType>(agent) = ResponseType::Dynamic;
        ctx.get<EntityType>(agent) = EntityType::Agent;
    }

    // Create origin marker gizmo - 3 colored boxes for XYZ axes
    // Box 0: Red box along X axis
    ctx.data().originMarkerBoxes[0] = ctx.makeRenderableEntity<RenderOnlyEntity>();
    ctx.get<Position>(ctx.data().originMarkerBoxes[0]) = Vector3{consts::rendering::axisMarkerOffset, 0, 0};  // Offset along X
    ctx.get<Rotation>(ctx.data().originMarkerBoxes[0]) = Quat{1, 0, 0, 0};
    ctx.get<Scale>(ctx.data().originMarkerBoxes[0]) = Diag3x3{consts::rendering::axisMarkerLength, consts::rendering::axisMarkerThickness, consts::rendering::axisMarkerThickness};  // Elongated along X
    ctx.get<ObjectID>(ctx.data().originMarkerBoxes[0]) = ObjectID{(int32_t)AssetIDs::AXIS_X};
    
    // Box 1: Green box along Y axis  
    ctx.data().originMarkerBoxes[1] = ctx.makeRenderableEntity<RenderOnlyEntity>();
    ctx.get<Position>(ctx.data().originMarkerBoxes[1]) = Vector3{0, consts::rendering::axisMarkerOffset, 0};  // Offset along Y
    ctx.get<Rotation>(ctx.data().originMarkerBoxes[1]) = Quat{1, 0, 0, 0};
    ctx.get<Scale>(ctx.data().originMarkerBoxes[1]) = Diag3x3{consts::rendering::axisMarkerThickness, consts::rendering::axisMarkerLength, consts::rendering::axisMarkerThickness};  // Elongated along Y
    ctx.get<ObjectID>(ctx.data().originMarkerBoxes[1]) = ObjectID{(int32_t)AssetIDs::AXIS_Y};
    
    // Box 2: Blue box along Z axis
    ctx.data().originMarkerBoxes[2] = ctx.makeRenderableEntity<RenderOnlyEntity>();
    ctx.get<Position>(ctx.data().originMarkerBoxes[2]) = Vector3{0, 0, consts::rendering::axisMarkerOffset};  // Offset along Z
    ctx.get<Rotation>(ctx.data().originMarkerBoxes[2]) = Quat{1, 0, 0, 0};
    ctx.get<Scale>(ctx.data().originMarkerBoxes[2]) = Diag3x3{consts::rendering::axisMarkerThickness, consts::rendering::axisMarkerThickness, consts::rendering::axisMarkerLength};  // Elongated along Z
    ctx.get<ObjectID>(ctx.data().originMarkerBoxes[2]) = ObjectID{(int32_t)AssetIDs::AXIS_Z};
}

// Although agents and walls persist between episodes, we still need to
// re-register them with the broadphase system and, in the case of the agents,
// reset their positions.
static void resetPersistentEntities(Engine &ctx)
{
    registerRigidBodyEntity(ctx, ctx.data().floorPlane, AssetIDs::PLANE);

     // Phase 1.1: Border wall registration removed - using hardcoded 16x16 room instead
     
     // Get spawn positions from compiled level
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


         ctx.get<Progress>(agent_entity).maxY = pos.y;

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

         ctx.get<StepsRemaining>(agent_entity).t = consts::episodeLen;
     }
}


// Generic entity creator that works with any asset ID
static Entity createEntityFromId(Engine& ctx, uint32_t objectId, 
                                Vector3 pos, Quat rot, Diag3x3 scale) {
    // Check for render-only assets (no physics)
    if (objectId == AssetIDs::AXIS_X || 
        objectId == AssetIDs::AXIS_Y || 
        objectId == AssetIDs::AXIS_Z) {
        Entity e = ctx.makeRenderableEntity<RenderOnlyEntity>();
        ctx.get<Position>(e) = pos;
        ctx.get<Rotation>(e) = rot;
        ctx.get<Scale>(e) = scale;
        ctx.get<ObjectID>(e) = ObjectID{(int32_t)objectId};
        return e;
    } else {
        // Physics entity
        Entity e = ctx.makeRenderableEntity<PhysicsEntity>();
        
        // Determine entity type based on object ID
        EntityType entityType = EntityType::None;
        if (objectId == AssetIDs::WALL) {
            entityType = EntityType::Wall;
        } else if (objectId == AssetIDs::CUBE) {
            entityType = EntityType::Cube;
        } else if (objectId == AssetIDs::AGENT) {
            entityType = EntityType::Agent;
        }
        
        setupRigidBodyEntity(ctx, e, pos, rot, objectId,
                           entityType, ResponseType::Static, scale);
        registerRigidBodyEntity(ctx, e, objectId);
        return e;
    }
}



// A room with 3 cubes as fixed obstacles
// static CountT makeCubeObstacleRoom(Engine &ctx,
//                                    Room &room,
//                                    float y_min,
//                                    float y_max)
// {
//     // Position cubes as obstacles in the room
//     float center_x = 0.f;
//     float center_y = (y_min + y_max) / 2.f;
//
//     float cube_a_x = center_x - 3.f;
//     float cube_a_y = center_y;
//
//     Entity cube_a = makeCube(ctx, cube_a_x, cube_a_y, 1.5f);
//
//     float cube_b_x = center_x;
//     float cube_b_y = center_y;
//
//     Entity cube_b = makeCube(ctx, cube_b_x, cube_b_y, 1.5f);
//
//     float cube_c_x = center_x + 3.f;
//     float cube_c_y = center_y;
//
//     Entity cube_c = makeCube(ctx, cube_c_x, cube_c_y, 1.5f);
//
//     room.entities[0] = cube_a;
//     room.entities[1] = cube_b;
//     room.entities[2] = cube_c;
//
//     return 3;
// }


// Create cube obstacles for the room
// static void makeRoom(Engine &ctx,
//                      LevelState &level,
//                      CountT room_idx)
// {
//     Room &room = level.rooms[room_idx];
//
//     float room_y_min = room_idx * consts::worldLength;
//     float room_y_max = (room_idx + 1) * consts::worldLength;
//
//     // Always create a cube obstacle room
//     CountT num_room_entities = makeCubeObstacleRoom(ctx, room, room_y_min, room_y_max);
//
//     // Need to set any extra entities to type none so random uninitialized data
//     // from prior episodes isn't exported to pytorch as agent observations.
//     for (CountT i = num_room_entities; i < CompiledLevel::MAX_TILES; i++) {
//         room.entities[i] = Entity::none();
//     }
// }

// REMOVED: Hardcoded room generation - now always use ASCII levels

// Original level generation with cube obstacles
// static void generateDefaultLevel(Engine &ctx)
// {
//     LevelState &level = ctx.singleton<LevelState>();
//     // Generate single room with cube obstacles
//     makeRoom(ctx, level, 0);
//     
//     // NOTE: max_entities should be set by compiler, not by level generator
// }

// Phase 2: Generate level from compiled level data
// Simple array iteration - GPU friendly
static void generateFromCompiled(Engine &ctx, CompiledLevel* level)
{
    LevelState &level_state = ctx.singleton<LevelState>();
    CountT entity_count = 0;
    
    // Use the scale from the compiled level to size tiles properly
    float tile_scale = level->scale;
    
    // Generate tiles from compiled data
    for (int32_t i = 0; i < level->num_tiles && entity_count < CompiledLevel::MAX_TILES; i++) {
        uint32_t objectId = level->object_ids[i];
        float x = level->tile_x[i];
        float y = level->tile_y[i];
        
        Entity entity = Entity::none();
        
        // Create entity based on object ID
        if (objectId != AssetIDs::INVALID && objectId != 0) {  // 0 means empty
            // Determine scale based on object type
            Diag3x3 scale = {tile_scale, tile_scale, tile_scale};
            float z = 0.0f;
            
            if (objectId == AssetIDs::WALL) {
                scale = {tile_scale, tile_scale, consts::rendering::wallHeight};  // Walls are taller
            } else if (objectId == AssetIDs::CUBE) {
                float s = consts::rendering::cubeHeightRatio * tile_scale / consts::rendering::cubeScaleFactor;
                scale = {s, s, s};
                z = s;  // Cubes rest on ground with bottom at z=0
            }
            
            entity = createEntityFromId(ctx, objectId, 
                                       Vector3{x, y, z}, 
                                       Quat{1,0,0,0}, scale);
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

static void generateLevel(Engine &ctx)
{
    // Always use compiled level - no fallback to hardcoded generation
    CompiledLevel& level = ctx.singleton<CompiledLevel>();
    if (level.num_tiles == 0) {
        // This should not happen - all managers should provide a level
        printf("ERROR: No compiled level provided! All simulations must use ASCII levels.\n");
        // Create a minimal emergency level to avoid crashes
        level.num_tiles = 4;
        level.width = consts::rendering::emergencyLevelSize; level.height = consts::rendering::emergencyLevelSize; level.scale = consts::rendering::emergencyLevelScale;
        // Emergency 3x3 room
        level.object_ids[consts::rendering::emergencyLevel::tile0] = AssetIDs::WALL; level.tile_x[consts::rendering::emergencyLevel::tile0] = -consts::rendering::emergencyLevelCoord; level.tile_y[consts::rendering::emergencyLevel::tile0] = -consts::rendering::emergencyLevelCoord;
        level.object_ids[consts::rendering::emergencyLevel::tile1] = AssetIDs::WALL; level.tile_x[consts::rendering::emergencyLevel::tile1] =  consts::rendering::emergencyLevelCoord; level.tile_y[consts::rendering::emergencyLevel::tile1] = -consts::rendering::emergencyLevelCoord;
        level.object_ids[consts::rendering::emergencyLevel::tile2] = AssetIDs::WALL; level.tile_x[consts::rendering::emergencyLevel::tile2] = -consts::rendering::emergencyLevelCoord; level.tile_y[consts::rendering::emergencyLevel::tile2] =  consts::rendering::emergencyLevelCoord;
        level.object_ids[consts::rendering::emergencyLevel::tile3] = AssetIDs::WALL; level.tile_x[consts::rendering::emergencyLevel::tile3] =  consts::rendering::emergencyLevelCoord; level.tile_y[consts::rendering::emergencyLevel::tile3] =  consts::rendering::emergencyLevelCoord;
    }
    
    // Generate from compiled level
    generateFromCompiled(ctx, &level);
}

// Randomly generate a new world for a training episode
void generateWorld(Engine &ctx)
{
    resetPersistentEntities(ctx);
    generateLevel(ctx);
}

}
