#include "level_gen.hpp"

namespace madEscape {

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

// Door-related constants removed - no longer needed

static inline float randInRangeCentered(Engine &ctx, float range)
{
    return ctx.data().rng.sampleUniform() * range - range / 2.f;
}

static inline float randBetween(Engine &ctx, float min, float max)
{
    return ctx.data().rng.sampleUniform() * (max - min) + min;
}

// Initialize the basic components needed for physics rigid body entities
static inline void setupRigidBodyEntity(
    Engine &ctx,
    Entity e,
    Vector3 pos,
    Quat rot,
    SimObject sim_obj,
    EntityType entity_type,
    ResponseType response_type = ResponseType::Dynamic,
    Diag3x3 scale = {1, 1, 1})
{
    ObjectID obj_id { (int32_t)sim_obj };

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
    SimObject sim_obj)
{
    ObjectID obj_id { (int32_t)sim_obj };
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
        SimObject::Plane,
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
                    100.f, 0.001f,
                    1.5f * math::up);
        }

        ctx.get<Scale>(agent) = Diag3x3 { 1, 1, 1 };
        ctx.get<ObjectID>(agent) = ObjectID { (int32_t)SimObject::Agent };
        ctx.get<ResponseType>(agent) = ResponseType::Dynamic;
        ctx.get<EntityType>(agent) = EntityType::Agent;
    }

    // Create origin marker gizmo - 3 colored boxes for XYZ axes
    // Box 0: Red box along X axis
    ctx.data().originMarkerBoxes[0] = ctx.makeRenderableEntity<RenderOnlyEntity>();
    ctx.get<Position>(ctx.data().originMarkerBoxes[0]) = Vector3{0.75f, 0, 0};  // Offset along X
    ctx.get<Rotation>(ctx.data().originMarkerBoxes[0]) = Quat{1, 0, 0, 0};
    ctx.get<Scale>(ctx.data().originMarkerBoxes[0]) = Diag3x3{1.5f, 0.3f, 0.3f};  // Elongated along X
    ctx.get<ObjectID>(ctx.data().originMarkerBoxes[0]) = ObjectID{(int32_t)SimObject::AxisX};
    
    // Box 1: Green box along Y axis  
    ctx.data().originMarkerBoxes[1] = ctx.makeRenderableEntity<RenderOnlyEntity>();
    ctx.get<Position>(ctx.data().originMarkerBoxes[1]) = Vector3{0, 0.75f, 0};  // Offset along Y
    ctx.get<Rotation>(ctx.data().originMarkerBoxes[1]) = Quat{1, 0, 0, 0};
    ctx.get<Scale>(ctx.data().originMarkerBoxes[1]) = Diag3x3{0.3f, 1.5f, 0.3f};  // Elongated along Y
    ctx.get<ObjectID>(ctx.data().originMarkerBoxes[1]) = ObjectID{(int32_t)SimObject::AxisY};
    
    // Box 2: Blue box along Z axis
    ctx.data().originMarkerBoxes[2] = ctx.makeRenderableEntity<RenderOnlyEntity>();
    ctx.get<Position>(ctx.data().originMarkerBoxes[2]) = Vector3{0, 0, 0.75f};  // Offset along Z
    ctx.get<Rotation>(ctx.data().originMarkerBoxes[2]) = Quat{1, 0, 0, 0};
    ctx.get<Scale>(ctx.data().originMarkerBoxes[2]) = Diag3x3{0.3f, 0.3f, 1.5f};  // Elongated along Z
    ctx.get<ObjectID>(ctx.data().originMarkerBoxes[2]) = ObjectID{(int32_t)SimObject::AxisZ};
}

// Although agents and walls persist between episodes, we still need to
// re-register them with the broadphase system and, in the case of the agents,
// reset their positions.
static void resetPersistentEntities(Engine &ctx)
{
    registerRigidBodyEntity(ctx, ctx.data().floorPlane, SimObject::Plane);

     // Phase 1.1: Border wall registration removed - using hardcoded 16x16 room instead

     for (CountT i = 0; i < consts::numAgents; i++) {
         Entity agent_entity = ctx.data().agents[i];
         registerRigidBodyEntity(ctx, agent_entity, SimObject::Agent);

         // Place agents in center of 16x16 room
         // In Phase 2, spawn positions will come from CompiledLevel
         Vector3 pos {
             i * 2.0f - 1.0f,  // Slight offset between agents
             0.0f,              // Center of room  
             1.0f,              // Above ground
         };

         ctx.get<Position>(agent_entity) = pos;
         // Phase 1.1: Fixed rotation for hardcoded room
         ctx.get<Rotation>(agent_entity) = Quat::angleAxis(
             0.0f,  // Face forward
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


// Helper function to create a wall entity - will be reused in Phase 2
static Entity makeWall(Engine &ctx,
                      float wall_x,
                      float wall_y,
                      float wall_z,
                      Diag3x3 wall_scale)
{
    Entity wall = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        wall,
        Vector3 { wall_x, wall_y, wall_z },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        EntityType::Wall,
        ResponseType::Static,
        wall_scale);
    registerRigidBodyEntity(ctx, wall, SimObject::Wall);
    return wall;
}

static Entity makeCube(Engine &ctx,
                       float cube_x,
                       float cube_y,
                       float scale = 1.f)
{
    Entity cube = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        cube,
        Vector3 {
            cube_x,
            cube_y,
            1.f * scale,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Cube,
        EntityType::Cube,
        ResponseType::Static,
        Diag3x3 {
            scale,
            scale,
            scale,
        });
    registerRigidBodyEntity(ctx, cube, SimObject::Cube);

    return cube;
}



// A room with 3 cubes as fixed obstacles
static CountT makeCubeObstacleRoom(Engine &ctx,
                                   Room &room,
                                   float y_min,
                                   float y_max)
{
    // Position cubes as obstacles in the room
    float center_x = 0.f;
    float center_y = (y_min + y_max) / 2.f;

    float cube_a_x = center_x - 3.f;
    float cube_a_y = center_y;

    Entity cube_a = makeCube(ctx, cube_a_x, cube_a_y, 1.5f);

    float cube_b_x = center_x;
    float cube_b_y = center_y;

    Entity cube_b = makeCube(ctx, cube_b_x, cube_b_y, 1.5f);

    float cube_c_x = center_x + 3.f;
    float cube_c_y = center_y;

    Entity cube_c = makeCube(ctx, cube_c_x, cube_c_y, 1.5f);

    room.entities[0] = cube_a;
    room.entities[1] = cube_b;
    room.entities[2] = cube_c;

    return 3;
}


// Create cube obstacles for the room
static void makeRoom(Engine &ctx,
                     LevelState &level,
                     CountT room_idx)
{
    Room &room = level.rooms[room_idx];

    float room_y_min = room_idx * consts::worldLength;
    float room_y_max = (room_idx + 1) * consts::worldLength;

    // Always create a cube obstacle room
    CountT num_room_entities = makeCubeObstacleRoom(ctx, room, room_y_min, room_y_max);

    // Need to set any extra entities to type none so random uninitialized data
    // from prior episodes isn't exported to pytorch as agent observations.
    for (CountT i = num_room_entities; i < consts::maxEntitiesPerRoom; i++) {
        room.entities[i] = Entity::none();
    }
}

// Phase 1.1: Hardcoded 16x16 room generation - prototype for Phase 2's generateFromCompiled
static void generateHardcodedRoom(Engine &ctx)
{
    LevelState &level = ctx.singleton<LevelState>();
    
    // Room parameters that will come from CompiledLevel in Phase 2
    static constexpr int32_t ROOM_SIZE = 16;
    static constexpr float TILE_SIZE = 2.0f;  // World units per tile
    static constexpr float WALL_HEIGHT = 2.0f;
    
    CountT entity_count = 0;
    
    // Create walls around perimeter
    // This loop structure mimics how we'll iterate through compiled tile data
    for (int32_t x = 0; x < ROOM_SIZE && entity_count < consts::maxEntitiesPerRoom; x++) {
        for (int32_t y = 0; y < ROOM_SIZE && entity_count < consts::maxEntitiesPerRoom; y++) {
            // Only create walls on edges
            if (x == 0 || x == ROOM_SIZE-1 || 
                y == 0 || y == ROOM_SIZE-1) {
                
                // Convert grid coordinates to world coordinates
                float world_x = (x - ROOM_SIZE/2.0f) * TILE_SIZE;  // Center room at origin
                float world_y = (y - ROOM_SIZE/2.0f) * TILE_SIZE;
                
                // Create wall with proper size to avoid gaps
                Entity wall = makeWall(ctx, world_x, world_y, WALL_HEIGHT / 2.0f, 
                    Diag3x3 { TILE_SIZE, TILE_SIZE, WALL_HEIGHT });
                
                // Store in level state
                level.rooms[0].entities[entity_count++] = wall;
            }
        }
    }
    
    // Fill remaining slots with none
    for (CountT i = entity_count; i < consts::maxEntitiesPerRoom; i++) {
        level.rooms[0].entities[i] = Entity::none();
    }
}

// Original level generation with cube obstacles
static void generateDefaultLevel(Engine &ctx)
{
    LevelState &level = ctx.singleton<LevelState>();
    // Generate single room with cube obstacles
    makeRoom(ctx, level, 0);
}

// Phase 2: Generate level from compiled level data
// Simple array iteration - GPU friendly
static void generateFromCompiled(Engine &ctx, CompiledLevel* level)
{
    LevelState &level_state = ctx.singleton<LevelState>();
    CountT entity_count = 0;
    
    // Generate tiles from compiled data
    for (int32_t i = 0; i < level->num_tiles && entity_count < consts::maxEntitiesPerRoom; i++) {
        TileType type = (TileType)level->tile_types[i];
        float x = level->tile_x[i];
        float y = level->tile_y[i];
        
        Entity entity = Entity::none();
        
        switch(type) {
            case TILE_WALL:
                entity = makeWall(ctx, x, y, 1.0f, Diag3x3{1.0f, 1.0f, 2.0f});
                break;
            case TILE_CUBE:
                entity = makeCube(ctx, x, y, 1.5f);
                break;
            case TILE_EMPTY:
            case TILE_SPAWN:
            case TILE_DOOR:
            case TILE_BUTTON:
            case TILE_GOAL:
                // Skip these for now - TILE_SPAWN handled in resetPersistentEntities
                break;
        }
        
        if (entity != Entity::none()) {
            level_state.rooms[0].entities[entity_count++] = entity;
        }
    }
    
    // Fill remaining slots with none
    for (CountT i = entity_count; i < consts::maxEntitiesPerRoom; i++) {
        level_state.rooms[0].entities[i] = Entity::none();
    }
}

static void generateLevel(Engine &ctx)
{
    // Phase 2: Check if we have a compiled level singleton
    if (ctx.singleton<CompiledLevel>().num_tiles > 0) {
        // Generate from compiled level
        generateFromCompiled(ctx, &ctx.singleton<CompiledLevel>());
    } else {
        // Fall back to hardcoded room (Phase 1.1)
        generateHardcodedRoom(ctx);
    }
}

// Randomly generate a new world for a training episode
void generateWorld(Engine &ctx)
{
    resetPersistentEntities(ctx);
    generateLevel(ctx);
}

}
