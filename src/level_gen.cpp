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

    // Create the outer wall entities
    // Behind
    ctx.data().borders[0] = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        ctx.data().borders[0],
        Vector3 {
            0,
            -5.f,  // Moved farther back to allow room for backward movement tests
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        EntityType::Wall,
        ResponseType::Static,
        Diag3x3 {
            consts::worldWidth + consts::wallWidth * 2,
            consts::wallWidth,
            2.f,
        });

    // Right
    ctx.data().borders[1] = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        ctx.data().borders[1],
        Vector3 {
            consts::worldWidth / 2.f + consts::wallWidth / 2.f,
            consts::worldLength / 2.f - 2.5f,  // Adjusted to span from behind wall to front wall
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        EntityType::Wall,
        ResponseType::Static,
        Diag3x3 {
            consts::wallWidth,
            consts::worldLength + 5.f,  // Extended to cover the gap created by moving behind wall
            2.f,
        });

    // Left
    ctx.data().borders[2] = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        ctx.data().borders[2],
        Vector3 {
            -consts::worldWidth / 2.f - consts::wallWidth / 2.f,
            consts::worldLength / 2.f - 2.5f,  // Adjusted to span from behind wall to front wall
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        EntityType::Wall,
        ResponseType::Static,
        Diag3x3 {
            consts::wallWidth,
            consts::worldLength + 5.f,  // Extended to cover the gap created by moving behind wall
            2.f,
        });

    // Front (fully enclosed - no gap)
    ctx.data().borders[3] = ctx.makeRenderableEntity<PhysicsEntity>();
    setupRigidBodyEntity(
        ctx,
        ctx.data().borders[3],
        Vector3 {
            0,
            consts::worldLength - consts::wallWidth / 2.f,
            0,
        },
        Quat { 1, 0, 0, 0 },
        SimObject::Wall,
        EntityType::Wall,
        ResponseType::Static,
        Diag3x3 {
            consts::worldWidth + consts::wallWidth * 2,
            consts::wallWidth,
            2.f,
        });

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

}

// Although agents and walls persist between episodes, we still need to
// re-register them with the broadphase system and, in the case of the agents,
// reset their positions.
static void resetPersistentEntities(Engine &ctx)
{
    registerRigidBodyEntity(ctx, ctx.data().floorPlane, SimObject::Plane);

     for (CountT i = 0; i < 4; i++) {
         Entity wall_entity = ctx.data().borders[i];
         registerRigidBodyEntity(ctx, wall_entity, SimObject::Wall);
     }

     for (CountT i = 0; i < consts::numAgents; i++) {
         Entity agent_entity = ctx.data().agents[i];
         registerRigidBodyEntity(ctx, agent_entity, SimObject::Agent);

         // Place the single agent near the starting wall (centered)
         Vector3 pos {
             0.f,  // Center of room
             randBetween(ctx, consts::agentRadius * 1.1f,  2.f),
             0.f,
         };

         ctx.get<Position>(agent_entity) = pos;
         ctx.get<Rotation>(agent_entity) = Quat::angleAxis(
             randInRangeCentered(ctx, math::pi / 4.f),
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

static void generateLevel(Engine &ctx)
{
    LevelState &level = ctx.singleton<LevelState>();

    // Generate single room with cube obstacles
    makeRoom(ctx, level, 0);
}

// Randomly generate a new world for a training episode
void generateWorld(Engine &ctx)
{
    resetPersistentEntities(ctx);
    generateLevel(ctx);
}

}
