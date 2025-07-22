#include <madrona/mw_gpu_entry.hpp>

#include "sim.hpp"
#include "level_gen.hpp"

#include <algorithm>

/*
 * This file uses a three-tier classification system for code:
 * 
 * [BOILERPLATE] - Pure Madrona framework code that never changes
 * [REQUIRED_INTERFACE] - Methods/structures every environment must implement
 * [GAME_SPECIFIC] - Implementation details unique to this escape room game
 * 
 * When creating a new environment, focus on:
 * - Implementing all [REQUIRED_INTERFACE] methods with your game's content
 * - Replacing all [GAME_SPECIFIC] code with your game's logic
 * - Leave all [BOILERPLATE] code unchanged
 */

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;

namespace RenderingSystem = madrona::render::RenderingSystem;

namespace madEscape {

// [REQUIRED_INTERFACE] Register all the ECS components and archetypes that will be
// used in the simulation - every environment must implement this method
void Sim::registerTypes(ECSRegistry &registry, const Config &cfg)
{
    // [BOILERPLATE] Always register base and physics types
    base::registerTypes(registry);
    phys::PhysicsSystem::registerTypes(registry);

    // [BOILERPLATE] Register rendering types if enabled
    RenderingSystem::registerTypes(registry, cfg.renderBridge);

    // [BOILERPLATE] Core components every environment needs
    registry.registerComponent<Action>();
    registry.registerComponent<Reward>();
    registry.registerComponent<Done>();
    
    // [GAME_SPECIFIC] Escape room specific components
    registry.registerComponent<SelfObservation>();
    registry.registerComponent<Progress>();
    registry.registerComponent<OtherAgents>();
    registry.registerComponent<PartnerObservations>();
    registry.registerComponent<RoomEntityObservations>();
    registry.registerComponent<StepsRemaining>();
    registry.registerComponent<EntityType>();

    // [REQUIRED_INTERFACE] Reset singleton - every episodic env needs this
    registry.registerSingleton<WorldReset>();
    
    // [GAME_SPECIFIC] Level management singleton
    registry.registerSingleton<LevelState>();

    // [GAME_SPECIFIC] Escape room archetypes
    registry.registerArchetype<Agent>();
    registry.registerArchetype<PhysicsEntity>();

    // [REQUIRED_INTERFACE] Export reset control
    registry.exportSingleton<WorldReset>(
        (uint32_t)ExportID::Reset);
    
    // [BOILERPLATE] Export core RL components
    registry.exportColumn<Agent, Action>(
        (uint32_t)ExportID::Action);
    registry.exportColumn<Agent, Reward>(
        (uint32_t)ExportID::Reward);
    registry.exportColumn<Agent, Done>(
        (uint32_t)ExportID::Done);
    
    // [GAME_SPECIFIC] Export escape room observations
    registry.exportColumn<Agent, SelfObservation>(
        (uint32_t)ExportID::SelfObservation);
    registry.exportColumn<Agent, PartnerObservations>(
        (uint32_t)ExportID::PartnerObservations);
    registry.exportColumn<Agent, RoomEntityObservations>(
        (uint32_t)ExportID::RoomEntityObservations);
    registry.exportColumn<Agent, StepsRemaining>(
        (uint32_t)ExportID::StepsRemaining);
}

// [GAME_SPECIFIC] Helper to clean up escape room entities
static inline void cleanupWorld(Engine &ctx)
{
    // Destroy current level entities
    LevelState &level = ctx.singleton<LevelState>();
    for (CountT i = 0; i < consts::numRooms; i++) {
        Room &room = level.rooms[i];
        for (CountT j = 0; j < consts::maxEntitiesPerRoom; j++) {
            if (room.entities[j] != Entity::none()) {
                ctx.destroyRenderableEntity(room.entities[j]);
            }
        }

        ctx.destroyRenderableEntity(room.walls[0]);
        ctx.destroyRenderableEntity(room.walls[1]);
    }
}

// [GAME_SPECIFIC] Helper to initialize a new escape room world
static inline void initWorld(Engine &ctx)
{
    // [BOILERPLATE] Always reset physics first
    phys::PhysicsSystem::reset(ctx);

    // [BOILERPLATE] Assign a new episode ID and RNG
    ctx.data().rng = RNG(rand::split_i(ctx.data().initRandKey,
        ctx.data().curWorldEpisode++, (uint32_t)ctx.worldID().idx));

    // [GAME_SPECIFIC] Defined in src/level_gen.hpp / src/level_gen.cpp
    generateWorld(ctx);
}

// [REQUIRED_INTERFACE] Reset system - every episodic environment needs this
// This system runs each frame and checks if the current episode is complete
// or if code external to the application has forced a reset by writing to the
// WorldReset singleton.
//
// If a reset is needed, cleanup the existing world and generate a new one.
inline void resetSystem(Engine &ctx, WorldReset &reset)
{
    int32_t should_reset = reset.reset;
    if (ctx.data().autoReset) {
        for (CountT i = 0; i < consts::numAgents; i++) {
            Entity agent = ctx.data().agents[i];
            Done done = ctx.get<Done>(agent);
            if (done.v) {
                should_reset = 1;
            }
        }
    }

    if (should_reset != 0) {
        reset.reset = 0;

        cleanupWorld(ctx);
        initWorld(ctx);
    }
}

// [GAME_SPECIFIC] Translates discrete actions from the Action component to forces
// used by the physics simulation.
inline void movementSystem(Engine &,
                           Action &action, 
                           Rotation &rot, 
                           ExternalForce &external_force,
                           ExternalTorque &external_torque)
{
    constexpr float move_max = 1000;
    constexpr float turn_max = 320;

    Quat cur_rot = rot;

    float move_amount = action.moveAmount *
        (move_max / (consts::numMoveAmountBuckets - 1));

    constexpr float move_angle_per_bucket =
        2.f * math::pi / float(consts::numMoveAngleBuckets);

    float move_angle = float(action.moveAngle) * move_angle_per_bucket;

    float f_x = move_amount * sinf(move_angle);
    float f_y = move_amount * cosf(move_angle);

    constexpr float turn_delta_per_bucket = 
        turn_max / (consts::numTurnBuckets / 2);
    float t_z =
        turn_delta_per_bucket * (action.rotate - consts::numTurnBuckets / 2);

    external_force = cur_rot.rotateVec({ f_x, f_y, 0 });
    external_torque = Vector3 { 0, 0, t_z };
}


// [GAME_SPECIFIC] Make the agents easier to control by zeroing out their velocity
// after each step.
inline void agentZeroVelSystem(Engine &,
                               Velocity &vel,
                               Action &)
{
    vel.linear.x = 0;
    vel.linear.y = 0;
    vel.linear.z = fminf(vel.linear.z, 0);

    vel.angular = Vector3::zero();
}

// [GAME_SPECIFIC] Helper functions for observation normalization
static inline float distObs(float v)
{
    return v / consts::worldLength;
}

static inline float globalPosObs(float v)
{
    return v / consts::worldLength;
}

static inline float angleObs(float v)
{
    return v / math::pi;
}

// [GAME_SPECIFIC] Translate xy delta to polar observations for learning.
static inline PolarObservation xyToPolar(Vector3 v)
{
    Vector2 xy { v.x, v.y };

    float r = xy.length();

    // Note that this is angle off y-forward
    float theta = atan2f(xy.x, xy.y);

    return PolarObservation {
        .r = distObs(r),
        .theta = angleObs(theta),
    };
}

static inline float encodeType(EntityType type)
{
    return (float)type / (float)EntityType::NumTypes;
}

static inline float computeZAngle(Quat q)
{
    float siny_cosp = 2.f * (q.w * q.z + q.x * q.y);
    float cosy_cosp = 1.f - 2.f * (q.y * q.y + q.z * q.z);
    return atan2f(siny_cosp, cosy_cosp);
}

// [REQUIRED_INTERFACE] This system packages all the egocentric observations together 
// for the policy inputs. Every environment must implement observation collection.
// [GAME_SPECIFIC] The specific observations collected and their format.
inline void collectObservationsSystem(Engine &ctx,
                                      Position pos,
                                      Rotation rot,
                                      const Progress &progress,
                                      const OtherAgents &other_agents,
                                      SelfObservation &self_obs,
                                      PartnerObservations &partner_obs,
                                      RoomEntityObservations &room_ent_obs)
{
    CountT cur_room_idx = CountT(pos.y / consts::roomLength);
    cur_room_idx = std::max(CountT(0), 
        std::min(consts::numRooms - 1, cur_room_idx));

    self_obs.roomX = pos.x / (consts::worldWidth / 2.f);
    self_obs.roomY = (pos.y - cur_room_idx * consts::roomLength) /
        consts::roomLength;
    self_obs.globalX = globalPosObs(pos.x);
    self_obs.globalY = globalPosObs(pos.y);
    self_obs.globalZ = globalPosObs(pos.z);
    self_obs.maxY = globalPosObs(progress.maxY);
    self_obs.theta = angleObs(computeZAngle(rot));

    Quat to_view = rot.inv();

#pragma unroll
    for (CountT i = 0; i < consts::numAgents - 1; i++) {
        Entity other = other_agents.e[i];

        Vector3 other_pos = ctx.get<Position>(other);
        Vector3 to_other = other_pos - pos;

        partner_obs.obs[i] = {
            .polar = xyToPolar(to_view.rotateVec(to_other)),
        };
    }

    const LevelState &level = ctx.singleton<LevelState>();
    const Room &room = level.rooms[cur_room_idx];

    for (CountT i = 0; i < consts::maxEntitiesPerRoom; i++) {
        Entity entity = room.entities[i];

        EntityObservation ob;
        if (entity == Entity::none()) {
            ob.polar = { 0.f, 1.f };
            ob.encodedType = encodeType(EntityType::None);
        } else {
            Vector3 entity_pos = ctx.get<Position>(entity);
            EntityType entity_type = ctx.get<EntityType>(entity);

            Vector3 to_entity = entity_pos - pos;
            ob.polar = xyToPolar(to_view.rotateVec(to_entity));
            ob.encodedType = encodeType(entity_type);
        }

        room_ent_obs.obs[i] = ob;
    }

}


// [REQUIRED_INTERFACE] Computes reward for each agent - every environment needs a reward system
// [GAME_SPECIFIC] Keeps track of the max distance achieved
// so far through the challenge. Continuous reward is provided for any new
// distance achieved.
inline void rewardSystem(Engine &,
                         Position pos,
                         Progress &progress,
                         Reward &out_reward)
{
    // Just in case agents do something crazy, clamp total reward
    float reward_pos = fminf(pos.y, consts::worldLength * 2);

    float old_max_y = progress.maxY;

    float new_progress = reward_pos - old_max_y;

    float reward;
    if (new_progress > 0) {
        reward = new_progress * consts::rewardPerDist;
        progress.maxY = reward_pos;
    } else {
        reward = consts::slackReward;
    }

    out_reward.v = reward;
}

// [REQUIRED_INTERFACE] Keep track of the number of steps remaining in the episode and
// notify training that an episode has completed by
// setting done = 1 on the final step of the episode
inline void stepTrackerSystem(Engine &,
                              StepsRemaining &steps_remaining,
                              Done &done)
{
    int32_t num_remaining = --steps_remaining.t;
    if (num_remaining == consts::episodeLen - 1) {
        done.v = 0;
    } else if (num_remaining == 0) {
        done.v = 1;
    }

}

// [BOILERPLATE] Helper function for sorting nodes in the taskgraph.
// Sorting is only supported / required on the GPU backend,
// since the CPU backend currently keeps separate tables for each world.
// This will likely change in the future with sorting required for both
// environments
#ifdef MADRONA_GPU_MODE
template <typename ArchetypeT>
TaskGraph::NodeID queueSortByWorld(TaskGraph::Builder &builder,
                                   Span<const TaskGraph::NodeID> deps)
{
    auto sort_sys =
        builder.addToGraph<SortArchetypeNode<ArchetypeT, WorldID>>(
            deps);
    auto post_sort_reset_tmp =
        builder.addToGraph<ResetTmpAllocNode>({sort_sys});

    return post_sort_reset_tmp;
}
#endif

// [REQUIRED_INTERFACE] Build the task graph - every environment must implement this
void Sim::setupTasks(TaskGraphManager &taskgraph_mgr, const Config &cfg)
{
    // [BOILERPLATE] Initialize task graph builder
    TaskGraphBuilder &builder = taskgraph_mgr.init(TaskGraphID::Step);

    // [GAME_SPECIFIC] Turn policy actions into movement
    auto move_sys = builder.addToGraph<ParallelForNode<Engine,
        movementSystem,
            Action,
            Rotation,
            ExternalForce,
            ExternalTorque
        >>({});

    // [BOILERPLATE] Build BVH for broadphase / raycasting
    auto broadphase_setup_sys = phys::PhysicsSystem::setupBroadphaseTasks(
        builder, {move_sys});

    // [BOILERPLATE] Physics collision detection and solver
    auto substep_sys = phys::PhysicsSystem::setupPhysicsStepTasks(builder,
        {broadphase_setup_sys}, consts::numPhysicsSubsteps);

    // [GAME_SPECIFIC] Improve controllability of agents by setting their velocity to 0
    // after physics is done.
    auto agent_zero_vel = builder.addToGraph<ParallelForNode<Engine,
        agentZeroVelSystem, Velocity, Action>>(
            {substep_sys});

    // [BOILERPLATE] Finalize physics subsystem work
    auto phys_done = phys::PhysicsSystem::setupCleanupTasks(
        builder, {agent_zero_vel});

    // [REQUIRED_INTERFACE] Compute initial reward now that physics has updated the world state
    auto reward_sys = builder.addToGraph<ParallelForNode<Engine,
         rewardSystem,
            Position,
            Progress,
            Reward
        >>({phys_done});

    // [REQUIRED_INTERFACE] Check if the episode is over
    auto done_sys = builder.addToGraph<ParallelForNode<Engine,
        stepTrackerSystem,
            StepsRemaining,
            Done
        >>({reward_sys});

    // [REQUIRED_INTERFACE] Conditionally reset the world if the episode is over
    auto reset_sys = builder.addToGraph<ParallelForNode<Engine,
        resetSystem,
            WorldReset
        >>({done_sys});

    // [BOILERPLATE] Clear temporary allocations
    auto clear_tmp = builder.addToGraph<ResetTmpAllocNode>({reset_sys});
    (void)clear_tmp;

#ifdef MADRONA_GPU_MODE
    // [BOILERPLATE] RecycleEntitiesNode is required on the GPU backend in order to reclaim
    // deleted entity IDs.
    auto recycle_sys = builder.addToGraph<RecycleEntitiesNode>({reset_sys});
    (void)recycle_sys;
#endif

    // [BOILERPLATE] This second BVH build is a limitation of the current taskgraph API.
    // It's only necessary if the world was reset, but we don't have a way
    // to conditionally queue taskgraph nodes yet.
    auto post_reset_broadphase = phys::PhysicsSystem::setupBroadphaseTasks(
        builder, {reset_sys});

    // [REQUIRED_INTERFACE] Finally, collect observations for the next step.
    auto collect_obs = builder.addToGraph<ParallelForNode<Engine,
        collectObservationsSystem,
            Position,
            Rotation,
            Progress,
            OtherAgents,
            SelfObservation,
            PartnerObservations,
            RoomEntityObservations
        >>({post_reset_broadphase});


    // [BOILERPLATE] Set up rendering tasks if enabled
    if (cfg.renderBridge) {
        RenderingSystem::setupTasks(builder, {reset_sys});
    }

#ifdef MADRONA_GPU_MODE
    // [BOILERPLATE] Sort entities, this could be conditional on reset like the second
    // BVH build above.
    // [GAME_SPECIFIC] The specific entity types to sort are game-specific
    auto sort_agents = queueSortByWorld<Agent>(
        builder, {collect_obs});
    auto sort_phys_objects = queueSortByWorld<PhysicsEntity>(
        builder, {sort_agents});
    (void)sort_phys_objects;
#else
    (void)collect_obs;
#endif
}

// [REQUIRED_INTERFACE] Constructor - every environment must implement this
Sim::Sim(Engine &ctx,
         const Config &cfg,
         const WorldInit &)
    : WorldBase(ctx)
{
    // [BOILERPLATE] Currently the physics system needs an upper bound on the number of
    // entities that will be stored in the BVH. We plan to fix this in
    // a future release.
    // [GAME_SPECIFIC] The calculation of max entities is game-specific
    constexpr CountT max_total_entities = consts::numAgents +
        consts::numRooms * (consts::maxEntitiesPerRoom + 3) +
        4; // side walls + floor

    // [BOILERPLATE] Initialize physics system
    // [GAME_SPECIFIC] Physics parameters (deltaT, substeps, gravity) are game-specific
    phys::PhysicsSystem::init(ctx, cfg.rigidBodyObjMgr,
        consts::deltaT, consts::numPhysicsSubsteps, -9.8f * math::up,
        max_total_entities);

    // [GAME_SPECIFIC] Store configuration
    initRandKey = cfg.initRandKey;
    autoReset = cfg.autoReset;

    // [BOILERPLATE] Initialize rendering if enabled
    enableRender = cfg.renderBridge != nullptr;

    if (enableRender) {
        RenderingSystem::init(ctx, cfg.renderBridge);
    }

    // [GAME_SPECIFIC] Initialize episode counter
    curWorldEpisode = 0;

    // [GAME_SPECIFIC] Creates agents, walls, etc.
    createPersistentEntities(ctx);

    // [GAME_SPECIFIC] Generate initial world state
    initWorld(ctx);
}

// [BOILERPLATE] This declaration is needed for the GPU backend in order to generate the
// CUDA kernel for world initialization, which needs to be specialized to the
// application's world data type (Sim) and config and initialization types.
// On the CPU it is a no-op.
MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Sim::Config, Sim::WorldInit);

}
