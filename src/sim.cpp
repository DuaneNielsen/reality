#include <madrona/mw_gpu_entry.hpp>

#include "sim.hpp"
#include "level_gen.hpp"


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
    registry.registerComponent<StepsRemaining>();
    registry.registerComponent<EntityType>();

    // [REQUIRED_INTERFACE] Reset singleton - every episodic env needs this
    registry.registerSingleton<WorldReset>();
    
    // [GAME_SPECIFIC] Level management singleton
    registry.registerSingleton<LevelState>();
    
    // [GAME_SPECIFIC] Phase 2: Test-driven level system singleton
    registry.registerSingleton<CompiledLevel>();

    // [GAME_SPECIFIC] Escape room archetypes
    registry.registerArchetype<Agent>();
    registry.registerArchetype<PhysicsEntity>();
    registry.registerArchetype<RenderOnlyEntity>();

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
    registry.exportColumn<Agent, StepsRemaining>(
        (uint32_t)ExportID::StepsRemaining);
    registry.exportColumn<Agent, Progress>(
        (uint32_t)ExportID::Progress);
}

// [GAME_SPECIFIC] Helper to clean up escape room entities
static inline void cleanupWorld(Engine &ctx)
{
    // Destroy current level entities (but not persistent ones)
    LevelState &level = ctx.singleton<LevelState>();
    Room &room = level.rooms[0];
    
    for (CountT j = 0; j < CompiledLevel::MAX_TILES; j++) {
        Entity e = room.entities[j];
        if (e != Entity::none()) {
            // Check if this is a persistent entity
            bool isPersistent = false;
            for (CountT k = 0; k < ctx.data().numPersistentLevelEntities; k++) {
                if (ctx.data().persistentLevelEntities[k] == e) {
                    isPersistent = true;
                    break;
                }
            }
            
            // Only destroy non-persistent entities
            if (!isPersistent) {
                ctx.destroyRenderableEntity(e);
            }
        }
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

    // [GAME_SPECIFIC] Phase 2: CompiledLevel singleton already initialized in Sim constructor

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
    // NOTE: Negative sign compensates for non-standard rotation encoding
    // where lower values = left turn, higher values = right turn
    float t_z =
        -turn_delta_per_bucket * (action.rotate - consts::numTurnBuckets / 2);

    Vector3 local_force = { f_x, f_y, 0 };
    external_force = cur_rot.rotateVec(local_force);
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
// NOTE: These functions now take the level boundaries as parameters
// to support dynamic level sizes

static inline float angleObs(float v)
{
    return v / math::pi;
}

// [GAME_SPECIFIC] Translate xy delta to polar observations for learning.
// static inline PolarObservation xyToPolar(Vector3 v)
// {
//     Vector2 xy { v.x, v.y };
//
//     float r = xy.length();
//
//     // Note that this is angle off y-forward
//     float theta = atan2f(xy.x, xy.y);
//
//     return PolarObservation {
//         .r = distObs(r),
//         .theta = angleObs(theta),
//     };
// }

// Removed encodeType - no longer needed without entity observations

static inline float computeZAngle(Quat q)
{
    float siny_cosp = consts::math::quaternionConversionFactor * (q.w * q.z + q.x * q.y);
    float cosy_cosp = 1.f - consts::math::quaternionConversionFactor * (q.y * q.y + q.z * q.z);
    return atan2f(siny_cosp, cosy_cosp);
}

// [REQUIRED_INTERFACE] This system packages all the egocentric observations together 
// for the policy inputs. Every environment must implement observation collection.
// [GAME_SPECIFIC] The specific observations collected and their format.
inline void collectObservationsSystem(Engine &ctx,
                                      Position pos,
                                      Rotation rot,
                                      const Progress &progress,
                                      SelfObservation &self_obs)
{
    // Get world boundaries from the compiled level singleton
    const CompiledLevel& level = ctx.singleton<CompiledLevel>();
    
    // Normalize positions based on actual world boundaries
    // Use the Y-axis range as the primary normalization factor (world length)
    float world_length = level.world_max_y - level.world_min_y;
    
    if (world_length <= 0.0f) {
        printf("ERROR: Invalid world boundaries - world_length = %f (min_y=%f, max_y=%f)\n",
               world_length, level.world_min_y, level.world_max_y);
    }
    
    // Normalize all positions consistently using world length
    // Adjust for world origin to match reward calculation
    self_obs.globalX = (pos.x - level.world_min_x) / world_length;
    self_obs.globalY = (pos.y - level.world_min_y) / world_length;
    self_obs.globalZ = (pos.z - level.world_min_z) / world_length;
    self_obs.maxY = (progress.maxY - level.world_min_y) / world_length;
    self_obs.theta = angleObs(computeZAngle(rot));
}


// [REQUIRED_INTERFACE] Computes reward for each agent - every environment needs a reward system
// [GAME_SPECIFIC] Tracks max Y position reached, but only gives reward at episode end
inline void rewardSystem(Engine &ctx,
                         Position pos,
                         Progress &progress,
                         Reward &out_reward,
                         Done &done,
                         StepsRemaining &steps_remaining)
{
    // Update max Y reached during episode
    if (pos.y > progress.maxY) {
        progress.maxY = pos.y;
    }

    // Only give reward at the end of the episode
    if (done.v == 1 || steps_remaining.t == 0) {
        // Get world boundaries from the compiled level singleton
        const CompiledLevel& level = ctx.singleton<CompiledLevel>();
        
        // Use actual world boundaries for normalization
        float world_length = level.world_max_y - level.world_min_y;
        
        if (world_length <= 0.0f) {
            printf("ERROR: Invalid world boundaries in reward calculation - world_length = %f\n", world_length);
        }
        
        float adjusted_progress = progress.maxY - level.world_min_y;
        float normalized_progress = adjusted_progress / world_length;
        
        out_reward.v = normalized_progress;
    } else {
        // No reward during the episode
        out_reward.v = 0.0f;
    }
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

    // [REQUIRED_INTERFACE] Check if the episode is over
    auto done_sys = builder.addToGraph<ParallelForNode<Engine,
        stepTrackerSystem,
            StepsRemaining,
            Done
        >>({phys_done});

    // [REQUIRED_INTERFACE] Compute reward - only given at episode end
    auto reward_sys = builder.addToGraph<ParallelForNode<Engine,
         rewardSystem,
            Position,
            Progress,
            Reward,
            Done,
            StepsRemaining
        >>({done_sys});

    // [REQUIRED_INTERFACE] Conditionally reset the world if the episode is over
    auto reset_sys = builder.addToGraph<ParallelForNode<Engine,
        resetSystem,
            WorldReset
        >>({reward_sys});

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
            SelfObservation
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
         const WorldInit &world_init)
    : WorldBase(ctx)
{
    // Initialize CompiledLevel singleton for BVH sizing
    CompiledLevel &compiled_level = ctx.singleton<CompiledLevel>();
    
    // Use per-world compiled level (highest priority), fallback to shared level
    compiled_level = world_init.compiledLevel;
    
    
    CountT max_total_entities = compiled_level.max_entities;

    // [BOILERPLATE] Initialize physics system
    // [GAME_SPECIFIC] Physics parameters (deltaT, substeps, gravity) are game-specific
    phys::PhysicsSystem::init(ctx, cfg.rigidBodyObjMgr,
        consts::deltaT, consts::numPhysicsSubsteps, -consts::physics::gravityAcceleration * math::up,
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
