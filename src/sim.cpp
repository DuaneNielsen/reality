#include <madrona/mw_gpu_entry.hpp>

#include "sim.hpp"
#include "level_gen.hpp"

// Need access to physics internals to insert collision detection
#include "../external/madrona/src/physics/physics_impl.hpp"
#include "../external/madrona/src/physics/xpbd.hpp"


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
    registry.registerComponent<CollisionDeath>();
    registry.registerComponent<TerminationReason>();
    
    // [GAME_SPECIFIC] Escape room specific components
    registry.registerComponent<SelfObservation>();
    registry.registerComponent<CompassObservation>();
    registry.registerComponent<Lidar>();
    registry.registerComponent<Progress>();
    registry.registerComponent<StepsTaken>();
    registry.registerComponent<EntityType>();
    registry.registerComponent<DoneOnCollide>();

    // [GAME_SPECIFIC] Target entity components
    registry.registerComponent<TargetTag>();
    registry.registerComponent<MotionParams>();

    // [REQUIRED_INTERFACE] Reset singleton - every episodic env needs this
    registry.registerSingleton<WorldReset>();
    
    // [GAME_SPECIFIC] Lidar visualization control singleton
    registry.registerSingleton<LidarVisControl>();
    
    // [GAME_SPECIFIC] Level management singleton
    registry.registerSingleton<LevelState>();
    
    // [GAME_SPECIFIC] Phase 2: Test-driven level system singleton
    registry.registerSingleton<CompiledLevel>();


    // [GAME_SPECIFIC] Escape room archetypes
    registry.registerArchetype<Agent>();
    registry.registerArchetype<PhysicsEntity>();
    registry.registerArchetype<RenderOnlyEntity>();
    registry.registerArchetype<LidarRayEntity>();
    registry.registerArchetype<CompassIndicatorEntity>();
    registry.registerArchetype<TargetEntity>();

    // [REQUIRED_INTERFACE] Export reset control
    registry.exportSingleton<WorldReset>(
        (uint32_t)ExportID::Reset);
    
    // [GAME_SPECIFIC] Export lidar visualization control
    registry.exportSingleton<LidarVisControl>(
        (uint32_t)ExportID::LidarVisControl);


    // [BOILERPLATE] Export core RL components
    registry.exportColumn<Agent, Action>(
        (uint32_t)ExportID::Action);
    registry.exportColumn<Agent, Reward>(
        (uint32_t)ExportID::Reward);
    registry.exportColumn<Agent, Done>(
        (uint32_t)ExportID::Done);
    registry.exportColumn<Agent, TerminationReason>(
        (uint32_t)ExportID::TerminationReason);

    // [GAME_SPECIFIC] Export escape room observations
    registry.exportColumn<Agent, SelfObservation>(
        (uint32_t)ExportID::SelfObservation);
    registry.exportColumn<Agent, CompassObservation>(
        (uint32_t)ExportID::CompassObservation);
    registry.exportColumn<Agent, Lidar>(
        (uint32_t)ExportID::Lidar);
    registry.exportColumn<Agent, StepsTaken>(
        (uint32_t)ExportID::StepsTaken);
    registry.exportColumn<Agent, Progress>(
        (uint32_t)ExportID::Progress);
    registry.exportColumn<Agent, Position>(
        (uint32_t)ExportID::AgentPosition);
    registry.exportColumn<TargetEntity, Position>(
        (uint32_t)ExportID::TargetPosition);
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
    if (ctx.data().autoReset && should_reset == 0) {
        // Check if any agent is done and schedule reset for next step
        for (CountT i = 0; i < consts::numAgents; i++) {
            Entity agent = ctx.data().agents[i];
            Done done = ctx.get<Done>(agent);
            if (done.v) {
                // Set reset flag for next step instead of immediate reset
                // This allows Python to observe the done=1 state and rewards
                reset.reset = 1;
                return;  // Don't reset this step, let it be processed next step
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

// [GAME_SPECIFIC] Normalize distance for observation
static inline float distObs(float v)
{
    // Normalize distance to [0, 1] range
    // Using lidar max range for consistency
    return fminf(v / consts::lidarMaxRange, 1.f);
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

// [GAME_SPECIFIC] Initialize progress values after reset and physics settling
// This ensures Progress contains valid values (not sentinel) for tensor export
inline void initProgressAfterReset(Engine &,
                                   Position pos,
                                   Progress &progress,
                                   TerminationReason &termination_reason)
{
    // Initialize progress with current position after physics has settled
    if (progress.maxY < -999990.0f) {
        progress.maxY = pos.y;
        progress.initialY = pos.y;

        // Only reset termination reason when progress is being initialized (i.e., after reset)
        termination_reason.code = -1;
    }
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
    // Use the appropriate axis range for each dimension
    float world_width = level.world_max_x - level.world_min_x;
    float world_length = level.world_max_y - level.world_min_y;
    float world_height = level.world_max_z - level.world_min_z;


    // Normalize each position using its corresponding axis range
    self_obs.globalX = (pos.x - level.world_min_x) / world_width;
    self_obs.globalY = (pos.y - level.world_min_y) / world_length;
    self_obs.globalZ = (pos.z - level.world_min_z) / world_height;
    // Use same normalization as reward system: (maxY - initialY) / (world_max_y - initialY)
    if (progress.initialY > -999990.0f) {
        float total_possible_progress = level.world_max_y - progress.initialY;
        self_obs.maxY = (progress.maxY - progress.initialY) / total_possible_progress;
    } else {
        self_obs.maxY = 0.0f;  // Not initialized yet
    }
    self_obs.theta = angleObs(computeZAngle(rot));
}

// [GAME_SPECIFIC] Computes compass one-hot encoding pointing toward target
// Updates the CompassObservation with a one-hot encoding pointing to the target entity
inline void compassSystem(Engine& ctx,
                          Entity agent_entity,
                          Position& agent_pos,
                          CompassObservation& compass_obs)
{
    // Find primary target entity using stored targets array
    bool found_target = false;
    Position target_pos = Vector3::zero();

    // Iterate through stored target entities
    for (CountT i = 0; i < ctx.data().numTargets; i++) {
        Entity target = ctx.data().targets[i];
        TargetTag& tag = ctx.get<TargetTag>(target);
        if (tag.id == 0) { // Primary target
            target_pos = ctx.get<Position>(target);
            found_target = true;
            break;
        }
    }

    float theta_radians;

    if (!found_target) {
        // Fallback to agent rotation if no target
        Rotation rot = ctx.get<Rotation>(agent_entity);
        theta_radians = computeZAngle(rot);
    } else {
        // Calculate angle to target
        float dx = target_pos.x - agent_pos.x;
        float dy = target_pos.y - agent_pos.y;
        theta_radians = atan2f(dy, dx);
    }

    // Apply compass equation: (64 - int(theta_in_radians / (2*pi))) % 128
    // This maps [-pi, pi] to compass buckets 0-127
    constexpr float two_pi = 2.0f * math::pi;

    // Normalize theta to [0, 2*pi) range first
    while (theta_radians < 0) theta_radians += two_pi;
    while (theta_radians >= two_pi) theta_radians -= two_pi;

    // Apply the compass formula
    int compass_bucket = (64 - (int)(theta_radians / two_pi * 128)) % 128;

    // Ensure bucket is in valid range [0, 127]
    if (compass_bucket < 0) compass_bucket += 128;

    // Zero out all compass values
    for (int i = 0; i < 128; i++) {
        compass_obs.compass[i] = 0.0f;
    }

    // Set the computed bucket to 1.0
    compass_obs.compass[compass_bucket] = 1.0f;
}

// [GAME_SPECIFIC] Compass indicator visualization system
inline void compassIndicatorSystem(Engine& ctx,
                                   Entity agent_entity,
                                   Position& agent_pos)
{
    // Find agent index by comparing entity against stored agent entities
    int32_t agent_idx = -1;
    for (int32_t i = 0; i < consts::numAgents; i++) {
        if (ctx.data().agents[i] == agent_entity) {
            agent_idx = i;
            break;
        }
    }

    // Skip if agent not found or rendering disabled
    if (agent_idx == -1 || !ctx.data().enableRender) {
        return;
    }

    // Get the compass indicator entity for this agent
    Entity indicator_entity = ctx.data().compassIndicators[agent_idx];

    // Find target position using stored targets array
    bool found_target = false;
    Position target_pos = Vector3::zero();

    // Iterate through stored target entities
    for (CountT i = 0; i < ctx.data().numTargets; i++) {
        Entity target = ctx.data().targets[i];
        TargetTag& tag = ctx.get<TargetTag>(target);
        if (tag.id == 0) { // Primary target
            target_pos = ctx.get<Position>(target);
            found_target = true;
            break;
        }
    }

    if (!found_target) {
        // Hide indicator when no target
        ctx.get<Scale>(indicator_entity) = Diag3x3{0, 0, 0};
        return;
    }

    // Calculate direction to target
    float dx = target_pos.x - agent_pos.x;
    float dy = target_pos.y - agent_pos.y;
    float distance = sqrtf(dx * dx + dy * dy);

    if (distance < 0.1f) {
        // Hide indicator when very close to target
        ctx.get<Scale>(indicator_entity) = Diag3x3{0, 0, 0};
        return;
    }

    // Normalize direction
    Vector3 direction = Vector3{dx, dy, 0} / distance;

    // Position indicator 1.5 units from agent (slightly offset from agent body)
    Vector3 indicator_pos = agent_pos + direction * 1.5f;

    // Set indicator position (slightly above ground)
    ctx.get<Position>(indicator_entity) = Vector3{indicator_pos.x, indicator_pos.y, agent_pos.z + 0.5f};

    // Calculate rotation to align cylinder horizontally with direction
    // We want the cylinder to point horizontally toward the target
    // Default cylinder mesh extends along Z axis, we want it to point along the horizontal XY plane
    Vector3 default_dir = Vector3{1, 0, 0};  // Point along +X axis by default
    Vector3 target_dir = Vector3{dx, dy, 0}.normalize();  // Horizontal direction

    Quat rotation;

    // Calculate angle between default direction and target direction
    float dot = default_dir.dot(target_dir);
    Vector3 cross = default_dir.cross(target_dir);

    if (cross.length2() > 0.001f) {
        // Normal case: create rotation
        float angle = acosf(fmaxf(-1.0f, fminf(1.0f, dot)));
        Vector3 axis = cross.normalize();
        rotation = Quat::angleAxis(angle, axis);
    } else if (dot < 0) {
        // Target direction opposite to default
        rotation = Quat::angleAxis(math::pi, Vector3{0, 0, 1});  // Rotate 180° around Z
    } else {
        // Target direction aligned with default
        rotation = Quat{1, 0, 0, 0};
    }

    // Apply additional rotation to make cylinder horizontal (rotate 90° around Y to lay it flat)
    Quat horizontal_rotation = Quat::angleAxis(math::pi / 2.0f, Vector3{0, 1, 0});
    rotation = rotation * horizontal_rotation;

    ctx.get<Rotation>(indicator_entity) = rotation;

    // Set scale: 2.0 units length, thicker than lidar rays
    // Z = length (2.0 / 3.0 since mesh is 3 units long)
    // X, Y = width (0.15 for visibility)
    float indicator_length = 2.0f / 3.0f;  // Cylinder mesh is 3 units long
    ctx.get<Scale>(indicator_entity) = Diag3x3{0.15f, 0.15f, indicator_length};
}

// [GAME_SPECIFIC] Template-based custom equation of motion (NVRTC-compatible)
template<int MotionType>
inline void applyMotionEquation(
    Engine& ctx,
    float dt,
    Position& pos,
    Velocity& vel,
    const MotionParams& params);

// Static motion specialization
template<>
inline void applyMotionEquation<0>(
    Engine& ctx, float dt, Position& pos, Velocity& vel, const MotionParams& params) {
    vel.linear = Vector3::zero();
    vel.angular = Vector3::zero();
}

// Figure-8 oscillator specialization - parameterized figure-8 motion
template<>
inline void applyMotionEquation<1>(
    Engine& ctx, float dt, Position& pos, Velocity& vel, const MotionParams& params) {
    // Access simulation time through agent's StepsTaken component
    Entity agent = ctx.data().agents[0];  // Use first agent for timing
    StepsTaken steps_taken = ctx.get<StepsTaken>(agent);
    float t = steps_taken.t * consts::deltaT;  // Convert steps to time

    Vector3 center = {params.center_x, params.center_y, params.center_z};

    // Parameterized figure-8: x = amp_x * cos(speed * t), y = amp_y * sin(2 * speed * t)
    float speed = params.omega_x;  // Use omega_x as speed control
    float amp_x = params.phase_x;  // Use phase_x as X amplitude
    float amp_y = params.phase_y;  // Use phase_y as Y amplitude

    float x_motion = cosf(speed * t);
    float y_motion = sinf(2.0f * speed * t);

    // Set position directly for figure-8
    pos.x = center.x + amp_x * x_motion;
    pos.y = center.y + amp_y * y_motion;
    pos.z = center.z;

    // Calculate velocity from position derivative
    // dx/dt = -amp_x * speed * sin(speed * t), dy/dt = amp_y * 2 * speed * cos(2 * speed * t)
    vel.linear.x = -amp_x * speed * sinf(speed * t);
    vel.linear.y = amp_y * 2.0f * speed * cosf(2.0f * speed * t);
    vel.linear.z = 0.0f;
}

// Circular motion specialization - parameterized circular motion
template<>
inline void applyMotionEquation<2>(
    Engine& ctx, float dt, Position& pos, Velocity& vel, const MotionParams& params) {
    // Access simulation time through agent's StepsTaken component
    Entity agent = ctx.data().agents[0];  // Use first agent for timing
    StepsTaken steps_taken = ctx.get<StepsTaken>(agent);
    float t = steps_taken.t * consts::deltaT;  // Convert steps to time

    Vector3 center = {params.center_x, params.center_y, params.center_z};

    // Circular motion parameters
    float angular_velocity = params.omega_x;  // Use omega_x as angular velocity (rad/s)
    float radius = params.phase_x;  // Use phase_x as radius
    float initial_angle = params.phase_y;  // Use phase_y as initial angle offset

    // Direction multiplier: 1 for counter-clockwise, -1 for clockwise
    // Stored in mass field during randomization setup
    float direction = (params.mass > 0.0f) ? 1.0f : -1.0f;

    // Calculate current angle: initial_angle + direction * angular_velocity * time
    float current_angle = initial_angle + direction * angular_velocity * t;

    // Set position for circular motion: center + radius * (cos(angle), sin(angle))
    pos.x = center.x + radius * cosf(current_angle);
    pos.y = center.y + radius * sinf(current_angle);
    pos.z = center.z;

    // Calculate velocity from position derivative
    // dx/dt = -radius * angular_velocity * direction * sin(angle)
    // dy/dt = radius * angular_velocity * direction * cos(angle)
    vel.linear.x = -radius * angular_velocity * direction * sinf(current_angle);
    vel.linear.y = radius * angular_velocity * direction * cosf(current_angle);
    vel.linear.z = 0.0f;
}

// [GAME_SPECIFIC] Custom motion system for target entities
inline void customMotionSystem(Engine& ctx,
    Position& pos,
    Velocity& vel,
    const MotionParams& params)
{
    float dt = consts::deltaT / consts::numPhysicsSubsteps;

    // Runtime dispatch (NVRTC-safe)
    switch(params.motion_type) {
        case 0: applyMotionEquation<0>(ctx, dt, pos, vel, params); break;
        case 1: applyMotionEquation<1>(ctx, dt, pos, vel, params); break;
        case 2: applyMotionEquation<2>(ctx, dt, pos, vel, params); break;
        default: applyMotionEquation<0>(ctx, dt, pos, vel, params); break;
    }
}

// [GAME_SPECIFIC] Launches consts::numLidarSamples per agent.
// This system is specially optimized in the GPU version:
// a warp of threads is dispatched for each invocation of the system
// and each thread in the warp traces one lidar ray for the agent.
inline void lidarSystem(Engine &ctx,
                        Entity e,
                        Lidar &lidar)
{
    Vector3 pos = ctx.get<Position>(e);
    Quat rot = ctx.get<Rotation>(e);
    auto &bvh = ctx.singleton<broadphase::BVH>();

    Vector3 agent_fwd = rot.rotateVec(math::fwd);
    Vector3 right = rot.rotateVec(math::right);
    
    // Get agent index for ray entity lookup (assuming single agent per world for now)
    int32_t agent_idx = 0;
    const LidarVisControl &lidar_vis_control = ctx.singleton<LidarVisControl>();
    bool visualize = (lidar_vis_control.enabled != 0) && ctx.data().enableRender;

    auto traceRay = [&](int32_t idx) {
        // 120-degree arc in front of agent (-60 to +60 degrees)
        float angle_range = 2.f * math::pi / 3.f; // 120 degrees in radians
        float theta = -angle_range / 2.f + (angle_range * float(idx) / float(consts::numLidarSamples - 1));
        
        // Rotate relative to agent's forward direction
        float cos_theta = cosf(theta);
        float sin_theta = sinf(theta);
        
        Vector3 ray_dir = (cos_theta * agent_fwd + sin_theta * right).normalize();
        Vector3 ray_origin = pos + consts::lidarHeightOffset * math::up;

        float hit_t;
        Vector3 hit_normal;
        Entity hit_entity =
            bvh.traceRay(ray_origin, ray_dir, &hit_t,
                         &hit_normal, consts::lidarMaxRange);

        if (hit_entity == Entity::none()) {
            lidar.samples[idx] = {
                .depth = 0.f,
            };
        } else {
            lidar.samples[idx] = {
                .depth = distObs(hit_t),
            };
        }
        
        // Handle ray visualization - always update ray entity if rendering is enabled
        if (ctx.data().enableRender) {
            Entity ray_entity = ctx.data().lidarRays[agent_idx][idx];
            
            if (!visualize) {
                // Hide all rays when visualization is disabled
                ctx.get<Scale>(ray_entity) = Diag3x3{0, 0, 0};
            } else if (hit_entity != Entity::none() && (idx % 8 == 0)) {
                // Show every 8th ray (16 rays total) when visualization is enabled
                // Position ray to start at origin and extend to hit point
                // Since cylinder extends ±0.5 * scale.z from center, position at midpoint
                Vector3 ray_midpoint = ray_origin + (ray_dir * hit_t * 0.5f);
                ctx.get<Position>(ray_entity) = ray_midpoint;
                
                // Align cylinder along ray direction
                // The cylinder mesh is oriented along Z axis by default (-1.5 to +1.5)
                Vector3 z_axis = Vector3{0, 0, 1};
                Quat rotation;
                
                // Calculate rotation from Z axis to ray direction
                Vector3 cross = z_axis.cross(ray_dir);
                float dot = z_axis.dot(ray_dir);
                
                if (cross.length2() > 0.001f) {
                    // Normal case: create rotation
                    float angle = acosf(dot);
                    Vector3 axis = cross.normalize();
                    rotation = Quat::angleAxis(angle, axis);
                } else if (dot < 0) {
                    // Ray pointing exactly opposite to Z
                    rotation = Quat::angleAxis(math::pi, Vector3{1, 0, 0});
                } else {
                    // Ray already aligned with Z
                    rotation = Quat{1, 0, 0, 0};
                }
                
                ctx.get<Rotation>(ray_entity) = rotation;
                
                // Scale: Z = length (hit distance / 3.0 since mesh is 3 units long)
                // X, Y = thin width
                float ray_length = hit_t / 3.0f;  // Cylinder mesh is 3 units long
                ctx.get<Scale>(ray_entity) = Diag3x3{0.04f, 0.04f, ray_length};
            } else {
                // Hide rays we're not displaying (no hit or not every 8th)
                ctx.get<Scale>(ray_entity) = Diag3x3{0, 0, 0};
            }
        }
    };


    // MADRONA_GPU_MODE guards GPU specific logic
#ifdef MADRONA_GPU_MODE
    // Can use standard cuda variables like threadIdx for 
    // block level programming (128 threads = 4 warps)
    int32_t idx = threadIdx.x % 128;

    if (idx < consts::numLidarSamples) {
        traceRay(idx);
    }
#else
    for (CountT i = 0; i < consts::numLidarSamples; i++) {
        traceRay(i);
    }
#endif
}

// [REQUIRED_INTERFACE] Computes reward for each agent - every environment needs a reward system
// [GAME_SPECIFIC] Provides completion rewards when agent reaches target entity
inline void rewardSystem(Engine &ctx,
                         Position pos,
                         Progress &progress,
                         Reward &out_reward,
                         Done &done,
                         CollisionDeath &collision_death,
                         TerminationReason &termination_reason,
                         StepsTaken &steps_taken)
{
    // Step 0 check: Always 0.0 reward on reset step
    if (steps_taken.t == 0) {
        out_reward.v = 0.0f;
        return;
    }

    // Find primary target entity (TargetTag.id == 0)
    bool found_target = false;
    Position target_pos = Vector3::zero();

    // Iterate through stored target entities
    for (CountT i = 0; i < ctx.data().numTargets; i++) {
        Entity target = ctx.data().targets[i];
        TargetTag& tag = ctx.get<TargetTag>(target);
        if (tag.id == 0) { // Primary target
            target_pos = ctx.get<Position>(target);
            found_target = true;
            break;
        }
    }

    // No target fallback - give no reward
    if (!found_target) {
        out_reward.v = 0.0f;
        return;
    }

    // Calculate Euclidean distance to target
    Vector3 agent_pos = Vector3{pos.x, pos.y, pos.z};
    Vector3 target_position = Vector3{target_pos.x, target_pos.y, target_pos.z};
    Vector3 diff = agent_pos - target_position;
    float distance_to_target = diff.length();


    // Check for episode completion - within 3.0 world units of target
    if (done.v == 0 && distance_to_target <= 3.0f) {
        done.v = 1;
        termination_reason.code = 1;  // goal_achieved
        out_reward.v = 1.0f;  // +1 reward for reaching the target
    } else {
        out_reward.v = 0.0f;  // No reward until target is reached
    }

    // Override with collision penalty if agent died
    // if (done.v == 1 && collision_death.died == 1) {
    //     out_reward.v = -0.1f;  // Collision death penalty overrides completion reward
    // }
}

// [GAME_SPECIFIC] Self-contained collision detection system
// Triggers episode termination when agent collides with ANY object (except floor)
inline void agentCollisionSystem(Engine &ctx,
                                Entity agent_entity,
                                EntityType agent_type,
                                Done &done,
                                CollisionDeath &collision_death,
                                TerminationReason &termination_reason)
{
    // Only process agents
    if (agent_type != EntityType::Agent) {
        return;
    }
    
    // If already done, skip processing
    if (done.v == 1) {
        return;
    }
    
    // Query ContactConstraints - these are available right after narrowphase
    auto contact_query = ctx.query<ContactConstraint>();
    
    ctx.iterateQuery(contact_query, [&](ContactConstraint &contact) {
        // Get the agent's location for comparison
        Loc agent_loc = ctx.loc(agent_entity);
        
        // Check if this agent is involved in the contact
        bool agent_is_ref = (contact.ref == agent_loc);
        bool agent_is_alt = (contact.alt == agent_loc);
        
        if (agent_is_ref || agent_is_alt) {
            // Get the other entity's location
            Loc other_loc = agent_is_ref ? contact.alt : contact.ref;
            
            // Get the floor's location for comparison
            Loc floor_loc = ctx.loc(ctx.data().floorPlane);
            
            // Check if it's the floor (floor is a special singleton entity)
            if (other_loc == floor_loc) {
                return; // Ignore floor collisions
            }
            
            // Check if this entity should trigger episode termination on collision
            // Since we have the location, we can access the component directly
            auto done_on_collide_ref = ctx.getCheck<DoneOnCollide>(other_loc);
            if (done_on_collide_ref.valid() && done_on_collide_ref.value().value) {
                done.v = 1;
                collision_death.died = 1;  // Mark that agent died from collision
                termination_reason.code = 2;  // collision_death
            }
        }
    });
}

// [REQUIRED_INTERFACE] Track the number of steps taken in the episode and
// notify training that an episode has completed by
// setting done = 1 when the episode limit is reached
inline void stepTrackerSystem(Engine &,
                              StepsTaken &steps_taken,
                              Done &done,
                              TerminationReason &termination_reason)
{
    uint32_t num_taken = ++steps_taken.t;
    // Done flag is reset by resetAgentPhysics during episode reset
    if (num_taken >= consts::episodeLen) {
        done.v = 1;  // Mark episode as done when step limit is reached
        termination_reason.code = 0;  // episode_steps_reached
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

    // [GAME_SPECIFIC] Apply custom motion equations to target entities
    auto custom_motion_sys = builder.addToGraph<ParallelForNode<Engine,
        customMotionSystem,
            Position,
            Velocity,
            MotionParams
        >>({move_sys});

    // [BOILERPLATE] Build BVH for broadphase / raycasting
    auto broadphase_setup_sys = phys::PhysicsSystem::setupBroadphaseTasks(
        builder, {custom_motion_sys});

    // Custom physics setup to expose ContactConstraints
    // We inline the XPBD substep loop to insert collision detection after narrowphase
    auto broadphase_prep = phys::broadphase::setupPreIntegrationTasks(
        builder, {broadphase_setup_sys});
    
    auto cur_node = broadphase_prep;

#ifdef MADRONA_GPU_MODE
    // Sort joints for better GPU performance
    cur_node = builder.addToGraph<SortArchetypeNode<phys::xpbd::Joint, WorldID>>({cur_node});
    cur_node = builder.addToGraph<ResetTmpAllocNode>({cur_node});
#endif

    // Run physics substeps with collision detection inserted
    for (CountT i = 0; i < consts::numPhysicsSubsteps; i++) {
        // Substep rigid body integration
        auto rgb_update = builder.addToGraph<ParallelForNode<Engine,
            phys::xpbd::substepRigidBodies, 
            Position, Rotation, Velocity, ObjectID,
            ResponseType, ExternalForce, ExternalTorque,
            phys::xpbd::SubstepPrevState, 
            phys::xpbd::PreSolvePositional,
            phys::xpbd::PreSolveVelocity>>({cur_node});
        
        // Run narrowphase collision detection
        auto run_narrowphase = phys::narrowphase::setupTasks(builder, {rgb_update});

#ifdef MADRONA_GPU_MODE
        // Sort contacts for better GPU performance
        run_narrowphase = builder.addToGraph<SortArchetypeNode<phys::xpbd::Contact, WorldID>>(
            {run_narrowphase});
        run_narrowphase = builder.addToGraph<ResetTmpAllocNode>({run_narrowphase});
#endif

        // HERE IS WHERE WE CAN ACCESS ContactConstraints!
        // Insert our collision detection system right after narrowphase
        auto collision_detect = builder.addToGraph<ParallelForNode<Engine,
            agentCollisionSystem,
            Entity,
            EntityType,
            Done,
            CollisionDeath,
            TerminationReason
        >>({run_narrowphase});
        
        // Continue with position solver
        auto solve_pos = builder.addToGraph<ParallelForNode<Engine,
            phys::xpbd::solvePositions, 
            phys::xpbd::SolverState>>({collision_detect});
        
        // Update velocities
        auto vel_set = builder.addToGraph<ParallelForNode<Engine,
            phys::xpbd::setVelocities, 
            Position, Rotation,
            phys::xpbd::SubstepPrevState, Velocity>>({solve_pos});
        
        // Solve velocity constraints
        auto solve_vel = builder.addToGraph<ParallelForNode<Engine,
            phys::xpbd::solveVelocities, 
            phys::xpbd::SolverState>>({vel_set});
        
        // Clear contacts for next substep
        auto clear_contacts = builder.addToGraph<
            ClearTmpNode<phys::xpbd::Contact>>({solve_vel});
        
        cur_node = builder.addToGraph<ResetTmpAllocNode>({clear_contacts});
    }
    
    // Finish physics post-processing
    auto clear_broadphase = builder.addToGraph<
        ClearTmpNode<phys::CandidateTemporary>>({cur_node});
    
    auto broadphase_post = phys::broadphase::setupPostIntegrationTasks(
        builder, {clear_broadphase});
    
    auto collision_sys = broadphase_post;

    // [GAME_SPECIFIC] Improve controllability of agents by setting their velocity to 0
    // after physics is done.
    auto agent_zero_vel = builder.addToGraph<ParallelForNode<Engine,
        agentZeroVelSystem, Velocity, Action>>(
            {collision_sys});

    // [BOILERPLATE] Finalize physics subsystem work
    auto phys_cleanup = phys::PhysicsSystem::setupCleanupTasks(
        builder, {agent_zero_vel});

    // [REQUIRED_INTERFACE] Check if the episode is over
    auto done_sys = builder.addToGraph<ParallelForNode<Engine,
        stepTrackerSystem,
            StepsTaken,
            Done,
            TerminationReason
        >>({phys_cleanup});

    // [REQUIRED_INTERFACE] Compute reward - only given at episode end
    auto reward_sys = builder.addToGraph<ParallelForNode<Engine,
         rewardSystem,
            Position,
            Progress,
            Reward,
            Done,
            CollisionDeath,
            TerminationReason,
            StepsTaken
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

    // [GAME_SPECIFIC] Initialize progress after reset and physics settling
    auto init_progress = builder.addToGraph<ParallelForNode<Engine,
        initProgressAfterReset,
            Position,
            Progress,
            TerminationReason
        >>({post_reset_broadphase});

    // [REQUIRED_INTERFACE] Finally, collect observations for the next step.
    auto collect_obs = builder.addToGraph<ParallelForNode<Engine,
        collectObservationsSystem,
            Position,
            Rotation,
            Progress,
            SelfObservation
        >>({init_progress});

    // [GAME_SPECIFIC] Update compass observations
    auto compass_sys = builder.addToGraph<ParallelForNode<Engine,
        compassSystem,
            Entity,
            Position,
            CompassObservation
        >>({collect_obs});

    // [GAME_SPECIFIC] Update compass indicator visualization
    auto compass_indicator_sys = builder.addToGraph<ParallelForNode<Engine,
        compassIndicatorSystem,
            Entity,
            Position
        >>({compass_sys});

    // [GAME_SPECIFIC] The lidar system
#ifdef MADRONA_GPU_MODE
    // [BOILERPLATE] Note the use of CustomParallelForNode to create a taskgraph node
    // that launches 128 threads (4 warps) for each invocation (1).
    // This allows all 128 lidar rays to be traced in parallel.
    auto lidar = builder.addToGraph<CustomParallelForNode<Engine,
        lidarSystem, 128, 1,
#else
    auto lidar = builder.addToGraph<ParallelForNode<Engine,
        lidarSystem,
#endif
            Entity,
            Lidar
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
        builder, {compass_sys, lidar});
    auto sort_phys_objects = queueSortByWorld<PhysicsEntity>(
        builder, {sort_agents});
    (void)sort_phys_objects;
#else
    (void)compass_sys;
    (void)lidar;
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
    
    // Initialize LidarVisControl singleton (default disabled for performance)
    LidarVisControl &lidar_vis_control = ctx.singleton<LidarVisControl>();
    lidar_vis_control.enabled = 0;  // Default to disabled
    
    
    CountT max_total_entities = compiled_level.max_entities;

    // [BOILERPLATE] Initialize physics system
    // [GAME_SPECIFIC] Physics parameters (deltaT, substeps, gravity) are game-specific
    phys::PhysicsSystem::init(ctx, cfg.rigidBodyObjMgr,
        consts::deltaT, consts::numPhysicsSubsteps, -consts::physics::gravityAcceleration * math::up,
        max_total_entities);

    // [GAME_SPECIFIC] Store configuration
    initRandKey = cfg.initRandKey;
    autoReset = cfg.autoReset;
    customVerticalFov = cfg.customVerticalFov;

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
