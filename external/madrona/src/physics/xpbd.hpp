#pragma once

#include "physics_impl.hpp"

namespace madrona::phys::xpbd {

// Internal types moved from xpbd.cpp for custom physics setup
struct XPBDContactState {
    float lambdaN[4];
};

struct Contact : Archetype<
    ContactConstraint,
    XPBDContactState 
> {};

struct Joint : Archetype<JointConstraint> {};

struct SolverState {
    Query<JointConstraint> jointQuery;
    Query<ContactConstraint, XPBDContactState> contactQuery;
};

struct SubstepPrevState {
    math::Vector3 prevPosition;
    math::Quat prevRotation;
};

struct PreSolvePositional {
    math::Vector3 x;
    math::Quat q;
};

struct PreSolveVelocity {
    math::Vector3 v;
    math::Vector3 omega;
};

struct XPBDRigidBodyState : Bundle<
    SubstepPrevState,
    PreSolvePositional,
    PreSolveVelocity
> {};

// Expose internal solver functions needed for custom physics setup
void substepRigidBodies(Context &ctx,
                       base::Position &pos,
                       base::Rotation &rot,
                       const Velocity &vel,
                       const base::ObjectID &obj_id,
                       ResponseType response_type,
                       ExternalForce &ext_force,
                       ExternalTorque &ext_torque,
                       SubstepPrevState &prev_state,
                       PreSolvePositional &presolve_pos,
                       PreSolveVelocity &presolve_vel);

void solvePositions(Context &ctx, SolverState &solver);

void setVelocities(Context &ctx,
                  const base::Position &x,
                  const base::Rotation &q,
                  const SubstepPrevState &prev_state,
                  Velocity &vel);

void solveVelocities(Context &ctx, SolverState &solver);

// Original public API
void registerTypes(ECSRegistry &registry);

void getSolverArchetypeIDs(uint32_t *contact_archetype_id,
                           uint32_t *joint_archetype_id);

void init(Context &ctx);

TaskGraphNodeID setupXPBDSolverTasks(
    TaskGraphBuilder &builder,
    TaskGraphNodeID broadphase,
    CountT num_substeps);

}
