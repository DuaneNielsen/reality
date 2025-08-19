#pragma once

#include "sim.hpp"

namespace madEscape {

/**
 * Creates persistent entities that exist for the entire simulation lifetime.
 * 
 * Called ONCE during Sim constructor initialization.
 * Creates: floor plane, agents, origin marker gizmo.
 * These entities are never destroyed, only reset between episodes.
 * 
 * @param ctx The engine context for this world
 */
void createPersistentEntities(Engine &ctx);

/**
 * Generates the world state for a new episode.
 * 
 * Called at the START of EACH episode:
 * - From initWorld() on first episode
 * - From resetSystem() when episode resets
 * 
 * Internally calls:
 * 1. resetPersistentEntities() - Re-registers persistent entities with physics
 * 2. generateLevel() - Creates per-episode entities from compiled level data
 * 
 * @param ctx The engine context for this world
 */
void generateWorld(Engine &ctx);

}
