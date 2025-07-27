# Plan to Remove OtherAgents Component

## Overview
The OtherAgents component stores static references to other agents in the world. Since we only have 2 agents and these references never change, we can compute them dynamically instead of storing them.

## Removal Steps

### 1. Update collectObservationsSystem (src/sim.cpp)
- Modify the system to compute other agent references dynamically
- Instead of using `other_agents.e[i]`, loop through `ctx.data().agents[]` array
- Skip the current agent by comparing positions (since we don't have access to entity self-reference)

### 2. Remove OtherAgents from Agent Archetype (src/types.hpp)
- Remove `OtherAgents` from line 187 in the Agent archetype definition
- Delete the OtherAgents struct definition (lines 131-135)

### 3. Remove OtherAgents Registration (src/sim.cpp)
- Remove `registry.registerComponent<OtherAgents>();` from line 48

### 4. Update Task Graph (src/sim.cpp)
- Remove `OtherAgents` from the collectObservationsSystem node in setupTasks (line 441)

### 5. Remove OtherAgents Initialization (src/level_gen.cpp)
- Delete the entire OtherAgents population code (lines 171-186)

## Implementation Strategy

The key challenge is identifying which agent is "self" in the observation system. Since Madrona systems don't provide direct access to the current entity, we'll use position matching:

```cpp
// In collectObservationsSystem, replace OtherAgents usage with:
for (CountT i = 0; i < consts::numAgents; i++) {
    Entity agent = ctx.data().agents[i];
    Vector3 agent_pos = ctx.get<Position>(agent);
    
    // Skip self by comparing positions
    if (agent_pos == pos) continue;
    
    // This is the other agent
    Vector3 to_other = agent_pos - pos;
    partner_obs.obs[0] = {
        .polar = xyToPolar(to_view.rotateVec(to_other)),
    };
}
```

## Benefits
- Removes one component from every agent
- Eliminates static data that was never updated
- Simplifies the Agent archetype
- Reduces memory usage and improves cache efficiency

## Potential Issues
- Position comparison might fail if two agents have identical positions
- Alternative: Pass entity ID through the system if Madrona supports it
- Could also use a unique agent ID component if needed