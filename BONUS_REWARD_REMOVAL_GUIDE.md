# Step-by-Step Guide to Remove Bonus Reward System from Madrona Escape Room

This guide documents the complete process of removing the bonus reward system from the Madrona Escape Room simulation. The bonus reward system was a cooperative gameplay element that provided a 25% reward multiplier when all agents maintained similar progress.

## Overview

The bonus reward system consisted of:
- A system function that compared agent progress distances
- A 25% reward multiplier when agents stayed within 2 units of each other
- Task graph integration between the base reward system and done system

## Step-by-Step Removal Process

### Step 1: Remove bonusRewardSystem Function

**File**: `src/sim.cpp`

1. **Remove the entire bonusRewardSystem function** (lines 462-484):
   ```cpp
   // DELETE THIS ENTIRE FUNCTION:
   // [GAME_SPECIFIC] Each agent gets a small bonus to it's reward if the other agent has
   // progressed a similar distance, to encourage them to cooperate.
   // This system reads the values of the Progress component written by
   // rewardSystem for other agents, so it must run after.
   inline void bonusRewardSystem(Engine &ctx,
                                 OtherAgents &others,
                                 Progress &progress,
                                 Reward &reward)
   {
       bool partners_close = true;
       for (CountT i = 0; i < consts::numAgents - 1; i++) {
           Entity other = others.e[i];
           Progress other_progress = ctx.get<Progress>(other);
           if (fabsf(other_progress.maxY - progress.maxY) > 2.f) {
               partners_close = false;
           }
       }
       if (partners_close && reward.v > 0.f) {
           reward.v *= 1.25f;
       }
   }
   ```

### Step 2: Remove System from Task Graph

**File**: `src/sim.cpp` in `setupTasks()` function

1. **Remove bonus_reward_sys node creation** (lines 574-579):
   ```cpp
   // DELETE THESE LINES:
   // [GAME_SPECIFIC] Assign partner's reward
   auto bonus_reward_sys = builder.addToGraph<ParallelForNode<Engine,
        bonusRewardSystem,
           OtherAgents,
           Progress,
           Reward
       >>({reward_sys});
   ```

### Step 3: Update Task Graph Dependencies

**File**: `src/sim.cpp` in `setupTasks()` function

1. **Update done_sys dependency** (line 586):
   ```cpp
   // Change from:
   auto done_sys = builder.addToGraph<ParallelForNode<Engine,
       stepTrackerSystem,
           StepsRemaining,
           Done
       >>({bonus_reward_sys});
   
   // To:
   auto done_sys = builder.addToGraph<ParallelForNode<Engine,
       stepTrackerSystem,
           StepsRemaining,
           Done
       >>({reward_sys});
   ```

### Step 4: Update Documentation

**File**: `CLAUDE.md`

1. **Update the Game Logic Pipeline section** (around line 220):
   ```markdown
   // Change from:
   3. **Game Logic Pipeline**:
      - `doorOpenSystem` → `rewardSystem` → `bonusRewardSystem` → `doneSystem` → `resetSystem`
   
   // To:
   3. **Game Logic Pipeline**:
      - `doorOpenSystem` → `rewardSystem` → `doneSystem` → `resetSystem`
   ```

## Components Analysis

The bonus reward system does not introduce any new components, archetypes, or exports. It only uses existing components:

- **Reads**: 
  - `OtherAgents` - Still needed for agent observations
  - `Progress` - Still needed for base reward calculation
- **Modifies**: 
  - `Reward` - Still needed for base reward system

No component removal is required.

## Verification Steps

After completing all changes:

1. **Build the project** to ensure no compilation errors:
   ```bash
   cd build && make -j$(nproc)
   ```

2. **Run the simulation** to verify rewards are still calculated:
   ```bash
   ./build/headless CPU 4 100
   ```

3. **Check reward behavior**:
   - Rewards should still be assigned for progress
   - Rewards should NOT be multiplied by 1.25
   - Agents can progress independently without penalty

4. **Test training** (if PyTorch is available):
   ```bash
   python scripts/train.py --num-worlds 1024 --num-updates 10
   ```

## Impact Assessment

### Gameplay Changes
- Removes cooperative reward bonus
- Agents no longer incentivized to stay together
- Each agent's reward depends only on individual progress

### Performance Impact
- Minor performance improvement by removing one system
- Reduces task graph complexity
- Fewer component reads per step

### Code Complexity
- Simplifies reward calculation logic
- Removes inter-agent dependency in reward system
- Makes reward system more modular

## Summary

This removal eliminates:
- 1 system function (`bonusRewardSystem`)
- 1 task graph node (`bonus_reward_sys`)
- 1 dependency update (done_sys now depends on reward_sys)
- 0 components (uses only existing components)
- 0 archetypes (no new entity types)
- 0 exports (no Python tensor exports)

The result is a simpler reward system where each agent's reward depends only on their individual progress, improving both performance and code maintainability.