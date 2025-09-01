# Comprehensive Test Plan: Reset, Episode Counter, and Done Flag Order of Operations

## Pre-Reading List

**Essential Reading (Must Read First):**
1. `src/sim.cpp:353-362` - `stepTrackerSystem()` - Core step counting and done flag logic
2. `src/sim.cpp:122-147` - `resetSystem()` - Reset detection and world cleanup logic
3. `src/level_gen.cpp:94-148` - `resetAgentPhysics()` - Agent state reset including done flag
4. `src/sim.cpp:490-510` - Task graph showing execution order: stepTracker → reward → reset
5. `tests/python/test_done_tensor_reset.py` - Current test coverage baseline

**Supporting Reading:**
6. `src/mgr.cpp:751-760` - `triggerReset()` - Manual reset trigger mechanism
7. `src/types.hpp:105-108` - `StepsRemaining` component definition
8. `src/types.hpp:99-102` - `Done` component definition  
9. `src/sim.cpp:277-290` - `rewardSystem()` - How rewards interact with done/steps
10. `madrona_escape_room/generated_constants.py:87` - `episodeLen = 200` constant

## System Architecture Analysis

### Execution Order (Per Step)
```
1. stepTrackerSystem():    StepsRemaining--, set Done=1 if steps==0
2. rewardSystem():         Calculate rewards based on Done+StepsRemaining state
3. resetSystem():          Check reset triggers, call resetAgentPhysics() if needed
   └─ resetAgentPhysics(): Reset StepsRemaining=200, Done=0, positions, etc.
```

### Reset Trigger Mechanisms
1. **Manual Reset**: `reset_tensor[world_idx] = 1` → `triggerReset(world_idx)`
2. **Auto-Reset**: `auto_reset=True` + any agent `done=1` → automatic reset
3. **Step Exhaustion**: `StepsRemaining` reaches 0 → `Done=1` → potential auto-reset
4. **Collision Termination**: Collision with `DoneOnCollide` entity → `Done=1` → potential auto-reset

### Critical Timing Dependencies
- **Same-Step Reset**: If reset triggers in step N, new episode starts in step N+1
- **Done Flag Persistence**: Done flag persists until resetAgentPhysics() clears it
- **Counter Reset**: StepsRemaining resets to episodeLen during resetAgentPhysics()

## Test Scenarios Matrix

### Category 1: Basic Reset Timing
| Test ID | Scenario | Expected Behavior | Edge Cases |
|---------|----------|-------------------|------------|
| RT-001 | Manual reset when done=0 | Immediate reset, steps=200, done=0 | Mid-episode reset |
| RT-002 | Manual reset when done=1 | Reset clears done to 0, steps=200 | Already terminated episode |
| RT-003 | Auto-reset off, done=1 | Done stays 1, no reset occurs | Manual intervention needed |
| RT-004 | Auto-reset on, done=1 | Automatic reset next step, done=0 | Seamless episode transition |

### Category 2: Step Counter Precision
| Test ID | Scenario | Expected Behavior | Edge Cases |
|---------|----------|-------------------|------------|
| SC-001 | Exact episodeLen steps | Step 200: done=1, step 201: reset if auto | Boundary condition |
| SC-002 | Reset at step 199 | Next step: steps=200, done=0 | One step before natural end |
| SC-003 | Reset at step 200 (done=1) | Next step: steps=200, done=0 | Reset on natural termination |
| SC-004 | Multiple resets per episode | Each reset: steps=200, done=0 | Reset spam protection |

### Category 3: Collision vs Step Termination
| Test ID | Scenario | Expected Behavior | Edge Cases |
|---------|----------|-------------------|------------|
| CT-001 | Collision at step 50 | done=1 early, steps=150 remaining | Early termination |
| CT-002 | Collision at step 200 | done=1 from collision, not step counter | Simultaneous conditions |
| CT-003 | Collision + manual reset | Reset overrides collision done=1 | Priority handling |
| CT-004 | No collision, step limit | done=1 from step counter only | Natural episode end |

### Category 4: Multi-World Synchronization  
| Test ID | Scenario | Expected Behavior | Edge Cases |
|---------|----------|-------------------|------------|
| MW-001 | Reset world 0, others continue | Only world 0 resets, others unaffected | Selective reset |
| MW-002 | Multiple worlds done simultaneously | Each world resets independently | Batch termination |
| MW-003 | Auto-reset with mixed done states | Only done worlds trigger reset | Partial world reset |
| MW-004 | Manual reset during auto-reset | Manual takes precedence | Conflicting signals |

### Category 5: Order of Operations Edge Cases
| Test ID | Scenario | Expected Behavior | Edge Cases |
|---------|----------|-------------------|------------|
| OO-001 | Reset same step as done=1 | Reset occurs after done flag set | Intra-step timing |
| OO-002 | Reward calculation during reset | Rewards calculated before reset | State consistency |
| OO-003 | Action processing during reset | Actions ignored during reset step | Input buffering |
| OO-004 | Observer tensor consistency | All tensors reflect post-reset state | Atomic updates |

### Category 6: State Persistence Validation
| Test ID | Scenario | Expected Behavior | Edge Cases |
|---------|----------|-------------------|------------|
| SP-001 | Done flag across reset boundary | done=1 → reset → done=0 | State clearing |
| SP-002 | Steps counter across reset | steps=X → reset → steps=200 | Counter restoration |
| SP-003 | Position/rotation reset | Agent returns to spawn position | Spatial reset |
| SP-004 | Progress/reward reset | Progress resets, reward given at end | Metric clearing |

## Implementation Strategy

### Phase 1: Core Mechanics (Week 1)
- **RT-001 to RT-004**: Basic reset timing validation
- **SC-001 to SC-004**: Step counter precision tests
- Focus on single-world scenarios first

### Phase 2: Termination Conditions (Week 2) 
- **CT-001 to CT-004**: Collision vs step termination interactions
- **OO-001 to OO-004**: Order of operations edge cases
- Validate task graph execution sequence

### Phase 3: Multi-World Scenarios (Week 3)
- **MW-001 to MW-004**: Multi-world synchronization tests
- **SP-001 to SP-004**: State persistence across resets
- Performance testing with many simultaneous resets

### Phase 4: Integration Testing (Week 4)
- End-to-end scenario combinations
- Stress testing with rapid reset cycles
- Regression testing against existing functionality

## Test Infrastructure Requirements

### Test Fixtures
```python
@pytest.fixture
def precise_timing_manager(cpu_manager):
    """Manager configured for precise step counting tests"""
    # Disable auto-reset for manual control
    # Enable step-by-step execution mode
    
@pytest.fixture
def multi_world_manager():
    """Manager with 8 worlds for synchronization tests"""
    
@pytest.fixture
def collision_level():
    """Custom level with collision-triggering objects at known positions"""
```

### Helper Functions
```python
def assert_step_sequence(mgr, expected_steps, expected_dones):
    """Validate step counter and done flag sequences"""
    
def trigger_collision_at_step(mgr, world_idx, target_step):
    """Force collision termination at specific step"""
    
def validate_tensor_consistency(mgr):
    """Ensure all tensors reflect consistent state post-reset"""
```

### Monitoring Tools
- **Step-by-step state capture**: Record done/steps/position every step
- **Reset event logging**: Track all reset triggers and their sources
- **Timing validation**: Verify order of operations within each step
- **Multi-world state comparison**: Detect cross-world interference

## Success Criteria

### Functional Requirements
✅ **Reset Immediacy**: Done flag cleared within 1 step of reset trigger  
✅ **Counter Accuracy**: Step counter always resets to exact episodeLen value  
✅ **State Isolation**: Resets affect only target world(s), not others  
✅ **Order Consistency**: stepTracker → reward → reset sequence maintained  

### Performance Requirements
✅ **Reset Latency**: <1ms additional overhead per reset operation  
✅ **Memory Stability**: No memory leaks during rapid reset cycles  
✅ **Multi-world Scaling**: Performance degrades <10% with 8+ worlds  

### Reliability Requirements
✅ **Race Condition Immunity**: No timing-dependent failures in 1000+ test runs  
✅ **State Consistency**: All tensor exports reflect identical post-reset state  
✅ **Error Recovery**: System recovers gracefully from invalid reset sequences  

## Risk Assessment

### High Risk Areas
1. **Intra-step timing**: Order of stepTracker vs resetSystem execution
2. **Tensor export consistency**: Done/steps/position tensors out of sync
3. **Auto-reset edge cases**: Multiple done agents triggering simultaneous resets
4. **Memory management**: Entity cleanup during reset operations

### Mitigation Strategies
1. **Task graph validation**: Explicit dependency verification in tests  
2. **Atomic state updates**: All-or-nothing reset operations  
3. **Reset serialization**: Queue and process resets sequentially  
4. **Memory leak detection**: Automated memory profiling in CI  

## Debugging Tools

### Debug Logging
```cpp
// Add to resetAgentPhysics()
MADRONA_LOG("Reset World %d: steps %d→%d, done %d→0", 
           world_idx, old_steps, consts::episodeLen, old_done);
```

### State Inspection
```python
def debug_episode_state(mgr, world_idx):
    """Print comprehensive world state for debugging"""
    print(f"World {world_idx}:")
    print(f"  Steps: {mgr.steps_remaining_tensor()[world_idx, 0, 0]}")
    print(f"  Done:  {mgr.done_tensor()[world_idx, 0, 0]}")
    print(f"  Pos:   {mgr.self_observation_tensor()[world_idx, 0, :3]}")
```

### Regression Detection
- **Golden state files**: Capture known-good state sequences for comparison
- **Automated bisection**: Identify commits that break reset timing
- **Performance regression**: Track reset operation timing over releases

---

**Document Version**: 1.0  
**Last Updated**: 2025-08-31  
**Reviewer**: Required before implementation  
**Estimated Effort**: 4 weeks (1 developer)