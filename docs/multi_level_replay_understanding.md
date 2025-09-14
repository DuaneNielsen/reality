# Multi-Level Replay Divergence Issue - Understanding

## Problem Description
When using the viewer in multi-level scenarios, switching worlds during replay causes agent trajectories to diverge from their original recorded paths.

## Initial Misunderstanding
I initially created tests that replayed the same actions on different levels, which obviously fails. This was not testing the actual bug.

## Correct Understanding of the Bug

### Multi-Level Setup
- **World 0**: Level A (e.g., open corridor) 
- **World 1**: Level B (e.g., maze layout)
- **World 2**: Level C (e.g., blocked paths)

### Recording Phase (Correct Behavior)
Each world should record its **own unique action stream** appropriate for its specific level:
- **World 0**: Forward movement actions (appropriate for open corridor)  
- **World 1**: Turn-left-forward actions (appropriate for maze navigation)
- **World 2**: Strafe-right actions (appropriate for avoiding blocked paths)

### Replay Phase (Where Bug Occurs)  
- **Expected**: Each world replays its recorded actions on its original level
- **Bug Trigger**: User switches between worlds in viewer during replay (Tab key, world selection)
- **Expected Behavior**: Each world continues its recorded trajectory on its original level
- **Actual Bug**: When switching worlds, some world's replay gets corrupted/diverges from recorded path

### The Actual Problem
When the viewer switches focus between worlds during replay, there's likely a synchronization bug where:
- World state gets mixed up between worlds
- Action streams get applied to wrong worlds during world switching
- Replay synchronization breaks during viewer world transitions
- Agent trajectories diverge from their recorded paths

## Correct Test Structure Should Be
1. **Multi-World Recording**: Record different action patterns on different levels simultaneously
2. **Control Replay**: Replay without world switching - all trajectories should match their recordings exactly
3. **Divergence Test**: Replay WITH viewer world switching - detect when trajectories break from recorded paths

This would test the actual synchronization bug between multi-level replay and viewer world switching, rather than the obvious failure case of applying wrong actions to wrong levels.

## Status
- **Incorrect implementation created**: Tests that verify obvious failure case
- **Need to implement**: Proper multi-world replay synchronization tests
- **Priority**: High - this affects user experience in viewer with multi-level scenarios

## Files Created (To Be Removed)
- `tests/cpp/unit/test_multi_level_replay_divergence.cpp` - Incorrect test implementation
- `tests/python/test_multi_level_replay_divergence.py` - Incorrect test implementation
- Modified `tests/cpp/CMakeLists.txt` - Added incorrect test to build