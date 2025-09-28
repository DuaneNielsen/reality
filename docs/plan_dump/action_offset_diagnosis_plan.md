# Plan to Diagnose the One-Step Action Offset Problem

## Problem Summary
Replay actions are being applied one step ahead of when they were recorded:
- Recording: Steps 1-10 use SLOW_LEFT, steps 11-15 use NONE
- Replay: Steps 1-11 use SLOW_LEFT (extra step!), steps 12-16 use NONE

## Potential Root Causes & Evidence Gathering

### 1. **Initial State Mismatch**
**Hypothesis:** The recording starts with a pre-initialized step, while replay starts from scratch
**Evidence to gather:**
- Check if Manager constructor calls `step()` for both recording and replay managers
- Compare step counters at initialization between recording vs replay
- Check if the initial reset/step 0 is handled differently

### 2. **Off-by-One in Action Index Calculation**
**Hypothesis:** The `replayStep()` function uses wrong index calculation
**Evidence to gather:**
- Trace the exact value of `currentReplayStep` when accessing actions
- Check if `currentReplayStep` starts at 0 or 1
- Verify the formula: `base_idx = (currentReplayStep * num_worlds * numActionComponents)`

### 3. **Recording Captures Actions at Wrong Time**
**Hypothesis:** Actions are recorded BEFORE or AFTER they should be
**Evidence to gather:**
- Check when `recordActions()` is called relative to `step()`
- Verify if actions are captured from the current frame or previous frame
- Compare action recording timing with action application timing

### 4. **Replay Increments Step Counter at Wrong Time**
**Hypothesis:** `currentReplayStep++` happens before/after action application incorrectly
**Evidence to gather:**
- Check if increment happens before or after `setAction()` calls
- Compare with how recording increments its frame counter
- Verify when step 0 actions are loaded vs when counter increments

### 5. **Test Sequence Logic Difference**
**Hypothesis:** The test applies actions differently for recording vs replay
**Evidence to gather:**
- Recording: Sets action THEN calls `step()`
- Replay: Calls `replay_step()` THEN `step()`
- Check if there's a timing mismatch in this sequence

## Diagnostic Steps to Execute

### 1. **Add debug logging to trace exact action indices:**
- Log `currentReplayStep` value when loading actions
- Log the exact action values being loaded from the replay data
- Log when `currentReplayStep` is incremented

### 2. **Compare initialization sequences:**
- Log step counters at Manager construction
- Trace the initial `step()` call in constructor
- Check if replay manager gets an extra initialization step

### 3. **Verify action recording timing:**
- Log when actions are captured during recording
- Check if recording happens before or after simulation step
- Verify the frame counter when actions are recorded

### 4. **Test with minimal reproduction:**
- Create a simple 3-step test (action A, action B, action C)
- Log every action set/load operation
- Compare exact timing between recording and replay

## Proposed Fix Approaches (depending on root cause)

### **If initial state mismatch:**
- Ensure replay starts from same initialized state as recording
- Skip or add an initialization step as needed

### **If off-by-one in index:**
- Adjust the action index calculation in `replayStep()`
- May need to use `currentReplayStep - 1` or similar

### **If recording timing issue:**
- Move action recording to correct position relative to `step()`
- Ensure actions are captured at the right frame

### **If increment timing issue:**
- Move `currentReplayStep++` to correct position
- Ensure it matches recording's frame increment logic

### **If test sequence issue:**
- Adjust test to ensure recording and replay use same pattern
- May need to call `replay_step()` at different point in loop

## Key Code Locations to Investigate

### Recording Side:
- `Manager::step()` lines 722-882 - Where actions are recorded
- `Manager::recordActions()` lines 1533-1569 - Action capture logic
- Recording test sequence in `run_rotation_sequence()` lines 19-100

### Replay Side:
- `Manager::replayStep()` lines 1622-1651 - Action loading logic
- `currentReplayStep` increment at line 1642
- Replay test sequence in `run_replay_sequence()` lines 233-270

### Initialization:
- `Manager::Manager()` lines 699-717 - Constructor with initial `step()`
- `Manager::fromReplay()` lines 1680-1842 - Replay manager creation
- Initial state setup differences between recording and replay managers