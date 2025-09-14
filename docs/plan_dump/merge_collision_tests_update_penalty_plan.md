# Plan: Merge Collision Tests, Update Collision Penalty, and Verify Spec

## Analysis Summary

I've analyzed both test files and the specification. Here's what needs to be done:

### 1. **Test File Merge Strategy**
- **Source**: `test_collision_termination.py` (344 lines) - contains 9 collision-specific tests
- **Target**: `test_reward_termination_system.py` (1125 lines) - comprehensive reward system tests
- **Approach**: Add collision tests as a new section in the reward termination file

### 2. **Tests That Need Penalty Updates**
From `test_collision_termination.py`:
- `test_collision_reward_penalty()` - Line 243: expects `-1.0`, needs `-0.1`
- `test_normal_episode_end_vs_collision_rewards()` - Line 293: expects `-1.0`, needs `-0.1`

From `test_reward_termination_system.py`:
- `test_collision_auto_reset_step_zero_reward()` - Line 839: expects `-1.0`, needs `-0.1`

### 3. **Specification Update**
- **File**: `docs/specs/reward_termination_system.md`
- **Line 62**: Update collision penalty from `-1.0` to `-0.1`
- **Line 328**: Update test assertion from `-1.0` to `-0.1`

### 4. **Implementation Plan**

1. **Merge collision tests** into `test_reward_termination_system.py`:
   - Add collision test class and methods
   - Update all collision penalty assertions to `-0.1`

2. **Update the specification**:
   - Change collision penalty from `-1.0` to `-0.1`
   - Update any related test expectations

3. **Remove old file**: Delete `test_collision_termination.py`

4. **Verify consistency**: Ensure all tests use `-0.1` and spec matches implementation

### 5. **Expected Outcomes**
- Single comprehensive test file for reward/collision/termination systems
- Updated spec reflecting the new `-0.1` collision penalty
- All tests passing with correct penalty expectations
- Consolidated test coverage for easier maintenance

## Detailed Steps

### Step 1: Merge Tests
1. Copy `TestCollisionTermination` class from `test_collision_termination.py`
2. Add it to `test_reward_termination_system.py` as a new section
3. Update all `-1.0` penalty expectations to `-0.1` in merged tests
4. Update existing test in `test_reward_termination_system.py` (line 839)

### Step 2: Update Specification
1. Edit `docs/specs/reward_termination_system.md`:
   - Line 62: Change collision penalty from `-1.0` to `-0.1`
   - Add rationale for softer penalty (reduced training instability)
   - Update any test expectations that reference the penalty

### Step 3: Clean Up
1. Delete `test_collision_termination.py`
2. Verify all tests pass with new penalty
3. Ensure spec accurately reflects implementation

### Step 4: Verification
1. Run merged tests to ensure they pass
2. Cross-check spec against implementation
3. Confirm penalty value consistency across all files

This approach will clean up the test structure while ensuring the specification accurately reflects the softer collision penalty.