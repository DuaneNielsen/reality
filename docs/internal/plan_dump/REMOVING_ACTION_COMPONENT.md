# Procedure for Removing an Action Component from Madrona ECS

This document describes the step-by-step process for removing an action component (like grab) from a Madrona-based simulation. This procedure was used to remove the grab action from the escape room environment.

## Overview

Removing an action component requires changes across multiple files:
- C++ struct definitions
- Manager interface and implementation
- All code that calls setAction
- Python unit tests
- Constants definitions

## Step-by-Step Procedure

### Step 1: Update the Action Structure
**File:** `src/types.hpp`

Remove the action field from the Action struct:

```cpp
// Before:
struct Action {
    int32_t moveAmount;
    int32_t moveAngle;
    int32_t rotate;
    int32_t grab;  // Remove this line
};

// After:
struct Action {
    int32_t moveAmount;
    int32_t moveAngle;
    int32_t rotate;
};
```

### Step 2: Add/Update Game Constants
**File:** `src/consts.hpp`

Add a constant for the number of action components to avoid magic numbers:

```cpp
// Add this constant:
inline constexpr madrona::CountT numActionComponents = 3;  // Was 4 with grab
```

### Step 3: Update the Manager Class Interface
**File:** `src/mgr.hpp`

Remove the parameter from the setAction method declaration:

```cpp
// Before:
void setAction(int32_t world_idx, int32_t agent_idx, 
               int32_t move_amount, int32_t move_angle, 
               int32_t rotate, int32_t grab);

// After:
void setAction(int32_t world_idx, int32_t agent_idx, 
               int32_t move_amount, int32_t move_angle, 
               int32_t rotate);
```

### Step 4: Update the Manager Implementation
**File:** `src/mgr.cpp`

1. Update the setAction method implementation to match the new signature:
```cpp
void Manager::setAction(int32_t world_idx,
                        int32_t agent_idx,
                        int32_t move_amount,
                        int32_t move_angle,
                        int32_t rotate)  // grab parameter removed
{
    Action action { 
        .moveAmount = move_amount,
        .moveAngle = move_angle,
        .rotate = rotate,
        // .grab line removed
    };
    // ... rest of implementation
}
```

2. Update the actionTensor() method to use the constant:
```cpp
Tensor Manager::actionTensor() const
{
    return impl_->exportTensor(ExportID::Action, TensorElementType::Int32,
        {
            impl_->cfg.numWorlds,
            consts::numAgents,
            consts::numActionComponents,  // Use constant instead of hardcoded value
        });
}
```

### Step 5: Update All C++ Code That Calls setAction
**File:** `src/viewer.cpp`

1. Remove grab-related variable declarations:
```cpp
// Remove this line:
int32_t g = 0;
```

2. Remove grab key handling:
```cpp
// Remove this block:
if (input.keyHit(Key::G)) {
    g = 1;
}
```

3. Update all setAction calls:
```cpp
// Before:
mgr.setAction(world_idx, agent_idx, move_amount, move_angle, r, g);

// After:
mgr.setAction(world_idx, agent_idx, move_amount, move_angle, r);
```

4. Fix replay log parsing (if applicable):
```cpp
// Remove reading of the 4th component
int32_t move_amount = (*replay_log)[base_idx];
int32_t move_angle = (*replay_log)[base_idx + 1];
int32_t turn = (*replay_log)[base_idx + 2];
// Remove: int32_t g = (*replay_log)[base_idx + 3];
```

**File:** `src/headless.cpp`

Update all setAction calls in the same way:
```cpp
// Before:
mgr.setAction(j, k, x, y, r, 0);

// After:
mgr.setAction(j, k, x, y, r);
```

### Step 6: Update Python Unit Tests
**File:** `tests/python/test_bindings.py`

1. Update expected action tensor shape:
```python
# Before:
assert actions.shape == (4, 1, 4), f"Expected shape (4, 1, 4), got {actions.shape}"

# After:
assert actions.shape == (4, 1, 3), f"Expected shape (4, 1, 3), got {actions.shape}"
```

2. Remove all references to the 4th action component:
```python
# Remove lines like:
actions[:, :, 3] = torch.randint(0, 2, (4, 1))  # Grab
```

3. Update action sequences in deterministic tests:
```python
# Before:
action_sequence = [
    (1, 0, 2, 0),  # Move forward
    (0, 0, 0, 1),  # Stop and grab
    # ...
]
for move_amt, move_angle, rotate, grab in action_sequence:
    actions[:, :, 0] = move_amt
    actions[:, :, 1] = move_angle
    actions[:, :, 2] = rotate
    actions[:, :, 3] = grab

# After:
action_sequence = [
    (1, 0, 2),  # Move forward
    (0, 0, 2),  # Stop (no grab)
    # ...
]
for move_amt, move_angle, rotate in action_sequence:
    actions[:, :, 0] = move_amt
    actions[:, :, 1] = move_angle
    actions[:, :, 2] = rotate
```

### Step 7: Rebuild and Test

1. Rebuild the project:
```bash
cd build && make -j8
```

2. Run the unit tests:
```bash
cd .. && uv run --extra test pytest tests/python -v
```

3. Test with the viewer:
```bash
./build/viewer
```

## Common Pitfalls to Avoid

1. **Forgetting to update the tensor export size** - This causes a mismatch between the C++ struct and Python tensor
2. **Missing some setAction calls** - Search for all occurrences in the codebase
3. **Not updating test assertions** - Tests will fail if they expect the old tensor shape
4. **Hardcoding action sizes** - Always use constants to make future changes easier
5. **Not updating both .hpp and .cpp files** - The declaration and implementation must match

## Verification Checklist

- [ ] Action struct updated in types.hpp
- [ ] Constant added/updated in consts.hpp
- [ ] Manager interface updated in mgr.hpp
- [ ] Manager implementation updated in mgr.cpp
- [ ] All setAction calls updated in viewer.cpp
- [ ] All setAction calls updated in headless.cpp
- [ ] Python tests updated for new tensor shape
- [ ] Project builds without errors
- [ ] All unit tests pass
- [ ] Viewer runs without crashes

## Notes

This procedure can be adapted for:
- Adding new action components (reverse the process)
- Modifying observation components
- Changing other tensor shapes in the ECS

The key principle is maintaining consistency across all layers of the system: struct definitions, tensor exports, function interfaces, and test expectations.