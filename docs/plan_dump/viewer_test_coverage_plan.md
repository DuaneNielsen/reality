# Viewer Test Coverage Improvement Plan

## Current Coverage: ~25%
- ✅ Tested: Option parsing, level/recording utilities
- ❌ Untested: Manager integration, input handling, replay system, trajectory tracking

## Implementation Phases

### Phase 1: Mock Infrastructure
Create `tests/cpp/fixtures/mock_components.hpp`:
- MockWindowManager (headless window)
- MockGPUHandle (fake GPU)
- ActionRecorder (capture Manager actions)
- InputSimulator (generate keyboard events)

### Phase 2: Integration Tests
**File**: `tests/cpp/integration/test_viewer_integration.cpp`
- Manager creation with parsed options
- Recording → file → replay round-trip
- Trajectory tracking with real simulation
- Pause/resume state management

### Phase 3: Input Processing Tests  
**File**: `tests/cpp/integration/test_viewer_input.cpp`
- WASD → action conversion
- Q/E rotation mapping
- Shift speed modifier
- R key reset functionality

### Phase 4: Error Handling Tests
**File**: `tests/cpp/integration/test_viewer_errors.cpp`
- Missing/corrupt files
- Invalid option combinations
- GPU initialization failures

### Phase 5: End-to-End Tests
**File**: `tests/cpp/e2e/test_viewer_workflows.cpp`
- Complete recording workflow
- Full replay verification
- Live simulation with tracking

## Expected Coverage: ~85%

## Key Constraints
- GPU tests must run in isolation (one Manager per process)
- No exceptions/RTTI in test code
- Use C API where possible