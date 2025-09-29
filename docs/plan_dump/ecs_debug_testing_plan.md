# ECS Debug Tracking System - Testing Plan

## Current Status: FAILING
- Build compiles successfully with debug tracking enabled
- Tests run extremely slowly or hang (normal tests should be instant)
- No validation that the system actually works
- GDB integration untested
- Core functionality unverified

## Issues to Investigate & Fix

### 1. Performance Issues (CRITICAL)
**Problem**: Tests that normally run instantly are now extremely slow
**Potential Causes**:
- Excessive mutex contention in hot paths
- Memory allocation tracking overhead too high
- Infinite loops or blocking operations in tracking code
- Secondary index operations (by_archetype_, by_world_) too expensive

**Action Items**:
- [ ] Profile the tracking code to find bottlenecks
- [ ] Add conditional compilation for hot path operations
- [ ] Consider lazy initialization of tracking structures
- [ ] Remove tracking from frequently called paths (consider sampling)

### 2. Functional Validation (MISSING)
**Problem**: No tests verify the system actually works
**Need to Test**:
- [ ] Address lookup returns correct component info
- [ ] Component registration tracking works
- [ ] Table allocation tracking works
- [ ] Archetype and world tracking works
- [ ] Memory map generation works
- [ ] GDB function integration works

## Test Implementation Strategy

### Phase 1: Basic Functionality Tests
Create minimal tests that verify core tracking without full simulation:

#### Test 1: Component Registration Tracking
```cpp
TEST(ECSDebugTracker, ComponentRegistrationTracking) {
    #ifdef MADRONA_ECS_DEBUG_TRACKING
    // Create minimal StateManager
    // Register a simple component
    // Verify component appears in debug tracker
    // Check component metadata is correct
    #else
    GTEST_SKIP() << "Debug tracking not enabled";
    #endif
}
```

#### Test 2: Address Lookup Basic
```cpp
TEST(ECSDebugTracker, BasicAddressLookup) {
    #ifdef MADRONA_ECS_DEBUG_TRACKING
    // Create simple ECS setup
    // Get address of a component
    // Use lookup to get info
    // Verify returned data matches expectations
    #endif
}
```

#### Test 3: Memory Map Generation
```cpp
TEST(ECSDebugTracker, MemoryMapGeneration) {
    #ifdef MADRONA_ECS_DEBUG_TRACKING
    // Create simple ECS setup
    // Generate memory map
    // Verify output contains expected entries
    // Check for no crashes or hangs
    #endif
}
```

### Phase 2: GDB Integration Tests
```cpp
TEST(ECSDebugTracker, GDBFunctionAccess) {
    #ifdef MADRONA_ECS_DEBUG_TRACKING
    // Test C function exports work
    // Call ecs_print_statistics()
    // Call ecs_memory_summary()
    // Verify no crashes
    #endif
}
```

### Phase 3: Performance Validation
```cpp
TEST(ECSDebugTracker, PerformanceImpact) {
    #ifdef MADRONA_ECS_DEBUG_TRACKING
    // Time basic operations with/without tracking
    // Ensure overhead is reasonable (<50% for debug builds)
    // Verify no deadlocks or infinite loops
    #endif
}
```

### Phase 4: Integration with Real Simulation
Only after basic tests pass:
```cpp
TEST(ECSDebugTracker, FullSimulationIntegration) {
    #ifdef MADRONA_ECS_DEBUG_TRACKING
    // Run minimal manager creation
    // Verify tracking works during real ECS operations
    // Test address lookup on actual simulation components
    #endif
}
```

## GDB Testing Plan

### Manual GDB Test Procedure
1. Build with debug tracking enabled
2. Create simple test executable that creates ECS manager
3. Run under GDB
4. Load GDB scripts: `source external/madrona/.gdbinit.ecs`
5. Test commands:
   - `ecs-help`
   - `ecs-stats`
   - `ecs-map`
   - `ecs-inspect <some_address>`

### Expected Outcomes
- GDB commands should load without errors
- `ecs-stats` should show tracker statistics
- `ecs-map` should show memory layout
- `ecs-inspect` should identify component at valid addresses

## Debugging Strategy

### Step 1: Isolate Performance Issues
- Comment out all tracking code except basic registration
- Run simple test to see if it's fast again
- Gradually re-enable tracking features to identify bottleneck

### Step 2: Add Debug Logging
```cpp
#ifdef MADRONA_ECS_DEBUG_TRACKING
#define DEBUG_LOG(msg) std::cout << "[ECS_DEBUG] " << msg << std::endl
#else
#define DEBUG_LOG(msg)
#endif
```

Add logging to:
- Constructor/destructor calls
- Mutex lock/unlock operations
- Major tracking operations
- Hot path entries/exits

### Step 3: Minimal Viable Test
Create the simplest possible test:
```cpp
TEST(ECSDebugTracker, MinimalViability) {
    #ifdef MADRONA_ECS_DEBUG_TRACKING
    // Just call ecs_memory_summary()
    // Should not crash or hang
    ecs_memory_summary();
    EXPECT_TRUE(true); // If we get here, basic functionality works
    #endif
}
```

## Success Criteria (Definition of Done)

### Minimum Viable System:
1. ✅ Compiles without errors
2. ❌ Simple test completes in <1 second
3. ❌ `ecs_memory_summary()` works without hanging
4. ❌ Basic address lookup returns reasonable data
5. ❌ GDB commands load and execute

### Full System Validation:
1. All basic functionality tests pass
2. Performance overhead <50% for debug builds
3. GDB integration works in real debugging session
4. Memory map shows actual ECS layout
5. Address lookup works for real simulation components

## Current Priority Actions

1. **IMMEDIATE**: Fix performance issues preventing basic testing
2. **NEXT**: Implement minimal viability test
3. **THEN**: Build up test suite incrementally
4. **FINALLY**: Validate GDB integration

## Timeline Estimate
- Performance fix: 1-2 hours
- Basic tests: 2-3 hours
- GDB validation: 1 hour
- Full test suite: 3-4 hours

**Total**: 7-10 hours to complete and validate system

## Risk Assessment

**High Risk**: Current performance issues suggest fundamental problems in the implementation
**Medium Risk**: Complex mutex/threading issues may be hard to debug
**Low Risk**: GDB integration should work once core system is stable

The system needs significant debugging and testing before it can be considered functional.