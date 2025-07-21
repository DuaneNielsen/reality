Carefully and diligently follow the guide below to document $ARGUMENTS

# Step-by-Step Process for Documenting Functions with GDB

This guide provides a systematic approach for documenting Madrona functions using the debugger, distinguishing between BOILERPLATE, GAME_SPECIFIC, and REQUIRED_INTERFACE code.

## Core Principle: Dynamic Tracing Over Static Reading

The debugger's PRIMARY value is showing actual runtime execution paths. ALWAYS:
- Step through functions to see which code actually executes
- Verify assumptions about control flow
- Discover runtime polymorphism (e.g., which impl_->run() is called)
- Identify conditional branches taken

Static code reading alone defeats the purpose of debugging.

## 1. **Identify the Entry Point Function**
- Choose a key function to document (e.g., `resetSystem`, `triggerReset`, `step`)
- Read the function signature and surrounding context to understand its purpose

## 2. **Set Up the Debugging Session**
```bash
# Start GDB
mcp__gdb__gdb_start(workingDir="/path/to/build")

# Load the executable with appropriate arguments
mcp__gdb__gdb_load(sessionId="...", program="./headless", arguments=["CPU", "1", "10"])

# Set breakpoint on the entry point function
mcp__gdb__gdb_set_breakpoint(sessionId="...", location="functionName")
```

## 3. **Run to the Breakpoint**
```bash
# Start execution
mcp__gdb__gdb_command(sessionId="...", command="run")

# Verify you're at the right location
mcp__gdb__gdb_command(sessionId="...", command="where")
```

## 4. **Step Through and Document**
For each step in the function:

### a) Examine Current Location
```bash
# Show current code
mcp__gdb__gdb_command(sessionId="...", command="list")

# Check call stack
mcp__gdb__gdb_backtrace(sessionId="...", limit=5)
```

### b) Determine Code Classification
- Look for classification markers: `[BOILERPLATE]`, `[GAME_SPECIFIC]`, `[REQUIRED_INTERFACE]`
- If no markers, analyze the code:
  - Framework calls (ECS, physics, rendering) → BOILERPLATE
  - Game logic (agents, rooms, rewards) → GAME_SPECIFIC
  - Interface methods (reset, step, action handling) → REQUIRED_INTERFACE

### c) Step or Continue

**CRITICAL: Always step INTO key functions to trace execution**
```bash
# ❌ WRONG: Just reading source code instead of stepping
# ✅ RIGHT: Step into to see actual execution
mcp__gdb__gdb_step(sessionId="...")

# For virtual functions or function pointers:
# - MUST step in to see which implementation runs
# - Example: impl_->run() could be CPUImpl::run() or CUDAImpl::run()

# Example: Tracing a virtual function
mcp__gdb__gdb_step(sessionId="...")  # Step INTO impl_->run()
mcp__gdb__gdb_command(sessionId="...", command="where")  # Verify which implementation

# Or step over utility functions
mcp__gdb__gdb_next(sessionId="...")

# Or continue to next breakpoint
mcp__gdb__gdb_continue(sessionId="...")
```

### When to Use Each GDB Command:

**`gdb_step` (step into)**
- Use when you WANT to trace into a function call
- Essential for understanding which implementation runs (polymorphism)
- Use for functions you're documenting or need to understand
- Example: Stepping into `impl_->run()` to see if it's CPU or CUDA

**`gdb_next` (step over)**
- Use when you DON'T need to trace into a function
- Good for utility functions, getters/setters, or well-understood calls
- Keeps you at the current function level
- Example: Stepping over `has_value()` or simple accessors

**`gdb_finish` (run until return)**
- Use when you've stepped INTO a function by mistake or are done examining it
- Executes until the current function returns to its caller
- WARNING: Don't use on the function you're trying to document!
- Example: Accidentally stepped into STL template code, use finish to get back

**`gdb_continue` (run to next breakpoint)**
- Use to skip large sections of code you don't need to trace
- Runs until hitting another breakpoint or program ends
- Good for jumping between key functions
- Example: After documenting init, continue to the step() function

### d) Set Additional Breakpoints
```bash
# For functions you want to explore
mcp__gdb__gdb_set_breakpoint(sessionId="...", location="interestingFunction")
```

## 5. **Find the Correct Location in CLAUDE.md**
**CRITICAL: Documentation should follow program control flow**

### a) Analyze the Call Stack
- Use the backtrace to understand where this function fits in the execution order
- Identify the parent function that calls this one

### b) Search for Related Documentation
```bash
# Search for the parent function
Grep(pattern="parentFunctionName", path="CLAUDE.md")

# Search for related phase/section
Grep(pattern="Initialization|Reset|Step|Task Graph", path="CLAUDE.md")
```

### c) Determine Insertion Point
- If called from an already documented function, insert after that section
- If part of a sequence (e.g., Step 1, Step 2), add as the next step
- If starting a new phase, create a new subsection
- Follow the execution timeline: initialization → reset → step → etc.

### d) Verify Logical Flow
- Read the sections before and after your insertion point
- Ensure your documentation maintains the narrative flow
- Example progression:
  1. Manager Creation → 2. World Init → 3. Entity Creation → 4. Reset Sequence → 5. Step Execution

## 6. **Document Based on Classification**

### For BOILERPLATE:
- Write English summary of the control flow
- List function names in call order
- Example:
  ```
  The reset sequence calls:
  1. `PhysicsSystem::reset()` - Clears collision state
  2. `RenderSystem::clear()` - Resets render buffers
  3. `broadphase.rebuild()` - Reconstructs spatial structures
  ```

### For GAME_SPECIFIC or REQUIRED_INTERFACE:
- Create code blocks showing the actual implementation
- Add inline comments explaining game logic
- Example:
  ```cpp
  // [GAME_SPECIFIC] Reward calculation
  if (agent_pos.y > progress.maxY) {
      reward.r = consts::rewardPerLevel;  // Give reward for progress
      progress.maxY = agent_pos.y;         // Update max progress
  }
  ```

## 7. **Organize Documentation Structure**
```markdown
### Function Name (`functionName()`)

Brief description of what the function does.

**Call Sequence:**
1. `firstCall()` - Purpose
2. `secondCall()` - Purpose
   - `nestedCall()` - If important

**Game-Specific Implementation:**
```cpp
// [GAME_SPECIFIC] Code blocks here
```

**Key Points:**
- Important behavior
- Side effects
- Dependencies
```

## 8. **Clean Up**
```bash
# Terminate GDB session
mcp__gdb__gdb_terminate(sessionId="...")

# Update todo list to track progress
TodoWrite(todos=[...mark items as completed...])
```

## 9. **Best Practices**
- Set breakpoints on both the entry function AND key subfunctions
- Use `info locals` to examine variable values when needed
- For complex flows, create a todo list item for each major function
- Read source files directly when GDB output is insufficient
- Document the "why" not just the "what" for game-specific code
- **Always maintain chronological/logical flow in documentation**

## Example Todo List for Documentation:
```
1. Set breakpoint on main entry function
2. Document BOILERPLATE initialization sequence
3. Step into first GAME_SPECIFIC function
4. Document game logic with code examples
5. Find correct insertion point in CLAUDE.md
6. Continue to REQUIRED_INTERFACE sections
7. Update CLAUDE.md with findings
8. Commit documentation changes
```

This process ensures thorough documentation while distinguishing between framework code (summarized) and game-specific code (shown in detail), and maintains a logical flow that mirrors the program's execution order.

## Common Mistakes to Avoid

1. **Reading source instead of stepping**: If you find yourself using Read() instead of step/next, you're not debugging properly
2. **Assuming code paths**: Virtual functions, function pointers, and conditionals require stepping to verify
3. **Marking tasks complete without tracing**: Each TODO item should correspond to actual debugging steps performed
4. **Skipping the "boring" parts**: Framework code often reveals important execution order and dependencies