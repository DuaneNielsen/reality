### Debugging with GDB

**Important**: Always use the MCP GDB server for debugging. The MCP server provides a clean interface for debugging sessions and handles GDB interactions properly.

```bash
# Build with debug symbols (CPU mode recommended for debugging)
mkdir build
/opt/cmake/bin/cmake -B build -DCMAKE_BUILD_TYPE=Debug -DMADRONA_CUDA_SUPPORT=OFF
make -C build -j$(nproc)

# Note: Ubuntu 20.04's default GDB (9.2) doesn't support DWARF 5 used by modern compilers
# If you see "DW_FORM_strx1" errors, upgrade to GDB 16.3+
```


When debugging in Claude Code, use the following MCP tools:
- `mcp__gdb__gdb_start` - Start a new debugging session
- `mcp__gdb__gdb_load` - Load executable with arguments
- `mcp__gdb__gdb_set_breakpoint` - Set breakpoints using readable names
- `mcp__gdb__gdb_continue` - Continue execution
- `mcp__gdb__gdb_step` - Step through code
- `mcp__gdb__gdb_backtrace` - View call stack
- `mcp__gdb__gdb_terminate` - End debugging session

Example debugging workflow:
```python
# 1. Start GDB session
mcp__gdb__gdb_start(workingDir="/path/to/build")

# 2. Load program
mcp__gdb__gdb_load(sessionId="...", program="./headless", arguments=["CPU", "1", "10"])

# 3. Set breakpoints using readable C++ names
mcp__gdb__gdb_set_breakpoint(sessionId="...", location="madEscape::Manager::Manager")
mcp__gdb__gdb_set_breakpoint(sessionId="...", location="loadPhysicsObjects")

# 4. Run and debug
mcp__gdb__gdb_continue(sessionId="...")
```

#### Key Breakpoint Locations

For debugging initialization issues:
- `main` - Program entry point
- `madEscape::Manager::Manager` - Manager constructor
- `madEscape::Manager::Impl::init` - Core initialization
- `loadPhysicsObjects` - Physics asset loading
- `loadRenderObjects` - Render asset loading (if rendering enabled)
- `madEscape::Sim::Sim` - Per-world simulator construction
- `madEscape::Sim::setupTasks` - Task graph configuration

For debugging simulation issues:
- `madEscape::Sim::step` - Main simulation step
- `movementSystem` - Agent movement processing
- `physicsSystem` - Physics simulation
- `rewardSystem` - Reward calculation
- `resetSystem` - Episode reset logic
