### Debugging with GDB

**Important**: Always use the MCP GDB server for debugging. The MCP server provides a clean interface for debugging sessions and handles GDB interactions properly.

## Installing the MCP GDB Server

### Prerequisites
- Claude Code CLI installed
- Node.js v20 or later
- GDB installed on your system

### Installation Steps

**Installation Notice**

When the GDB MCP server is not installed, Claude Code will show a session hook message. Simply tell Claude to read the GDB_GUIDE.md and follow the installation instructions.

**Manual Installation**

If you need to install manually:

1. **Initialize the submodule** (if not already done):
```bash
git submodule update --init external/mcp-gdb
```

2. **Build the MCP GDB server**:
```bash
# From the project root directory
cd external/mcp-gdb
npm install
npm run build
cd ../..  # Return to project root
```

**Note**: The build must be run from within the `external/mcp-gdb` directory for the build paths to resolve correctly.

3. **Add to Claude Code** (using absolute path):
```bash
claude mcp add gdb node "$(pwd)/external/mcp-gdb/build/index.js"
```

4. **Restart Claude Code**:
```bash
# Exit current session and restart Claude Code for MCP server to be loaded
```

5. **Verify installation**:
```bash
claude mcp list
# Should show: gdb: node /full/path/to/external/mcp-gdb/build/index.js - ✓ Connected
```

**Troubleshooting**

If the server shows as "Failed to connect":
- Ensure you used the absolute path when adding the server
- Test the server manually: `cd external/mcp-gdb && node build/index.js` 
- Remove and re-add with correct path: `claude mcp remove gdb && claude mcp add gdb node "$(pwd)/external/mcp-gdb/build/index.js"`

```bash
# Build with debug symbols (CPU mode recommended for debugging)
mkdir build
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DMADRONA_CUDA_SUPPORT=OFF
make -C build -j8

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

## Debugging Python Tests with GDB

When debugging C++ code that's called from Python tests, follow this specific process:

### The Problem with Common Approaches

Don't make these mistakes:
1. **Running shared library directly in GDB** - Shared libraries need an executable to load them
2. **Debugging the test runner directly** - Overly complex and doesn't load right symbols
3. **Not setting up pending breakpoints** - C++ symbols aren't loaded until shared library loads

### Correct Process for Python Test Debugging

#### Step 1: Find Your Python Executable
```bash
# For uv-managed projects
uv run which python
# Output: /home/duane/madrona_escape_room/.venv/bin/python
```

#### Step 2: Load Python in GDB
Using MCP GDB tools:
```python
mcp__gdb__gdb_start()
mcp__gdb__gdb_load(
    sessionId="your_session_id", 
    program="/home/duane/madrona_escape_room/.venv/bin/python",
    arguments=["-m", "pytest", "tests/python/test_file.py::test_function", "-v", "-s"]
)
```

#### Step 3: Enable Pending Breakpoints (CRUCIAL)
```python
mcp__gdb__gdb_command(sessionId="your_session_id", command="set breakpoint pending on")
```

#### Step 4: Set C++ Breakpoints
```python
mcp__gdb__gdb_set_breakpoint(sessionId="your_session_id", location="batch_renderer.cpp:178")
# Shows as "pending" until shared library loads
```

#### Step 5: Run and Debug
```python
mcp__gdb__gdb_continue(sessionId="your_session_id")
```

### Execution Flow
1. Python starts and loads pytest
2. Python imports your modules
3. C++ shared library loads (breakpoints become active)
4. Test executes and hits your C++ breakpoints

### Real Example: Shader Path Bug Discovery
```gdb
Thread 1 "python" hit Breakpoint 1, madrona::render::makeDrawShaders (
    dev=..., 
    repeat_sampler=0xb2ba420, 
    clamp_sampler=0xb2ba420, 
    depth_only=false  # <-- KEY: Using RGB shader, not depth shader!
) at batch_renderer.cpp:184
```

This revealed tests were using `batch_draw_rgb.hlsl` instead of `batch_draw_depth.hlsl`.

### Key Success Factors
- ✅ Use correct Python executable (from your venv)
- ✅ Always enable pending breakpoints first
- ✅ Set breakpoints before running
- ✅ Be patient with Python/pytest startup
- ✅ Check function arguments for execution flow insights

### Common Mistakes to Avoid
- ❌ Loading shared library directly: `gdb build/libmadrona_escape_room_c_api.so`
- ❌ Not enabling pending breakpoints
- ❌ Using system Python instead of venv Python
- ❌ Setting breakpoints after program already ran
