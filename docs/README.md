# Madrona Escape Room Documentation

This documentation covers the Madrona Escape Room - a high-performance 3D single-agent reinforcement learning environment built on the Madrona Engine.

## Documentation Structure

### 🏗️ Architecture
Core system design and execution flow documentation.

- [**ECS Architecture**](architecture/ECS_ARCHITECTURE.md) - Madrona's archetype-based ECS design and performance characteristics
- [**Initialization Sequence**](architecture/INITIALIZATION_SEQUENCE.md) - Detailed step-by-step manager creation and world initialization process
- [**Reset Sequence**](architecture/RESET_SEQUENCE.md) - Episode reset mechanics
- [**Step Sequence**](architecture/STEP_SEQUENCE.md) - Simulation step execution
- [**Collision System**](architecture/COLLISION_SYSTEM.md) - Physics and collision detection

### 🛠️ Development
Guides for developers working on the codebase.

- [**Level Format**](development/LEVEL_FORMAT.md) - Level format specification
- [**OpenGL Development Setup**](development/OPENGL_DEVELOPMENT_SETUP.md) - Install required packages to build the 3D viewer
- [**Add Component**](development/ADD_COMPONENT.md) - How to add new ECS components
- [**Add System**](development/ADD_SYSTEM.md) - How to add new ECS systems
- [**Export Component**](development/EXPORT_COMPONENT.md) - Export components to Python bindings
- [**Progress Component**](development/using_progress_component.md) - How to access and monitor agent progress in Python
- [**Trajectory Logging**](development/using_trajectory_logging.md) - Enable and monitor agent position tracking during simulation
- [**Recording and Debugging**](development/using_recording_debugging.md) - Use context managers to record actions and track trajectories for debugging workflows
- [**Testing Guide**](development/TESTING_GUIDE.md) - Learn testing patterns and understand GPU manager constraints for reliable tests
- [**C++ Testing Guide**](development/CPP_TESTING_GUIDE.md) - Write and run C++ unit tests using GoogleTest framework
- [**Performance Test Guide**](development/PERFORMANCE_TEST_GUIDE.md) - Run performance benchmarks and maintain baseline thresholds
- [**GDB Guide**](development/GDB_GUIDE.md) - Debug C++ code step-by-step using the MCP GDB server
- [**C Stdout Capture**](development/C_STDOUT_CAPTURE.md) - Capture C-level stdout output in Python environments for debugging
- [**Python REPL MCP Setup**](development/PYTHON_REPL_MCP_SETUP.md) - Interactive development with MCP server for debugging and exploration
- [**Asset Loading**](development/ASSET_LOADING.md) - Understand how physics and rendering assets are loaded and managed by the engine

### 🚀 Deployment
Production deployment, packaging, and platform-specific guides.

- [**Python Bindings Guide**](deployment/PYTHON_BINDINGS_GUIDE.md) - Complete guide to the ctypes-based Python bindings with API reference and context managers
- [**ctypes Packaging Guide**](deployment/CTYPES_PACKAGING_GUIDE.md) - Required files and libraries for distributing ctypes bindings
- [**Packaging TODO**](deployment/PACKAGING_TODO.md) - Outstanding packaging tasks
- [**CUDA Setup Guide**](deployment/CUDA_SETUP_GUIDE.md) - CUDA 12.5 setup and common library issues
- [**CUDA Linking Issues**](deployment/DEBUGGING_CUDA_LINKING_ISSUES.md) - Troubleshooting CUDA problems
- [**CUDA Deadlock Analysis**](deployment/CUDA_DEADLOCK_SOLUTION_ANALYSIS.md) - Diagnose and resolve CUDA initialization deadlocks in Python bindings
- [**CFFI CUDA Research**](deployment/CFFI_CUDA_SEGFAULT_RESEARCH.md) - Debug and understand CUDA/Python CFFI segmentation faults
- [**Headless Mode**](deployment/HEADLESS_MODE.md) - Run simulations without graphics for server deployment and batch processing
- [**Headless Quick Reference**](deployment/HEADLESS_QUICK_REFERENCE.md) - Quick command-line reference for headless simulation options

### 🔧 Tools
Documentation for development and visualization tools.

- [**Viewer Guide**](tools/VIEWER_GUIDE.md) - Use the 3D viewer for visualization, manual agent control, and recording/replaying sessions
- [**File Inspector Guide**](tools/FILE_INSPECTOR_GUIDE.md) - Unified tool for examining and validating recording files (.rec) and level files (.lvl)

## Quick Links

- **Environment Specification**: See [ENVIRONMENT.md](../ENVIRONMENT.md) for action/observation space and usage
- **Getting Started**: See the main [README.md](../README.md) in the project root
- **Build Instructions**: Check [CLAUDE.md](../CLAUDE.md) for build commands
- **Testing**: Refer to test commands in [CLAUDE.md](../CLAUDE.md)

## Contributing

When adding new documentation:
1. Place it in the appropriate category folder
2. Update this README if adding a new major document
3. Follow the existing naming conventions
4. Include clear examples and code snippets where helpful