# Madrona Escape Room Documentation

This documentation covers the Madrona Escape Room - a high-performance 3D multi-agent reinforcement learning environment built on the Madrona Engine.

## Documentation Structure

### 🏗️ Architecture
Core system design and execution flow documentation.

- [**ECS Architecture**](architecture/ECS_ARCHITECTURE.md) - Entity Component System overview
- [**Initialization Sequence**](architecture/INITIALIZATION_SEQUENCE.md) - System startup flow
- [**Reset Sequence**](architecture/RESET_SEQUENCE.md) - Episode reset mechanics
- [**Step Sequence**](architecture/STEP_SEQUENCE.md) - Simulation step execution
- [**Collision System**](architecture/COLLISION_SYSTEM.md) - Physics and collision detection

### 🛠️ Development
Guides for developers working on the codebase.

#### Instructions
- [**Add Component**](development/instructions/ADD_COMPONENT.md) - How to add new ECS components
- [**Add System**](development/instructions/ADD_SYSTEM.md) - How to add new ECS systems
- [**Export Component**](development/instructions/EXPORT_COMPONENT.md) - Export components to Python bindings

#### Components
- [**Progress Component**](development/components/using_progress_component.md) - Tracking agent progress
- [**Trajectory Logging**](development/components/using_trajectory_logging.md) - Recording agent trajectories

#### Debugging
- [**GDB Guide**](development/debugging/GDB_GUIDE.md) - Step-through debugging with GDB

#### Advanced
- [**Asset Loading**](development/advanced/ASSET_LOADING.md) - How assets are loaded and managed

### 🚀 Deployment
Production deployment, packaging, and platform-specific guides.

#### Packaging
- [**ctypes Packaging Guide**](deployment/packaging/CTYPES_PACKAGING_GUIDE.md) - Packaging ctypes bindings
- [**Python Bindings Architecture**](deployment/packaging/PYTHON_BINDINGS_ARCHITECTURE.md) - Bindings overview
- [**Bindings README**](deployment/packaging/README_BINDINGS.md) - Bindings usage guide
- [**Packaging TODO**](deployment/packaging/PACKAGING_TODO.md) - Outstanding packaging tasks

#### CUDA
- [**CUDA Setup Guide**](deployment/cuda/CUDA_SETUP_GUIDE.md) - CUDA environment setup
- [**CUDA Linking Issues**](deployment/cuda/DEBUGGING_CUDA_LINKING_ISSUES.md) - Troubleshooting CUDA problems
- [**CUDA Deadlock Analysis**](deployment/cuda/CUDA_DEADLOCK_SOLUTION_ANALYSIS.md) - Deadlock solutions
- [**CFFI CUDA Research**](deployment/cuda/CFFI_CUDA_SEGFAULT_RESEARCH.md) - Segfault investigation

#### Headless
- [**Headless Mode**](deployment/headless/HEADLESS_MODE.md) - Running without graphics
- [**Headless Quick Reference**](deployment/headless/HEADLESS_QUICK_REFERENCE.md) - Command reference

### 🔧 Tools
Documentation for development and visualization tools.

- [**Viewer Guide**](tools/VIEWER_GUIDE.md) - Interactive viewer usage

### 🔒 Internal
Internal development documentation and planning archives.

- [**Plan Dump**](internal/plan_dump/) - Development planning documents

## Quick Links

- **Getting Started**: See the main [README.md](../README.md) in the project root
- **Build Instructions**: Check [CLAUDE.md](../CLAUDE.md) for build commands
- **Testing**: Refer to test commands in [CLAUDE.md](../CLAUDE.md)

## Contributing

When adding new documentation:
1. Place it in the appropriate category folder
2. Update this README if adding a new major document
3. Follow the existing naming conventions
4. Include clear examples and code snippets where helpful