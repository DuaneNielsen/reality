# OpenGL Development Setup for Viewer

## Overview

The Madrona Escape Room viewer requires OpenGL development packages to build. This guide explains how to set up the necessary dependencies on Ubuntu/Debian systems, particularly when using NVIDIA graphics cards.

## Problem Description

When building the project, the viewer executable may not be built if OpenGL development dependencies are missing. The build system checks for the `madrona_viz` target, which requires:

1. GLFW library (`glfw` target)
2. OpenGL development headers
3. GLU (OpenGL Utility) headers

Without these, you'll see that only `headless` is built, but not `viewer`.

## Required Packages

Install the following packages to enable viewer building:

```bash
# OpenGL development headers and libraries
sudo apt install libgl1-mesa-dev libglu1-mesa-dev

# GLFW3 windowing library for OpenGL
sudo apt install libglfw3-dev

# X11 extension libraries required by GLFW
sudo apt install libxinerama-dev libxcursor-dev libxi-dev libxrandr-dev

# pkg-config for library detection (recommended)
sudo apt install pkg-config

# Optional: OpenGL debugging utilities
sudo apt install mesa-utils
```

## NVIDIA Compatibility

**These packages are fully compatible with NVIDIA graphics cards.** Here's why:

### Development vs Runtime

- **Development packages** (`-dev`): Provide headers and link-time libraries
  - Headers define the OpenGL API interface (same across all vendors)
  - Link libraries are thin wrappers for dynamic loading

- **Runtime libraries**: Provide the actual OpenGL implementation
  - NVIDIA: `libGLX_nvidia.so`, `libEGL_nvidia.so`, `libnvidia-glcore.so`
  - Mesa: `libGLX_mesa.so`, `libEGL_mesa.so` (used as fallback)

### How It Works

1. **At compile time**: Your code links against the Mesa dev libraries
2. **At runtime**: The GL dispatch library (`libGLdispatch.so`) automatically routes OpenGL calls to the active driver (NVIDIA in your case)

This vendor-neutral approach (GLVND - GL Vendor Neutral Dispatch) allows you to:
- Build with standard Mesa development packages
- Run with optimized NVIDIA drivers
- Switch between drivers without recompiling

## Verification

After installation, verify the setup:

```bash
# Check that GL headers are installed
ls -la /usr/include/GL/gl.h

# Check that GLFW is available
pkg-config --modversion glfw3

# Check OpenGL runtime (requires X11 running)
glxinfo | grep "OpenGL vendor"
# Should show: OpenGL vendor string: NVIDIA Corporation
```

## Rebuilding After Installation

After installing the dependencies:

```bash
cd build
# Re-run CMake to detect newly installed libraries
cmake ..
# Rebuild
make -j8
# Verify viewer was built
ls -la viewer
```

## Troubleshooting

### Viewer Still Not Building

Check if CMake found GLFW:
```bash
cmake -LA . | grep -i glfw
```

Check the viz subdirectory was processed:
```bash
grep -i "madrona_viz" build/CMakeFiles/CMakeOutput.log
```

### X11 vs Wayland

The viewer requires X11. If using Wayland, you may need:
```bash
# For Wayland compatibility layer
sudo apt install libglfw3-wayland
```

### Missing X11 Extensions

If you get CMake errors about missing Xinerama, Xcursor, Xi, or Xrandr:
```bash
# Install all required X11 extension development packages
sudo apt install libxinerama-dev libxcursor-dev libxi-dev libxrandr-dev
```

### Multiple GL Implementations

If you see duplicate GL libraries:
```bash
# Check which GL libraries are available
ldconfig -p | grep libGL.so
```

The system should automatically select the NVIDIA version when the NVIDIA driver is active.

## Related Documentation

- [CUDA Setup Guide](../cuda/CUDA_SETUP_GUIDE.md) - For GPU simulation support
- [Viewer Guide](../../tools/VIEWER_GUIDE.md) - Using the interactive viewer