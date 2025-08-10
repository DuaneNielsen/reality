# Installation Guide for Claude

This guide provides the step-by-step process for helping users install the Madrona Escape Room project.

## Prerequisites Check
```bash
# Verify we're in the project directory
pwd  # Should show: /path/to/madrona_escape_room

# Check current git submodule status
git submodule status
```

## Step 1: Initialize Git Submodules
```bash
git submodule update --init --recursive
```
**Action**: Run this command automatically

## Step 2: System Dependencies (Ubuntu)
**Action**: Ask the user to run this command (requires sudo):
```bash
sudo apt install libx11-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev mesa-common-dev
```
**Message to user**: "Please run the following command to install OpenGL development libraries (requires sudo privileges):"

## Step 2a: CUDA Setup (Optional - for GPU acceleration)
**Prerequisites for GPU mode**: CUDA 12.5 is required for GPU acceleration.

**IMPORTANT**: 
- **Only CUDA 12.5 is supported**
- CUDA 12.6 has LLVM bugs and won't work
- CUDA 12.8 is not yet supported

**Action**: If user wants GPU support, ask them to:

1. **Install CUDA 12.5** (if not already installed):
   - Download from NVIDIA's archive
   - Install to `/usr/local/cuda-12.5/`

2. **Verify CUDA 12.5 installation**:
   ```bash
   ls -la /usr/local/cuda-12.5/targets/x86_64-linux/lib/libnvrtc-builtins*
   ```

3. **Configure system library path** (requires sudo):
   ```bash
   sudo bash -c 'echo "/usr/local/cuda-12.5/targets/x86_64-linux/lib" > /etc/ld.so.conf.d/cuda-12-5.conf'
   sudo ldconfig
   ```

4. **Verify CUDA setup**:
   ```bash
   ldconfig -p | grep nvrtc-builtins
   ```

**Alternative** (if no sudo access):
```bash
export LD_LIBRARY_PATH=/usr/local/cuda-12.5/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
```

**Message to user**: "For GPU acceleration, you need CUDA 12.5. Other CUDA versions are not supported. Skip this step if you only need CPU mode."

## Step 3: Setup Python Virtual Environment
```bash
# Create and activate virtual environment using uv
uv venv
source .venv/bin/activate
```
**Action**: Run these commands automatically

## Step 4: Build C++ Components
```bash
# Create build directory
mkdir build

# Configure with CMake
cmake -B build

# Build with make
make -C build -j8 -s
```
**Action**: Run these commands automatically
**Note**: If cmake fails with compiler errors about `-nostdlib++` or `-march=x86-64-v3`, use the alternative:
```bash
cmake -B build -DCMAKE_TOOLCHAIN_FILE=external/madrona/cmake/toolchain/madrona-toolchain.cmake
```

## Step 5: Install Python Package
```bash
uv pip install -e .
```
**Action**: Run this command automatically

## Step 6: Verify Build
```bash
# Check if binaries were created
ls -la build/viewer build/headless

# Test Python import
uv run python -c "import madrona_escape_room; print('Import successful')"
```
**Action**: Run these verification commands automatically

## Step 7: Install Test Dependencies and Run Tests
Reference: `docs/development/testing/TESTING_GUIDE.md`

```bash
# Install development dependencies (including pytest)
uv sync --group dev

# Run CPU tests first (always)
uv run pytest tests/python/ -v --no-gpu

# Only if CPU tests pass, run GPU tests
uv run pytest tests/python/ -v -k "gpu"
```
**Action**: Run these test commands automatically in sequence
**Note**: GPU tests only run if CPU tests pass and CUDA is available

## Final Verification
```bash
# Test the viewer
./build/viewer --help

# Test quick smoke test (CPU)
./tests/quick_test.sh

# Test GPU mode (if CUDA 12.5 is installed)
./build/headless --cuda 0 --num-worlds 1 --num-steps 100 --rand-actions
```
**Action**: Run these final verification commands
**Note**: GPU test will only work if CUDA 12.5 is properly configured

## Troubleshooting Notes
- If build fails: Check if using correct cmake toolchain
- If tests fail: Refer to `docs/development/testing/TESTING_GUIDE.md`
- If viewer fails: Ensure OpenGL libraries are installed
- If GPU mode fails: Refer to `docs/deployment/cuda/CUDA_SETUP_GUIDE.md`
  - Common issue: `libnvrtc-builtins.so.12.5` not found
  - Solution: Configure system library path for CUDA 12.5
  - Only CUDA 12.5 is supported (not 12.6 or 12.8)