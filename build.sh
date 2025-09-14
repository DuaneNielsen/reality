#!/bin/bash

# Madrona Escape Room Build Script
# Handles cleaning, building, and rebuilding the project with integrated madrona

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

print_usage() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "COMMANDS:"
    echo "  clean     Clean all build artifacts from project and submodules"
    echo "  build     Quick incremental build (default if no command given)"
    echo "  fullbuild Full build with CMake reconfiguration and codegen"
    echo "  rebuild   Clean then full build"
    echo "  help      Show this help message"
    echo ""
    echo "OPTIONS:"
    echo "  --test    Run tests after building"
    echo "  --jobs N  Use N parallel jobs for building (default: 16)"
    echo "  --verbose Show detailed output (default: errors only)"
    echo ""
    echo "Examples:"
    echo "  $0              # Quick incremental build"
    echo "  $0 build        # Quick incremental build" 
    echo "  $0 fullbuild    # Full build with CMake + codegen"
    echo "  $0 clean        # Clean all build artifacts"
    echo "  $0 rebuild      # Clean then full build"
    echo "  $0 build --test # Quick build and run tests"
}

log_info() {
    # Suppress info messages - only show on verbose mode
    if [ "${VERBOSE:-false}" = true ]; then
        echo -e "${BLUE}[INFO]${NC} $1"
    fi
}

log_success() {
    # Suppress success messages - only show on verbose mode  
    if [ "${VERBOSE:-false}" = true ]; then
        echo -e "${GREEN}[SUCCESS]${NC} $1"
    fi
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

clean_build_artifacts() {
    log_info "Cleaning build artifacts..."
    
    # Clean CUDA kernel cache if it exists (before removing build directory)
    if [ -f "$PROJECT_ROOT/build/madrona_kernels.cache" ]; then
        log_info "Removing CUDA kernel cache: $PROJECT_ROOT/build/madrona_kernels.cache"
        rm -f "$PROJECT_ROOT/build/madrona_kernels.cache"
    fi
    
    # Clean main build directory
    if [ -d "$PROJECT_ROOT/build" ]; then
        log_info "Removing main build directory: $PROJECT_ROOT/build"
        rm -rf "$PROJECT_ROOT/build"
    fi
    
    # Clean madrona build directory (madrona is now directly integrated)
    if [ -d "$PROJECT_ROOT/external/madrona/build" ]; then
        log_info "Removing madrona build directory: $PROJECT_ROOT/external/madrona/build"
        rm -rf "$PROJECT_ROOT/external/madrona/build"
    fi
    
    # Clean Python cache files
    log_info "Cleaning Python cache files..."
    find "$PROJECT_ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "$PROJECT_ROOT" -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete 2>/dev/null || true
    
    # Clean generated files
    if [ -f "$PROJECT_ROOT/madrona_escape_room/generated_constants.py" ]; then
        log_info "Removing generated constants file"
        rm -f "$PROJECT_ROOT/madrona_escape_room/generated_constants.py"
    fi
    
    if [ -f "$PROJECT_ROOT/madrona_escape_room/generated_dataclasses.py" ]; then
        log_info "Removing generated dataclasses file"
        rm -f "$PROJECT_ROOT/madrona_escape_room/generated_dataclasses.py"
    fi
    
    # Clean shared libraries and compiled extensions
    log_info "Cleaning shared libraries and compiled extensions..."
    find "$PROJECT_ROOT" -maxdepth 1 -name "*.so*" -delete 2>/dev/null || true
    find "$PROJECT_ROOT/madrona_escape_room" -name "*.so*" -delete 2>/dev/null || true
    
    # Clean .build_venv directory
    if [ -d "$PROJECT_ROOT/.build_venv" ]; then
        log_info "Removing build venv directory"
        rm -rf "$PROJECT_ROOT/.build_venv"
    fi
    
    # Clean git untracked files in remaining submodules (but be careful)
    cd "$PROJECT_ROOT"
    # Note: madrona is now directly integrated, not a submodule

    if [ -d "external/mcp-gdb/.git" ]; then
        log_info "Cleaning untracked files in mcp-gdb submodule"
        cd external/mcp-gdb
        git clean -fd 2>/dev/null || log_warning "Could not clean mcp-gdb submodule"
        cd "$PROJECT_ROOT"
    fi
    
    log_success "Clean completed successfully"
}

check_prerequisites() {
    # Check prerequisites
    if ! command -v cmake &> /dev/null; then
        log_error "CMake not found. Please install CMake."
        exit 1
    fi
    
    if ! command -v make &> /dev/null; then
        log_error "Make not found. Please install build-essential."
        exit 1
    fi
}

check_uv_prerequisite() {
    if ! command -v uv &> /dev/null; then
        log_error "uv not found. Please install uv for Python package management."
        exit 1
    fi
}

run_tests_if_requested() {
    local run_tests=${1:-false}
    
    # Run tests if requested
    if [ "$run_tests" = true ]; then
        log_info "Running tests..."
        
        # Run C++ tests
        if [ -f "$PROJECT_ROOT/build/mad_escape_tests" ]; then
            log_info "Running C++ tests..."
            cd "$PROJECT_ROOT"
            ./build/mad_escape_tests
        fi
        
        # Run Python tests
        if [ -d "$PROJECT_ROOT/tests/python" ]; then
            log_info "Running Python tests..."
            cd "$PROJECT_ROOT"
            uv run --group dev pytest tests/python/ -v || log_warning "Some Python tests failed"
        fi
    fi
}

quick_build() {
    local jobs=${1:-16}
    local run_tests=${2:-false}
    
    log_info "Quick incremental build with $jobs parallel jobs..."
    
    cd "$PROJECT_ROOT"
    check_prerequisites
    
    # Always delete CUDA kernel cache to ensure correct optimization level
    if [ -f "$PROJECT_ROOT/build/madrona_kernels.cache" ]; then
        log_info "Removing CUDA kernel cache to ensure correct optimization level"
        rm -f "$PROJECT_ROOT/build/madrona_kernels.cache"
    fi
    
    # Quick build - only run make if build directory exists
    if [ ! -d "$PROJECT_ROOT/build" ]; then
        log_warning "Build directory doesn't exist. Running full build instead..."
        full_build "$jobs" "$run_tests"
        return
    fi
    
    # Build with make (incremental)
    log_info "Building with make (incremental, using $jobs jobs)..."
    if [ "${VERBOSE:-false}" = true ]; then
        make -C build -j"$jobs"
    else
        # Show build progress while logging full output
        make -C build -j"$jobs" 2>&1 | tee build_output.log | grep -E "(Built target|Linking.*executable|Linking.*library|Generating.*\.py)"
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            # Show errors if build failed
            echo "Build failed. Full log available in build_output.log"
            grep -E "(error|Error|ERROR|FAILED)" build_output.log || tail -20 build_output.log
            return 1
        fi
        log_info "Build log saved to build_output.log"
    fi
    
    run_tests_if_requested "$run_tests"
    
    if [ "${VERBOSE:-false}" = true ]; then
        log_success "Quick build completed successfully"
    else
        echo "Build successful"
    fi
    echo "Build log: build_output.log"
}

full_build() {
    local jobs=${1:-16}
    local run_tests=${2:-false}
    
    log_info "Full build with CMake reconfiguration and $jobs parallel jobs..."
    
    cd "$PROJECT_ROOT"
    check_prerequisites
    check_uv_prerequisite
    
    # Always delete CUDA kernel cache to ensure correct optimization level
    if [ -f "$PROJECT_ROOT/build/madrona_kernels.cache" ]; then
        log_info "Removing CUDA kernel cache to ensure correct optimization level"
        rm -f "$PROJECT_ROOT/build/madrona_kernels.cache"
    fi
    
    # Configure with CMake (creates build directory automatically)
    # Include version hashes needed for madrona toolchain and dependencies
    log_info "Configuring with CMake..."
    CMAKE_ARGS="-B build -DCMAKE_BUILD_TYPE=RelWithDebInfo -DMADRONA_DEPS_VERSION=8d57788 -DMADRONA_TOOLCHAIN_VERSION=8c0b55b"
    if [ "${VERBOSE:-false}" = true ]; then
        cmake $CMAKE_ARGS
    else
        cmake $CMAKE_ARGS 2>&1 | grep -E "(error|Error|ERROR|FATAL)" || true
    fi
    
    # Build with make
    log_info "Building with make (using $jobs jobs)..."
    if [ "${VERBOSE:-false}" = true ]; then
        make -C build -j"$jobs"
    else
        # Show build progress while logging full output
        make -C build -j"$jobs" 2>&1 | tee build_output.log | grep -E "(Built target|Linking.*executable|Linking.*library|Generating.*\.py)"
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            # Show errors if build failed
            echo "Build failed. Full log available in build_output.log"
            grep -E "(error|Error|ERROR|FAILED)" build_output.log || tail -20 build_output.log
            return 1
        fi
        log_info "Build log saved to build_output.log"
    fi
    
    # Python constants are now generated automatically during CMake build
    
    # Install Python package in development mode
    log_info "Installing Python package in development mode..."
    cd "$PROJECT_ROOT"
    if [ "${VERBOSE:-false}" = true ]; then
        uv pip install -e . || log_warning "Could not install Python package"
    else
        uv pip install -e . 2>&1 | tee -a build_output.log | grep -E "(Installing|Uninstalling|Built|error|Error|ERROR|FAILED)" || true
        if [ ${PIPESTATUS[0]} -ne 0 ]; then
            log_warning "Could not install Python package"
        fi
    fi
    
    run_tests_if_requested "$run_tests"
    
    if [ "${VERBOSE:-false}" = true ]; then
        log_success "Full build completed successfully"
    else
        echo "Build successful"
    fi
    echo "Build log: build_output.log"
}

# Parse command line arguments
COMMAND="build"
RUN_TESTS=false
JOBS=16
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        clean|build|fullbuild|rebuild|help)
            COMMAND=$1
            shift
            ;;
        --test)
            RUN_TESTS=true
            shift
            ;;
        --jobs)
            JOBS=$2
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Execute the requested command
case $COMMAND in
    clean)
        clean_build_artifacts
        ;;
    build)
        quick_build "$JOBS" "$RUN_TESTS"
        ;;
    fullbuild)
        full_build "$JOBS" "$RUN_TESTS"
        ;;
    rebuild)
        clean_build_artifacts
        full_build "$JOBS" "$RUN_TESTS"
        ;;
    help)
        print_usage
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        print_usage
        exit 1
        ;;
esac

# Only show completion message in verbose mode
if [ "${VERBOSE:-false}" = true ]; then
    log_success "Script completed successfully"
fi