#!/bin/bash

# Madrona Escape Room Build Script
# Handles cleaning, building, and rebuilding the project and all submodules

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
    echo "  build     Build the project (default if no command given)"
    echo "  rebuild   Clean then build"
    echo "  help      Show this help message"
    echo ""
    echo "OPTIONS:"
    echo "  --test    Run tests after building"
    echo "  --jobs N  Use N parallel jobs for building (default: 16)"
    echo ""
    echo "Examples:"
    echo "  $0              # Build the project"
    echo "  $0 clean        # Clean all build artifacts"
    echo "  $0 rebuild      # Clean then build"
    echo "  $0 build --test # Build and run tests"
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

clean_build_artifacts() {
    log_info "Cleaning build artifacts..."
    
    # Clean main build directory
    if [ -d "$PROJECT_ROOT/build" ]; then
        log_info "Removing main build directory: $PROJECT_ROOT/build"
        rm -rf "$PROJECT_ROOT/build"
    fi
    
    # Clean madrona submodule build directory
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
    
    # Clean git untracked files in submodules (but be careful)
    cd "$PROJECT_ROOT"
    if [ -d "external/madrona/.git" ]; then
        log_info "Cleaning untracked files in madrona submodule"
        cd external/madrona
        git clean -fd 2>/dev/null || log_warning "Could not clean madrona submodule"
        cd "$PROJECT_ROOT"
    fi
    
    if [ -d "external/mcp-gdb/.git" ]; then
        log_info "Cleaning untracked files in mcp-gdb submodule"
        cd external/mcp-gdb
        git clean -fd 2>/dev/null || log_warning "Could not clean mcp-gdb submodule"
        cd "$PROJECT_ROOT"
    fi
    
    log_success "Clean completed successfully"
}

build_project() {
    local jobs=${1:-16}
    local run_tests=${2:-false}
    
    log_info "Building project with $jobs parallel jobs..."
    
    cd "$PROJECT_ROOT"
    
    # Check prerequisites
    if ! command -v cmake &> /dev/null; then
        log_error "CMake not found. Please install CMake."
        exit 1
    fi
    
    if ! command -v make &> /dev/null; then
        log_error "Make not found. Please install build-essential."
        exit 1
    fi
    
    if ! command -v uv &> /dev/null; then
        log_error "uv not found. Please install uv for Python package management."
        exit 1
    fi
    
    # Configure with CMake (creates build directory automatically)
    log_info "Configuring with CMake..."
    cmake -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
    
    # Build with make
    log_info "Building with make (using $jobs jobs)..."
    make -C build -j"$jobs" -s
    
    # Generate Python constants if the script exists
    if [ -f "$PROJECT_ROOT/codegen/generate_python_constants.py" ]; then
        log_info "Generating Python constants..."
        cd "$PROJECT_ROOT"
        uv run python codegen/generate_python_constants.py || log_warning "Could not generate Python constants"
    fi
    
    # Install Python package in development mode
    log_info "Installing Python package in development mode..."
    cd "$PROJECT_ROOT"
    uv pip install -e . || log_warning "Could not install Python package"
    
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
    
    log_success "Build completed successfully"
}

# Parse command line arguments
COMMAND="build"
RUN_TESTS=false
JOBS=16

while [[ $# -gt 0 ]]; do
    case $1 in
        clean|build|rebuild|help)
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
        build_project "$JOBS" "$RUN_TESTS"
        ;;
    rebuild)
        clean_build_artifacts
        build_project "$JOBS" "$RUN_TESTS"
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

log_success "Script completed successfully"