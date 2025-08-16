#!/bin/bash

# Script to build and run C++ unit tests

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse command line arguments
TEST_FILTER=""
BUILD_ONLY=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --cpu-only)
            TEST_FILTER="--gtest_filter=*CPU*"
            shift
            ;;
        --gpu-only)
            TEST_FILTER="--gtest_filter=*GPU*"
            shift
            ;;
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --cpu-only    Run only CPU tests"
            echo "  --gpu-only    Run only GPU tests"
            echo "  --build-only  Build tests but don't run them"
            echo "  --verbose,-v  Show verbose output"
            echo "  --help,-h     Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Get the repository root directory
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"

echo -e "${GREEN}=== C++ Unit Test Runner ===${NC}"
echo "Repository root: ${REPO_ROOT}"
echo "Build directory: ${BUILD_DIR}"

# Check if build directory exists
if [ ! -d "${BUILD_DIR}" ]; then
    echo -e "${YELLOW}Build directory doesn't exist. Creating it...${NC}"
    mkdir -p "${BUILD_DIR}"
fi

# Configure with tests enabled
echo -e "\n${GREEN}Configuring with tests enabled...${NC}"
cd "${BUILD_DIR}"

if [ "$VERBOSE" = true ]; then
    cmake -DBUILD_TESTS=ON ..
else
    cmake -DBUILD_TESTS=ON .. > /dev/null 2>&1
fi

# Build the tests
echo -e "${GREEN}Building C++ tests...${NC}"
if [ "$VERBOSE" = true ]; then
    make mad_escape_tests -j8
else
    make mad_escape_tests -j8 -s
fi

if [ "$BUILD_ONLY" = true ]; then
    echo -e "${GREEN}Tests built successfully!${NC}"
    exit 0
fi

# Run the tests
echo -e "\n${GREEN}Running C++ tests...${NC}"
cd "${REPO_ROOT}"

# Set test executable path
TEST_EXEC="${BUILD_DIR}/mad_escape_tests"

if [ ! -f "${TEST_EXEC}" ]; then
    echo -e "${RED}Error: Test executable not found at ${TEST_EXEC}${NC}"
    exit 1
fi

# Run tests with appropriate filter
if [ -n "$TEST_FILTER" ]; then
    echo "Filter: ${TEST_FILTER}"
fi

# Run the tests from repo root so level files can be found
cd "${REPO_ROOT}"

# Run the tests and capture the exit code
set +e  # Temporarily disable exit on error
"${TEST_EXEC}" ${TEST_FILTER} --gtest_color=yes
TEST_RESULT=$?
set -e

# Report results
echo ""
if [ $TEST_RESULT -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
else
    echo -e "${RED}✗ Some tests failed!${NC}"
    exit $TEST_RESULT
fi