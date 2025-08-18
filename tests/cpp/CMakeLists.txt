cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Enable testing
enable_testing()

# Note: GoogleTest is already built and configured by Madrona's build system
# in external/madrona/external/CMakeLists.txt with proper toolchain linking.
# We just need to use the gtest_main target that Madrona provides.

# Collect test source files
set(TEST_SOURCES
    fixtures/test_base.cpp
    unit/test_c_api_cpu.cpp
    unit/test_c_api_gpu.cpp
    unit/test_option_parser.cpp
    unit/test_level_utilities.cpp
    unit/test_recording_utilities.cpp
    unit/test_file_inspector.cpp
    unit/test_viewer_core.cpp
    unit/test_viewercore_trajectory.cpp
    unit/test_direct_cpp.cpp
    integration/test_viewer_integration.cpp
    integration/test_viewer_input.cpp
    integration/test_viewer_errors.cpp
    e2e/test_viewer_workflows.cpp
    test_asset_registry.cpp
)

# Create test executable
add_executable(mad_escape_tests
    ${TEST_SOURCES}
)

# Include directories for test fixtures
target_include_directories(mad_escape_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/optionparser
)

# Link libraries
target_link_libraries(mad_escape_tests
    PRIVATE
        madrona_escape_room_c_api
        mad_escape_mgr
        viewer_core
        gtest_main
        gtest
)

# Match project compile options (no RTTI, no exceptions)
target_compile_options(mad_escape_tests PRIVATE
    -fno-rtti
    -fno-exceptions
)

# Set C++ standard (need C++20 for Madrona headers)
set_target_properties(mad_escape_tests PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Set RPATH to find libc++ from the toolchain
# Find the actual libc++ location based on the toolchain being used
if(EXISTS "${CMAKE_SOURCE_DIR}/external/madrona/external/madrona-toolchain/bundled-toolchain/toolchain/lib/x86_64-pc-linux-gnu")
    set(LIBCXX_PATH "${CMAKE_SOURCE_DIR}/external/madrona/external/madrona-toolchain/bundled-toolchain/toolchain/lib/x86_64-pc-linux-gnu")
else()
    set(LIBCXX_PATH "${CMAKE_BINARY_DIR}")
endif()

set_target_properties(mad_escape_tests PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
    INSTALL_RPATH "$ORIGIN:${LIBCXX_PATH}:${CMAKE_BINARY_DIR}"
)

# Register tests with CTest but don't auto-discover (requires running the executable)
# Tests will be discovered at runtime instead
add_test(NAME mad_escape_tests 
         COMMAND mad_escape_tests
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Add custom target for running C++ tests only
add_custom_target(run_cpp_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --label-regex cpp --output-on-failure
    DEPENDS mad_escape_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running C++ tests..."
)

# Add custom target for running CPU tests only
add_custom_target(run_cpp_cpu_tests
    COMMAND mad_escape_tests --gtest_filter="*CPU*" --gtest_color=yes
    DEPENDS mad_escape_tests
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running C++ CPU tests..."
)

# Add custom target for running GPU tests only
add_custom_target(run_cpp_gpu_tests
    COMMAND mad_escape_tests --gtest_filter="*GPU*" --gtest_color=yes
    DEPENDS mad_escape_tests
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running C++ GPU tests..."
)