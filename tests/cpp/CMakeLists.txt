cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

# Enable testing
enable_testing()

# Note: GoogleTest is already built and configured by Madrona's build system
# in external/madrona/external/CMakeLists.txt with proper toolchain linking.
# We just need to use the gtest_main target that Madrona provides.

# Collect CPU test source files (tests that don't require GPU)
set(CPU_TEST_SOURCES
    fixtures/test_base.cpp
    fixtures/debug_capture.cpp
    unit/test_c_api_cpu.cpp
    unit/test_option_parser.cpp
    unit/test_level_utilities.cpp
    unit/test_recording_utilities.cpp
    unit/test_file_inspector.cpp
    unit/test_viewer_core.cpp
    unit/test_viewercore_trajectory.cpp
    unit/test_direct_cpp.cpp
    unit/test_asset_refactor.cpp
    unit/test_persistence.cpp
    integration/test_viewer_integration.cpp
    integration/test_viewer_input.cpp
    integration/test_viewer_errors.cpp  # Has mixed tests but GPU test will skip gracefully
    e2e/test_viewer_workflows.cpp
    test_asset_registry.cpp
)

# Collect GPU test source files (tests that require GPU)
set(GPU_TEST_SOURCES
    fixtures/test_base.cpp
    unit/test_manager_gpu.cpp
)

# Collect GPU stress test source files (expensive tests with long compilation times)
set(GPU_STRESS_TEST_SOURCES
    fixtures/test_base.cpp
    unit/test_manager_gpu_stress.cpp
)

# Create CPU test executable
add_executable(mad_escape_tests
    ${CPU_TEST_SOURCES}
)

# Create GPU test executable
add_executable(mad_escape_gpu_tests
    ${GPU_TEST_SOURCES}
)

# Create GPU stress test executable
add_executable(mad_escape_gpu_stress_tests
    ${GPU_STRESS_TEST_SOURCES}
)

# Include directories for CPU test fixtures
target_include_directories(mad_escape_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/optionparser
    ${CMAKE_BINARY_DIR}  # For default_level_data.h
)

# Include directories for GPU test fixtures
target_include_directories(mad_escape_gpu_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/optionparser
    ${CMAKE_BINARY_DIR}  # For default_level_data.h
)

# Include directories for GPU stress test fixtures
target_include_directories(mad_escape_gpu_stress_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/fixtures
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/external/optionparser
    ${CMAKE_BINARY_DIR}  # For default_level_data.h
)

# Link libraries for CPU tests
target_link_libraries(mad_escape_tests
    PRIVATE
        madrona_escape_room_c_api
        mad_escape_mgr
        viewer_core
        gtest_main
        gtest
)

# Link libraries for GPU tests
target_link_libraries(mad_escape_gpu_tests
    PRIVATE
        madrona_escape_room_c_api
        mad_escape_mgr
        viewer_core
        gtest_main
        gtest
)

# Link libraries for GPU stress tests
target_link_libraries(mad_escape_gpu_stress_tests
    PRIVATE
        madrona_escape_room_c_api
        mad_escape_mgr
        viewer_core
        gtest_main
        gtest
)

# Match project compile options (no RTTI, no exceptions) for CPU tests
target_compile_options(mad_escape_tests PRIVATE
    -fno-rtti
    -fno-exceptions
)

# Match project compile options (no RTTI, no exceptions) for GPU tests
target_compile_options(mad_escape_gpu_tests PRIVATE
    -fno-rtti
    -fno-exceptions
)

# Match project compile options (no RTTI, no exceptions) for GPU stress tests
target_compile_options(mad_escape_gpu_stress_tests PRIVATE
    -fno-rtti
    -fno-exceptions
)

# Set C++ standard (need C++20 for Madrona headers) for CPU tests
set_target_properties(mad_escape_tests PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Set C++ standard (need C++20 for Madrona headers) for GPU tests
set_target_properties(mad_escape_gpu_tests PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
)

# Set C++ standard (need C++20 for Madrona headers) for GPU stress tests
set_target_properties(mad_escape_gpu_stress_tests PROPERTIES
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

set_target_properties(mad_escape_gpu_tests PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
    INSTALL_RPATH "$ORIGIN:${LIBCXX_PATH}:${CMAKE_BINARY_DIR}"
)

set_target_properties(mad_escape_gpu_stress_tests PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
    INSTALL_RPATH "$ORIGIN:${LIBCXX_PATH}:${CMAKE_BINARY_DIR}"
)

# Register CPU tests with CTest
add_test(NAME mad_escape_tests 
         COMMAND mad_escape_tests
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Register GPU tests with CTest
add_test(NAME mad_escape_gpu_tests 
         COMMAND mad_escape_gpu_tests
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Register GPU stress tests with CTest
add_test(NAME mad_escape_gpu_stress_tests 
         COMMAND mad_escape_gpu_stress_tests
         WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Add custom target for running CPU tests only
add_custom_target(run_cpp_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --label-regex cpp --output-on-failure
    DEPENDS mad_escape_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running CPU tests..."
)

# Add custom target for running GPU tests only (fast tests)
add_custom_target(run_gpu_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -R mad_escape_gpu_tests --output-on-failure
    DEPENDS mad_escape_gpu_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running GPU tests (fast)..."
)

# Add custom target for running GPU stress tests only (slow tests)
add_custom_target(run_gpu_stress_tests
    COMMAND ${CMAKE_CTEST_COMMAND} -R mad_escape_gpu_stress_tests --output-on-failure
    DEPENDS mad_escape_gpu_stress_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running GPU stress tests (slow)..."
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