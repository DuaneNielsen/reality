cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
cmake_policy(VERSION 3.18)
include("${CMAKE_CURRENT_SOURCE_DIR}/external/madrona/cmake/madrona_init.cmake")

project(Madrona3DExample LANGUAGES C CXX)

include(setup)
include(dependencies)

# Option to build C++ tests
option(BUILD_TESTS "Build C++ unit tests" OFF)
if(BUILD_TESTS)
    # Enable Madrona's tests to get GoogleTest built with the correct toolchain
    set(MADRONA_ENABLE_TESTS ON CACHE BOOL "" FORCE)
endif()

add_subdirectory(external)
add_subdirectory(src)

if(BUILD_TESTS)
    add_subdirectory(tests/cpp)
endif()
