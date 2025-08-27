set(SIMULATOR_SRCS
    types.hpp
    sim.hpp sim.inl sim.cpp
    level_gen.hpp level_gen.cpp
)

add_library(mad_escape_cpu_impl STATIC
    ${SIMULATOR_SRCS}
)

target_link_libraries(mad_escape_cpu_impl
    PUBLIC
        madrona_mw_core
    PRIVATE
        madrona_common
        madrona_mw_physics
        madrona_rendering_system
)

add_library(mad_escape_mgr STATIC
    mgr.hpp mgr.cpp
    asset_ids.hpp
    asset_registry.hpp asset_data.cpp
)

target_link_libraries(mad_escape_mgr 
    PUBLIC
        madrona_python_utils
    PRIVATE
        mad_escape_cpu_impl
        madrona_mw_cpu
        madrona_common
        madrona_importer
        madrona_physics_loader
        madrona_render
)

if (TARGET madrona_mw_gpu)
    madrona_build_compile_defns(
        OUT_TARGET
            mad_escape_gpu_srcs
        SOURCES_DEFN
            GPU_HIDESEEK_SRC_LIST
        FLAGS_DEFN
            GPU_HIDESEEK_COMPILE_FLAGS 
        SRCS
            ${SIMULATOR_SRCS}
    )

    target_link_libraries(mad_escape_mgr PRIVATE
        mad_escape_gpu_srcs
        madrona_mw_gpu
    )
endif ()

target_compile_definitions(mad_escape_mgr PRIVATE
    -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
)

# Nanobind bindings removed - using ctypes bindings instead
# See include/madrona_escape_room_c_api.h and src/madrona_escape_room_c_api.cpp

if (TARGET madrona_viz)
    # ViewerCore library for testable viewer logic
    add_library(viewer_core STATIC
        viewer_core.hpp viewer_core.cpp)
    
    target_link_libraries(viewer_core PUBLIC
        mad_escape_mgr
        madrona_mw_core)
    
    add_executable(viewer viewer.cpp)
    target_link_libraries(viewer PRIVATE
        viewer_core
        madrona_mw_core mad_escape_mgr madrona_viz)
    target_include_directories(viewer PRIVATE ${CMAKE_BINARY_DIR})
    add_dependencies(viewer generate_default_level)

    target_compile_definitions(viewer PRIVATE
        -DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../data/"
    )
    
    target_include_directories(viewer PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../external/optionparser
    )
endif()

add_executable(headless headless.cpp)
target_link_libraries(headless madrona_mw_core mad_escape_mgr)

target_include_directories(headless PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../external/optionparser
    ${CMAKE_BINARY_DIR}
)
add_dependencies(headless generate_default_level)

add_executable(file_inspector file_inspector.cpp)
target_link_libraries(file_inspector madrona_mw_core mad_escape_mgr)

add_executable(default_level default_level.cpp)
target_link_libraries(default_level madrona_mw_core mad_escape_mgr)

# Generate default_level.lvl by running the executable
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/default_level.lvl
    COMMAND default_level ${CMAKE_BINARY_DIR}/default_level.lvl
    DEPENDS default_level
    COMMENT "Generating default_level.lvl"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    VERBATIM
)

# Generate C header with embedded level data
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/default_level_data.h
    COMMAND xxd -i default_level.lvl > ${CMAKE_BINARY_DIR}/default_level_data.h
    DEPENDS ${CMAKE_BINARY_DIR}/default_level.lvl
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Generating default_level_data.h from binary"
    VERBATIM
)

add_custom_target(generate_default_level ALL
    DEPENDS ${CMAKE_BINARY_DIR}/default_level.lvl ${CMAKE_BINARY_DIR}/default_level_data.h
)

# Test executables moved to tests/cpp/CMakeLists.txt
# To build tests, use: cmake -DBUILD_TESTS=ON ..

# C API wrapper library
add_library(madrona_escape_room_c_api SHARED
    madrona_escape_room_c_api.cpp
    struct_export.cpp  # Dummy file to export structs for pahole
)

target_include_directories(madrona_escape_room_c_api
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link all dependencies statically
# First, link the static libraries with --whole-archive
target_link_libraries(madrona_escape_room_c_api
    PRIVATE
        -Wl,--whole-archive
        mad_escape_mgr
        mad_escape_cpu_impl
        ${CMAKE_BINARY_DIR}/external/madrona/src/core/libmadrona_mw_core.a
        ${CMAKE_BINARY_DIR}/external/madrona/src/mw/libmadrona_mw_cpu.a
        ${CMAKE_BINARY_DIR}/external/madrona/src/common/libmadrona_common.a
        ${CMAKE_BINARY_DIR}/external/madrona/src/importer/libmadrona_importer.a
        ${CMAKE_BINARY_DIR}/external/madrona/src/physics/libmadrona_physics_loader.a
        ${CMAKE_BINARY_DIR}/external/madrona/src/physics/libmadrona_physics_assets.a
        ${CMAKE_BINARY_DIR}/external/madrona/src/physics/libmadrona_mw_physics.a
        ${CMAKE_BINARY_DIR}/external/madrona/src/render/libmadrona_rendering_system.a
        ${CMAKE_BINARY_DIR}/external/madrona/src/render/libmadrona_render.a
        ${CMAKE_BINARY_DIR}/external/madrona/src/render/libmadrona_render_core.a
        ${CMAKE_BINARY_DIR}/external/madrona/src/render/libmadrona_render_asset_processor.a
        ${CMAKE_BINARY_DIR}/external/madrona/src/python/libmadrona_python_utils.a
        ${CMAKE_BINARY_DIR}/external/madrona/src/common/libmadrona_mem.a
        ${CMAKE_BINARY_DIR}/external/madrona/src/common/libmadrona_err.a
        ${CMAKE_BINARY_DIR}/external/madrona/src/common/libmadrona_bvh_builder.a
        -Wl,--no-whole-archive
        # External dependencies that might only be available as shared libraries
        # We'll try to link them statically if possible
        ${CMAKE_DL_LIBS}
        pthread
        m
)

# For the external shared libraries, we need to handle them specially
# Try to find static versions first
find_library(EMBREE_STATIC_LIB NAMES embree4_static embree4.a PATHS ${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR}/external NO_DEFAULT_PATH)
find_library(DXCOMPILER_STATIC_LIB NAMES dxcompiler_static dxcompiler.a PATHS ${CMAKE_BINARY_DIR} ${CMAKE_BINARY_DIR}/external NO_DEFAULT_PATH)

if(EMBREE_STATIC_LIB)
    target_link_libraries(madrona_escape_room_c_api PRIVATE ${EMBREE_STATIC_LIB})
else()
    # If no static version, we'll have to embed the shared library
    # Use the full path to avoid linker issues
    if(EXISTS "${CMAKE_BINARY_DIR}/libembree4.so.4")
        target_link_libraries(madrona_escape_room_c_api PRIVATE "${CMAKE_BINARY_DIR}/libembree4.so.4")
    else()
        # Try to find embree in the system
        find_library(EMBREE_LIB NAMES embree4 embree3)
        if(EMBREE_LIB)
            target_link_libraries(madrona_escape_room_c_api PRIVATE ${EMBREE_LIB})
        endif()
    endif()
endif()

if(DXCOMPILER_STATIC_LIB)
    target_link_libraries(madrona_escape_room_c_api PRIVATE ${DXCOMPILER_STATIC_LIB})
else()
    # If no static version, we'll have to embed the shared library
    target_link_libraries(madrona_escape_room_c_api PRIVATE ${CMAKE_BINARY_DIR}/libdxcompiler.so)
endif()

# Handle the shader compiler
if(EXISTS ${CMAKE_BINARY_DIR}/external/madrona/src/render/libmadrona_render_shader_compiler.a)
    target_link_libraries(madrona_escape_room_c_api PRIVATE 
        -Wl,--whole-archive
        ${CMAKE_BINARY_DIR}/external/madrona/src/render/libmadrona_render_shader_compiler.a
        -Wl,--no-whole-archive
    )
else()
    target_link_libraries(madrona_escape_room_c_api PRIVATE ${CMAKE_BINARY_DIR}/libmadrona_render_shader_compiler.so)
endif()

# Handle the std_mem library
if(EXISTS ${CMAKE_BINARY_DIR}/external/madrona/src/common/libmadrona_std_mem.a)
    target_link_libraries(madrona_escape_room_c_api PRIVATE 
        -Wl,--whole-archive
        ${CMAKE_BINARY_DIR}/external/madrona/src/common/libmadrona_std_mem.a
        -Wl,--no-whole-archive
    )
else()
    target_link_libraries(madrona_escape_room_c_api PRIVATE ${CMAKE_BINARY_DIR}/libmadrona_std_mem.so)
endif()

# Ensure we compile without RTTI and exceptions
# Add debug symbols for pahole to extract struct information
target_compile_options(madrona_escape_room_c_api PRIVATE
    -fno-rtti
    -fno-exceptions
    -fvisibility=hidden
    -g  # Add debug symbols for pahole
)

# Don't try to statically link C++ runtime when using Madrona toolchain
# The toolchain uses libc++ not libstdc++

# Define building DLL for proper export
target_compile_definitions(madrona_escape_room_c_api PRIVATE
    BUILDING_MER_DLL
)

# Find CUDA toolkit for portable CUDA library paths
find_package(CUDAToolkit QUIET)

# Set proper rpath for the shared library
# Include CUDA library directory if CUDA is found
if(CUDAToolkit_FOUND)
    set_target_properties(madrona_escape_room_c_api PROPERTIES
        INSTALL_RPATH "$ORIGIN:${CUDAToolkit_LIBRARY_DIR}"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
else()
    set_target_properties(madrona_escape_room_c_api PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# Check for uv (Python package manager) which is required for code generation
find_program(UV_EXECUTABLE uv)
if(NOT UV_EXECUTABLE)
    message(FATAL_ERROR 
        "\n"
        "uv (Python package manager) not found but is required for code generation.\n"
        "\n"
        "To install uv, run one of the following:\n"
        "  - Linux/macOS: curl -LsSf https://astral.sh/uv/install.sh | sh\n"
        "  - Windows: powershell -c \"irm https://astral.sh/uv/install.ps1 | iex\"\n"
        "  - Using pip: pip install uv\n"
        "\n"
        "For more installation options, see: https://docs.astral.sh/uv/getting-started/installation/\n"
        "\n"
        "After installation, you may need to:\n"
        "  1. Restart your terminal or run: source ~/.bashrc (or ~/.zshrc)\n"
        "  2. Re-run CMake configuration\n"
    )
endif()
message(STATUS "Found uv: ${UV_EXECUTABLE}")

# Set up isolated Python environment for build
set(BUILD_VENV_DIR ${CMAKE_BINARY_DIR}/.build_venv)
set(BUILD_PYTHON ${BUILD_VENV_DIR}/bin/python3)

# Create isolated venv for build if it doesn't exist
if(NOT EXISTS ${BUILD_VENV_DIR})
    message(STATUS "Creating isolated Python environment for build at ${BUILD_VENV_DIR}")
    execute_process(
        COMMAND ${UV_EXECUTABLE} venv ${BUILD_VENV_DIR}
        RESULT_VARIABLE VENV_RESULT
        OUTPUT_VARIABLE VENV_OUTPUT
        ERROR_VARIABLE VENV_ERROR
    )
    if(NOT VENV_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to create build venv: ${VENV_ERROR}")
    endif()
    
    message(STATUS "Installing codegen dependencies in build environment")
    # Install dependencies from codegen/pyproject.toml
    execute_process(
        COMMAND ${UV_EXECUTABLE} pip install --python ${BUILD_PYTHON} libclang>=14.0.0
        RESULT_VARIABLE INSTALL_RESULT
        OUTPUT_VARIABLE INSTALL_OUTPUT
        ERROR_VARIABLE INSTALL_ERROR
    )
    if(NOT INSTALL_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to install build dependencies: ${INSTALL_ERROR}")
    endif()
else()
    message(STATUS "Using existing build venv at ${BUILD_VENV_DIR}")
endif()

# Generate Python dataclass and constants from the compiled library using pahole
# This ensures Python structures exactly match C++ memory layout
# Create a custom target that always runs
add_custom_target(generate_python_bindings ALL
    COMMAND ${CMAKE_COMMAND} -E echo "Generating Python dataclass definitions from binary..."
    COMMAND ${BUILD_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/../codegen/generate_dataclass_structs.py
            $<TARGET_FILE:madrona_escape_room_c_api>
            ${CMAKE_CURRENT_SOURCE_DIR}/../madrona_escape_room/generated_dataclasses.py
    COMMAND ${CMAKE_COMMAND} -E echo "Generating Python constants from headers..."
    COMMAND ${BUILD_PYTHON} ${CMAKE_CURRENT_SOURCE_DIR}/../codegen/generate_python_constants.py
            ${CMAKE_CURRENT_SOURCE_DIR}/consts.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/types.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../madrona_escape_room/generated_constants.py
    DEPENDS madrona_escape_room_c_api
    COMMENT "Auto-generating Python bindings using pahole and libclang (using isolated build venv)"
    VERBATIM
)

# Python DLPack extension
add_library(madrona_escape_room_dlpack MODULE
    dlpack_extension.cpp
)

# Set proper Python module naming
set_target_properties(madrona_escape_room_dlpack PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_madrona_escape_room_dlpack"
    SUFFIX ".cpython-312-x86_64-linux-gnu.so"
)

target_include_directories(madrona_escape_room_dlpack PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
    ${CMAKE_CURRENT_SOURCE_DIR}
    /usr/include/python3.12
)

target_compile_definitions(madrona_escape_room_dlpack PRIVATE
    MADRONA_CLANG
)

target_link_libraries(madrona_escape_room_dlpack PRIVATE
    madrona_common
    madrona_python_utils
)

# Copy all required libraries to package directory
add_custom_command(TARGET madrona_escape_room_dlpack POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:madrona_escape_room_dlpack>
        ${CMAKE_CURRENT_SOURCE_DIR}/../madrona_escape_room/
    COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:madrona_escape_room_c_api>
        ${CMAKE_CURRENT_SOURCE_DIR}/../madrona_escape_room/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/libembree4.so.4
        ${CMAKE_CURRENT_SOURCE_DIR}/../madrona_escape_room/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/libdxcompiler.so
        ${CMAKE_CURRENT_SOURCE_DIR}/../madrona_escape_room/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/libmadrona_render_shader_compiler.so
        ${CMAKE_CURRENT_SOURCE_DIR}/../madrona_escape_room/
    COMMAND ${CMAKE_COMMAND} -E copy
        ${CMAKE_BINARY_DIR}/libmadrona_std_mem.so
        ${CMAKE_CURRENT_SOURCE_DIR}/../madrona_escape_room/
    COMMENT "Copying all required libraries to Python package directory"
)

