#!/usr/bin/env python3
"""
Build script for CFFI bindings of Madrona Escape Room
"""

import os
import sys
from pathlib import Path
from cffi import FFI

def build_cffi():
    ffibuilder = FFI()
    
    # Get paths
    root_dir = Path(__file__).parent.parent
    include_dir = root_dir / "include"
    src_dir = root_dir / "src"
    build_dir = root_dir / "build"
    
    # Read the C header
    header_path = include_dir / "madrona_escape_room_c_api.h"
    with open(header_path, 'r') as f:
        header_content = f.read()
    
    # Remove preprocessor directives and extern "C" blocks for CFFI
    lines = []
    skip_until_endif = False
    in_header_guard = False
    
    for line in header_content.split('\n'):
        # Skip header guards
        if line.strip().startswith('#ifndef MADRONA_ESCAPE_ROOM_C_API_H'):
            in_header_guard = True
            continue
        if line.strip().startswith('#define MADRONA_ESCAPE_ROOM_C_API_H'):
            continue
        if line.strip() == '#endif // MADRONA_ESCAPE_ROOM_C_API_H':
            continue
            
        # Skip C++ specific blocks
        if line.strip().startswith('#ifdef __cplusplus'):
            skip_until_endif = True
            continue
        if skip_until_endif and line.strip().startswith('#endif'):
            skip_until_endif = False
            continue
        if skip_until_endif:
            continue
            
        # Skip extern "C" declarations
        if line.strip() in ['extern "C" {', '}']:
            continue
            
        # Skip all preprocessor directives
        if line.strip().startswith('#'):
            continue
            
        lines.append(line)
    
    cleaned_header = '\n'.join(lines)
    
    # Remove MER_EXPORT macros for CFFI
    cleaned_header = cleaned_header.replace('MER_EXPORT ', '')
    
    # Define the API
    ffibuilder.cdef(cleaned_header)
    
    # Set up the build with constants
    # Set rpath to $ORIGIN so the CFFI module can find the library in the same directory
    extra_link_args = []
    if os.name == 'posix':  # Linux/Unix
        extra_link_args = ["-Wl,-rpath,$ORIGIN"]
    
    ffibuilder.set_source(
        "_madrona_escape_room_cffi",
        """
        #include "madrona_escape_room_c_api.h"
        """,
        sources=[],  # We'll link against the already built library
        include_dirs=[str(include_dir)],
        libraries=["madrona_escape_room_c_api"],
        library_dirs=[str(build_dir)],
        extra_compile_args=[],  # No C++ flags for C compilation
        extra_link_args=extra_link_args,
    )
    
    # Add constants that need to be exposed
    # CFFI doesn't automatically expose #define constants
    constants = [
        "MER_SELF_OBSERVATION_SIZE",
        "MER_STEPS_REMAINING_SIZE", 
        "MER_AGENT_ID_SIZE",
        "MER_TOTAL_OBSERVATION_SIZE",
        "MER_NUM_AGENTS",
        "MER_NUM_ROOMS",
        "MER_MAX_ENTITIES_PER_ROOM",
        "MER_EPISODE_LENGTH",
        # Action constants
        "MER_MOVE_STOP",
        "MER_MOVE_SLOW",
        "MER_MOVE_MEDIUM",
        "MER_MOVE_FAST",
        "MER_MOVE_FORWARD",
        "MER_MOVE_FORWARD_RIGHT",
        "MER_MOVE_RIGHT",
        "MER_MOVE_BACKWARD_RIGHT",
        "MER_MOVE_BACKWARD",
        "MER_MOVE_BACKWARD_LEFT",
        "MER_MOVE_LEFT",
        "MER_MOVE_FORWARD_LEFT",
        "MER_ROTATE_FAST_LEFT",
        "MER_ROTATE_SLOW_LEFT",
        "MER_ROTATE_NONE",
        "MER_ROTATE_SLOW_RIGHT",
        "MER_ROTATE_FAST_RIGHT",
    ]
    
    # Add constants to the module
    for const in constants:
        ffibuilder.cdef(f"#define {const} ...")
    
    # Build the CFFI module
    cffi_tmp_dir = build_dir / "cffi_tmp"
    cffi_tmp_dir.mkdir(exist_ok=True)
    
    ffibuilder.compile(verbose=True, tmpdir=str(cffi_tmp_dir))
    
    print("CFFI build completed successfully!")
    return cffi_tmp_dir

if __name__ == "__main__":
    build_cffi()