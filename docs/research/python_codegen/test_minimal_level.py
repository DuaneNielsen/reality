# ruff: noqa
#!/usr/bin/env python3
"""Minimal test with just a few walls."""

import ctypes
import os
import sys

# Generate structs
script_dir = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(script_dir, "generate_structs_from_binary.py")).read())

print("Starting minimal level test...")

# Create minimal level
level = CompiledLevel()
level.width = 4
level.height = 4
level.world_scale = 1.0
level.done_on_collide = False
level.max_entities = 10
level.num_tiles = 4

# Set name
name = b"minimal_test"
for i in range(len(name)):
    level.level_name[i] = name[i]

# World bounds
level.world_min_x = -10.0
level.world_max_x = 10.0
level.world_min_y = -10.0
level.world_max_y = 10.0
level.world_min_z = 0.0
level.world_max_z = 10.0

# Spawn
level.num_spawns = 1
level.spawn_x[0] = 0.0
level.spawn_y[0] = 0.0
level.spawn_facing[0] = 0.0

# Initialize all tiles to defaults
for i in range(1024):
    level.tile_scale_x[i] = 1.0
    level.tile_scale_y[i] = 1.0
    level.tile_scale_z[i] = 1.0
    level.tile_rot_w[i] = 1.0
    level.tile_rot_x[i] = 0.0
    level.tile_rot_y[i] = 0.0
    level.tile_rot_z[i] = 0.0
    level.tile_response_type[i] = 2
    level.tile_rand_x[i] = 0.0
    level.tile_rand_y[i] = 0.0
    level.tile_rand_z[i] = 0.0
    level.tile_rand_rot_z[i] = 0.0
    level.tile_rand_scale_x[i] = 0.0
    level.tile_rand_scale_y[i] = 0.0
    level.tile_rand_scale_z[i] = 0.0

# Just 4 walls
for i in range(4):
    level.object_ids[i] = 2  # WALL
    level.tile_x[i] = float(i * 2 - 3)  # -3, -1, 1, 3
    level.tile_y[i] = 5.0
    level.tile_z[i] = 0.0
    level.tile_persistent[i] = True
    level.tile_render_only[i] = False
    level.tile_entity_type[i] = 2  # Wall
    level.tile_response_type[i] = 2  # Static

print(f"Level created: {level.num_tiles} tiles")

# Load library
lib_path = os.path.join(os.path.dirname(script_dir), "build", "libmadrona_escape_room_c_api.so")
lib = ctypes.CDLL(lib_path)


# Minimal Config
class Config(ctypes.Structure):
    _fields_ = [
        ("execMode", ctypes.c_int),
        ("gpuID", ctypes.c_int),
        ("numWorlds", ctypes.c_uint32),
        ("randSeed", ctypes.c_uint32),
        ("autoReset", ctypes.c_bool),
        ("enableBatchRenderer", ctypes.c_bool),
        ("_pad1", ctypes.c_byte * 2),
        ("batchRenderViewWidth", ctypes.c_uint32),
        ("batchRenderViewHeight", ctypes.c_uint32),
        ("_pad2", ctypes.c_byte * 4),
        ("extRenderAPI", ctypes.c_void_p),
        ("extRenderDev", ctypes.c_void_p),
        ("enableTrajectoryTracking", ctypes.c_bool),
        ("_pad3", ctypes.c_byte * 7),
    ]


class Handle(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_void_p)]


# Validate
lib.mer_validate_compiled_level.argtypes = [ctypes.POINTER(CompiledLevel)]
lib.mer_validate_compiled_level.restype = ctypes.c_int

print("Validating...")
result = lib.mer_validate_compiled_level(ctypes.byref(level))
print(f"Validation result: {result}")

if result == 0:
    # Try to create manager
    lib.mer_create_manager.argtypes = [
        ctypes.POINTER(Handle),
        ctypes.POINTER(Config),
        ctypes.POINTER(CompiledLevel),
        ctypes.c_uint32,
    ]
    lib.mer_create_manager.restype = ctypes.c_int

    config = Config()
    config.execMode = 0  # CPU
    config.gpuID = 0
    config.numWorlds = 1
    config.randSeed = 42
    config.autoReset = True
    config.enableBatchRenderer = False
    config.batchRenderViewWidth = 64
    config.batchRenderViewHeight = 64
    config.extRenderAPI = None
    config.extRenderDev = None
    config.enableTrajectoryTracking = False

    handle = Handle()
    print("Creating manager...")
    result = lib.mer_create_manager(
        ctypes.byref(handle), ctypes.byref(config), ctypes.byref(level), 1
    )
    print(f"Create result: {result}")

    if result == 0 and handle.ptr:
        print("SUCCESS! Manager created with handle:", hex(handle.ptr))

        # Run a step
        lib.mer_step.argtypes = [ctypes.c_void_p]
        lib.mer_step.restype = ctypes.c_int  # Returns an error code!

        print("Running step...")
        print(f"Handle value: {handle.ptr}")
        step_result = lib.mer_step(handle.ptr)
        print(f"Step result: {step_result}")
        if step_result == 0:
            print("Step completed successfully!")

        # Cleanup
        lib.mer_destroy_manager.argtypes = [ctypes.c_void_p]
        lib.mer_destroy_manager.restype = None
        lib.mer_destroy_manager(handle.ptr)
        print("Manager destroyed")
