# ruff: noqa
#!/usr/bin/env python3
"""Test SimManager with structs generated from pahole."""

import ctypes
import os
import sys

import numpy as np

# First, run the struct generator to get our types
exec(open("scratch/generate_structs_from_binary.py").read())

# Load the C API library
lib_path = os.path.join(os.getcwd(), "build/libmadrona_escape_room_c_api.so")
if not os.path.exists(lib_path):
    print(f"Library not found at {lib_path}")
    sys.exit(1)

lib = ctypes.CDLL(lib_path)


# Define the C API structures and functions
class Handle(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_void_p)]


class Config(ctypes.Structure):
    _fields_ = [
        ("execMode", ctypes.c_int),  # enum ExecMode at offset 0
        ("gpuID", ctypes.c_int),  # at offset 4
        ("numWorlds", ctypes.c_uint32),  # at offset 8
        ("randSeed", ctypes.c_uint32),  # at offset 12
        ("autoReset", ctypes.c_bool),  # at offset 16
        ("enableBatchRenderer", ctypes.c_bool),  # at offset 17
        ("_pad1", ctypes.c_byte * 2),  # 2-byte padding
        ("batchRenderViewWidth", ctypes.c_uint32),  # at offset 20
        ("batchRenderViewHeight", ctypes.c_uint32),  # at offset 24
        ("_pad2", ctypes.c_byte * 4),  # 4-byte padding
        ("extRenderAPI", ctypes.c_void_p),  # at offset 32
        ("extRenderDev", ctypes.c_void_p),  # at offset 40
        ("enableTrajectoryTracking", ctypes.c_bool),  # at offset 48
        ("_pad3", ctypes.c_byte * 7),  # 7-byte padding
    ]


# Define function signatures
lib.mer_create_manager.argtypes = [
    ctypes.POINTER(Handle),
    ctypes.POINTER(Config),
    ctypes.POINTER(CompiledLevel),
    ctypes.c_uint32,
]
lib.mer_create_manager.restype = ctypes.c_int

lib.mer_step.argtypes = [ctypes.POINTER(Handle)]
lib.mer_step.restype = None

lib.mer_destroy_manager.argtypes = [ctypes.POINTER(Handle)]
lib.mer_destroy_manager.restype = None

lib.mer_validate_compiled_level.argtypes = [ctypes.POINTER(CompiledLevel)]
lib.mer_validate_compiled_level.restype = ctypes.c_int

lib.mer_result_to_string.argtypes = [ctypes.c_int]
lib.mer_result_to_string.restype = ctypes.c_char_p

# Get tensor functions
lib.mer_get_self_observation_tensor.argtypes = [ctypes.POINTER(Handle)]
lib.mer_get_self_observation_tensor.restype = ctypes.c_void_p

lib.mer_get_action_tensor.argtypes = [ctypes.POINTER(Handle)]
lib.mer_get_action_tensor.restype = ctypes.c_void_p

lib.mer_get_reward_tensor.argtypes = [ctypes.POINTER(Handle)]
lib.mer_get_reward_tensor.restype = ctypes.c_void_p

lib.mer_get_done_tensor.argtypes = [ctypes.POINTER(Handle)]
lib.mer_get_done_tensor.restype = ctypes.c_void_p


# Create a simple test level
def create_test_level():
    """Create a minimal test level."""
    level = CompiledLevel()

    # Basic metadata
    level.num_tiles = 4
    level.max_entities = 1
    level.width = 10
    level.height = 10
    level.world_scale = 1.0
    level.done_on_collide = False

    # Set level name (char array)
    level_name = b"TestLevel"
    # Copy bytes to char array
    for i in range(min(len(level_name), 64)):
        level.level_name[i] = level_name[i]
    # Null terminate if there's room
    if len(level_name) < 64:
        level.level_name[len(level_name)] = 0

    # World bounds
    level.world_min_x = -10.0
    level.world_max_x = 10.0
    level.world_min_y = -10.0
    level.world_max_y = 10.0
    level.world_min_z = 0.0
    level.world_max_z = 5.0

    # Single spawn point
    level.num_spawns = 1
    level.spawn_x[0] = 0.0
    level.spawn_y[0] = 0.0
    level.spawn_facing[0] = 0.0

    # Create 4 wall tiles (a small box)
    positions = [
        (-2.0, 0.0, 1.0),  # Left wall
        (2.0, 0.0, 1.0),  # Right wall
        (0.0, -2.0, 1.0),  # Back wall
        (0.0, 2.0, 1.0),  # Front wall
    ]

    for i, (x, y, z) in enumerate(positions):
        level.object_ids[i] = 2  # WALL asset ID
        level.tile_x[i] = x
        level.tile_y[i] = y
        level.tile_z[i] = z
        level.tile_persistent[i] = True
        level.tile_render_only[i] = False
        level.tile_entity_type[i] = 2  # EntityType::Wall
        level.tile_response_type[i] = 2  # ResponseType::Static
        level.tile_scale_x[i] = 1.0
        level.tile_scale_y[i] = 1.0
        level.tile_scale_z[i] = 1.0
        # Rotation quaternion (identity = no rotation)
        level.tile_rot_w[i] = 1.0
        level.tile_rot_x[i] = 0.0
        level.tile_rot_y[i] = 0.0
        level.tile_rot_z[i] = 0.0
        # Randomization (all zero = no randomization)
        level.tile_rand_x[i] = 0.0
        level.tile_rand_y[i] = 0.0
        level.tile_rand_z[i] = 0.0
        level.tile_rand_rot_z[i] = 0.0
        level.tile_rand_scale_x[i] = 0.0
        level.tile_rand_scale_y[i] = 0.0
        level.tile_rand_scale_z[i] = 0.0

    return level


def main():
    print("Creating test level with pahole-generated structs...")
    level = create_test_level()

    print(f"Level struct size: {ctypes.sizeof(level)} bytes")
    print(f"Level name: {bytes(level.level_name).split(b'\\0')[0].decode()}")
    print(f"Num tiles: {level.num_tiles}")
    print(f"World bounds: X[{level.world_min_x}, {level.world_max_x}]")

    # Validate the level first
    print("\nValidating level...")
    validation_result = lib.mer_validate_compiled_level(ctypes.byref(level))
    if validation_result != 0:
        error_msg = lib.mer_result_to_string(validation_result)
        print(f"Level validation failed: {error_msg.decode() if error_msg else 'Unknown error'}")
        return
    print("Level validation passed!")

    # Create manager configuration
    config = Config()
    config.execMode = 1  # 0 = CUDA, 1 = CPU (based on enum ExecMode)
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

    # Create the manager
    handle = Handle()
    print("\nCreating SimManager...")

    result = lib.mer_create_manager(
        ctypes.byref(handle),
        ctypes.byref(config),
        ctypes.byref(level),
        1,  # num_levels = 1
    )

    if result != 0:
        print(f"Failed to create manager! Error code: {result}")
        print("\nDebug info:")
        print(f"  Handle ptr: {handle.ptr}")
        print(f"  Config: numWorlds={config.numWorlds}, execMode={config.execMode}")
        print(f"  Level size: {ctypes.sizeof(level)}")
        print(
            f"  First tile: obj_id={level.object_ids[0]}, pos=({level.tile_x[0]}, {level.tile_y[0]}, {level.tile_z[0]})"
        )
        return

    print("SimManager created successfully!")

    # Get observation tensor to verify it's working
    obs_ptr = lib.mer_get_self_observation_tensor(ctypes.byref(handle))
    if obs_ptr:
        print("Got observation tensor pointer:", hex(obs_ptr))

    # Run a few steps
    print("\nRunning 5 simulation steps...")
    for i in range(5):
        lib.mer_step(ctypes.byref(handle))
        print(f"  Step {i + 1} completed")

    # Clean up
    print("\nDestroying manager...")
    lib.mer_destroy_manager(ctypes.byref(handle))
    print("Done!")


if __name__ == "__main__":
    main()
