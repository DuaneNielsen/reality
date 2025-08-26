"""
Auto-generated Python dataclass structures from compiled binary.
Uses cdataclass for C compatibility with Pythonic interface.
DO NOT EDIT - this file is automatically regenerated.
"""

import ctypes
from dataclasses import dataclass, field
from typing import List, Tuple

from cdataclass import BigEndianCDataMixIn, NativeEndianCDataMixIn, meta


# Factory functions for pre-sized arrays
def _make_bool_array_1024():
    """Factory for 1024-element bool array"""
    return [False] * 1024


def _make_float_array_1024():
    """Factory for 1024-element float array"""
    return [0.0] * 1024


def _make_float_array_8():
    """Factory for 8-element float array"""
    return [0.0] * 8


def _make_int_array_1024():
    """Factory for 1024-element int array"""
    return [0] * 1024


def _make_int_array_7():
    """Factory for 7-element int array"""
    return [0] * 7


def _make_quat_array_1024():
    """Factory for 1024-element quaternion array (identity quaternions)"""
    return [(1.0, 0.0, 0.0, 0.0)] * 1024


# Dataclass structures
@dataclass
class CompiledLevel(NativeEndianCDataMixIn):
    num_tiles: int = field(metadata=meta(ctypes.c_int32), default=0)
    max_entities: int = field(metadata=meta(ctypes.c_int32), default=0)
    width: int = field(metadata=meta(ctypes.c_int32), default=0)
    height: int = field(metadata=meta(ctypes.c_int32), default=0)
    world_scale: float = field(metadata=meta(ctypes.c_float), default=0)
    done_on_collide: bool = field(metadata=meta(ctypes.c_bool), default=False)
    level_name: bytes = field(metadata=meta(ctypes.c_char * 64), default=b"")
    _pad_85: bytes = field(metadata=meta(ctypes.c_byte * 3), default=b"\x00" * 3)
    world_min_x: float = field(metadata=meta(ctypes.c_float), default=0)
    world_max_x: float = field(metadata=meta(ctypes.c_float), default=0)
    world_min_y: float = field(metadata=meta(ctypes.c_float), default=0)
    world_max_y: float = field(metadata=meta(ctypes.c_float), default=0)
    world_min_z: float = field(metadata=meta(ctypes.c_float), default=0)
    world_max_z: float = field(metadata=meta(ctypes.c_float), default=0)
    num_spawns: int = field(metadata=meta(ctypes.c_int32), default=0)
    spawn_x: List[float] = field(
        metadata=meta(ctypes.c_float * 8), default_factory=_make_float_array_8
    )
    spawn_y: List[float] = field(
        metadata=meta(ctypes.c_float * 8), default_factory=_make_float_array_8
    )
    spawn_facing: List[float] = field(
        metadata=meta(ctypes.c_float * 8), default_factory=_make_float_array_8
    )
    object_ids: List[int] = field(
        metadata=meta(ctypes.c_int32 * 1024), default_factory=_make_int_array_1024
    )
    tile_x: List[float] = field(
        metadata=meta(ctypes.c_float * 1024), default_factory=_make_float_array_1024
    )
    tile_y: List[float] = field(
        metadata=meta(ctypes.c_float * 1024), default_factory=_make_float_array_1024
    )
    tile_z: List[float] = field(
        metadata=meta(ctypes.c_float * 1024), default_factory=_make_float_array_1024
    )
    tile_persistent: List[bool] = field(
        metadata=meta(ctypes.c_bool * 1024), default_factory=_make_bool_array_1024
    )
    tile_render_only: List[bool] = field(
        metadata=meta(ctypes.c_bool * 1024), default_factory=_make_bool_array_1024
    )
    tile_entity_type: List[int] = field(
        metadata=meta(ctypes.c_int32 * 1024), default_factory=_make_int_array_1024
    )
    tile_response_type: List[int] = field(
        metadata=meta(ctypes.c_int32 * 1024), default_factory=_make_int_array_1024
    )
    tile_scale_x: List[float] = field(
        metadata=meta(ctypes.c_float * 1024), default_factory=_make_float_array_1024
    )
    tile_scale_y: List[float] = field(
        metadata=meta(ctypes.c_float * 1024), default_factory=_make_float_array_1024
    )
    tile_scale_z: List[float] = field(
        metadata=meta(ctypes.c_float * 1024), default_factory=_make_float_array_1024
    )
    tile_rotation: List[Tuple[float, float, float, float]] = field(
        metadata=meta((ctypes.c_float * 4) * 1024), default_factory=_make_quat_array_1024
    )
    tile_rand_x: List[float] = field(
        metadata=meta(ctypes.c_float * 1024), default_factory=_make_float_array_1024
    )
    tile_rand_y: List[float] = field(
        metadata=meta(ctypes.c_float * 1024), default_factory=_make_float_array_1024
    )
    tile_rand_z: List[float] = field(
        metadata=meta(ctypes.c_float * 1024), default_factory=_make_float_array_1024
    )
    tile_rand_rot_z: List[float] = field(
        metadata=meta(ctypes.c_float * 1024), default_factory=_make_float_array_1024
    )
    tile_rand_scale_x: List[float] = field(
        metadata=meta(ctypes.c_float * 1024), default_factory=_make_float_array_1024
    )
    tile_rand_scale_y: List[float] = field(
        metadata=meta(ctypes.c_float * 1024), default_factory=_make_float_array_1024
    )
    tile_rand_scale_z: List[float] = field(
        metadata=meta(ctypes.c_float * 1024), default_factory=_make_float_array_1024
    )


@dataclass
class Action(NativeEndianCDataMixIn):
    moveAmount: int = field(metadata=meta(ctypes.c_int32), default=0)
    moveAngle: int = field(metadata=meta(ctypes.c_int32), default=0)
    rotate: int = field(metadata=meta(ctypes.c_int32), default=0)


@dataclass
class SelfObservation(NativeEndianCDataMixIn):
    globalX: float = field(metadata=meta(ctypes.c_float), default=0)
    globalY: float = field(metadata=meta(ctypes.c_float), default=0)
    globalZ: float = field(metadata=meta(ctypes.c_float), default=0)
    maxY: float = field(metadata=meta(ctypes.c_float), default=0)
    theta: float = field(metadata=meta(ctypes.c_float), default=0)


@dataclass
class Done(NativeEndianCDataMixIn):
    v: int = field(metadata=meta(ctypes.c_int32), default=0)


@dataclass
class Reward(NativeEndianCDataMixIn):
    v: float = field(metadata=meta(ctypes.c_float), default=0)


@dataclass
class Progress(NativeEndianCDataMixIn):
    maxY: float = field(metadata=meta(ctypes.c_float), default=0)


@dataclass
class StepsRemaining(NativeEndianCDataMixIn):
    t: int = field(metadata=meta(ctypes.c_uint32), default=0)


@dataclass
class ReplayMetadata(NativeEndianCDataMixIn):
    magic: int = field(metadata=meta(ctypes.c_uint32), default=0)
    version: int = field(metadata=meta(ctypes.c_uint32), default=0)
    sim_name: bytes = field(metadata=meta(ctypes.c_char * 64), default=b"")
    level_name: bytes = field(metadata=meta(ctypes.c_char * 64), default=b"")
    num_worlds: int = field(metadata=meta(ctypes.c_uint32), default=0)
    num_agents_per_world: int = field(metadata=meta(ctypes.c_uint32), default=0)
    num_steps: int = field(metadata=meta(ctypes.c_uint32), default=0)
    actions_per_step: int = field(metadata=meta(ctypes.c_uint32), default=0)
    timestamp: int = field(metadata=meta(ctypes.c_uint64), default=0)
    seed: int = field(metadata=meta(ctypes.c_uint32), default=0)
    reserved: List[int] = field(
        metadata=meta(ctypes.c_uint32 * 7), default_factory=_make_int_array_7
    )


@dataclass
class ManagerConfig(NativeEndianCDataMixIn):
    exec_mode: int = field(metadata=meta(ctypes.c_int), default=0)
    gpu_id: int = field(metadata=meta(ctypes.c_int), default=0)
    num_worlds: int = field(metadata=meta(ctypes.c_uint32), default=0)
    rand_seed: int = field(metadata=meta(ctypes.c_uint32), default=0)
    auto_reset: bool = field(metadata=meta(ctypes.c_bool), default=False)
    enable_batch_renderer: bool = field(metadata=meta(ctypes.c_bool), default=False)
    _pad_18: bytes = field(metadata=meta(ctypes.c_byte * 2), default=b"\x00" * 2)
    batch_render_view_width: int = field(metadata=meta(ctypes.c_uint32), default=0)
    batch_render_view_height: int = field(metadata=meta(ctypes.c_uint32), default=0)


# Size validation
assert (
    CompiledLevel.size() == 84180
), f"CompiledLevel size mismatch: {CompiledLevel.size()} != 84180"

assert Action.size() == 12, f"Action size mismatch: {Action.size()} != 12"

assert (
    SelfObservation.size() == 20
), f"SelfObservation size mismatch: {SelfObservation.size()} != 20"

assert Done.size() == 4, f"Done size mismatch: {Done.size()} != 4"

assert Reward.size() == 4, f"Reward size mismatch: {Reward.size()} != 4"

assert Progress.size() == 4, f"Progress size mismatch: {Progress.size()} != 4"

assert StepsRemaining.size() == 4, f"StepsRemaining size mismatch: {StepsRemaining.size()} != 4"

assert ReplayMetadata.size() == 192, f"ReplayMetadata size mismatch: {ReplayMetadata.size()} != 192"

assert ManagerConfig.size() == 28, f"ManagerConfig size mismatch: {ManagerConfig.size()} != 28"


# Helper function to convert any dataclass to ctypes for C API
def to_ctypes(obj):
    """Convert dataclass to ctypes Structure for C API."""
    if hasattr(obj, "to_ctype"):
        return obj.to_ctype()
    return obj  # Already a ctypes structure
