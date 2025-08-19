#!/usr/bin/env python3
"""Test script to debug asset registry"""

from madrona_escape_room import ctypes_bindings

# Call the function we want to debug
count = ctypes_bindings.lib.mer_get_physics_assets_count()
print(f"Physics asset count: {count}")

if count > 0:
    name = ctypes_bindings.lib.mer_get_physics_asset_name(0)
    if name:
        print(f"First asset: {name.decode('utf-8')}")
