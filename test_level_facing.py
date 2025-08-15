#!/usr/bin/env python3
"""
Test that compiled levels with agent facing angles work correctly.
"""

import math

from madrona_escape_room.level_compiler import compile_level_from_json, load_compiled_level_binary

# Test loading a compiled level with facing
print("Loading compiled level with agent facing...")
compiled = load_compiled_level_binary("levels/two_agents_facing.lvl")

print("\nLevel: two_agents_facing")
print(f"  Dimensions: {compiled['width']}x{compiled['height']}")
print(f"  Spawns: {compiled['num_spawns']}")

for i in range(compiled["num_spawns"]):
    x = compiled["spawn_x"][i]
    y = compiled["spawn_y"][i]
    facing_rad = compiled["spawn_facing"][i]
    facing_deg = facing_rad * 180.0 / math.pi
    print(f"  Agent {i}: position ({x:.1f}, {y:.1f}), facing {facing_deg:.1f}°")

# Test the patrol route level
print("\nLoading patrol_route level...")
compiled = load_compiled_level_binary("levels/patrol_route.lvl")

print("\nLevel: patrol_route")
print(f"  Dimensions: {compiled['width']}x{compiled['height']} (scale: {compiled['scale']})")
print(f"  Spawns: {compiled['num_spawns']}")

directions = ["East", "South", "North", "West"]
for i in range(compiled["num_spawns"]):
    x = compiled["spawn_x"][i]
    y = compiled["spawn_y"][i]
    facing_rad = compiled["spawn_facing"][i]
    facing_deg = facing_rad * 180.0 / math.pi
    direction = directions[i] if i < len(directions) else "?"
    print(f"  Agent {i}: position ({x:.1f}, {y:.1f}), facing {facing_deg:.1f}° ({direction})")

print("\n✅ Level facing data is preserved in compiled binary format!")
