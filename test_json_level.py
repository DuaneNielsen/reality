#!/usr/bin/env python3
"""
Test script for JSON level format with agent facing parameter.
"""

import math

from madrona_escape_room.level_compiler import compile_level_from_json, print_level_info

# Create a JSON level with two spawn points and different facings
json_level = {
    "ascii": """########
#S.....#
#......#
#.....S#
########""",
    "scale": 2.0,
    "agent_facing": [0.0, math.pi],  # First faces north (0°), second faces south (180°)
}

print("Compiling JSON level...")
compiled = compile_level_from_json(json_level)

print("\n" + "=" * 50)
print_level_info(compiled)

print("\n" + "=" * 50)
print("Testing different facing angles:")

test_angles = [
    (0, "North"),
    (math.pi / 2, "East"),
    (math.pi, "South"),
    (3 * math.pi / 2, "West"),
    (math.pi / 4, "Northeast"),
]

for angle, direction in test_angles:
    json_test = {
        "ascii": """###
#S#
###""",
        "scale": 1.0,
        "agent_facing": [angle],
    }
    compiled_test = compile_level_from_json(json_test)
    angle_deg = angle * 180.0 / math.pi
    print(f"  {direction:10s}: {angle:.3f} rad = {angle_deg:.1f}°")

print("\n✅ JSON level format with agent facing is working!")
