#!/usr/bin/env python3
"""
Asset Inspector for Madrona Escape Room Level Files

This script inspects level files and provides detailed information about
specific asset types, including positions, scales, and spatial analysis.
"""

import argparse
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from madrona_escape_room.ctypes_bindings import (
    get_physics_asset_object_id,
    get_physics_assets_list,
    get_render_asset_object_id,
    get_render_assets_list,
)
from madrona_escape_room.level_compiler import load_compiled_level_binary


def get_asset_id(asset_name):
    """Get the object ID for an asset by name."""
    # Try physics assets first
    obj_id = get_physics_asset_object_id(asset_name)
    if obj_id >= 0:
        return obj_id, "physics"

    # Try render assets
    obj_id = get_render_asset_object_id(asset_name)
    if obj_id >= 0:
        return obj_id, "render-only"

    return -1, None


def analyze_asset(level_path, asset_name, verbose=False):
    """Analyze a specific asset type in a level file."""

    # Get asset ID
    asset_id, asset_type = get_asset_id(asset_name)
    if asset_id < 0:
        print(f"Error: Asset '{asset_name}' not found in asset registry")
        print("\nAvailable physics assets:")
        for name in get_physics_assets_list():
            print(f"  - {name}")
        print("\nAvailable render-only assets:")
        for name in get_render_assets_list():
            print(f"  - {name}")
        return 1

    print(f"Asset: {asset_name}")
    print(f"  Object ID: {asset_id}")
    print(f"  Type: {asset_type}")
    print()

    # Load the level file
    if not os.path.exists(level_path):
        print(f"Error: Level file not found: {level_path}")
        return 1

    level = load_compiled_level_binary(level_path)

    print(f"Level: {level['level_name']}")
    print(f"  Dimensions: {level['width']}x{level['height']}")
    print(f"  Scale: {level['scale']}")
    print(f"  Total tiles: {level['num_tiles']}")
    print()

    # Find all instances of this asset
    instances = []
    for i in range(level["num_tiles"]):
        if level["object_ids"][i] == asset_id:
            instances.append(
                {
                    "index": i,
                    "x": level["tile_x"][i],
                    "y": level["tile_y"][i],
                    "z": level["tile_z"][i],
                    "scale_x": level["tile_scale_x"][i],
                    "scale_y": level["tile_scale_y"][i],
                    "scale_z": level["tile_scale_z"][i],
                    "persistent": level["tile_persistent"][i],
                    "render_only": level["tile_render_only"][i],
                    "entity_type": level["tile_entity_type"][i],
                    "response_type": level["tile_response_type"][i],
                }
            )

    print(f"Found {len(instances)} {asset_name} instances")

    if len(instances) == 0:
        print(f"No {asset_name} objects found in this level")
        return 0

    # Sort by position for analysis
    instances.sort(key=lambda obj: (obj["y"], obj["x"]))

    # Calculate boundaries
    min_x = min(obj["x"] for obj in instances)
    max_x = max(obj["x"] for obj in instances)
    min_y = min(obj["y"] for obj in instances)
    max_y = max(obj["y"] for obj in instances)
    min_z = min(obj["z"] for obj in instances)
    max_z = max(obj["z"] for obj in instances)

    print("\nSpatial boundaries:")
    print(f"  X: {min_x:.2f} to {max_x:.2f} (range: {max_x - min_x:.2f})")
    print(f"  Y: {min_y:.2f} to {max_y:.2f} (range: {max_y - min_y:.2f})")
    print(f"  Z: {min_z:.2f} to {max_z:.2f} (range: {max_z - min_z:.2f})")

    # Analyze scale variations
    scales_x = set(obj["scale_x"] for obj in instances)
    scales_y = set(obj["scale_y"] for obj in instances)
    scales_z = set(obj["scale_z"] for obj in instances)

    print("\nScale variations:")
    print(f"  Scale X: {sorted(scales_x)}")
    print(f"  Scale Y: {sorted(scales_y)}")
    print(f"  Scale Z: {sorted(scales_z)}")

    # Property analysis
    persistent_count = sum(1 for obj in instances if obj["persistent"])
    render_only_count = sum(1 for obj in instances if obj["render_only"])

    print("\nProperties:")
    print(f"  Persistent: {persistent_count}/{len(instances)}")
    print(f"  Render-only: {render_only_count}/{len(instances)}")

    # Entity types
    entity_types = {}
    for obj in instances:
        et = obj["entity_type"]
        entity_types[et] = entity_types.get(et, 0) + 1

    entity_type_names = {
        0: "None",
        1: "Object",
        2: "Wall",
        3: "Agent",
        4: "Floor",
    }

    print("\nEntity types:")
    for et, count in sorted(entity_types.items()):
        name = entity_type_names.get(et, f"Unknown({et})")
        print(f"  {name}: {count}")

    # Response types
    response_types = {}
    for obj in instances:
        rt = obj["response_type"]
        response_types[rt] = response_types.get(rt, 0) + 1

    response_type_names = {
        0: "Dynamic",
        1: "Kinematic",
        2: "Static",
    }

    print("\nResponse types:")
    for rt, count in sorted(response_types.items()):
        name = response_type_names.get(rt, f"Unknown({rt})")
        print(f"  {name}: {count}")

    # Spatial distribution analysis for walls
    if asset_name == "wall" and len(instances) > 1:
        print("\nWall-specific analysis:")

        # Check for edge walls
        edge_tolerance = 0.1
        top_walls = [obj for obj in instances if abs(obj["y"] - max_y) < edge_tolerance]
        bottom_walls = [obj for obj in instances if abs(obj["y"] - min_y) < edge_tolerance]
        left_walls = [obj for obj in instances if abs(obj["x"] - min_x) < edge_tolerance]
        right_walls = [obj for obj in instances if abs(obj["x"] - max_x) < edge_tolerance]

        print("  Edge walls:")
        print(f"    Top: {len(top_walls)}")
        print(f"    Bottom: {len(bottom_walls)}")
        print(f"    Left: {len(left_walls)}")
        print(f"    Right: {len(right_walls)}")

        # Check spacing
        if len(top_walls) > 1:
            top_walls.sort(key=lambda obj: obj["x"])
            spacings = [
                top_walls[i + 1]["x"] - top_walls[i]["x"] for i in range(len(top_walls) - 1)
            ]
            if spacings:
                avg_spacing = sum(spacings) / len(spacings)
                print(f"  Average horizontal spacing (top edge): {avg_spacing:.2f}")

        # Check for corners
        corners = []
        for obj in instances:
            # Top-left
            if abs(obj["x"] - min_x) < edge_tolerance and abs(obj["y"] - max_y) < edge_tolerance:
                corners.append(("TL", obj))
            # Top-right
            elif abs(obj["x"] - max_x) < edge_tolerance and abs(obj["y"] - max_y) < edge_tolerance:
                corners.append(("TR", obj))
            # Bottom-left
            elif abs(obj["x"] - min_x) < edge_tolerance and abs(obj["y"] - min_y) < edge_tolerance:
                corners.append(("BL", obj))
            # Bottom-right
            elif abs(obj["x"] - max_x) < edge_tolerance and abs(obj["y"] - min_y) < edge_tolerance:
                corners.append(("BR", obj))

        print(f"  Corners found: {len(corners)}")
        for corner_type, obj in corners:
            print(f"    {corner_type}: ({obj['x']:.2f}, {obj['y']:.2f})")

    # Verbose output - list all instances
    if verbose:
        print("\nDetailed instance list:")
        print(
            f"{'Index':<6} {'X':<8} {'Y':<8} {'Z':<8} {'Scale X':<8} "
            f"{'Scale Y':<8} {'Scale Z':<8} {'Persist':<8} {'Render':<8}"
        )
        print("-" * 80)
        for obj in instances[:20]:  # Limit to first 20 in verbose mode
            print(
                f"{obj['index']:<6} {obj['x']:<8.2f} {obj['y']:<8.2f} {obj['z']:<8.2f} "
                f"{obj['scale_x']:<8.2f} {obj['scale_y']:<8.2f} {obj['scale_z']:<8.2f} "
                f"{'Yes' if obj['persistent'] else 'No':<8} "
                f"{'Yes' if obj['render_only'] else 'No':<8}"
            )

        if len(instances) > 20:
            print(f"... and {len(instances) - 20} more instances")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Inspect assets in Madrona Escape Room level files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect walls in default level
  python inspect_asset.py wall

  # Inspect cubes in a specific level with verbose output
  python inspect_asset.py cube levels/maze_with_cubes.lvl -v

  # List all available assets
  python inspect_asset.py --list-assets

  # Inspect all assets in a level
  python inspect_asset.py --all levels/default.lvl
        """,
    )

    parser.add_argument(
        "asset_name", nargs="?", help="Name of the asset to inspect (e.g., wall, cube, cylinder)"
    )

    parser.add_argument(
        "level_file",
        nargs="?",
        default="levels/default.lvl",
        help="Path to the level file to inspect (default: levels/default.lvl)",
    )

    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Show detailed instance information"
    )

    parser.add_argument(
        "--list-assets", action="store_true", help="List all available assets and exit"
    )

    parser.add_argument("--all", action="store_true", help="Analyze all asset types in the level")

    args = parser.parse_args()

    # List assets mode
    if args.list_assets:
        print("Available physics assets:")
        for name in get_physics_assets_list():
            asset_id, _ = get_asset_id(name)
            print(f"  {name:<20} (ID: {asset_id})")

        print("\nAvailable render-only assets:")
        for name in get_render_assets_list():
            asset_id, _ = get_asset_id(name)
            print(f"  {name:<20} (ID: {asset_id})")
        return 0

    # Analyze all assets mode
    if args.all:
        if not args.level_file:
            print("Error: Level file required for --all mode")
            return 1

        print(f"Analyzing all assets in {args.level_file}")
        print("=" * 60)

        all_assets = get_physics_assets_list() + get_render_assets_list()
        for asset in all_assets:
            print(f"\n{'=' * 60}")
            analyze_asset(args.level_file, asset, args.verbose)

        return 0

    # Normal mode - single asset
    if not args.asset_name:
        parser.print_help()
        return 1

    return analyze_asset(args.level_file, args.asset_name, args.verbose)


if __name__ == "__main__":
    sys.exit(main())
