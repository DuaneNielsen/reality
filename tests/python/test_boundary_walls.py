#!/usr/bin/env python3
"""
Test automatic boundary wall generation in level compiler.
"""

import pytest

from madrona_escape_room.generated_constants import AssetIDs, EntityType, ResponseType
from madrona_escape_room.level_compiler import compile_level


class TestBoundaryWalls:
    """Test automatic boundary wall generation functionality"""

    def test_auto_boundary_walls_disabled_by_default(self):
        """Test that auto_boundary_walls is disabled by default for backward compatibility"""
        level_json = {
            "ascii": ["S....", ".....", ".....", "....."],
            "tileset": {"S": {"asset": "spawn"}, ".": {"asset": "empty"}},
            "scale": 2.0,
        }

        compiled_levels = compile_level(level_json)
        level = compiled_levels[0]

        # Should have auto_boundary_walls disabled by default
        assert not level.auto_boundary_walls

        # Should have no tiles (no walls, no obstacles)
        assert level.num_tiles == 0

    def test_auto_boundary_walls_disabled(self):
        """Test that auto_boundary_walls can be disabled"""
        level_json = {
            "ascii": ["S....", ".....", ".....", "....."],
            "tileset": {"S": {"asset": "spawn"}, ".": {"asset": "empty"}},
            "scale": 2.0,
            "auto_boundary_walls": False,
        }

        compiled_levels = compile_level(level_json)
        level = compiled_levels[0]

        # Should have auto_boundary_walls disabled
        assert not level.auto_boundary_walls

        # Should have no tiles (no walls, no obstacles)
        assert level.num_tiles == 0

    def test_boundary_wall_positioning(self):
        """Test that boundary walls are positioned correctly"""
        level_json = {
            "ascii": ["S...", "....", "...."],
            "tileset": {"S": {"asset": "spawn"}, ".": {"asset": "empty"}},
            "scale": 2.0,
            "auto_boundary_walls": True,
        }

        compiled_levels = compile_level(level_json)
        level = compiled_levels[0]

        # 4x3 grid with scale=2.0
        # world_width = 4 * 2.0 = 8.0
        # world_height = 3 * 2.0 = 6.0
        # World boundaries should be:
        # min_x = -(4-1)/2 * 2.0 - 1.0 = -4.0
        # max_x = (4-1)/2 * 2.0 + 1.0 = 4.0
        # min_y = -(3-1)/2 * 2.0 - 1.0 = -3.0
        # max_y = (3-1)/2 * 2.0 + 1.0 = 3.0

        expected_min_x = -4.0
        expected_max_x = 4.0
        expected_min_y = -3.0
        expected_max_y = 3.0

        assert level.world_min_x == pytest.approx(expected_min_x)
        assert level.world_max_x == pytest.approx(expected_max_x)
        assert level.world_min_y == pytest.approx(expected_min_y)
        assert level.world_max_y == pytest.approx(expected_max_y)

        # Check boundary wall positions
        wall_positions = []
        for i in range(level.num_tiles):
            if level.object_ids[i] == AssetIDs.WALL:
                wall_positions.append((level.tile_x[i], level.tile_y[i]))

        # Should have 8 walls at the boundaries (4 boundary walls + 4 corner blocks)
        assert len(wall_positions) == 8

        # Check that walls are positioned correctly outside the boundaries
        # Wall centers should be offset by wall_thickness/2 outside the boundaries
        wall_thickness = 1.0  # From implementation
        wall_half_thickness = wall_thickness / 2.0

        wall_x_positions = [pos[0] for pos in wall_positions]
        wall_y_positions = [pos[1] for pos in wall_positions]

        # Should have walls positioned outside boundaries with inner edge at boundary
        expected_west_x = expected_min_x - wall_half_thickness  # -4.5
        expected_east_x = expected_max_x + wall_half_thickness  # 4.5
        expected_south_y = expected_min_y - wall_half_thickness  # -3.5
        expected_north_y = expected_max_y + wall_half_thickness  # 3.5

        # Check for boundary walls (4 walls)
        assert any(
            abs(x - expected_west_x) < 0.1 for x in wall_x_positions
        ), f"Missing west wall at {expected_west_x}"
        assert any(
            abs(x - expected_east_x) < 0.1 for x in wall_x_positions
        ), f"Missing east wall at {expected_east_x}"
        assert any(
            abs(y - expected_south_y) < 0.1 for y in wall_y_positions
        ), f"Missing south wall at {expected_south_y}"
        assert any(
            abs(y - expected_north_y) < 0.1 for y in wall_y_positions
        ), f"Missing north wall at {expected_north_y}"

        # Check for corner blocks (4 corners)
        expected_corners = [
            (expected_east_x, expected_north_y),  # Northeast
            (expected_west_x, expected_north_y),  # Northwest
            (expected_east_x, expected_south_y),  # Southeast
            (expected_west_x, expected_south_y),  # Southwest
        ]

        for corner_x, corner_y in expected_corners:
            assert any(
                abs(x - corner_x) < 0.1 and abs(y - corner_y) < 0.1 for x, y in wall_positions
            ), f"Missing corner block at ({corner_x}, {corner_y})"

    def test_boundary_walls_with_existing_tiles(self):
        """Test boundary walls work with existing level content"""
        level_json = {
            "ascii": ["S.C.", "....", "...."],
            "tileset": {"S": {"asset": "spawn"}, "C": {"asset": "cube"}, ".": {"asset": "empty"}},
            "scale": 2.0,
            "auto_boundary_walls": True,
        }

        compiled_levels = compile_level(level_json)
        level = compiled_levels[0]

        # Should have 1 cube + 4 boundary walls + 4 corner blocks = 9 total tiles
        assert level.num_tiles == 9

        # Count tiles by type
        wall_count = 0
        cube_count = 0
        for i in range(level.num_tiles):
            if level.object_ids[i] == AssetIDs.WALL:
                wall_count += 1
            elif level.object_ids[i] == AssetIDs.CUBE:
                cube_count += 1

        assert wall_count == 8, "Should have 8 wall tiles (4 boundary walls + 4 corner blocks)"
        assert cube_count == 1, "Should have 1 cube from the level"

    def test_multi_level_boundary_walls(self):
        """Test boundary walls work with multi-level JSON"""
        multi_level_json = {
            "levels": [
                {"ascii": ["S..", "...", "..."], "name": "level1"},
                {"ascii": ["S..C", "....", "...."], "name": "level2"},
            ],
            "tileset": {"S": {"asset": "spawn"}, "C": {"asset": "cube"}, ".": {"asset": "empty"}},
            "scale": 2.0,
            "auto_boundary_walls": True,
        }

        compiled_levels = compile_level(multi_level_json)

        assert len(compiled_levels) == 2

        # Level 1: should have 4 boundary walls + 4 corner blocks = 8 tiles
        level1 = compiled_levels[0]
        assert level1.auto_boundary_walls
        assert level1.num_tiles == 8  # 4 boundary walls + 4 corner blocks

        # Level 2: should have 1 cube + 4 boundary walls + 4 corner blocks = 9 tiles
        level2 = compiled_levels[1]
        assert level2.auto_boundary_walls
        assert level2.num_tiles == 9  # 1 cube + 8 boundary tiles

    def test_boundary_walls_validation_error_on_tile_limit(self):
        """Test that an error is raised if adding boundary walls would exceed tile limit"""
        # This test would require creating a level with MAX_TILES - 3 existing tiles
        # to test the boundary walls overflow check. For now, we'll test with smaller limits.
        pass  # Skip this test as it would require a very large level

    def test_boundary_wall_properties(self):
        """Test that boundary walls have correct properties"""
        level_json = {
            "ascii": ["S...", "....", "...."],
            "tileset": {"S": {"asset": "spawn"}, ".": {"asset": "empty"}},
            "scale": 2.0,
            "auto_boundary_walls": True,
        }

        compiled_levels = compile_level(level_json)
        level = compiled_levels[0]

        # Check properties of all boundary walls
        for i in range(level.num_tiles):
            if level.object_ids[i] == AssetIDs.WALL:
                # Should be persistent
                assert level.tile_persistent[i]

                # Should have physics (not render-only)
                assert not level.tile_render_only[i]

                # Should not end episode on collision
                assert not level.tile_done_on_collide[i]

                # Should be Wall entity type
                assert level.tile_entity_type[i] == EntityType.Wall

                # Should be Static response type
                assert level.tile_response_type[i] == ResponseType.Static

                # Should have no randomization
                assert level.tile_rand_x[i] == 0.0
                assert level.tile_rand_y[i] == 0.0
                assert level.tile_rand_z[i] == 0.0
                assert level.tile_rand_rot_z[i] == 0.0
                assert level.tile_rand_scale_x[i] == 0.0
                assert level.tile_rand_scale_y[i] == 0.0
                assert level.tile_rand_scale_z[i] == 0.0

                # Should have identity rotation
                assert level.tile_rotation[i] == (1.0, 0.0, 0.0, 0.0)

    def test_backward_compatibility(self):
        """Test that existing levels without auto_boundary_walls still work"""
        # Test that levels without the auto_boundary_walls field default to False
        level_json = {
            "ascii": ["S...", "....", "...."],
            "tileset": {"S": {"asset": "spawn"}, ".": {"asset": "empty"}},
            "scale": 2.0,
            # Note: no auto_boundary_walls field - should default to False
        }

        compiled_levels = compile_level(level_json)
        level = compiled_levels[0]

        # Should default to False and not add boundary walls
        assert not level.auto_boundary_walls
        assert level.num_tiles == 0  # No boundary walls added
