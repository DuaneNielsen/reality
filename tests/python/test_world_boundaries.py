#!/usr/bin/env python3
"""
Test world boundary calculations in level compiler.
Ensures boundaries are correctly computed for different level sizes and scales.
"""

import pytest

from madrona_escape_room.level_compiler import compile_ascii_level


class TestWorldBoundaries:
    """Test that world boundaries are correctly calculated for various level configurations"""

    def test_small_square_room(self):
        """Test a small 4x4 room with default scale"""
        level = """####
#S.#
#..#
####"""

        compiled = compile_ascii_level(level, scale=2.5)

        # For a 4x4 grid with scale=2.5:
        # Tile centers range from -(4-1)/2 * 2.5 to (4-1)/2 * 2.5
        # = -3.75 to 3.75
        # With wall extent of scale/2 = 1.25:
        # Boundaries should be -5.0 to 5.0

        assert compiled.world_min_x == pytest.approx(-5.0)
        assert compiled.world_max_x == pytest.approx(5.0)
        assert compiled.world_min_y == pytest.approx(-5.0)
        assert compiled.world_max_y == pytest.approx(5.0)
        assert compiled.world_min_z == 0.0
        assert compiled.world_max_z == 25.0  # 10 * scale

    def test_rectangular_room(self):
        """Test a rectangular 8x6 room"""
        level = """########
#S.....#
#......#
#......#
#......#
########"""

        compiled = compile_ascii_level(level, scale=2.5)

        # Width = 8, Height = 6
        # X: tile centers from -(8-1)/2 * 2.5 to (8-1)/2 * 2.5 = -8.75 to 8.75
        # With extent: -10.0 to 10.0
        # Y: tile centers from -(6-1)/2 * 2.5 to (6-1)/2 * 2.5 = -6.25 to 6.25
        # With extent: -7.5 to 7.5

        assert compiled.world_min_x == pytest.approx(-10.0)
        assert compiled.world_max_x == pytest.approx(10.0)
        assert compiled.world_min_y == pytest.approx(-7.5)
        assert compiled.world_max_y == pytest.approx(7.5)

    def test_large_room_default_scale(self):
        """Test a 16x16 room matching default_level.cpp"""
        level = """################
#S.............#
#..............#
#..............#
#..............#
#..............#
#..............#
#..............#
#..............#
#..............#
#..............#
#..............#
#..............#
#..............#
#..............#
################"""

        compiled = compile_ascii_level(level, scale=2.5)

        # 16x16 grid with scale=2.5
        # Tile centers from -(16-1)/2 * 2.5 to (16-1)/2 * 2.5 = -18.75 to 18.75
        # With extent of 1.25: -20.0 to 20.0
        # This should match default_level.cpp

        assert compiled.world_min_x == pytest.approx(-20.0)
        assert compiled.world_max_x == pytest.approx(20.0)
        assert compiled.world_min_y == pytest.approx(-20.0)
        assert compiled.world_max_y == pytest.approx(20.0)

    def test_32x32_room(self):
        """Test the 32x32 room used in test_reward_normalization"""
        # Create a 32x32 room
        top_bottom = "#" * 32
        middle = "#" + "S" + "." * 29 + "#"
        other_rows = ["#" + "." * 30 + "#"] * 29  # 29 middle rows + 1 spawn row = 30 inner rows

        lines = [top_bottom, middle] + other_rows + [top_bottom]
        level = "\n".join(lines)

        compiled = compile_ascii_level(level, scale=2.5)

        # 32x32 grid with scale=2.5
        # Tile centers from -(32-1)/2 * 2.5 to (32-1)/2 * 2.5 = -38.75 to 38.75
        # With extent of 1.25: -40.0 to 40.0

        assert compiled.world_min_x == pytest.approx(-40.0)
        assert compiled.world_max_x == pytest.approx(40.0)
        assert compiled.world_min_y == pytest.approx(-40.0)
        assert compiled.world_max_y == pytest.approx(40.0)

    def test_different_scales(self):
        """Test that boundaries scale correctly with different scale factors"""
        level = """######
#S...#
#....#
######"""

        # Test scale=1.0
        compiled_1 = compile_ascii_level(level, scale=1.0)
        # 6x4 grid, scale=1.0
        # X: -(6-1)/2 * 1.0 = -2.5, extent 0.5 -> -3.0 to 3.0
        # Y: -(4-1)/2 * 1.0 = -1.5, extent 0.5 -> -2.0 to 2.0
        assert compiled_1.world_min_x == pytest.approx(-3.0)
        assert compiled_1.world_max_x == pytest.approx(3.0)
        assert compiled_1.world_min_y == pytest.approx(-2.0)
        assert compiled_1.world_max_y == pytest.approx(2.0)
        assert compiled_1.world_max_z == 10.0  # 10 * scale

        # Test scale=5.0
        compiled_5 = compile_ascii_level(level, scale=5.0)
        # X: -(6-1)/2 * 5.0 = -12.5, extent 2.5 -> -15.0 to 15.0
        # Y: -(4-1)/2 * 5.0 = -7.5, extent 2.5 -> -10.0 to 10.0
        assert compiled_5.world_min_x == pytest.approx(-15.0)
        assert compiled_5.world_max_x == pytest.approx(15.0)
        assert compiled_5.world_min_y == pytest.approx(-10.0)
        assert compiled_5.world_max_y == pytest.approx(10.0)
        assert compiled_5.world_max_z == 50.0  # 10 * scale

    def test_odd_vs_even_dimensions(self):
        """Test boundary calculations for odd vs even grid dimensions"""

        # Odd dimensions: 5x5
        level_odd = """#####
#S..#
#...#
#...#
#####"""

        compiled_odd = compile_ascii_level(level_odd, scale=2.0)
        # 5x5 grid, scale=2.0
        # Tile centers from -(5-1)/2 * 2.0 to (5-1)/2 * 2.0 = -4.0 to 4.0
        # With extent of 1.0: -5.0 to 5.0
        assert compiled_odd.world_min_x == pytest.approx(-5.0)
        assert compiled_odd.world_max_x == pytest.approx(5.0)
        assert compiled_odd.world_min_y == pytest.approx(-5.0)
        assert compiled_odd.world_max_y == pytest.approx(5.0)

        # Even dimensions: 6x6
        level_even = """######
#S...#
#....#
#....#
#....#
######"""

        compiled_even = compile_ascii_level(level_even, scale=2.0)
        # 6x6 grid, scale=2.0
        # Tile centers from -(6-1)/2 * 2.0 to (6-1)/2 * 2.0 = -5.0 to 5.0
        # With extent of 1.0: -6.0 to 6.0
        assert compiled_even.world_min_x == pytest.approx(-6.0)
        assert compiled_even.world_max_x == pytest.approx(6.0)
        assert compiled_even.world_min_y == pytest.approx(-6.0)
        assert compiled_even.world_max_y == pytest.approx(6.0)

    def test_actual_tile_positions_within_boundaries(self):
        """Verify that level content tiles fall within the calculated boundaries

        Note: Boundary walls (when auto_boundary_walls=True) are positioned outside
        the world boundaries by design, with their inner edge at the boundary.
        """
        level = """########
#S.....#
#..CC..#
#......#
########"""

        compiled = compile_ascii_level(level, scale=3.0)

        # Get boundaries
        min_x = compiled.world_min_x
        max_x = compiled.world_max_x
        min_y = compiled.world_min_y
        max_y = compiled.world_max_y

        # Check that all tiles are within boundaries
        # Account for wall scaling (walls are scaled by 'scale' factor)
        wall_half_extent = 3.0 / 2.0  # scale / 2

        for i in range(compiled.num_tiles):
            tile_x = compiled.tile_x[i]
            tile_y = compiled.tile_y[i]

            # Skip auto-generated boundary walls - positioned outside world boundaries by design
            # Their inner edge aligns with the world boundaries to contain the level
            if compiled.auto_boundary_walls:
                is_boundary_wall = (
                    compiled.object_ids[i] == 2  # Wall asset ID
                    and compiled.tile_persistent[i]  # Boundary walls are persistent
                    and (
                        abs(tile_x - min_x) < 0.6
                        or abs(tile_x - max_x) < 0.6
                        or abs(tile_y - min_y) < 0.6
                        or abs(tile_y - max_y) < 0.6
                    )
                )
                if is_boundary_wall:
                    continue  # Skip boundary walls

            # The tile center should be within the boundaries minus the extent
            assert (
                tile_x >= min_x + wall_half_extent - 0.01
            ), f"Tile {i} X position {tile_x} extends beyond min boundary {min_x}"
            assert (
                tile_x <= max_x - wall_half_extent + 0.01
            ), f"Tile {i} X position {tile_x} extends beyond max boundary {max_x}"
            assert (
                tile_y >= min_y + wall_half_extent - 0.01
            ), f"Tile {i} Y position {tile_y} extends beyond min boundary {min_y}"
            assert (
                tile_y <= max_y - wall_half_extent + 0.01
            ), f"Tile {i} Y position {tile_y} extends beyond max boundary {max_y}"

    def test_boundary_fields_exist(self):
        """Ensure all boundary fields are present in compiled output"""
        level = """###
#S#
###"""

        compiled = compile_ascii_level(level)

        # All boundary fields should exist
        required_fields = [
            "world_min_x",
            "world_max_x",
            "world_min_y",
            "world_max_y",
            "world_min_z",
            "world_max_z",
        ]

        for field in required_fields:
            assert hasattr(compiled, field), f"Missing boundary field: {field}"
            field_value = getattr(compiled, field)
            assert isinstance(
                field_value, float
            ), f"Boundary field {field} should be float, got {type(field_value)}"

    def test_compiled_level_boundary_attributes(self):
        """Test that boundary values are correctly set as CompiledLevel attributes"""
        level = """######
#S...#
#....#
######"""

        compiled = compile_ascii_level(level, scale=2.0)

        # Verify all boundary attributes exist and are correct
        # For a 6x4 grid with scale=2.0:
        # X: tile centers from -(6-1)/2 * 2.0 to (6-1)/2 * 2.0 = -5.0 to 5.0
        # With extent of 1.0: -6.0 to 6.0
        # Y: tile centers from -(4-1)/2 * 2.0 to (4-1)/2 * 2.0 = -3.0 to 3.0
        # With extent of 1.0: -4.0 to 4.0
        assert hasattr(compiled, "world_min_x"), "CompiledLevel missing world_min_x attribute"
        assert hasattr(compiled, "world_max_x"), "CompiledLevel missing world_max_x attribute"
        assert hasattr(compiled, "world_min_y"), "CompiledLevel missing world_min_y attribute"
        assert hasattr(compiled, "world_max_y"), "CompiledLevel missing world_max_y attribute"
        assert hasattr(compiled, "world_min_z"), "CompiledLevel missing world_min_z attribute"
        assert hasattr(compiled, "world_max_z"), "CompiledLevel missing world_max_z attribute"

        assert compiled.world_min_x == pytest.approx(-6.0)
        assert compiled.world_max_x == pytest.approx(6.0)
        assert compiled.world_min_y == pytest.approx(-4.0)
        assert compiled.world_max_y == pytest.approx(4.0)
        assert compiled.world_min_z == pytest.approx(0.0)
        assert compiled.world_max_z == pytest.approx(20.0)  # 10 * scale
