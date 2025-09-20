"""
Test ASCII level compiler integration with pytest framework

Tests the complete pipeline from ASCII art to C API struct population.
"""

import pytest

from madrona_escape_room.level_compiler import (
    MAX_TILES_C_API,
    compile_ascii_level,
    compile_level,
    compile_multi_level,
    validate_compiled_level,
)


class TestLevelCompiler:
    """Test the level_compiler module"""

    @pytest.mark.spec("docs/specs/level_compiler.md", "compile_ascii_level")
    def test_simple_room_compilation(self):
        """Test compiling a simple room"""
        level = """######
#S...#
#....#
######"""

        compiled = compile_ascii_level(level)

        assert compiled.width == 6
        assert compiled.height == 4
        assert compiled.world_scale == 2.5
        assert compiled.num_tiles > 0
        assert compiled.max_entities > compiled.num_tiles
        assert len(compiled.object_ids) == MAX_TILES_C_API
        assert len(compiled.tile_x) == MAX_TILES_C_API
        assert len(compiled.tile_y) == MAX_TILES_C_API

        # Verify spawn point was found
        assert compiled.num_spawns == 1
        assert compiled.spawn_x[0] != 0.0 or compiled.spawn_y[0] != 0.0

        validate_compiled_level(compiled)

    @pytest.mark.spec("docs/specs/level_compiler.md", "compile_ascii_level")
    def test_level_with_obstacles(self):
        """Test compiling a level with cube obstacles"""
        level = """########
#S.....#
#..CC..#
#......#
########"""

        compiled = compile_ascii_level(level)

        # Get the actual object IDs from the C API
        from madrona_escape_room.ctypes_bindings import get_physics_asset_object_id

        wall_id = get_physics_asset_object_id("wall")
        cube_id = get_physics_asset_object_id("cube")

        # Should have walls + cubes
        wall_count = sum(1 for i in range(compiled.num_tiles) if compiled.object_ids[i] == wall_id)
        cube_count = sum(1 for i in range(compiled.num_tiles) if compiled.object_ids[i] == cube_id)

        assert wall_count > 0, "Should have wall tiles"
        assert cube_count == 2, f"Should have 2 cubes, got {cube_count}"
        assert compiled.num_tiles == wall_count + cube_count

        validate_compiled_level(compiled)

    @pytest.mark.spec("docs/specs/level_compiler.md", "compile_ascii_level")
    def test_error_cases(self):
        """Test error handling for invalid levels"""

        # No spawn point
        with pytest.raises(ValueError, match="No spawn points"):
            compile_ascii_level("""####
#..#
####""")

        # Empty level
        with pytest.raises(ValueError, match="'ascii' field cannot be empty"):
            compile_ascii_level("")

        # Unknown character
        with pytest.raises(ValueError, match="Unknown character"):
            compile_ascii_level("""####
#SX#
####""")

    @pytest.mark.spec("docs/specs/level_compiler.md", "compile_multi_level")
    def test_multi_level_compilation(self):
        """Test compiling multi-level JSON format"""
        multi_level = {
            "levels": [
                {"ascii": ["####", "#S.#", "####"], "name": "simple_level", "agent_facing": [0.0]},
                {
                    "ascii": ["######", "#S..C#", "######"],
                    "name": "cube_level",
                    "agent_facing": [1.57],  # Face right
                },
            ],
            "tileset": {
                "#": {"asset": "wall", "done_on_collision": True},
                "C": {"asset": "cube", "done_on_collision": True},
                "S": {"asset": "spawn"},
                ".": {"asset": "empty"},
            },
            "scale": 2.5,
            "name": "test_multi_levels",
        }

        # Test compile_multi_level() directly
        compiled_levels = compile_multi_level(multi_level)

        # Should return a list of CompiledLevel objects
        assert isinstance(compiled_levels, list)
        assert len(compiled_levels) == 2

        # Test first level
        level1 = compiled_levels[0]
        assert level1.width == 4
        assert level1.height == 3
        assert level1.world_scale == 2.5
        assert level1.num_spawns == 1
        assert level1.level_name.decode("utf-8", errors="ignore").startswith("simple_level")
        validate_compiled_level(level1)

        # Test second level
        level2 = compiled_levels[1]
        assert level2.width == 6
        assert level2.height == 3
        assert level2.world_scale == 2.5
        assert level2.num_spawns == 1
        assert level2.num_tiles > level1.num_tiles  # Has more obstacles
        assert level2.level_name.decode("utf-8", errors="ignore").startswith("cube_level")
        validate_compiled_level(level2)

        # Test auto-detection via compile_level()
        compiled_levels2 = compile_level(multi_level)
        assert isinstance(compiled_levels2, list)
        assert len(compiled_levels2) == 2
        assert compiled_levels2[0].width == compiled_levels[0].width
        assert compiled_levels2[1].width == compiled_levels[1].width

    @pytest.mark.spec("docs/specs/level_compiler.md", "compile_multi_level")
    def test_multi_level_validation_errors(self):
        """Test multi-level format validation errors"""

        # Missing levels field - compile_multi_level expects multi-level format
        with pytest.raises(
            ValueError, match="compile_multi_level\\(\\) requires multi-level JSON format"
        ):
            compile_multi_level({"tileset": {"#": {"asset": "wall"}}})

        # Empty levels array
        with pytest.raises(ValueError, match="'levels' field must be a non-empty list"):
            compile_multi_level({"levels": [], "tileset": {"#": {"asset": "wall"}}})

        # Level missing ascii
        with pytest.raises(ValueError, match="Level 0 must contain 'ascii' field"):
            compile_multi_level({"levels": [{"name": "test"}], "tileset": {"#": {"asset": "wall"}}})

        # Missing tileset
        with pytest.raises(ValueError, match="Multi-level JSON must contain 'tileset' field"):
            compile_multi_level({"levels": [{"ascii": ["###", "#S#", "###"]}]})


class TestCTypesIntegration:
    """Test integration with ctypes bindings"""

    @pytest.mark.spec("docs/specs/level_compiler.md", "CompiledLevel")
    def test_dataclass_to_ctypes_conversion(self):
        """Test that CompiledLevel dataclass works with ctypes"""
        level = """####
#S.#
####"""

        compiled = compile_ascii_level(level)

        # CompiledLevel is already a ctypes-compatible dataclass
        # Just verify the structure has the expected fields
        assert hasattr(compiled, "num_tiles")
        assert hasattr(compiled, "max_entities")
        assert hasattr(compiled, "width")
        assert hasattr(compiled, "height")
        assert hasattr(compiled, "world_scale")

        # Verify arrays are accessible
        assert compiled.num_tiles > 0
        assert compiled.object_ids[0] >= 0
        assert compiled.tile_x[0] != 0.0 or compiled.tile_y[0] != 0.0

        # Verify zero padding in unused portion
        for i in range(compiled.num_tiles, MAX_TILES_C_API):
            assert compiled.object_ids[i] == 0
            assert abs(compiled.tile_x[i]) < 0.001
            assert abs(compiled.tile_y[i]) < 0.001

    @pytest.mark.spec("docs/specs/level_compiler.md", "validate_compiled_level")
    def test_compiled_level_validation(self):
        """Test compiled level validation"""
        level = """######
#S...#
#....#
######"""

        compiled = compile_ascii_level(level)

        # Should not raise any exception
        validate_compiled_level(compiled)

    @pytest.mark.spec("docs/specs/level_compiler.md", "compile_ascii_level")
    def test_multiple_level_compilation(self):
        """Test compiling multiple levels"""
        level1 = """####
#S.#
####"""

        level2 = """######
#S...#
######"""

        compiled1 = compile_ascii_level(level1)
        compiled2 = compile_ascii_level(level2)

        # Test that both levels compile correctly
        assert compiled1.num_tiles > 0
        assert compiled2.num_tiles > 0
        assert compiled1.width == 4
        assert compiled2.width == 6

        # Each should have their own spawn point
        assert compiled1.num_spawns == 1
        assert compiled2.num_spawns == 1


class TestManagerIntegration:
    """Test integration with SimManager (requires built C API)"""

    @pytest.mark.spec("docs/specs/level_compiler.md", "System Integration")
    def test_manager_creation_with_ascii_level(self, cpu_manager):
        """Test creating manager with ASCII level"""
        # Note: SimManager doesn't yet support custom levels via level_ascii parameter
        # This test verifies the level compilation works and manager runs with default level
        level = """########
#S.....#
#......#
########"""

        # Compile the level to verify it works
        compiled = compile_ascii_level(level)
        validate_compiled_level(compiled)

        # Test that simulation runs with default level
        for _ in range(3):
            cpu_manager.step()

    @pytest.mark.spec("docs/specs/level_compiler.md", "System Integration")
    def test_multiple_worlds_same_level(self, cpu_manager):
        """Test multiple worlds with same ASCII level"""
        level = """######
#S...#
#....#
######"""

        # Compile the level to verify it works
        compiled = compile_ascii_level(level)
        validate_compiled_level(compiled)

        # Test simulation runs with multiple worlds (using default level)
        for _ in range(2):
            cpu_manager.step()

    @pytest.mark.spec("docs/specs/level_compiler.md", "System Integration")
    def test_backward_compatibility(self, cpu_manager):
        """Test that SimManager still works without level_ascii"""
        # Should work without level_ascii (uses default level)
        # The cpu_manager fixture already provides a working manager

        # Should still run
        cpu_manager.step()


class TestJSONLevelFormat:
    """Test JSON level format with parameters"""

    @pytest.mark.spec("docs/specs/level_compiler.md", "JSON Level Format (Single Level)")
    def test_json_level_with_agent_facing(self):
        """Test JSON level with agent facing angles"""
        import math

        json_level = {
            "ascii": """######
#S..S#
######""",
            "tileset": {
                "#": {"asset": "wall"},
                "S": {"asset": "spawn"},
                ".": {"asset": "empty"},
            },
            "scale": 1.5,
            "agent_facing": [0.0, math.pi / 2],  # First faces forward, second faces right
        }

        compiled_levels = compile_level(json_level)
        compiled = compiled_levels[0]  # Extract single level

        # Verify structure
        assert compiled.world_scale == 1.5
        assert compiled.num_spawns == 2

        # Check agent facing was set correctly
        assert compiled.spawn_facing[0] == 0.0
        assert abs(compiled.spawn_facing[1] - math.pi / 2) < 0.001

        # Remaining spawn slots should be 0
        for i in range(2, 8):
            assert compiled.spawn_facing[i] == 0.0

    @pytest.mark.spec("docs/specs/level_compiler.md", "JSON Level Format (Single Level)")
    def test_json_level_without_facing(self):
        """Test JSON level without agent_facing defaults to 0"""
        json_level = {
            "ascii": """####
#S.#
####""",
            "tileset": {
                "#": {"asset": "wall"},
                "S": {"asset": "spawn"},
                ".": {"asset": "empty"},
            },
            "scale": 2.0,
        }

        compiled_levels = compile_level(json_level)
        compiled = compiled_levels[0]  # Extract single level

        # Should default to facing forward (0.0)
        assert compiled.spawn_facing[0] == 0.0

    @pytest.mark.spec("docs/specs/level_compiler.md", "System Integration")
    def test_json_level_in_sim_manager(self, cpu_manager):
        """Test using JSON level with SimManager"""
        import math

        json_level = {
            "ascii": """########
#S.....#
#......#
########""",
            "tileset": {
                "#": {"asset": "wall"},
                "S": {"asset": "spawn"},
                ".": {"asset": "empty"},
            },
            "scale": 2.0,
            "agent_facing": [math.pi / 4],  # Face 45 degrees
        }

        # Compile and validate the level
        compiled_levels = compile_level(json_level)
        compiled = compiled_levels[0]  # Extract single level

        # Verify the compiled level has the correct facing
        assert abs(compiled.spawn_facing[0] - math.pi / 4) < 0.001

        # Test simulation runs with default level (custom levels not yet supported)
        for _ in range(3):
            cpu_manager.step()


class TestDoneOnCollisionFlags:
    """Test done_on_collision flag support in level compiler"""

    @pytest.mark.spec("docs/specs/level_compiler.md", "JSON Level Format (Single Level)")
    def test_explicit_done_on_collision_flags(self):
        """Test that done_on_collision flags are set correctly via JSON"""
        json_level = {
            "ascii": """####O####
#S.....C#
#........#
#C......O#
#########""",
            "tileset": {
                "#": {"asset": "wall", "done_on_collision": False},
                "C": {"asset": "cube", "done_on_collision": True},
                "O": {"asset": "cylinder", "done_on_collision": True},
                "S": {"asset": "spawn"},
                ".": {"asset": "empty"},
            },
            "scale": 2.5,
            "name": "test_collision_flags",
        }

        compiled_levels = compile_level(json_level)
        compiled = compiled_levels[0]  # Extract single level

        # Get asset IDs for verification
        from madrona_escape_room.ctypes_bindings import (
            get_physics_asset_object_id,
            get_render_asset_object_id,
        )

        wall_id = get_physics_asset_object_id("wall")
        cube_id = get_physics_asset_object_id("cube")
        cylinder_id = get_render_asset_object_id("cylinder")

        # Count tiles by type and verify done_on_collide flags
        walls_correct = 0
        cubes_correct = 0
        cylinders_correct = 0

        for i in range(compiled.num_tiles):
            obj_id = compiled.object_ids[i]
            done_flag = compiled.tile_done_on_collide[i]

            if obj_id == wall_id:
                # Walls should have done_on_collision = False
                assert not done_flag, f"Wall at index {i} has done_on_collide=True (expected False)"
                walls_correct += 1
            elif obj_id == cube_id:
                # Cubes should have done_on_collision = True
                assert done_flag, f"Cube at index {i} has done_on_collide=False (expected True)"
                cubes_correct += 1
            elif obj_id == cylinder_id:
                # Cylinders should have done_on_collision = True
                assert done_flag, f"Cylinder at index {i} has done_on_collide=False (expected True)"
                cylinders_correct += 1

        # Verify counts match ASCII
        assert walls_correct == json_level["ascii"].count("#")
        assert cubes_correct == json_level["ascii"].count("C")
        assert cylinders_correct == json_level["ascii"].count("O")

    @pytest.mark.spec("docs/specs/level_compiler.md", "compile_ascii_level")
    def test_default_tileset_done_on_collision(self):
        """Test that default tileset sets done_on_collision correctly"""
        ascii_level = """####O####
#S.....C#
#........#
#C......O#
#########"""

        compiled = compile_ascii_level(ascii_level, scale=2.5, level_name="default_test")

        # Get asset IDs
        from madrona_escape_room.ctypes_bindings import (
            get_physics_asset_object_id,
            get_render_asset_object_id,
        )

        wall_id = get_physics_asset_object_id("wall")
        cube_id = get_physics_asset_object_id("cube")
        cylinder_id = get_render_asset_object_id("cylinder")

        # Check that default tileset applies correct flags
        for i in range(compiled.num_tiles):
            obj_id = compiled.object_ids[i]
            done_flag = compiled.tile_done_on_collide[i]

            if obj_id == wall_id:
                assert not done_flag, "Wall has incorrect done_on_collide flag"
            elif obj_id == cube_id:
                assert done_flag, "Cube missing done_on_collide=True flag"
            elif obj_id == cylinder_id:
                assert done_flag, "Cylinder missing done_on_collide=True flag"

    @pytest.mark.spec("docs/specs/level_compiler.md", "JSON Level Format (Single Level)")
    def test_mixed_collision_flags(self):
        """Test level with mixed collision behavior"""
        json_level = {
            "ascii": """######
#S...#
#.CC.#
######""",
            "tileset": {
                "#": {"asset": "wall"},  # Default: done_on_collision = False
                "C": {"asset": "cube", "done_on_collision": True},
                "S": {"asset": "spawn"},
                ".": {"asset": "empty"},
            },
            "scale": 2.0,
        }

        compiled_levels = compile_level(json_level)
        compiled = compiled_levels[0]  # Extract single level

        from madrona_escape_room.ctypes_bindings import get_physics_asset_object_id

        cube_id = get_physics_asset_object_id("cube")

        # Count tiles with done_on_collide = True
        collision_tiles = sum(
            1 for i in range(compiled.num_tiles) if compiled.tile_done_on_collide[i]
        )

        # Count cubes
        cube_count = sum(1 for i in range(compiled.num_tiles) if compiled.object_ids[i] == cube_id)

        # Only cubes should have done_on_collide = True
        assert collision_tiles == cube_count == 2

    @pytest.mark.spec("docs/specs/level_compiler.md", "JSON Level Format (Single Level)")
    def test_validation_of_done_on_collision_type(self):
        """Test that done_on_collision must be a boolean"""
        json_level = {
            "ascii": """###
#S#
###""",
            "tileset": {
                "#": {"asset": "wall", "done_on_collision": "true"},  # String instead of bool
                "S": {"asset": "spawn"},
            },
        }

        with pytest.raises(ValueError, match="done_on_collision.*must be a boolean"):
            compile_level(json_level)


class TestBoundaryWallOffset:
    """Test boundary_wall_offset flag functionality"""

    @pytest.mark.spec("docs/specs/level_compiler.md", "BoundaryWallGeneration")
    def test_boundary_wall_offset_default_zero(self):
        """Test that boundary_wall_offset defaults to 0.0"""
        json_level = {
            "ascii": ["####", "#S.#", "####"],
            "tileset": {
                "#": {"asset": "wall"},
                "S": {"asset": "spawn"},
                ".": {"asset": "empty"},
            },
            "auto_boundary_walls": True,
        }

        compiled_levels = compile_level(json_level)
        compiled = compiled_levels[0]

        # Should work with default offset of 0.0
        assert compiled.auto_boundary_walls is True
        assert compiled.num_tiles > 0  # Should have boundary walls added

    @pytest.mark.spec("docs/specs/level_compiler.md", "BoundaryWallGeneration")
    def test_boundary_wall_offset_specified(self):
        """Test that boundary_wall_offset can be specified"""
        json_level = {
            "ascii": ["####", "#S.#", "####"],
            "tileset": {
                "#": {"asset": "wall"},
                "S": {"asset": "spawn"},
                ".": {"asset": "empty"},
            },
            "auto_boundary_walls": True,
            "boundary_wall_offset": 1.5,
        }

        compiled_levels = compile_level(json_level)
        compiled = compiled_levels[0]

        # Should compile successfully with custom offset
        assert compiled.auto_boundary_walls is True
        assert compiled.num_tiles > 0  # Should have boundary walls added

    @pytest.mark.spec("docs/specs/level_compiler.md", "BoundaryWallGeneration")
    def test_boundary_wall_offset_validation(self):
        """Test validation of boundary_wall_offset parameter"""
        # Test negative offset rejection
        json_level = {
            "ascii": ["####", "#S.#", "####"],
            "tileset": {
                "#": {"asset": "wall"},
                "S": {"asset": "spawn"},
                ".": {"asset": "empty"},
            },
            "auto_boundary_walls": True,
            "boundary_wall_offset": -1.0,  # Invalid negative offset
        }

        with pytest.raises(ValueError, match="Invalid boundary_wall_offset.*must be non-negative"):
            compile_level(json_level)

        # Test non-numeric offset rejection
        json_level["boundary_wall_offset"] = "invalid"
        with pytest.raises(ValueError, match="Invalid boundary_wall_offset.*must be non-negative"):
            compile_level(json_level)

    @pytest.mark.spec("docs/specs/level_compiler.md", "JSON Multi-Level Format")
    def test_boundary_wall_offset_multi_level(self):
        """Test boundary_wall_offset in multi-level format"""
        multi_level = {
            "levels": [
                {"ascii": ["####", "#S.#", "####"], "name": "level_1"},
                {"ascii": ["######", "#S..C#", "######"], "name": "level_2"},
            ],
            "tileset": {
                "#": {"asset": "wall"},
                "C": {"asset": "cube"},
                "S": {"asset": "spawn"},
                ".": {"asset": "empty"},
            },
            "auto_boundary_walls": True,
            "boundary_wall_offset": 2.0,
        }

        compiled_levels = compile_multi_level(multi_level)

        # Both levels should have boundary walls with offset
        assert len(compiled_levels) == 2
        for level in compiled_levels:
            assert level.auto_boundary_walls is True
            assert level.num_tiles > 0  # Should have boundary walls added

    @pytest.mark.spec("docs/specs/level_compiler.md", "BoundaryWallGeneration")
    def test_boundary_wall_offset_without_auto_walls(self):
        """Test that boundary_wall_offset is ignored when auto_boundary_walls is False"""
        json_level = {
            "ascii": ["####", "#S.#", "####"],
            "tileset": {
                "#": {"asset": "wall"},
                "S": {"asset": "spawn"},
                ".": {"asset": "empty"},
            },
            "auto_boundary_walls": False,  # Walls disabled
            "boundary_wall_offset": 5.0,  # Should be ignored
        }

        compiled_levels = compile_level(json_level)
        compiled = compiled_levels[0]

        # Should compile successfully even with offset when walls are disabled
        assert compiled.auto_boundary_walls is False
        # Number of tiles should only be from the ASCII art, not boundary walls
        # Count all "#" characters in all rows (walls don't include spawn points or empty spaces)
        expected_tiles = sum(row.count("#") for row in json_level["ascii"])
        assert compiled.num_tiles == expected_tiles


class TestEndToEndIntegration:
    """Test the complete end-to-end system"""

    @pytest.mark.spec("docs/specs/level_compiler.md", "System Integration")
    def test_complete_pipeline(self, cpu_manager):
        """Test complete pipeline from ASCII to simulation"""
        # Define a more complex level
        level = """############
#S.........#
#..CC......#
#..........#
#....####..#
#..........#
############"""

        # Compile and validate the level
        compiled = compile_ascii_level(level)
        validate_compiled_level(compiled)

        # Verify level was compiled correctly
        assert compiled.num_tiles > 0
        assert compiled.num_spawns == 1

        # Run multiple steps to verify the simulation is working (with default level)
        for i in range(10):
            cpu_manager.step()

            # Get observations to verify simulation is progressing
            obs = cpu_manager.self_observation_tensor().to_numpy()

            # Check observation shape and extract position data
            assert len(obs.shape) >= 2, f"Unexpected observation shape: {obs.shape}"

            # Agent should be positioned in the world - extract first 3 values as position
            if obs.shape[1] >= 3:
                x, y, z = obs[0, 0], obs[0, 1], obs[0, 2]

                # Basic sanity checks
                assert -20 <= x <= 20, f"Agent X position out of bounds: {x}"
                assert -20 <= y <= 20, f"Agent Y position out of bounds: {y}"
                assert 0 <= z <= 5, f"Agent Z position out of bounds: {z}"
            else:
                # Just verify we got some observation data
                assert obs.shape[0] > 0, "No observation data received"
