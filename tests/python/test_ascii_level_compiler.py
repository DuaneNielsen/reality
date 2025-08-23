"""
Test ASCII level compiler integration with pytest framework

Tests the complete pipeline from ASCII art to C API struct population.
"""

import pytest

from madrona_escape_room.level_compiler import (
    MAX_TILES_C_API,
    compile_level,
    validate_compiled_level,
)


class TestLevelCompiler:
    """Test the level_compiler module"""

    def test_simple_room_compilation(self):
        """Test compiling a simple room"""
        level = """######
#S...#
#....#
######"""

        compiled = compile_level(level)

        assert compiled["width"] == 6
        assert compiled["height"] == 4
        assert compiled["scale"] == 2.5
        assert compiled["num_tiles"] > 0
        assert compiled["max_entities"] > compiled["num_tiles"]
        assert len(compiled["object_ids"]) == MAX_TILES_C_API
        assert len(compiled["tile_x"]) == MAX_TILES_C_API
        assert len(compiled["tile_y"]) == MAX_TILES_C_API

        # Verify spawn point was found
        assert "_spawn_points" in compiled
        assert len(compiled["_spawn_points"]) == 1

        validate_compiled_level(compiled)

    def test_level_with_obstacles(self):
        """Test compiling a level with cube obstacles"""
        level = """########
#S.....#
#..CC..#
#......#
########"""

        compiled = compile_level(level)

        # Get the actual object IDs from the C API
        from madrona_escape_room.ctypes_bindings import get_physics_asset_object_id

        wall_id = get_physics_asset_object_id("wall")
        cube_id = get_physics_asset_object_id("cube")

        # Should have walls + cubes
        wall_count = sum(
            1 for i in range(compiled["num_tiles"]) if compiled["object_ids"][i] == wall_id
        )
        cube_count = sum(
            1 for i in range(compiled["num_tiles"]) if compiled["object_ids"][i] == cube_id
        )

        assert wall_count > 0, "Should have wall tiles"
        assert cube_count == 2, f"Should have 2 cubes, got {cube_count}"
        assert compiled["num_tiles"] == wall_count + cube_count

        validate_compiled_level(compiled)

    def test_error_cases(self):
        """Test error handling for invalid levels"""

        # No spawn point
        with pytest.raises(ValueError, match="No spawn points"):
            compile_level("""####
#..#
####""")

        # Empty level
        with pytest.raises(ValueError, match="Empty level"):
            compile_level("")

        # Unknown character
        with pytest.raises(ValueError, match="Unknown character"):
            compile_level("""####
#SX#
####""")


class TestCTypesIntegration:
    """Test integration with ctypes bindings"""

    def test_dict_to_struct_conversion(self):
        """Test converting compiled dict to ctypes struct"""
        level = """####
#S.#
####"""

        compiled = compile_level(level)

        from madrona_escape_room.ctypes_bindings import dict_to_compiled_level

        struct = dict_to_compiled_level(compiled)

        # Verify all fields copied correctly
        assert struct.num_tiles == compiled["num_tiles"]
        assert struct.max_entities == compiled["max_entities"]
        assert struct.width == compiled["width"]
        assert struct.height == compiled["height"]
        assert abs(struct.world_scale - compiled["scale"]) < 0.001

        # Verify arrays copied correctly
        for i in range(compiled["num_tiles"]):
            assert struct.object_ids[i] == compiled["object_ids"][i]
            assert abs(struct.tile_x[i] - compiled["tile_x"][i]) < 0.001
            assert abs(struct.tile_y[i] - compiled["tile_y"][i]) < 0.001

        # Verify zero padding
        for i in range(compiled["num_tiles"], MAX_TILES_C_API):
            assert struct.object_ids[i] == 0
            assert abs(struct.tile_x[i]) < 0.001
            assert abs(struct.tile_y[i]) < 0.001

    def test_c_api_validation(self):
        """Test C API validation function"""
        level = """######
#S...#
#....#
######"""

        compiled = compile_level(level)

        from madrona_escape_room.ctypes_bindings import (
            dict_to_compiled_level,
            validate_compiled_level_ctypes,
        )

        struct = dict_to_compiled_level(compiled)

        # Should not raise any exception
        validate_compiled_level_ctypes(struct)

    def test_array_creation(self):
        """Test creating ctypes arrays for multiple worlds"""
        level1 = """####
#S.#
####"""

        level2 = """######
#S...#
######"""

        compiled1 = compile_level(level1)
        compiled2 = compile_level(level2)

        from madrona_escape_room.ctypes_bindings import create_compiled_levels_array

        # Test single level
        array, count = create_compiled_levels_array([compiled1])
        assert count == 1
        assert array[0].num_tiles == compiled1["num_tiles"]

        # Test multiple levels
        array, count = create_compiled_levels_array([compiled1, compiled2])
        assert count == 2
        assert array[0].num_tiles == compiled1["num_tiles"]
        assert array[1].num_tiles == compiled2["num_tiles"]

        # Test empty array
        array, count = create_compiled_levels_array([])
        assert array is None
        assert count == 0


class TestManagerIntegration:
    """Test integration with SimManager (requires built C API)"""

    def test_manager_creation_with_ascii_level(self):
        """Test creating manager with ASCII level"""
        level = """########
#S.....#
#......#
########"""

        # This will test the full integration through SimManager
        from madrona_escape_room import SimManager, madrona

        mgr = SimManager(
            exec_mode=madrona.ExecMode.CPU,
            gpu_id=0,
            num_worlds=1,
            rand_seed=42,
            auto_reset=True,
            level_ascii=level,
        )

        # Test that simulation runs
        for _ in range(3):
            mgr.step()

        # Clean up handled by __del__

    def test_multiple_worlds_same_level(self):
        """Test multiple worlds with same ASCII level"""
        level = """######
#S...#
#....#
######"""

        from madrona_escape_room import SimManager, madrona

        mgr = SimManager(
            exec_mode=madrona.ExecMode.CPU,
            gpu_id=0,
            num_worlds=2,  # Multiple worlds
            rand_seed=42,
            auto_reset=True,
            level_ascii=level,
        )

        # Test simulation runs with multiple worlds
        for _ in range(2):
            mgr.step()

    def test_backward_compatibility(self):
        """Test that SimManager still works without level_ascii"""
        from madrona_escape_room import SimManager, madrona

        # Should work without level_ascii (uses default level)
        mgr = SimManager(
            exec_mode=madrona.ExecMode.CPU,
            gpu_id=0,
            num_worlds=1,
            rand_seed=42,
            auto_reset=True,
            # No level_ascii parameter
        )

        # Should still run
        mgr.step()


class TestJSONLevelFormat:
    """Test JSON level format with parameters"""

    def test_json_level_with_agent_facing(self):
        """Test JSON level with agent facing angles"""
        import math

        from madrona_escape_room.level_compiler import compile_level_from_json

        json_level = {
            "ascii": """######
#S..S#
######""",
            "scale": 1.5,
            "agent_facing": [0.0, math.pi / 2],  # First faces forward, second faces right
        }

        compiled = compile_level_from_json(json_level)

        # Verify structure
        assert compiled["scale"] == 1.5
        assert compiled["num_spawns"] == 2

        # Check agent facing was set correctly
        assert compiled["spawn_facing"][0] == 0.0
        assert abs(compiled["spawn_facing"][1] - math.pi / 2) < 0.001

        # Remaining spawn slots should be 0
        for i in range(2, 8):
            assert compiled["spawn_facing"][i] == 0.0

    def test_json_level_without_facing(self):
        """Test JSON level without agent_facing defaults to 0"""
        from madrona_escape_room.level_compiler import compile_level_from_json

        json_level = {
            "ascii": """####
#S.#
####""",
            "scale": 2.0,
        }

        compiled = compile_level_from_json(json_level)

        # Should default to facing forward (0.0)
        assert compiled["spawn_facing"][0] == 0.0

    def test_json_level_in_sim_manager(self):
        """Test using JSON level with SimManager"""
        import math

        from madrona_escape_room import SimManager, madrona
        from madrona_escape_room.level_compiler import compile_level_from_json

        json_level = {
            "ascii": """########
#S.....#
#......#
########""",
            "scale": 2.0,
            "agent_facing": [math.pi / 4],  # Face 45 degrees
        }

        # For now, we just compile and validate the level
        # SimManager doesn't yet support pre-compiled levels directly
        compiled = compile_level_from_json(json_level)

        # Verify the compiled level has the correct facing
        assert abs(compiled["spawn_facing"][0] - math.pi / 4) < 0.001

        # Test that we can use the ASCII part with SimManager
        mgr = SimManager(
            exec_mode=madrona.ExecMode.CPU,
            gpu_id=0,
            num_worlds=1,
            rand_seed=42,
            auto_reset=True,
            level_ascii=json_level["ascii"],  # Just use the ASCII for now
        )

        # Test simulation runs
        for _ in range(3):
            mgr.step()


class TestEndToEndIntegration:
    """Test the complete end-to-end system"""

    def test_complete_pipeline(self):
        """Test complete pipeline from ASCII to simulation"""
        # Define a more complex level
        level = """############
#S.........#
#..CC......#
#..........#
#....####..#
#..........#
############"""

        from madrona_escape_room import SimManager, madrona

        # Create manager with custom level
        mgr = SimManager(
            exec_mode=madrona.ExecMode.CPU,
            gpu_id=0,
            num_worlds=1,
            rand_seed=42,
            auto_reset=True,
            level_ascii=level,
        )

        # Run multiple steps to verify the level is working
        for i in range(10):
            mgr.step()

            # Get observations to verify simulation is progressing
            obs = mgr.self_observation_tensor().to_numpy()

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
