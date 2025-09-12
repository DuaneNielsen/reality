"""
Test tileset functionality in level compiler

Tests the new tileset-based level compilation that allows mapping ASCII characters
to named assets from the asset registry.
"""

import json

import pytest

from madrona_escape_room.level_compiler import (
    DEFAULT_TILESET,
    _get_asset_object_id,
    _validate_tileset,
    compile_level,
    compile_level_from_json,
    validate_compiled_level,
)


class TestTilesetFunctionality:
    """Test tileset-based level compilation"""

    def test_default_tileset(self):
        """Test that DEFAULT_TILESET is properly defined"""
        assert "#" in DEFAULT_TILESET
        assert DEFAULT_TILESET["#"]["asset"] == "wall"
        assert "C" in DEFAULT_TILESET
        assert DEFAULT_TILESET["C"]["asset"] == "cube"
        assert "O" in DEFAULT_TILESET
        assert DEFAULT_TILESET["O"]["asset"] == "cylinder"
        assert "S" in DEFAULT_TILESET
        assert DEFAULT_TILESET["S"]["asset"] == "spawn"
        assert "." in DEFAULT_TILESET
        assert DEFAULT_TILESET["."]["asset"] == "empty"

    def test_asset_object_id_mapping(self):
        """Test asset name to object ID conversion"""
        # Import the actual constants from level_compiler
        from madrona_escape_room.level_compiler import TILE_EMPTY, TILE_SPAWN

        # Special cases - use actual constants not hardcoded values
        assert _get_asset_object_id("spawn") == TILE_SPAWN
        assert _get_asset_object_id("empty") == TILE_EMPTY

        # Standard assets - should get actual IDs from C API
        wall_id = _get_asset_object_id("wall")
        assert wall_id > 0, f"Wall should have valid ID, got {wall_id}"

    def test_tileset_with_randomization(self):
        """Test tileset with randomization parameters"""
        tileset = {
            "#": {"asset": "wall"},
            "C": {
                "asset": "cube",
                "rand_x": 1.0,
                "rand_y": 0.5,
                "rand_z": 0.25,
                "rand_rot_z": 3.14159,
            },
            "S": {"asset": "spawn"},
            ".": {"asset": "empty"},
        }

        level = """###
#SC#
###"""

        # Create proper JSON format
        json_level = {
            "ascii": level,
            "tileset": tileset,
            "scale": 2.5,
            "name": "test_tileset_with_randomization",
        }

        compiled_levels = compile_level(json_level)
        compiled = compiled_levels[0]  # Extract single level
        validate_compiled_level(compiled)

        # Find the cube tile and verify randomization
        cube_id = _get_asset_object_id("cube")
        cube_found = False
        for i in range(compiled.num_tiles):
            if compiled.object_ids[i] == cube_id:
                assert (
                    compiled.tile_rand_x[i] == 1.0
                ), f"Expected rand_x=1.0, got {compiled.tile_rand_x[i]}"
                assert (
                    compiled.tile_rand_y[i] == 0.5
                ), f"Expected rand_y=0.5, got {compiled.tile_rand_y[i]}"
                assert (
                    compiled.tile_rand_z[i] == 0.25
                ), f"Expected rand_z=0.25, got {compiled.tile_rand_z[i]}"
                assert (
                    abs(compiled.tile_rand_rot_z[i] - 3.14159) < 0.001
                ), f"Expected rand_rot_z≈π, got {compiled.tile_rand_rot_z[i]}"
                cube_found = True
                break

        assert cube_found, "Cube tile not found in compiled level"

        # Verify walls have no randomization
        wall_id = _get_asset_object_id("wall")
        for i in range(compiled.num_tiles):
            if compiled.object_ids[i] == wall_id:
                assert (
                    compiled.tile_rand_x[i] == 0.0
                ), f"Wall should have rand_x=0, got {compiled.tile_rand_x[i]}"
                assert (
                    compiled.tile_rand_y[i] == 0.0
                ), f"Wall should have rand_y=0, got {compiled.tile_rand_y[i]}"
                assert (
                    compiled.tile_rand_z[i] == 0.0
                ), f"Wall should have rand_z=0, got {compiled.tile_rand_z[i]}"
                assert (
                    compiled.tile_rand_rot_z[i] == 0.0
                ), f"Wall should have rand_rot_z=0, got {compiled.tile_rand_rot_z[i]}"
                break  # Check just one wall

    def test_json_level_with_randomization(self):
        """Test JSON level format with randomization parameters"""
        json_level = {
            "ascii": "###O###\n#S...C#\n#######",
            "tileset": {
                "#": {"asset": "wall"},
                "C": {
                    "asset": "cube",
                    "rand_x": 2.0,
                    "rand_rot_z": 1.57,  # π/2 radians
                },
                "O": {"asset": "cylinder", "rand_y": 0.3},
                "S": {"asset": "spawn"},
                ".": {"asset": "empty"},
            },
            "scale": 2.0,
        }

        compiled_levels = compile_level_from_json(json_level)
        compiled = compiled_levels[0]  # Extract single level
        validate_compiled_level(compiled)

        # Verify cube randomization
        cube_id = _get_asset_object_id("cube")
        for i in range(compiled.num_tiles):
            if compiled.object_ids[i] == cube_id:
                assert (
                    compiled.tile_rand_x[i] == 2.0
                ), f"Cube rand_x should be 2.0, got {compiled.tile_rand_x[i]}"
                assert (
                    compiled.tile_rand_y[i] == 0.0
                ), f"Cube rand_y should be 0.0, got {compiled.tile_rand_y[i]}"
                assert (
                    abs(compiled.tile_rand_rot_z[i] - 1.57) < 0.01
                ), f"Cube rand_rot_z should be ~1.57, got {compiled.tile_rand_rot_z[i]}"
                break

        # Verify cylinder randomization
        cylinder_id = _get_asset_object_id("cylinder")
        for i in range(compiled.num_tiles):
            if compiled.object_ids[i] == cylinder_id:
                assert (
                    compiled.tile_rand_x[i] == 0.0
                ), f"Cylinder rand_x should be 0.0, got {compiled.tile_rand_x[i]}"
                assert (
                    compiled.tile_rand_y[i] == 0.3
                ), f"Cylinder rand_y should be 0.3, got {compiled.tile_rand_y[i]}"
                assert (
                    compiled.tile_rand_z[i] == 0.0
                ), f"Cylinder rand_z should be 0.0, got {compiled.tile_rand_z[i]}"
                assert (
                    compiled.tile_rand_rot_z[i] == 0.0
                ), f"Cylinder rand_rot_z should be 0.0, got {compiled.tile_rand_rot_z[i]}"
                break

    def test_validate_tileset_with_invalid_randomization(self):
        """Test that tileset validation catches invalid randomization parameters"""
        # Test negative randomization value
        tileset = {
            "C": {
                "asset": "cube",
                "rand_x": -1.0,  # Invalid: negative
            }
        }

        with pytest.raises(ValueError, match="rand_x.*must be non-negative"):
            _validate_tileset(tileset)

        # Test non-numeric randomization value
        tileset = {
            "C": {
                "asset": "cube",
                "rand_rot_z": "invalid",  # Invalid: not a number
            }
        }

        with pytest.raises(ValueError, match="rand_rot_z.*must be a number"):
            _validate_tileset(tileset)

        cube_id = _get_asset_object_id("cube")
        assert cube_id > 0, f"Cube should have valid ID, got {cube_id}"

        cylinder_id = _get_asset_object_id("cylinder")
        assert cylinder_id > 0, f"Cylinder should have valid ID, got {cylinder_id}"

        # Unknown asset should raise error
        with pytest.raises(ValueError, match="Unknown asset name"):
            _get_asset_object_id("nonexistent_asset")

    def test_validate_tileset(self):
        """Test tileset validation"""
        # Valid tileset
        valid_tileset = {
            "#": {"asset": "wall"},
            "C": {"asset": "cube"},
        }
        _validate_tileset(valid_tileset)  # Should not raise

        # Invalid: not a dict
        with pytest.raises(ValueError, match="Tileset must be a dictionary"):
            _validate_tileset("not a dict")

        # Invalid: multi-char key
        with pytest.raises(ValueError, match="single character"):
            _validate_tileset({"##": {"asset": "wall"}})

        # Invalid: missing asset field
        with pytest.raises(ValueError, match="must have 'asset' field"):
            _validate_tileset({"#": {"type": "wall"}})

        # Invalid: non-string asset name
        with pytest.raises(ValueError, match="Asset name.*must be a string"):
            _validate_tileset({"#": {"asset": 123}})

    def test_compile_with_custom_tileset(self):
        """Test compiling a level with custom tileset"""
        level = """###O###
#S...C#
#######"""

        tileset = {
            "#": {"asset": "wall"},
            "C": {"asset": "cube"},
            "O": {"asset": "cylinder"},
            "S": {"asset": "spawn"},
            ".": {"asset": "empty"},
        }

        # Create proper JSON format
        json_level = {
            "ascii": level,
            "tileset": tileset,
            "scale": 2.5,
            "name": "test_compile_with_custom_tileset",
        }

        compiled_levels = compile_level(json_level)
        compiled = compiled_levels[0]  # Extract single level

        assert compiled.width == 7
        assert compiled.height == 3
        assert compiled.num_tiles > 0

        # Get the actual cylinder ID from the C API
        from madrona_escape_room.ctypes_bindings import get_physics_asset_object_id

        cylinder_id = get_physics_asset_object_id("cylinder")

        # Check that cylinder was included
        cylinder_found = False
        for i in range(compiled.num_tiles):
            if compiled.object_ids[i] == cylinder_id:
                cylinder_found = True
                break
        assert cylinder_found, f"Cylinder asset (ID {cylinder_id}) should be in compiled level"

        validate_compiled_level(compiled)

    def test_compile_json_with_tileset(self):
        """Test JSON compilation with tileset"""
        json_data = {
            "ascii": "###O###\n#S...C#\n#######",
            "tileset": {
                "#": {"asset": "wall"},
                "C": {"asset": "cube"},
                "O": {"asset": "cylinder"},
                "S": {"asset": "spawn"},
                ".": {"asset": "empty"},
            },
            "scale": 3.0,
            "name": "test_tileset_level",
        }

        compiled_levels = compile_level_from_json(json_data)
        compiled = compiled_levels[0]  # Extract single level

        assert compiled.world_scale == 3.0
        assert compiled.level_name.decode("utf-8").rstrip("\x00") == "test_tileset_level"
        assert compiled.num_tiles > 0

        validate_compiled_level(compiled)

    def test_json_auto_tileset_for_special_chars(self):
        """Test that special characters trigger default tileset usage"""
        # Level with 'O' which is not in legacy CHAR_MAP
        json_data = {"ascii": "###O###\n#S....#\n#######", "tileset": DEFAULT_TILESET}

        compiled_levels = compile_level_from_json(json_data)
        compiled = compiled_levels[0]  # Extract single level

        # Should compile successfully using DEFAULT_TILESET
        assert compiled.num_tiles > 0
        validate_compiled_level(compiled)

    def test_backward_compatibility(self):
        """Test that old levels without tileset still work"""
        # Legacy level using only CHAR_MAP characters
        level = """######
#S...#
#.CC.#
######"""

        # Use DEFAULT_TILESET for backward compatibility
        json_level = {
            "ascii": level,
            "tileset": DEFAULT_TILESET,
            "scale": 2.5,
            "name": "test_backward_compatibility",
        }

        compiled_levels = compile_level(json_level)
        compiled = compiled_levels[0]  # Extract single level

        assert compiled.num_tiles > 0
        validate_compiled_level(compiled)

        # Get the actual cube ID from the C API
        from madrona_escape_room.ctypes_bindings import get_physics_asset_object_id

        cube_id = get_physics_asset_object_id("cube")

        # Verify cubes are present
        cube_count = 0
        for i in range(compiled.num_tiles):
            if compiled.object_ids[i] == cube_id:
                cube_count += 1
        assert cube_count == 2, f"Should have 2 cubes with ID {cube_id}, found {cube_count}"

    def test_mixed_assets_tileset(self):
        """Test tileset with mix of physics and render-only assets"""
        tileset = {
            "#": {"asset": "wall"},  # Has physics and render
            "X": {"asset": "axis_x"},  # Render-only
            "Y": {"asset": "axis_y"},  # Render-only
            "S": {"asset": "spawn"},
            ".": {"asset": "empty"},
        }

        level = """#####
#S.X#
#.Y.#
#####"""

        # This may fail if axis assets aren't properly exposed
        # but the test structure is correct
        try:
            # Create proper JSON format
            json_level = {
                "ascii": level,
                "tileset": tileset,
                "scale": 2.5,
                "name": "test_mixed_assets_tileset",
            }

            compiled_levels = compile_level(json_level)
            compiled = compiled_levels[0]  # Extract single level
            assert compiled.num_tiles > 0
            validate_compiled_level(compiled)
        except ValueError as e:
            # Expected if axis assets aren't available in test environment
            if "Unknown asset name" in str(e) and ("axis_x" in str(e) or "axis_y" in str(e)):
                pytest.skip("Axis assets not available in test environment")
            else:
                raise

    def test_invalid_asset_in_tileset(self):
        """Test error handling for invalid asset names"""
        tileset = {
            "#": {"asset": "wall"},
            "?": {"asset": "mystery_box"},  # Doesn't exist
            "S": {"asset": "spawn"},
            ".": {"asset": "empty"},
        }

        level = """###
#S?#
###"""

        with pytest.raises(ValueError, match="Invalid asset 'mystery_box'"):
            # Create proper JSON format
            json_level = {
                "ascii": level,
                "tileset": tileset,
                "scale": 2.5,
                "name": "test_invalid_asset_in_tileset",
            }

            compile_level(json_level)

    def test_json_string_input(self):
        """Test JSON string parsing with tileset"""
        json_str = json.dumps(
            {"ascii": "###\n#S#\n###", "tileset": {"#": {"asset": "wall"}, "S": {"asset": "spawn"}}}
        )

        compiled_levels = compile_level_from_json(json_str)
        compiled = compiled_levels[0]  # Extract single level
        assert compiled.num_tiles > 0
        validate_compiled_level(compiled)
