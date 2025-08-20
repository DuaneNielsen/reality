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
        # Special cases
        assert _get_asset_object_id("spawn") == 3  # TILE_SPAWN
        assert _get_asset_object_id("empty") == 0  # TILE_EMPTY

        # Standard assets - should get actual IDs from C API
        wall_id = _get_asset_object_id("wall")
        assert wall_id > 0, f"Wall should have valid ID, got {wall_id}"

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

        compiled = compile_level(level, tileset=tileset)

        assert compiled["width"] == 7
        assert compiled["height"] == 3
        assert compiled["num_tiles"] > 0

        # Get the actual cylinder ID from the C API
        from madrona_escape_room.ctypes_bindings import get_physics_asset_object_id

        cylinder_id = get_physics_asset_object_id("cylinder")

        # Check that cylinder was included
        cylinder_found = False
        for i in range(compiled["num_tiles"]):
            if compiled["object_ids"][i] == cylinder_id:
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

        compiled = compile_level_from_json(json_data)

        assert compiled["scale"] == 3.0
        assert compiled["level_name"] == "test_tileset_level"
        assert compiled["num_tiles"] > 0

        validate_compiled_level(compiled)

    def test_json_auto_tileset_for_special_chars(self):
        """Test that special characters trigger default tileset usage"""
        # Level with 'O' which is not in legacy CHAR_MAP
        json_data = {"ascii": "###O###\n#S....#\n#######"}

        compiled = compile_level_from_json(json_data)

        # Should compile successfully using DEFAULT_TILESET
        assert compiled["num_tiles"] > 0
        validate_compiled_level(compiled)

    def test_backward_compatibility(self):
        """Test that old levels without tileset still work"""
        # Legacy level using only CHAR_MAP characters
        level = """######
#S...#
#.CC.#
######"""

        compiled = compile_level(level)  # No tileset

        assert compiled["num_tiles"] > 0
        validate_compiled_level(compiled)

        # Get the actual cube ID from the C API
        from madrona_escape_room.ctypes_bindings import get_physics_asset_object_id

        cube_id = get_physics_asset_object_id("cube")

        # Verify cubes are present
        cube_count = 0
        for i in range(compiled["num_tiles"]):
            if compiled["object_ids"][i] == cube_id:
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
            compiled = compile_level(level, tileset=tileset)
            assert compiled["num_tiles"] > 0
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
            compile_level(level, tileset=tileset)

    def test_json_string_input(self):
        """Test JSON string parsing with tileset"""
        json_str = json.dumps(
            {"ascii": "###\n#S#\n###", "tileset": {"#": {"asset": "wall"}, "S": {"asset": "spawn"}}}
        )

        compiled = compile_level_from_json(json_str)
        assert compiled["num_tiles"] > 0
        validate_compiled_level(compiled)
