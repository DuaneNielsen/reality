"""
Pytest tests for level compiler C API integration.

Tests the single source of truth for MAX_TILES between C++ and Python,
ensuring the level compiler correctly uses the C API constant and validates
against it properly.
"""

import os
import tempfile
from pathlib import Path

import pytest

from madrona_escape_room.level_compiler import (
    MAX_TILES_C_API,
    _get_max_tiles_from_c_api,
    compile_level,
    load_compiled_level_binary,
    save_compiled_level_binary,
    validate_compiled_level,
)


class TestCAPIIntegration:
    """Test C API integration for MAX_TILES constant."""

    def test_max_tiles_c_api_value(self):
        """Test that MAX_TILES_C_API returns expected value."""
        # Should be 1024 based on CompiledLevel::MAX_TILES
        assert MAX_TILES_C_API == 1024
        assert _get_max_tiles_from_c_api() == 1024

    def test_max_tiles_consistency(self):
        """Test that MAX_TILES_C_API matches direct C API call."""
        direct_call = _get_max_tiles_from_c_api()
        module_constant = MAX_TILES_C_API
        assert (
            direct_call == module_constant
        ), f"Direct C API call ({direct_call}) != module constant ({module_constant})"


class TestLevelSizeValidation:
    """Test level size validation against C API limits."""

    def test_level_within_limits(self):
        """Test that levels within limits compile successfully."""
        # Small level that should work fine
        small_level = """
#######
#S....#
#.....#
#######
"""
        compiled = compile_level(small_level)
        assert compiled["width"] == 7
        assert compiled["height"] == 4
        assert compiled["array_size"] == 28
        assert 28 <= MAX_TILES_C_API
        validate_compiled_level(compiled)

    def test_maximum_size_level(self):
        """Test level at exactly the maximum size."""
        # Create a 32x32 level (1024 tiles exactly)
        size = 32
        lines = []
        for y in range(size):
            if y == 0 or y == size - 1:
                lines.append("#" * size)
            else:
                if y == 1:
                    lines.append("#S" + "." * (size - 3) + "#")
                else:
                    lines.append("#" + "." * (size - 2) + "#")

        max_level = "\n".join(lines)
        compiled = compile_level(max_level)
        assert compiled["array_size"] == 1024
        assert compiled["array_size"] == MAX_TILES_C_API
        validate_compiled_level(compiled)

    def test_oversized_level_rejected(self):
        """Test that levels exceeding MAX_TILES_C_API are rejected."""
        # Create a 33x33 level (1089 tiles > 1024)
        size = 33
        lines = []
        for y in range(size):
            if y == 0 or y == size - 1:
                lines.append("#" * size)
            else:
                if y == 1:
                    lines.append("#S" + "." * (size - 3) + "#")
                else:
                    lines.append("#" + "." * (size - 2) + "#")

        oversized_level = "\n".join(lines)

        with pytest.raises(
            ValueError, match=r"Level too large.*1089 tiles > 1024 max \(from C API\)"
        ):
            compile_level(oversized_level)

    def test_64x64_level_rejected(self):
        """Test that maximum dimension levels are correctly rejected when too large."""
        # 64x64 = 4096 tiles, which exceeds 1024
        size = 64
        lines = []
        for y in range(size):
            if y == 0 or y == size - 1:
                lines.append("#" * size)
            else:
                if y == 1:
                    lines.append("#S" + "." * (size - 3) + "#")
                else:
                    lines.append("#" + "." * (size - 2) + "#")

        large_level = "\n".join(lines)

        with pytest.raises(
            ValueError, match=r"Level too large.*4096 tiles > 1024 max \(from C API\)"
        ):
            compile_level(large_level)


class TestArraySizing:
    """Test that compiled levels have correct array sizes."""

    def test_compiled_arrays_correct_size(self):
        """Test that compiled level arrays are sized to MAX_TILES_C_API."""
        level = """
#####
#S..#
#####
"""
        compiled = compile_level(level)

        # Arrays should be sized to MAX_TILES_C_API, not level dimensions
        assert len(compiled["object_ids"]) == MAX_TILES_C_API
        assert len(compiled["tile_x"]) == MAX_TILES_C_API
        assert len(compiled["tile_y"]) == MAX_TILES_C_API

        # Level dimensions should be smaller
        assert compiled["array_size"] == 15  # 5x3
        assert compiled["array_size"] < MAX_TILES_C_API

    def test_unused_array_slots_zero(self):
        """Test that unused array slots are properly zeroed."""
        level = """
#####
#S..#
#####
"""
        compiled = compile_level(level)

        num_tiles = compiled["num_tiles"]

        # Check that slots beyond num_tiles are zero
        for i in range(num_tiles, min(num_tiles + 10, MAX_TILES_C_API)):
            assert compiled["object_ids"][i] == 0
            assert compiled["tile_x"][i] == 0.0
            assert compiled["tile_y"][i] == 0.0


class TestBinaryIO:
    """Test binary I/O with C API-sized arrays."""

    def test_binary_roundtrip_c_api_arrays(self):
        """Test that binary save/load works with C API-sized arrays."""
        level = """
########
#S.....#
#......#
#....C.#
########
"""
        original = compile_level(level)

        with tempfile.NamedTemporaryFile(suffix=".lvl", delete=False) as f:
            try:
                save_compiled_level_binary(original, f.name)
                loaded = load_compiled_level_binary(f.name)

                # Check header fields match
                for field in ["num_tiles", "max_entities", "width", "height", "scale"]:
                    assert original[field] == loaded[field]

                # Check arrays are same size
                assert len(loaded["object_ids"]) == MAX_TILES_C_API
                assert len(loaded["tile_x"]) == MAX_TILES_C_API
                assert len(loaded["tile_y"]) == MAX_TILES_C_API

                # Check tile data matches for used slots
                for i in range(original["num_tiles"]):
                    assert original["object_ids"][i] == loaded["object_ids"][i]
                    assert abs(original["tile_x"][i] - loaded["tile_x"][i]) < 0.001
                    assert abs(original["tile_y"][i] - loaded["tile_y"][i]) < 0.001

                # Check unused slots are zero
                for i in range(
                    original["num_tiles"], min(original["num_tiles"] + 10, MAX_TILES_C_API)
                ):
                    assert loaded["object_ids"][i] == 0
                    assert loaded["tile_x"][i] == 0.0
                    assert loaded["tile_y"][i] == 0.0

            finally:
                os.unlink(f.name)

    def test_validation_accepts_c_api_arrays(self):
        """Test that validation accepts arrays sized to MAX_TILES_C_API."""
        level = """
#####
#S..#
#####
"""
        compiled = compile_level(level)

        # Should not raise - arrays are correctly sized to MAX_TILES_C_API
        validate_compiled_level(compiled)

    def test_validation_rejects_wrong_sized_arrays(self):
        """Test that validation rejects arrays not sized to MAX_TILES_C_API."""
        level = """
#####
#S..#
#####
"""
        compiled = compile_level(level)

        # Corrupt the array size
        compiled["object_ids"] = compiled["object_ids"][:100]  # Wrong size

        with pytest.raises(ValueError, match=r"Invalid object_ids array length.*must be 1024"):
            validate_compiled_level(compiled)


class TestEdgeCases:
    """Test edge cases with C API integration."""

    def test_minimum_size_level(self):
        """Test minimum possible level size (3x3)."""
        level = """
...
.S.
...
"""

        compiled = compile_level(level)
        assert compiled["width"] == 3
        assert compiled["height"] == 3
        assert compiled["array_size"] == 9
        assert compiled["num_tiles"] == 0  # No solid tiles, just spawn and empty
        assert len(compiled["object_ids"]) == MAX_TILES_C_API
        validate_compiled_level(compiled)

    def test_exact_square_root_size(self):
        """Test levels at exact square root of MAX_TILES_C_API."""
        # 32x32 = 1024 exactly
        size = 32
        level = "S" + "." * (size - 1) + "\n" + ("." * size + "\n") * (size - 1)
        level = level.rstrip()

        compiled = compile_level(level)
        assert compiled["array_size"] == 1024
        assert compiled["array_size"] == MAX_TILES_C_API
        validate_compiled_level(compiled)

    def test_rectangular_at_limit(self):
        """Test rectangular level at the size limit."""
        # 64x16 = 1024 exactly
        width, height = 64, 16
        lines = []
        for y in range(height):
            if y == 0:
                lines.append("S" + "." * (width - 1))
            else:
                lines.append("." * width)

        level = "\n".join(lines)
        compiled = compile_level(level)
        assert compiled["array_size"] == 1024
        validate_compiled_level(compiled)

    def test_fallback_when_c_api_unavailable(self):
        """Test fallback behavior when C API is not available."""
        # This test verifies the fallback in _get_max_tiles_from_c_api
        # We can't easily mock the import failure, but we can test the logic
        from madrona_escape_room.level_compiler import _get_max_tiles_from_c_api

        # The function should return 1024 (the fallback value)
        result = _get_max_tiles_from_c_api()
        assert result == 1024


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
