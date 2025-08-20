"""
Test asset descriptors C API functions.
"""

import pytest

from madrona_escape_room.ctypes_bindings import (
    get_physics_asset_object_id,
    get_physics_assets_list,
    get_render_asset_object_id,
    get_render_assets_list,
)


class TestAssetDescriptorAPI:
    """Test C API functions for asset descriptors."""

    def test_get_physics_assets_list(self):
        """Test getting list of physics asset names."""
        assets = get_physics_assets_list()

        # Dynamically check that we have physics assets
        assert len(assets) > 0, "Should have at least one physics asset"

        # Check that core required assets are present
        required_assets = ["cube", "wall", "agent", "plane"]
        for asset in required_assets:
            assert asset in assets, f"Required physics asset '{asset}' not found"

        # The system may have additional assets (like cylinder), which is fine
        print(f"Found {len(assets)} physics assets: {assets}")

    def test_get_render_assets_list(self):
        """Test getting list of render asset names."""
        assets = get_render_assets_list()

        # Dynamically check that we have render assets
        assert len(assets) > 0, "Should have at least one render asset"

        # Check that core required assets are present
        required_assets = ["cube", "wall", "agent", "plane"]
        for asset in required_assets:
            assert asset in assets, f"Required render asset '{asset}' not found"

        # Check that axis visualization assets are present
        axis_assets = ["axis_x", "axis_y", "axis_z"]
        for asset in axis_assets:
            assert asset in assets, f"Axis render asset '{asset}' not found"

        # The system may have additional assets (like cylinder), which is fine
        print(f"Found {len(assets)} render assets: {assets}")

    def test_physics_asset_object_id_lookup(self):
        """Test looking up physics asset object IDs by name."""
        # Get all physics assets dynamically
        assets = get_physics_assets_list()

        # Test that all listed assets have valid IDs
        asset_ids = {}
        for asset_name in assets:
            asset_id = get_physics_asset_object_id(asset_name)
            assert asset_id >= 0, f"Physics asset '{asset_name}' should have valid object ID"
            asset_ids[asset_name] = asset_id

        # Object IDs should be unique
        unique_ids = set(asset_ids.values())
        assert len(unique_ids) == len(asset_ids), "Object IDs should be unique"

        # Test core assets specifically
        assert get_physics_asset_object_id("cube") >= 0, "Cube should have valid ID"
        assert get_physics_asset_object_id("wall") >= 0, "Wall should have valid ID"
        assert get_physics_asset_object_id("agent") >= 0, "Agent should have valid ID"
        assert get_physics_asset_object_id("plane") >= 0, "Plane should have valid ID"

        # Test invalid name
        invalid_id = get_physics_asset_object_id("nonexistent")
        assert invalid_id == -1, "Invalid name should return -1"

    def test_render_asset_object_id_lookup(self):
        """Test looking up render asset object IDs by name."""
        # Test valid names
        cube_id = get_render_asset_object_id("cube")
        assert cube_id >= 0, "Cube render asset should have valid object ID"

        wall_id = get_render_asset_object_id("wall")
        assert wall_id >= 0, "Wall render asset should have valid object ID"

        agent_id = get_render_asset_object_id("agent")
        assert agent_id >= 0, "Agent render asset should have valid object ID"

        plane_id = get_render_asset_object_id("plane")
        assert plane_id >= 0, "Plane render asset should have valid object ID"

        axis_x_id = get_render_asset_object_id("axis_x")
        assert axis_x_id >= 0, "X-axis render asset should have valid object ID"

        axis_y_id = get_render_asset_object_id("axis_y")
        assert axis_y_id >= 0, "Y-axis render asset should have valid object ID"

        axis_z_id = get_render_asset_object_id("axis_z")
        assert axis_z_id >= 0, "Z-axis render asset should have valid object ID"

        # Test invalid name
        invalid_id = get_render_asset_object_id("nonexistent")
        assert invalid_id == -1, "Invalid name should return -1"

    def test_physics_render_consistency(self):
        """Test that physics and render assets with same names have same object IDs."""
        # These assets exist in both physics and render
        common_assets = ["cube", "wall", "agent", "plane"]

        for asset_name in common_assets:
            physics_id = get_physics_asset_object_id(asset_name)
            render_id = get_render_asset_object_id(asset_name)

            assert physics_id == render_id, (
                f"Asset '{asset_name}' should have same object ID in physics ({physics_id}) "
                f"and render ({render_id})"
            )

    def test_case_sensitivity(self):
        """Test that asset name lookups are case-sensitive."""
        # Should match exact case
        assert get_physics_asset_object_id("cube") >= 0
        assert get_physics_asset_object_id("Cube") == -1
        assert get_physics_asset_object_id("CUBE") == -1

        assert get_render_asset_object_id("agent") >= 0
        assert get_render_asset_object_id("Agent") == -1
        assert get_render_asset_object_id("AGENT") == -1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
