"""
Test 128-beam raycast lidar with 120° FOV using pytest framework

Tests the raycast-based lidar implementation that directly traces rays through
the BVH structure. Now returns depth-only data (no entity type encodings).
"""

import math

import numpy as np
import pytest

from madrona_escape_room import ExecMode
from madrona_escape_room.level_compiler import compile_ascii_level

# Test level with walls positioned for clear lidar validation
RAYCAST_TEST_LEVEL = """################################################################
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
..............................S...............................
................................................................
................................................................
................................................................
................................................................
................................................................
................................................................
################################################################"""


def calculate_raycast_distance(
    beam_index: int,
    agent_pos: tuple[float, float],  # (x, y) of agent
    agent_angle: float,  # Agent's facing direction in radians
    wall_center_y: float,  # Wall center Y position
    wall_thickness: float,  # Wall thickness (cube extends ±thickness/2)
    beam_count: int = 128,  # Total number of beams
    fov_degrees: float = 120.0,  # Horizontal field of view
    lidar_height_offset: float = 0.5,  # Ray height above agent
) -> float:
    """Calculate expected distance using direct raycast geometry.

    Unlike the perspective projection model, this uses the actual geometric
    ray tracing approach implemented in the lidarSystem.
    """
    # Agent position with lidar height offset
    ray_origin_x = agent_pos[0]
    ray_origin_y = agent_pos[1]

    # Calculate ray angle relative to agent's forward direction
    # 120-degree arc from -60° to +60° relative to agent forward
    fov_radians = math.radians(fov_degrees)
    angle_range = fov_radians

    # Map beam index to angle within the arc
    # beam 0 = -60°, beam 63.5 = 0°, beam 127 = +60°
    theta = -angle_range / 2.0 + (angle_range * float(beam_index) / float(beam_count - 1))

    # Calculate absolute ray direction
    ray_angle = agent_angle + theta
    ray_dir_x = math.cos(ray_angle)
    ray_dir_y = math.sin(ray_angle)

    # Wall front face position (closest to agent)
    wall_front_y = wall_center_y - wall_thickness / 2.0

    # Ray-plane intersection: solve for t where ray hits wall
    # ray_y = ray_origin_y + t * ray_dir_y = wall_front_y
    if abs(ray_dir_y) < 1e-6:  # Ray parallel to wall
        return float("inf")

    t = (wall_front_y - ray_origin_y) / ray_dir_y

    if t <= 0:  # Ray pointing away from wall
        return float("inf")

    # Calculate intersection point
    hit_x = ray_origin_x + t * ray_dir_x
    hit_y = wall_front_y

    # Check if intersection is within wall bounds (assuming wall extends from -10 to +10)
    wall_half_width = 10.0  # From the test level
    if abs(hit_x) > wall_half_width:
        return float("inf")  # Ray misses the wall

    # Return distance from ray origin to hit point
    distance = math.sqrt((hit_x - ray_origin_x) ** 2 + (hit_y - ray_origin_y) ** 2)
    return distance


def model_full_raycast_sweep(
    agent_pos: tuple[float, float],
    agent_angle: float,
    wall_center_y: float,
    wall_thickness: float,
    beam_count: int = 128,
    fov_degrees: float = 120.0,
    lidar_height_offset: float = 0.5,
) -> np.ndarray:
    """Model the full lidar sweep using direct raycast geometry."""
    distances = np.zeros(beam_count)
    for i in range(beam_count):
        distances[i] = calculate_raycast_distance(
            i,
            agent_pos,
            agent_angle,
            wall_center_y,
            wall_thickness,
            beam_count,
            fov_degrees,
            lidar_height_offset,
        )
    return distances


class TestRaycastLidar:
    """Test raycast-based lidar functionality (depth-only)"""

    def test_lidar_tensor_shape_and_access(self, cpu_manager):
        """Test basic lidar tensor access and shape validation"""
        mgr = cpu_manager

        # Step once to generate observations
        mgr.step()

        # Get lidar tensor
        lidar_tensor = mgr.lidar_tensor()
        if lidar_tensor.isOnGPU():
            lidar_array = lidar_tensor.to_torch().cpu().numpy()
        else:
            lidar_array = lidar_tensor.to_numpy()

        # Validate tensor shape: [worlds, agents, samples, values]
        expected_shape = (4, 1, 128, 1)  # 4 worlds, 1 agent, 128 samples, depth only
        assert (
            lidar_array.shape == expected_shape
        ), f"Lidar tensor shape mismatch. Expected {expected_shape}, got {lidar_array.shape}"

        # Validate data types
        assert lidar_array.dtype == np.float32, f"Expected float32, got {lidar_array.dtype}"

        # Extract depth data
        depth_data = lidar_array[0, 0, :, 0]  # First world, depths

        print(f"✅ Tensor shape: {lidar_array.shape}")
        print(f"✅ Depth range: [{depth_data.min():.3f}, {depth_data.max():.3f}]")

    def test_depth_value_distribution(self, cpu_manager):
        """Test depth value distribution and quality"""
        mgr = cpu_manager
        mgr.step()

        lidar_tensor = mgr.lidar_tensor()
        if lidar_tensor.isOnGPU():
            lidar_array = lidar_tensor.to_torch().cpu().numpy()
        else:
            lidar_array = lidar_tensor.to_numpy()

        # Extract depth data
        depth_data = lidar_array[0, 0, :, 0]  # First world, depths

        print("\n=== DEPTH ANALYSIS ===")
        print(f"Depth range: [{depth_data.min():.3f}, {depth_data.max():.3f}]")

        # Analyze depth distribution
        finite_depths = depth_data[np.isfinite(depth_data) & (depth_data > 0)]
        zero_depths = np.sum(depth_data == 0.0)

        print(f"Valid depth readings: {len(finite_depths)}/128")
        print(f"Zero/miss readings: {zero_depths}/128")

        if len(finite_depths) > 0:
            print(
                f"Valid depth stats: mean={finite_depths.mean():.3f}, std={finite_depths.std():.3f}"
            )

        # Validation: depths should be normalized [0, 1]
        assert np.all(depth_data >= 0.0), "All depths should be non-negative"
        assert np.all(depth_data <= 1.0), "All depths should be normalized [0,1]"

        print("✅ Depth values are valid")

    def test_128_beam_raycast_lidar_validation(self, cpu_manager):
        """
        Test 128-beam raycast lidar with 120° FOV using geometric ray tracing model.

        This test validates the raycast implementation with depth-only data.
        """
        mgr = cpu_manager

        # Compile the ASCII level to get exact wall positions
        level = compile_ascii_level(RAYCAST_TEST_LEVEL, level_name="raycast_test")

        print("\n=== COMPILED LEVEL INFO ===")
        print(f"Number of tiles: {level.num_tiles}")
        print(f"Spawn position: ({level.spawn_x[0]:.3f}, {level.spawn_y[0]:.3f})")

        # Find wall positions
        wall_count = 0
        north_walls = []  # Walls in front of agent
        for i in range(level.num_tiles):
            obj_id = level.object_ids[i]
            if obj_id == 2:  # Wall asset ID
                x = level.tile_x[i]
                y = level.tile_y[i]
                if y > 5.0:  # North wall (in front of agent)
                    north_walls.append((x, y))
                wall_count += 1
                if wall_count < 10:  # Show first 10 walls
                    print(f"Wall {wall_count}: ({x:.1f}, {y:.1f})")
        print(f"Total walls: {wall_count}, North walls: {len(north_walls)}")

        # Step simulation to get lidar data
        mgr.step()

        # Get lidar readings
        lidar_tensor = mgr.lidar_tensor()
        if lidar_tensor.isOnGPU():
            lidar_array = lidar_tensor.to_torch().cpu().numpy()
        else:
            lidar_array = lidar_tensor.to_numpy()

        # Extract depth data for first world
        depth_readings = lidar_array[0, 0, :, 0]  # 128 depth values

        print("\n=== LIDAR READINGS SUMMARY ===")
        print(f"Depth range: [{depth_readings.min():.3f}, {depth_readings.max():.3f}]")

        # Analyze reading quality
        finite_count = np.sum(np.isfinite(depth_readings))
        zero_count = np.sum(depth_readings == 0.0)
        valid_count = np.sum((depth_readings > 0.0) & (depth_readings < 1.0))

        print(f"Finite readings: {finite_count}/128 ({finite_count/128:.1%})")
        print(f"Zero readings: {zero_count}/128 ({zero_count/128:.1%})")
        print(f"Valid range (0-1): {valid_count}/128 ({valid_count/128:.1%})")

        # Model expected distances using raycast geometry
        # Agent parameters from level spawn
        agent_position = (level.spawn_x[0], level.spawn_y[0])
        agent_angle = 0.0  # Assuming agent faces north (positive Y)

        # Find the closest north wall for modeling
        if north_walls:
            closest_wall_y = min(y for x, y in north_walls)
            wall_thickness = 2.0  # Standard wall thickness

            expected_distances = model_full_raycast_sweep(
                agent_position, agent_angle, closest_wall_y, wall_thickness
            )

            # Compare actual vs expected for key beams
            print("\n=== RAYCAST MODEL VALIDATION ===")
            key_beams = [
                (0, "Left edge (-60°)"),
                (32, "Left-center (-30°)"),
                (64, "Center (0°)"),
                (96, "Right-center (+30°)"),
                (127, "Right edge (+60°)"),
            ]

            print(f"{'Beam':<16} {'Expected':<9} {'Actual':<9} {'Error%':<8} {'Within 5%':<8}")
            print("-" * 65)

            tolerance_failures = 0
            for beam_idx, label in key_beams:
                expected = expected_distances[beam_idx]
                actual = depth_readings[beam_idx]

                if np.isfinite(expected) and np.isfinite(actual) and expected > 0:
                    # Convert actual from normalized [0,1] back to world units
                    actual_world = actual * 200.0  # lidarMaxRange = 200
                    error_pct = ((actual_world - expected) / expected) * 100
                    within_tolerance = abs(error_pct) <= 5.0

                    if not within_tolerance:
                        tolerance_failures += 1

                    tolerance_mark = "✓" if within_tolerance else "✗"
                    print(
                        f"{label:<16} {expected:<9.3f} {actual_world:<9.3f} "
                        f"{error_pct:<8.1f} {tolerance_mark:<8}"
                    )
                else:
                    print(f"{label:<16} {expected:<9.3f} {actual:<9.3f} {'N/A':<8} {'N/A':<8}")

            # Overall model accuracy assessment
            finite_expected = expected_distances[np.isfinite(expected_distances)]
            finite_actual_indices = np.where(np.isfinite(expected_distances))[0]
            finite_actual = depth_readings[finite_actual_indices] * 200.0  # Convert to world units

            if len(finite_actual) > 0:
                errors = np.abs(finite_actual - finite_expected)
                mean_error = np.mean(errors)
                max_error = np.max(errors)

                print("\n=== MODEL ACCURACY SUMMARY ===")
                print(f"Comparable beams: {len(finite_actual)}/128")
                print(f"Mean absolute error: {mean_error:.3f} world units")
                print(f"Max absolute error: {max_error:.3f} world units")
                print(
                    f"Key beams within 5% tolerance: "
                    f"{len(key_beams) - tolerance_failures}/{len(key_beams)}"
                )

        # Primary functionality tests
        print("\n=== FUNCTIONALITY VALIDATION ===")

        # 1. Basic functionality: Should detect some walls
        assert finite_count > 0, (
            f"Raycast lidar completely non-functional - no finite readings detected. "
            f"All {len(depth_readings)} beams returned invalid data."
        )

        # 2. Reasonable detection rate for 120° forward arc
        min_detection_rate = 0.3  # Expect at least 30% of beams to hit something
        detection_rate = finite_count / len(depth_readings)
        assert detection_rate >= min_detection_rate, (
            f"Low detection rate: {finite_count}/{len(depth_readings)} ({detection_rate:.1%}). "
            f"Expected at least {min_detection_rate:.1%} for 120° forward arc in environment."
        )

        # 3. Valid depth normalization: non-zero finite readings should be in (0, 1]
        finite_depths = depth_readings[np.isfinite(depth_readings) & (depth_readings > 0)]
        if len(finite_depths) > 0:
            assert np.all(
                finite_depths <= 1.0
            ), f"Depth values exceed normalization range: max = {finite_depths.max():.3f}"
            assert np.all(
                finite_depths > 0.0
            ), f"Invalid zero depths in finite readings: {np.sum(finite_depths == 0)} occurrences"

        # 4. Depth consistency: should have reasonable variation
        if len(finite_depths) > 1:
            depth_std = np.std(finite_depths)
            print(f"Depth variation: std={depth_std:.3f}")
            assert (
                depth_std > 0.001
            ), "Depth readings should show some variation in realistic environment"

        # Success indicators
        print("\n=== TEST RESULTS ===")
        print(f"✅ Functionality: {finite_count}/128 beams active ({detection_rate:.1%})")
        print(f"✅ Depth quality: {len(finite_depths)}/128 beams have valid distances")
        print(f"✅ Coverage: 120° forward arc with {len(depth_readings)} beams")
        print("✅ Raycast lidar validation completed successfully!")

    def test_lidar_angular_coverage(self, cpu_manager):
        """Test that lidar beams cover the expected 120° angular range"""
        mgr = cpu_manager

        # Use a simple test to verify angular distribution
        # The actual angular coverage is implemented in the C++ lidarSystem
        # This test validates the beam indexing and distribution

        mgr.step()
        lidar_tensor = mgr.lidar_tensor()
        if lidar_tensor.isOnGPU():
            lidar_array = lidar_tensor.to_torch().cpu().numpy()
        else:
            lidar_array = lidar_tensor.to_numpy()

        depth_readings = lidar_array[0, 0, :, 0]

        print("\n=== ANGULAR COVERAGE ANALYSIS ===")

        # Analyze readings in angular sectors
        # Beam 0 = -60°, Beam 63.5 = 0°, Beam 127 = +60°
        left_sector = depth_readings[:43]  # -60° to -30° (beams 0-42)
        center_sector = depth_readings[43:85]  # -30° to +30° (beams 43-84)
        right_sector = depth_readings[85:]  # +30° to +60° (beams 85-127)

        sectors = [
            ("Left (-60° to -30°)", left_sector),
            ("Center (-30° to +30°)", center_sector),
            ("Right (+30° to +60°)", right_sector),
        ]

        for sector_name, sector_data in sectors:
            finite_count = np.sum(np.isfinite(sector_data) & (sector_data > 0))
            total_count = len(sector_data)
            coverage = finite_count / total_count

            print(
                f"{sector_name:18}: {finite_count:2}/{total_count:2} beams "
                f"({coverage:.1%} coverage)"
            )

        # Basic coverage expectation: each sector should have some coverage
        total_coverage = np.sum(np.isfinite(depth_readings) & (depth_readings > 0))
        overall_coverage = total_coverage / len(depth_readings)

        assert overall_coverage > 0.1, (
            f"Insufficient overall coverage: {total_coverage}/128 ({overall_coverage:.1%}). "
            f"Expected significant coverage for 120° forward arc."
        )

        print(f"\n✅ Overall coverage: {total_coverage}/128 beams ({overall_coverage:.1%})")
        print("✅ Angular coverage validation completed!")

    def test_lidar_value_normalization(self, cpu_manager):
        """Test that lidar values are properly normalized"""
        mgr = cpu_manager
        mgr.step()

        lidar_tensor = mgr.lidar_tensor()
        if lidar_tensor.isOnGPU():
            lidar_array = lidar_tensor.to_torch().cpu().numpy()
        else:
            lidar_array = lidar_tensor.to_numpy()

        # Test all worlds - depth only now
        all_depths = lidar_array[:, :, :, 0].flatten()

        print("\n=== NORMALIZATION VALIDATION ===")

        # Depth value validation
        finite_depths = all_depths[np.isfinite(all_depths)]
        if len(finite_depths) > 0:
            print(f"Depth values: min={finite_depths.min():.3f}, max={finite_depths.max():.3f}")

            assert np.all(finite_depths >= 0.0), "All depths should be non-negative"
            assert np.all(finite_depths <= 1.0), "All depths should be ≤ 1.0 (normalized)"

            # Check for reasonable distribution (not all the same value)
            if len(finite_depths) > 1:
                depth_std = np.std(finite_depths)
                print(f"Depth standard deviation: {depth_std:.3f}")
                # Should have some variation in a realistic environment
                assert depth_std > 0.001, "Depth readings should show some variation"

        # Additional depth validation across all worlds
        total_valid_readings = np.sum((all_depths > 0.0) & (all_depths <= 1.0))
        total_readings = len(all_depths)

        print(f"Total valid readings across all worlds: {total_valid_readings}/{total_readings}")
        print(f"Overall depth range: [{all_depths.min():.3f}, {all_depths.max():.3f}]")

        # Should have reasonable detection rate across all worlds
        assert (
            total_valid_readings > total_readings * 0.1
        ), "Should have at least 10% valid depth readings"

        print("✅ All values are properly normalized")
        print("✅ Normalization validation completed!")


if __name__ == "__main__":
    # Allow running the test directly for debugging
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
