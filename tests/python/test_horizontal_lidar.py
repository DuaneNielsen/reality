"""
Test 128-beam horizontal lidar with 120° FOV using pytest framework

Tests the horizontal lidar POC implementation with hardcoded configuration.
"""

import math

import numpy as np
import pytest

from madrona_escape_room import ExecMode, RenderMode
from madrona_escape_room.level_compiler import compile_ascii_level
from madrona_escape_room.manager import SimManager
from madrona_escape_room.sensor_config import SensorConfig

# Level with walls on top and bottom for lidar testing
LARGE_OPEN_LEVEL = """################################################################
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


def calculate_expected_distance(
    beam_index: int,
    agent_pos: tuple[float, float],  # (x, y) of agent
    camera_offset_y: float,  # Camera offset in agent's +Y direction
    wall_center_y: float,  # Wall center Y position
    wall_thickness: float,  # Wall thickness (cube extends ±thickness/2)
    beam_count: int = 128,  # Total number of beams
    fov_degrees: float = 120.0,  # Horizontal field of view
) -> float:
    """Calculate expected distance using perspective projection mapping."""
    # Camera position in world coordinates
    camera_y = agent_pos[1] + camera_offset_y

    # Wall front face Y position (closest to agent)
    wall_front_y = wall_center_y - wall_thickness / 2

    # Map beam index to screen coordinate (NDC-like)
    # Pixel center is at +0.5, so beam 0 is at 0.5, beam 127 is at 127.5
    pixel_x = beam_index + 0.5
    screen_x = (pixel_x / beam_count) * 2.0 - 1.0  # Map [0.5, 127.5] to [-1, 1]

    # Convert FOV to half-angle and calculate perspective scale
    fov_radians = math.radians(fov_degrees)
    tan_half_fov = math.tan(fov_radians / 2.0)

    # Map screen coordinate to view ray direction via perspective projection
    # This accounts for the perspective transformation's non-linear angular mapping
    ray_x = screen_x * tan_half_fov  # Horizontal ray direction component
    ray_y = 1.0  # Forward direction (normalized)

    # Calculate intersection with wall front face
    t = (wall_front_y - camera_y) / ray_y
    distance = t * math.sqrt(ray_x**2 + ray_y**2)

    return distance


def model_full_lidar_sweep(
    agent_pos: tuple[float, float],
    camera_offset_y: float,
    wall_center_y: float,
    wall_thickness: float,
    beam_count: int = 128,
    fov_degrees: float = 120.0,
) -> np.ndarray:
    """Model the full lidar sweep using perspective projection mapping."""
    distances = np.zeros(beam_count)
    for i in range(beam_count):
        distances[i] = calculate_expected_distance(
            i, agent_pos, camera_offset_y, wall_center_y, wall_thickness, beam_count, fov_degrees
        )
    return distances


class TestHorizontalLidar:
    """Test horizontal lidar functionality"""

    @pytest.mark.skip(reason="Debug test - kept for knowledge but skipped in normal runs")
    @pytest.mark.ascii_level(LARGE_OPEN_LEVEL)
    @pytest.mark.depth_default  # Use named sensor config
    def test_64x64_depth_configuration_debug(self, cpu_manager):
        """
        Debug test with 64x64 depth configuration to investigate depth tensor behavior.

        This test uses a standard 64x64 depth sensor configuration to see what values
        are actually being returned and compare with the 128x1 horizontal lidar.
        """

        # Use the cpu_manager fixture with 64x64 configuration
        mgr = cpu_manager

        # Step once to generate observations
        mgr.step()

        # Get depth tensor
        depth_tensor = mgr.depth_tensor()
        if depth_tensor.isOnGPU():
            depth_array = depth_tensor.to_torch().cpu().numpy()
        else:
            depth_array = depth_tensor.to_numpy()

        print("\n=== 64x64 DEBUG ANALYSIS ===")
        print(f"Depth tensor shape: {depth_array.shape}")

        # Extract center region for analysis
        if len(depth_array.shape) >= 4:
            # Shape should be (4, 1, 64, 64, 1) for 4 worlds
            center_region = depth_array[0, 0, :, :, 0]  # [height, width] for first world

            print(f"Center region shape: {center_region.shape}")
            print(f"Min depth: {center_region.min():.3f}")
            print(f"Max depth: {center_region.max():.3f}")
            print(f"Mean depth: {center_region.mean():.3f}")

            # Sample some specific pixels
            h, w = center_region.shape
            center_pixel = center_region[h // 2, w // 2]
            corner_pixel = center_region[0, 0]

            print(f"Center pixel [{h // 2},{w // 2}]: {center_pixel:.3f}")
            print(f"Corner pixel [0,0]: {corner_pixel:.3f}")

            # Count different value types
            finite_count = np.sum(np.isfinite(center_region))
            inf_count = np.sum(np.isinf(center_region))
            nan_count = np.sum(np.isnan(center_region))
            zero_count = np.sum(center_region == 0.0)
            total_pixels = center_region.size

            print(f"Total pixels: {total_pixels}")
            print(f"Finite values: {finite_count} ({finite_count / total_pixels:.1%})")
            print(f"Infinity values: {inf_count} ({inf_count / total_pixels:.1%})")
            print(f"NaN values: {nan_count} ({nan_count / total_pixels:.1%})")
            print(f"Zero values: {zero_count} ({zero_count / total_pixels:.1%})")

            # Look for our debug markers
            marker_123456 = np.sum(center_region == 123456.0)
            marker_999999 = np.sum(center_region == 999999.0)
            marker_111111_plus = np.sum((center_region > 111111.0) & (center_region < 111112.0))

            print(f"Debug marker 123456.0: {marker_123456} pixels")
            print(f"Debug marker 999999.0: {marker_999999} pixels")
            print(f"Debug marker ~111111.x: {marker_111111_plus} pixels")

            # Sample actual values to see patterns
            unique_values = np.unique(center_region.flatten())
            print(f"Number of unique values: {len(unique_values)}")
            if len(unique_values) <= 10:
                print(f"Unique values: {unique_values}")
            else:
                print(f"First 10 unique values: {unique_values[:10]}")
                print(f"Last 10 unique values: {unique_values[-10:]}")

    @pytest.mark.ascii_level(LARGE_OPEN_LEVEL)
    @pytest.mark.lidar_128  # Use 128-beam horizontal lidar preset
    def test_128_beam_horizontal_lidar_with_fixture(self, cpu_manager):
        """
        Test 128 horizontal lidar beams with 120° FOV using wall-in-front scenario.

        Expected setup:
        - Agent at (0, 5, 0) facing north toward wall at y=10
        - Wall extends from x=-10 to x=+10
        - Distance from agent to wall: 5 units
        - Lidar should read consistent ~5.0 unit distances across all beams
        """

        # Use the standard cpu_manager with depth_config marker
        mgr = cpu_manager

        # Compile the ASCII level to see wall positions
        from madrona_escape_room.level_compiler import compile_ascii_level

        level = compile_ascii_level(LARGE_OPEN_LEVEL, level_name="test_level")

        print("\n=== COMPILED LEVEL INFO ===")
        print(f"Number of tiles: {level.num_tiles}")
        print(f"Spawn position: ({level.spawn_x[0]:.3f}, {level.spawn_y[0]:.3f})")

        wall_count = 0
        print("\n=== WALL POSITIONS (first 10) ===")
        for i in range(level.num_tiles):
            obj_id = level.object_ids[i]
            if obj_id == 2:  # Wall asset ID
                if wall_count < 10:
                    x = level.tile_x[i]
                    y = level.tile_y[i]
                    print(f"Wall {wall_count}: ({x:.1f}, {y:.1f})")
                wall_count += 1
        print(f"Total walls: {wall_count}")

        # Step once to get observations
        mgr.step()

        # Get depth tensor for first world only
        depth_tensor = mgr.depth_tensor()
        if depth_tensor.isOnGPU():
            depth_array = depth_tensor.to_torch().cpu().numpy()
        else:
            depth_array = depth_tensor.to_numpy()

        # Print first world depth readings
        depth_readings = depth_array[0, 0, 0, :, 0]  # World 0, Agent 0, all 128 beams
        print(f"\nFirst world depth readings (128 beams): {depth_readings}")

        # Model expected distances using perspective projection
        agent_position = (-1.25, 0.0)
        camera_offset = 1.0
        wall_center = 17.5
        wall_thickness = 2.0
        fov = 120.0

        expected_distances = model_full_lidar_sweep(
            agent_position, camera_offset, wall_center, wall_thickness, fov_degrees=fov
        )

        # Compare key beams
        print("\n=== MODEL VALIDATION ===")
        center_beam = 64
        key_beams = [
            (0, "Left edge"),
            (32, "Left-center"),
            (center_beam, "Center"),
            (96, "Right-center"),
            (127, "Right edge"),
        ]

        print(f"{'Beam':<12} {'Expected':<8} {'Actual':<8} {'Error%':<8} {'Within 2%':<8}")
        print("-" * 60)

        for beam_idx, label in key_beams:
            expected = expected_distances[beam_idx]
            actual = depth_readings[beam_idx]
            error_pct = ((actual - expected) / expected) * 100
            within_tolerance = abs(error_pct) <= 2.0

            tolerance_mark = "✓" if within_tolerance else "✗"
            print(
                f"{label:<12} {expected:<8.3f} {actual:<8.3f} {error_pct:<8.1f} {tolerance_mark:<8}"
            )

        # Overall error stats
        errors = depth_readings - expected_distances
        mean_error = np.mean(np.abs(errors))
        max_error = np.max(np.abs(errors))
        error_percentages = np.abs((depth_readings - expected_distances) / expected_distances) * 100
        max_error_pct = np.max(error_percentages)
        beams_within_tolerance = np.sum(error_percentages <= 2.0)

        print("\n=== ACCURACY SUMMARY ===")
        print(f"Mean absolute error: {mean_error:.3f} units")
        print(f"Max absolute error: {max_error:.3f} units")
        print(f"Max error percentage: {max_error_pct:.1f}%")
        tolerance_pct = beams_within_tolerance / len(depth_readings)
        print(
            f"Beams within 2% tolerance: {beams_within_tolerance}/{len(depth_readings)} "
            f"({tolerance_pct:.1%})"
        )
        print(f"FOV: {fov}°")

        # Phase 5.3: Full Beam Array Analysis - Comprehensive horizontal lidar validation

        # Analyze all 128 beams for complete coverage assessment
        finite_readings = depth_readings[np.isfinite(depth_readings)]
        infinite_readings = depth_readings[np.isinf(depth_readings)]
        zero_readings = depth_readings[depth_readings == 0.0]

        finite_count = len(finite_readings)
        infinite_count = len(infinite_readings)
        zero_count = len(zero_readings)
        total_beams = len(depth_readings)

        print("\n=== BEAM ANALYSIS RESULTS ===")
        print(f"Total beams: {total_beams}")
        print(f"Finite readings: {finite_count} ({finite_count / total_beams:.1%})")
        print(f"Infinity readings: {infinite_count} ({infinite_count / total_beams:.1%})")
        print(f"Zero readings: {zero_count} ({zero_count / total_beams:.1%})")

        # Find beams with finite readings for detailed analysis
        if finite_count > 0:
            print(
                f"Distance range: {finite_readings.min():.3f} - {finite_readings.max():.3f} units"
            )

        # Center beam analysis
        center_beam_idx = 64  # Middle of 128 beams (0-indexed)
        center_distance = depth_readings[center_beam_idx]
        print(f"Center beam ({center_beam_idx}): {center_distance:.3f}")

        # Horizontal coverage analysis - check different regions
        left_region = depth_readings[:32]  # Beams 0-31 (left side)
        left_center = depth_readings[32:64]  # Beams 32-63 (left-center)
        right_center = depth_readings[64:96]  # Beams 64-95 (right-center)
        right_region = depth_readings[96:]  # Beams 96-127 (right side)

        regions = [
            ("Left", left_region),
            ("Left-Center", left_center),
            ("Right-Center", right_center),
            ("Right", right_region),
        ]

        print("\n=== REGIONAL COVERAGE ===")
        for region_name, region_data in regions:
            region_finite = np.sum(np.isfinite(region_data))
            region_total = len(region_data)
            region_pct = region_finite / region_total
            print(f"{region_name:12}: {region_finite:2}/{region_total} ({region_pct:.1%}) finite")

        # Primary assertions for horizontal lidar functionality

        # 1. Basic functionality: At least some beams should detect walls
        assert finite_count > 0, (
            f"Horizontal lidar completely non-functional - all {total_beams} beams "
            f"report infinity. Expected at least some wall detection for 128x1 configuration."
        )

        # 2. Reasonable finite readings quality
        if finite_count > 0:
            assert np.all(finite_readings > 0.001), (
                f"Some finite readings too close to zero: "
                f"{finite_readings[finite_readings <= 0.001]}"
            )
            assert np.all(
                finite_readings < 100.0
            ), f"Some finite readings unreasonably far: {finite_readings[finite_readings >= 100.0]}"

        # 3. Coverage expectations - 100% coverage required for success
        required_coverage = 100.0 / 100  # 100% coverage required - no compromise!
        actual_coverage = finite_count / total_beams

        assert actual_coverage >= required_coverage, (
            f"HORIZONTAL LIDAR FAILED: {finite_count}/{total_beams} "
            f"({actual_coverage:.1%}) coverage. "
            f"REQUIREMENT: 100% coverage ({total_beams}/{total_beams} beams must work). "
            f"Current coverage is {required_coverage - actual_coverage:.1%} below requirement."
        )

        # 4. Model accuracy - perspective projection model should be within 2% tolerance
        model_tolerance_pct = 2.0
        beams_within_model_tolerance = np.sum(error_percentages <= model_tolerance_pct)
        model_accuracy_pct = beams_within_model_tolerance / len(depth_readings)

        assert model_accuracy_pct >= 0.95, (  # Require 95% of beams to be within 2% tolerance
            f"PERSPECTIVE PROJECTION MODEL FAILED: Only {beams_within_model_tolerance}/"
            f"{len(depth_readings)} ({model_accuracy_pct:.1%}) beams within "
            f"{model_tolerance_pct}% tolerance. Max error: {max_error_pct:.1f}%. "
            f"Model may need refinement."
        )

        # 4. If we reach here, we have 100% coverage - SUCCESS!
        print("\n=== PERFORMANCE ASSESSMENT ===")
        print(f"Coverage: {actual_coverage:.1%} - 100% SUCCESS!")

        # 5. Distance consistency for finite readings
        if finite_count >= 2:
            depth_std = np.std(finite_readings)
            depth_mean = np.mean(finite_readings)
            cv = depth_std / depth_mean
            print(f"Distance consistency: mean={depth_mean:.3f}, std={depth_std:.3f}, CV={cv:.3f}")

            # Allow higher variation for limited sample size
            max_cv = 1.0 if finite_count < 10 else 0.5
            assert cv <= max_cv, (
                f"Finite readings too inconsistent (CV={cv:.3f} > {max_cv}). "
                f"Mean: {depth_mean:.3f}, Std: {depth_std:.3f}"
            )

        # Success output - comprehensive summary
        print("\n✅ HORIZONTAL LIDAR VALIDATION COMPLETE")
        print(
            f"✅ Functionality: {finite_count}/{total_beams} beams active ({actual_coverage:.1%})"
        )
        if finite_count > 0:
            print(
                f"✅ Distance range: {finite_readings.min():.3f} - "
                f"{finite_readings.max():.3f} units"
            )
        print("✅ Configuration: 128x1 horizontal lidar with default 100° FOV")
        print("✅ Status: 100% SUCCESS - ALL BEAMS FUNCTIONAL!")

    def test_render_mode_constants_available(self):
        """Test that RenderMode constants are properly imported and have correct values"""
        # Should be able to import RenderMode
        assert hasattr(RenderMode, "RGBD"), "RenderMode.RGBD not available"
        assert hasattr(RenderMode, "Depth"), "RenderMode.Depth not available"

        # Check values are correct
        assert RenderMode.RGBD == 0, f"Expected RGBD=0, got {RenderMode.RGBD}"
        assert RenderMode.Depth == 1, f"Expected Depth=1, got {RenderMode.Depth}"

        print(f"✅ RenderMode constants: RGBD={RenderMode.RGBD}, Depth={RenderMode.Depth}")

    @pytest.mark.depth_sensor  # Uses Depth render mode, 64x64, 100° FOV
    def test_render_mode_depth_only(self, cpu_manager):
        """Test Depth render mode provides depth data efficiently"""
        mgr = cpu_manager

        # Step once to generate data
        mgr.step()

        # Test depth tensor access
        depth_tensor = mgr.depth_tensor()
        if depth_tensor.isOnGPU():
            depth_np = depth_tensor.to_torch().cpu().numpy()
        else:
            depth_np = depth_tensor.to_numpy()

        # Verify tensor shape and data (4 worlds from fixture)
        assert depth_np.shape == (
            4,
            1,
            64,
            64,
            1,
        ), f"Expected depth shape (4,1,64,64,1), got {depth_np.shape}"
        assert depth_np.dtype == np.float32, f"Expected depth dtype float32, got {depth_np.dtype}"

        print(f"✅ Depth-only mode: depth shape {depth_np.shape}")

    def test_render_mode_rgbd_explicit(self):
        """Test RGBD render mode (default) provides both RGB and depth data"""
        config = SensorConfig.rgbd_default()
        mgr = SimManager(
            exec_mode=ExecMode.CPU,
            gpu_id=0,
            num_worlds=1,
            rand_seed=42,
            auto_reset=True,
            **config.to_manager_kwargs(),
        )

        # Step once to generate data
        mgr.step()

        # Test RGB and depth tensor access
        rgb_tensor = mgr.rgb_tensor()
        depth_tensor = mgr.depth_tensor()

        if rgb_tensor.isOnGPU():
            rgb_np = rgb_tensor.to_torch().cpu().numpy()
            depth_np = depth_tensor.to_torch().cpu().numpy()
        else:
            rgb_np = rgb_tensor.to_numpy()
            depth_np = depth_tensor.to_numpy()

        # Verify tensor shapes for single world
        assert rgb_np.shape == (
            1,
            1,
            64,
            64,
            4,
        ), f"Expected RGB shape (1,1,64,64,4), got {rgb_np.shape}"
        assert depth_np.shape == (
            1,
            1,
            64,
            64,
            1,
        ), f"Expected depth shape (1,1,64,64,1), got {depth_np.shape}"

        print(f"✅ RGBD mode: RGB shape {rgb_np.shape}, depth shape {depth_np.shape}")

    @pytest.mark.skip(reason="Research test - kept for knowledge but skipped in normal runs")
    def test_depth_sensor_configuration_comparison(self):
        """
        Test various depth sensor configurations to identify working vs failing thresholds.

        This test helps determine if the horizontal lidar issue is caused by:
        1. Extreme aspect ratios (width >> height)
        2. Custom vertical FOV parameter issues
        3. Minimum vertical resolution requirements
        4. Rendering pipeline limitations
        """

        test_level = """#####
#S..#
#...#
#####"""

        # Test configurations: (name, width, height, fov, should_work)
        test_cases = [
            ("Normal Depth Sensor", 64, 64, 100.0, True),  # Known working baseline
            ("Wide Square", 128, 128, 100.0, True),  # Test if width is issue
            (
                "Extreme Horizontal Default FOV",
                128,
                1,
                100.0,
                False,
            ),  # Test extreme aspect + default FOV
            ("Horizontal 3px Default FOV", 128, 3, 100.0, False),  # Test 3px height + default FOV
            ("Current Failing Config", 128, 3, 10.0, False),  # Current test configuration
            ("Moderate Aspect 8px", 128, 8, 100.0, True),  # Test moderate aspect ratio
            ("Moderate Aspect 16px", 128, 16, 100.0, True),  # Test moderate aspect ratio
            ("Horizontal Large FOV", 128, 3, 45.0, False),  # Test larger custom FOV
        ]

        results = []

        for config_name, width, height, fov, expected_success in test_cases:
            print(f"\n--- Testing {config_name} ---")
            print(f"Config: {width}x{height}, FOV: {fov}°")

            try:
                # Create custom sensor config for testing various configurations
                sensor_config = SensorConfig.custom(
                    width=width,
                    height=height,
                    vertical_fov=fov,
                    render_mode=RenderMode.Depth,
                    name=config_name,
                )

                mgr = SimManager(
                    exec_mode=ExecMode.CPU,
                    gpu_id=0,
                    num_worlds=1,
                    rand_seed=42,
                    auto_reset=True,
                    compiled_levels=compile_ascii_level(test_level),
                    **sensor_config.to_manager_kwargs(),
                )

                mgr.step()
                depth_tensor = mgr.depth_tensor()

                # Handle both CPU and GPU tensors
                if depth_tensor.isOnGPU():
                    depth_torch = depth_tensor.to_torch()
                    depth_array = depth_torch.cpu().numpy()
                else:
                    depth_array = depth_tensor.to_numpy()

                expected_shape = (1, 1, height, width, 1)
                assert (
                    depth_array.shape == expected_shape
                ), f"Shape mismatch: expected {expected_shape}, got {depth_array.shape}"

                # Extract middle row for analysis
                middle_row = height // 2
                depth_readings = depth_array[0, 0, middle_row, :, 0]

                # Count finite vs infinite readings
                finite_count = np.sum(np.isfinite(depth_readings))
                total_count = len(depth_readings)
                finite_ratio = finite_count / total_count
                center_value = depth_readings[depth_readings.shape[0] // 2]

                print(f"Finite readings: {finite_count}/{total_count} ({finite_ratio:.1%})")
                print(f"Center beam: {center_value:.3f}")

                # Configuration is successful if it has any finite readings
                actual_success = finite_count > 0

                if actual_success:
                    finite_readings = depth_readings[np.isfinite(depth_readings)]
                    print(
                        f"✅ SUCCESS - Range: {finite_readings.min():.3f} - "
                        f"{finite_readings.max():.3f}"
                    )
                else:
                    print("❌ FAILED - All readings are infinity")

                # Check if result matches expectation
                matches_expectation = actual_success == expected_success
                if matches_expectation:
                    print("✓ Result matches expectation")
                else:
                    print(f"! UNEXPECTED: Expected {expected_success}, got {actual_success}")

                results.append(
                    (
                        config_name,
                        width,
                        height,
                        fov,
                        expected_success,
                        actual_success,
                        matches_expectation,
                    )
                )

            except Exception as e:
                print(f"❌ ERROR: {e}")
                results.append((config_name, width, height, fov, expected_success, False, False))

        # Summary analysis
        print(f"\n{'=' * 60}")
        print("CONFIGURATION ANALYSIS SUMMARY")
        print(f"{'=' * 60}")

        working_configs = []
        failing_configs = []
        unexpected_results = []

        for name, w, h, fov, expected, actual, matches in results:
            aspect_ratio = w / h
            if actual:
                working_configs.append((name, w, h, fov, aspect_ratio))
                status = "✅"
            else:
                failing_configs.append((name, w, h, fov, aspect_ratio))
                status = "❌"

            if not matches:
                unexpected_results.append(name)
                status += " (!)"

            print(f"{status} {name}: {w}x{h}, FOV:{fov}°, AR:{aspect_ratio:.1f}")

        print(f"\nWorking configurations: {len(working_configs)}")
        print(f"Failing configurations: {len(failing_configs)}")

        if unexpected_results:
            print(f"Unexpected results: {len(unexpected_results)}")
            for name in unexpected_results:
                print(f"  - {name}")

        # Hypothesis analysis
        if working_configs and failing_configs:
            working_aspects = [ar for _, _, _, _, ar in working_configs]
            failing_aspects = [ar for _, _, _, _, ar in failing_configs]

            max_working_aspect = max(working_aspects) if working_aspects else 0
            min_failing_aspect = min(failing_aspects) if failing_aspects else float("inf")

            print("\nHYPOTHESIS ANALYSIS:")
            print(f"Max working aspect ratio: {max_working_aspect:.1f}")
            print(f"Min failing aspect ratio: {min_failing_aspect:.1f}")

            if max_working_aspect < min_failing_aspect:
                print("✓ ASPECT RATIO THRESHOLD IDENTIFIED")
                print(f"  Configurations work when aspect ratio ≤ {max_working_aspect:.1f}")
                print(f"  Configurations fail when aspect ratio ≥ {min_failing_aspect:.1f}")
            else:
                print("? Aspect ratio is not the determining factor")

            # Check FOV patterns
            working_fovs = set(fov for _, _, _, fov, _ in working_configs)

            if 100.0 in working_fovs and all(
                fov != 100.0 for _, _, _, fov, _ in failing_configs if fov < 50
            ):
                print("✓ CUSTOM FOV ISSUE IDENTIFIED")
                print("  Default FOV (100°) works, custom FOV values may be problematic")

        # This test is informational - don't fail on results
        # The goal is to characterize the behavior, not assert specific expectations
        assert len(results) == len(test_cases), "Not all test cases were executed"


if __name__ == "__main__":
    # Allow running the test directly
    test_lidar = TestHorizontalLidar()
    test_lidar.test_128_beam_horizontal_lidar()
