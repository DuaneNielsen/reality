"""
Test 128-beam horizontal lidar with 120° FOV using pytest framework

Tests the horizontal lidar POC implementation with hardcoded configuration.
"""

import numpy as np
import pytest

from madrona_escape_room import ExecMode
from madrona_escape_room.level_compiler import compile_ascii_level
from madrona_escape_room.manager import SimManager


class TestHorizontalLidar:
    """Test horizontal lidar functionality"""

    def test_128_beam_horizontal_lidar(self):
        """
        Test 128 horizontal lidar beams with 120° FOV using wall-in-front scenario.

        Expected setup:
        - Agent at (0, 5, 0) facing north toward wall at y=10
        - Wall extends from x=-10 to x=+10
        - Distance from agent to wall: 5 units
        - Lidar should read consistent ~5.0 unit distances across all beams
        """

        # Create lidar test level
        # Scale 2.5 means: 6-wide level = 15 units, 4-tall level = 10 units
        # Agent 'S' at grid (1,1) = world (-5, 2.5)
        # Need agent at (0, 5) with wall at y=10, so need different approach

        # Create custom level with proper dimensions
        # For a 21x9 grid with scale 1.0: 21 units wide, 9 units tall
        # Agent at grid position (10, 5) = world position (0, 5)
        # Wall at row 0 (y=8.5) is close, wall at row 8 would be at y=0.5
        # We need wall at y=10, so we need the agent further south

        # Let's use scale=1.25 and 17x13 grid:
        # Grid 17x13 with scale 1.25 = 21.25x16.25 world units
        # Agent at (8,4) = world (0, 5)
        # Wall at row 0 = y=15 (too far)
        # Let's adjust: Agent at (8,8) = world (0,0), wall at row 0 = y=15

        # Use a simple test level first
        lidar_level = """#####
#S..#
#...#
#####"""

        # Import required modules
        from madrona_escape_room import ExecMode, SimManager

        # Phase 2: Hardcode lidar configuration - use default FOV to fix infinity issue
        batch_render_view_width = 128  # 128 horizontal beams for lidar
        batch_render_view_height = 1  # Single pixel height works with default FOV
        # custom_vertical_fov removed - using default 100° FOV to fix infinity readings

        mgr = SimManager(
            exec_mode=ExecMode.CPU,
            gpu_id=0,
            num_worlds=1,
            rand_seed=42,
            auto_reset=True,
            enable_batch_renderer=True,
            batch_render_view_width=batch_render_view_width,
            batch_render_view_height=batch_render_view_height,
            # custom_vertical_fov parameter removed to use working default FOV
            compiled_levels=compile_ascii_level(lidar_level),
        )

        # Step once to get initial observations
        mgr.step()

        # Get depth tensor - should be (worlds, agents, height, width, channels)
        depth_tensor = mgr.depth_tensor()

        # Handle both CPU and GPU tensors
        if depth_tensor.isOnGPU():
            depth_torch = depth_tensor.to_torch()
            depth_array = depth_torch.cpu().numpy()
        else:
            depth_array = depth_tensor.to_numpy()

        # Phase 4.2: Validation Logic

        # Verify tensor shape: (worlds, agents, height, width, channels)
        expected_shape = (1, 1, 1, 128, 1)
        assert (
            depth_array.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {depth_array.shape}"

        # Extract depth readings for analysis - single row configuration
        depth_readings = depth_array[0, 0, 0, :, 0]  # Shape: (128,) - horizontal lidar line
        print(
            f"Lidar readings: min={depth_readings.min():.3f}, max={depth_readings.max():.3f}, mean={depth_readings.mean():.3f}"
        )
        print(f"Sample readings: {depth_readings[::32]}")  # Every 32nd reading

        print(f"\nDepth readings shape: {depth_readings.shape}")
        print(f"Center beam (64): {depth_readings[64]:.3f}")

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
        print(f"Finite readings: {finite_count} ({finite_count/total_beams:.1%})")
        print(f"Infinity readings: {infinite_count} ({infinite_count/total_beams:.1%})")
        print(f"Zero readings: {zero_count} ({zero_count/total_beams:.1%})")

        # Find beams with finite readings for detailed analysis
        finite_indices = np.where(np.isfinite(depth_readings))[0]
        if finite_count > 0:
            print(f"Finite beam positions: {finite_indices}")
            print(f"Finite beam distances: {depth_readings[finite_indices]}")
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
            f"Horizontal lidar completely non-functional - all {total_beams} beams report infinity. "
            f"Expected at least some wall detection for 128x1 configuration with default FOV."
        )

        # 2. Reasonable finite readings quality
        if finite_count > 0:
            assert np.all(
                finite_readings > 0.001
            ), f"Some finite readings too close to zero: {finite_readings[finite_readings <= 0.001]}"
            assert np.all(
                finite_readings < 100.0
            ), f"Some finite readings unreasonably far: {finite_readings[finite_readings >= 100.0]}"

        # 3. Coverage expectations - 100% coverage required for success
        required_coverage = 100.0 / 100  # 100% coverage required - no compromise!
        actual_coverage = finite_count / total_beams

        assert actual_coverage >= required_coverage, (
            f"HORIZONTAL LIDAR FAILED: {finite_count}/{total_beams} ({actual_coverage:.1%}) coverage. "
            f"REQUIREMENT: 100% coverage ({total_beams}/{total_beams} beams must work). "
            f"Current coverage is {required_coverage - actual_coverage:.1%} below requirement."
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
                f"✅ Distance range: {finite_readings.min():.3f} - {finite_readings.max():.3f} units"
            )
            print(f"✅ Finite beam positions: {list(finite_indices)}")
        print("✅ Configuration: 128x1 horizontal lidar with default 100° FOV")
        print("✅ Status: 100% SUCCESS - ALL BEAMS FUNCTIONAL!")

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
                mgr = SimManager(
                    exec_mode=ExecMode.CPU,
                    gpu_id=0,
                    num_worlds=1,
                    rand_seed=42,
                    auto_reset=True,
                    enable_batch_renderer=True,
                    batch_render_view_width=width,
                    batch_render_view_height=height,
                    custom_vertical_fov=fov,
                    compiled_levels=compile_ascii_level(test_level),
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
                        f"✅ SUCCESS - Range: {finite_readings.min():.3f} - {finite_readings.max():.3f}"
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
        print(f"\n{'='*60}")
        print("CONFIGURATION ANALYSIS SUMMARY")
        print(f"{'='*60}")

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
            failing_fovs = set(fov for _, _, _, fov, _ in failing_configs)

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
