- Clearly report number of tests passed, failed, and skipped
   - **IMPORTANT**: When presenting the results of GoogleTests, output the failed tests, and an example command how to run one of them
   - **IMPORTANT**: For Python tests, always include the full test name format: `test_file.py::test_name` (not just `test_name`)
   - **IMPORTANT**: clearly provide the output formatting of the agent so they can be reproduced
   eg: uv run --group dev pytest tests/python/test_bindings.py::test_deterministic_actions -v
   eg: ./build/mad_escape_tests --gtest_filter="ViewerCoreTrajectoryTest.TrajectoryPointsMatchRecordedFrames" --disable-capture --gtest_print_time=1 --gtest_output=json -v

   **Python/Pytest Output Format**:
   When reporting pytest results, use this structured format:

   ## Test Results Summary

   ### ✅ C++ Tests: X/X passed
   ### ❌ Python Tests: X passed, X failed, X skipped

   **Failed Tests:**
   - ❌ `test_file.py::test_name` - Brief error description
   - ❌ `test_file.py::test_name` - Brief error description

   **Reproduction:**
   ```bash
   uv run --group dev pytest tests/python/test_file.py::test_name -v
   ```

   **Example Output:**

   ## Test Results Summary

   ### ✅ C++ Tests: 18/18 passed
   ### ❌ Python Tests: 145 passed, 4 failed, 15 skipped

   **Failed Tests:**
   - ❌ `test_bindings.py::test_deterministic_actions` - Position comparison failure
   - ❌ `test_spawn_locations.py::test_single_spawn_center` - Wrong spawn X: -20.0 (expected -6.25)
   - ❌ `test_reward_system.py::test_reward_normalization` - Reward calculation assertion
   - ❌ `test_level_compiler.py::test_compiled_level_structure_validation` - Level validation error

   **Reproduction:**
   ```bash
   uv run --group dev pytest tests/python/test_bindings.py::test_deterministic_actions -v
   uv run --group dev pytest tests/python/test_spawn_locations.py::test_single_spawn_center -v
   uv run --group dev pytest tests/python/test_reward_system.py::test_reward_normalization -v
   uv run --group dev pytest tests/python/test_level_compiler.py::test_compiled_level_structure_validation -v
   ./build/mad_escape_tests --gtest_filter="ViewerCoreTrajectoryTest.TrajectoryPointsMatchRecordedFrames" --disable-capture --gtest_print_time=1 --gtest_output=json -v
   ```
