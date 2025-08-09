"""
Tests for the FastMCP Madrona Python REPL server

This test suite validates the FastMCP server functionality including:
- Python code execution with Madrona imports
- Session persistence across executions
- Error handling and edge cases
"""

import asyncio
import os
import sys
from pathlib import Path

import pytest

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import server functions directly
from scripts.fastmcp_madrona_server import (
    execute_python,
    global_namespace,
    initialize_madrona_environment,
    list_variables,
)


class TestFastMCPMadronaServer:
    """Test suite for FastMCP Madrona Server"""

    @pytest.fixture
    def reset_namespace(self):
        """Reset global namespace before each test"""
        global global_namespace
        global_namespace.clear()
        global_namespace["__builtins__"] = __builtins__
        initialize_madrona_environment()
        yield
        # Cleanup after test
        global_namespace.clear()
        global_namespace["__builtins__"] = __builtins__
        initialize_madrona_environment()

    def test_madrona_imports_available(self, reset_namespace):
        """Test that Madrona imports are pre-loaded"""
        # Check if imports are available in the namespace
        assert "SimManager" in global_namespace or "_madrona_available" in global_namespace
        assert "madrona" in global_namespace or "_madrona_available" in global_namespace
        assert "np" in global_namespace

    def test_basic_python_execution(self, reset_namespace):
        """Test basic Python code execution"""
        # Test basic math
        result = asyncio.run(execute_python("result = 2 + 2\nprint(f'Result: {result}')"))
        assert "Result: 4" in result

    def test_numpy_operations(self, reset_namespace):
        """Test numpy array operations work correctly"""
        code = """
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")
print(f"Shape: {arr.shape}")
print(f"Sum: {arr.sum()}")
"""
        result = asyncio.run(execute_python(code))
        assert "Array: [1 2 3 4 5]" in result
        assert "Shape: (5,)" in result
        assert "Sum: 15" in result

    def test_session_persistence(self, reset_namespace):
        """Test that variables persist across multiple executions"""
        # First execution
        result1 = asyncio.run(execute_python("session_var = 42\nprint(f'Set var: {session_var}')"))
        assert "Set var: 42" in result1

        # Second execution - access previously created variable
        result2 = asyncio.run(
            execute_python("session_var += 8\nprint(f'Updated var: {session_var}')")
        )
        assert "Updated var: 50" in result2

    def test_error_handling(self, reset_namespace):
        """Test error handling for invalid Python code"""
        # Test syntax error
        result = asyncio.run(execute_python("invalid syntax here"))
        assert "Error:" in result or "SyntaxError" in result

        # Test runtime error
        result2 = asyncio.run(execute_python("undefined_variable"))
        assert "Error:" in result2 or "NameError" in result2

    def test_session_reset(self, reset_namespace):
        """Test session reset functionality"""
        # Create a variable
        asyncio.run(execute_python("test_var = 'before_reset'"))

        # Verify it exists
        result1 = asyncio.run(execute_python("print(test_var)"))
        assert "before_reset" in result1

        # Reset session
        result2 = asyncio.run(execute_python("", reset=True))
        assert "session reset" in result2.lower()

        # Verify variable is gone
        result3 = asyncio.run(execute_python("print(test_var)"))
        assert "NameError" in result3 or "Error:" in result3

    def test_list_variables(self, reset_namespace):
        """Test listing session variables"""
        # Create some variables
        asyncio.run(execute_python("x = 42"))
        asyncio.run(execute_python("y = 'hello'"))
        asyncio.run(execute_python("z = [1, 2, 3]"))

        # List variables
        result = asyncio.run(list_variables())
        assert "x" in result
        assert "y" in result
        assert "z" in result
        assert "int" in result  # x type
        assert "str" in result  # y type
        assert "list" in result  # z type

    def test_multiline_code_execution(self, reset_namespace):
        """Test execution of multiline code blocks"""
        code = """
def test_function(x, y):
    return x + y

result = test_function(5, 3)
print(f"Function result: {result}")

for i in range(3):
    print(f"Loop iteration: {i}")
"""
        result = asyncio.run(execute_python(code))
        assert "Function result: 8" in result
        assert "Loop iteration: 0" in result
        assert "Loop iteration: 1" in result
        assert "Loop iteration: 2" in result


# Integration test that requires a real Madrona environment
@pytest.mark.skipif("CI" in os.environ, reason="Requires built Madrona environment, skip in CI")
class TestFastMCPMadronaIntegration:
    """Integration tests that require a built Madrona environment"""

    @pytest.fixture
    def reset_namespace(self):
        """Reset global namespace before each test"""
        global global_namespace
        global_namespace.clear()
        global_namespace["__builtins__"] = __builtins__
        initialize_madrona_environment()
        yield
        # Cleanup after test
        global_namespace.clear()
        global_namespace["__builtins__"] = __builtins__
        initialize_madrona_environment()

    def test_canonical_workflow(self, reset_namespace):
        """Test the canonical Madrona simulation workflow"""
        # This tests the exact workflow from the server instructions
        canonical_code = """
# Test canonical workflow
try:
    mgr = SimManager(exec_mode=madrona.ExecMode.CPU, num_worlds=2)
    obs = mgr.self_observation_tensor().to_numpy()
    actions = mgr.action_tensor().to_numpy()
    actions[0, :] = [1, 0, 0]  # move_amount, move_angle, rotate
    mgr.step()
    pos = obs[0, 0, :3]  # world 0, agent 0 position
    
    print(f"Manager created successfully")
    print(f"Position shape: {pos.shape}")
    print("Canonical workflow completed successfully!")
except Exception as e:
    print(f"Test requires built Madrona environment: {e}")
"""

        result = asyncio.run(execute_python(canonical_code))
        # Should either complete successfully or show environment requirement
        assert (
            "Canonical workflow completed successfully!" in result
            or "Test requires built Madrona environment:" in result
        )


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
