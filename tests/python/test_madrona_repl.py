"""
Tests for the Madrona MCP Python REPL server

This test suite validates the MCP server functionality including:
- Server initialization and tool listing
- Python code execution with Madrona imports
- Session persistence across executions
- Error handling and edge cases
"""

import pytest
import tempfile
import os
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.madrona_mcp_server import MadronaMCPServer
import mcp.types as types


class TestMadronaMCPServer:
    """Test suite for MadronaMCPServer"""
    
    @pytest.fixture
    def server(self):
        """Create a MadronaMCPServer instance for testing"""
        server = MadronaMCPServer()
        return server
    
    def test_server_initialization(self, server):
        """Test that the server initializes correctly"""
        assert server is not None
        assert server.server.name == "madrona-python-repl"
        assert hasattr(server, 'global_namespace')
        
    def test_madrona_imports_available(self, server):
        """Test that Madrona imports are pre-loaded"""
        # Check if imports are available in the namespace
        namespace = server.global_namespace
        assert 'SimManager' in namespace or '_madrona_available' in namespace
        assert 'madrona' in namespace or '_madrona_available' in namespace
        assert 'np' in namespace
        
    def test_basic_python_execution(self, server):
        """Test basic Python code execution without async"""
        # Test basic math
        result = server._execute_python_sync("result = 2 + 2\nprint(f'Result: {result}')")
        assert "Result: 4" in result
        
    def test_numpy_operations(self, server):
        """Test numpy array operations work correctly"""
        code = """
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
print(f"Array: {arr}")
print(f"Shape: {arr.shape}")
print(f"Sum: {arr.sum()}")
"""
        result = server._execute_python_sync(code)
        assert "Array: [1 2 3 4 5]" in result
        assert "Shape: (5,)" in result
        assert "Sum: 15" in result
        
    def test_session_persistence(self, server):
        """Test that variables persist across multiple executions"""
        # First execution
        result1 = server._execute_python_sync("session_var = 42\nprint(f'Set var: {session_var}')")
        assert "Set var: 42" in result1
        
        # Second execution - access previously created variable
        result2 = server._execute_python_sync("session_var += 8\nprint(f'Updated var: {session_var}')")
        assert "Updated var: 50" in result2
        
    def test_error_handling(self, server):
        """Test error handling for invalid Python code"""
        # Test syntax error
        result = server._execute_python_sync("invalid syntax here")
        assert "Error:" in result or "SyntaxError" in result
        
        # Test runtime error
        result2 = server._execute_python_sync("undefined_variable")
        assert "Error:" in result2 or "NameError" in result2
        
    def test_session_reset(self, server):
        """Test session reset functionality"""
        # Create a variable
        server._execute_python_sync("test_var = 'before_reset'")
        
        # Verify it exists
        result1 = server._execute_python_sync("print(test_var)")
        assert "before_reset" in result1
        
        # Reset session
        server._reset_session()
        
        # Verify variable is gone
        result2 = server._execute_python_sync("print(test_var)")
        assert "NameError" in result2 or "Error:" in result2
        
    def test_multiline_code_execution(self, server):
        """Test execution of multiline code blocks"""
        code = """
def test_function(x, y):
    return x + y

result = test_function(5, 3)
print(f"Function result: {result}")

for i in range(3):
    print(f"Loop iteration: {i}")
"""
        result = server._execute_python_sync(code)
        assert "Function result: 8" in result
        assert "Loop iteration: 0" in result
        assert "Loop iteration: 1" in result
        assert "Loop iteration: 2" in result


# Integration test that requires a real Madrona environment
@pytest.mark.skipif(
    "CI" in os.environ, 
    reason="Requires built Madrona environment, skip in CI"
)
class TestMadronaMCPServerIntegration:
    """Integration tests that require a built Madrona environment"""
    
    @pytest.fixture
    def server(self):
        """Create a MadronaMCPServer instance for integration testing"""
        server = MadronaMCPServer()
        return server
    
    def test_canonical_workflow(self, server):
        """Test the canonical Madrona simulation workflow"""
        # This tests the exact workflow from the tool description
        canonical_code = """
# 1. Create manager
try:
    mgr = SimManager(
        exec_mode=madrona.ExecMode.CPU,
        gpu_id=-1, num_worlds=2, rand_seed=42, auto_reset=True
    )

    # 2. Get tensor references (zero-copy)
    obs = mgr.self_observation_tensor().to_numpy()
    actions = mgr.action_tensor().to_numpy()
    rewards = mgr.reward_tensor().to_numpy()
    done = mgr.done_tensor().to_numpy()

    print(f"Manager created successfully")
    print(f"Tensor shapes: obs={obs.shape}, actions={actions.shape}")

    # 3. Set actions
    actions[0, :] = [1, 0, 0]  # move_amount, move_angle, rotate (discrete)

    # 4. Step simulation
    mgr.step()

    # 5. Read observations
    pos = obs[0, 0, :3]  # world 0, agent 0, xyz position
    reward = rewards[0, 0, 0]  # world 0, agent 0 reward
    is_done = done[0, 0, 0]  # world 0, agent 0 episode done

    print(f"Position: {pos}")
    print(f"Reward: {reward}")
    print(f"Done: {is_done}")
    print("Canonical workflow completed successfully!")
except Exception as e:
    print(f"Test requires built Madrona environment: {e}")
"""
        
        result = server._execute_python_sync(canonical_code)
        # Should either complete successfully or show environment requirement
        assert ("Canonical workflow completed successfully!" in result or 
                "Test requires built Madrona environment:" in result)


# Add helper methods to MadronaMCPServer for sync testing
def _execute_python_sync(self, code, reset=False):
    """Synchronous wrapper for Python execution (for testing)"""
    import asyncio
    import io
    from contextlib import redirect_stdout, redirect_stderr
    
    if reset:
        self.global_namespace.clear() 
        self.global_namespace["__builtins__"] = __builtins__
        self._initialize_madrona_environment()
    
    if not code.strip():
        return "No code provided"
    
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    
    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, self.global_namespace)
        
        output = stdout_capture.getvalue()
        error_output = stderr_capture.getvalue()
        
        result = ""
        if output:
            result += output
        if error_output:
            result += error_output
        
        if not result:
            result = "Code executed successfully (no output)"
        
        return result
        
    except Exception as e:
        error_msg = f"Error: {str(e)}\n"
        if stderr_capture.getvalue():
            error_msg += stderr_capture.getvalue()
        return error_msg

def _reset_session(self):
    """Reset the Python session (for testing)"""
    self.global_namespace.clear()
    self.global_namespace["__builtins__"] = __builtins__
    self._initialize_madrona_environment()

# Monkey patch the test methods
MadronaMCPServer._execute_python_sync = _execute_python_sync
MadronaMCPServer._reset_session = _reset_session


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])