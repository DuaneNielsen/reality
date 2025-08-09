#!/usr/bin/env python3
"""
FastMCP-based Madrona Python REPL Server

Clean implementation using FastMCP framework with concise tool descriptions
and automatic Madrona environment initialization.
"""

import io
import traceback
from contextlib import redirect_stderr, redirect_stdout

from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
mcp = FastMCP(
    name="madrona-python-repl",
    instructions="""Use this server for Madrona simulation tasks. This server provides Python \
execution with pre-loaded Madrona environment including SimManager, madrona, and numpy. \
Use when you need to:
- Run Madrona escape room simulations
- Test environment behavior and agent actions
- Benchmark simulation performance
- Analyze simulation data or trajectories
- Experiment with different simulation parameters

Canonical example:
mgr = SimManager(exec_mode=madrona.ExecMode.CPU, num_worlds=2)
obs = mgr.self_observation_tensor().to_numpy()
actions = mgr.action_tensor().to_numpy()
actions[0, :] = [1, 0, 0]  # move_amount, move_angle, rotate
mgr.step()
pos = obs[0, 0, :3]  # world 0, agent 0 position

See ENVIRONMENT.md for complete action/observation space details.""",
)

# Global namespace for persistent session
global_namespace = {"__builtins__": __builtins__}


def initialize_madrona_environment():
    """Initialize Madrona imports in the global namespace."""
    init_code = """
import sys
import os
import numpy as np

# Add project to path
project_path = os.getcwd()
if project_path not in sys.path:
    sys.path.insert(0, project_path)

try:
    import madrona_escape_room
    from madrona_escape_room import SimManager, madrona
    import numpy as np
    _madrona_available = True
except ImportError as e:
    _madrona_available = False
    _import_error = str(e)
"""

    try:
        exec(init_code, global_namespace)
    except Exception:
        pass  # Silent initialization


# Initialize on startup
initialize_madrona_environment()


@mcp.tool()
async def execute_python(code: str, reset: bool = False) -> str:
    """Execute Python code with pre-loaded Madrona environment.

    Args:
        code: Python code to execute
        reset: Reset the Python session (clear all variables)
    """
    global global_namespace

    if reset:
        global_namespace.clear()
        global_namespace["__builtins__"] = __builtins__
        initialize_madrona_environment()
        return "Python session reset. Madrona environment reinitialized."

    if not code.strip():
        return "No code provided"

    # Capture output
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    # Show the input code with clear formatting
    result = "=" * 50 + "\n"
    result += "EXECUTING:\n"
    result += "=" * 50 + "\n"

    # Indent the code for better readability
    indented_code = "\n".join(f"  {line}" for line in code.strip().split("\n"))
    result += indented_code + "\n\n"

    try:
        with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
            exec(code, global_namespace)

        output = stdout_capture.getvalue()
        error_output = stderr_capture.getvalue()

        # Format output section
        if output or error_output:
            result += "=" * 50 + "\n"
            result += "OUTPUT:\n"
            result += "=" * 50 + "\n"
            if output:
                result += output
            if error_output:
                result += error_output
        else:
            result += "=" * 50 + "\n"
            result += "SUCCESS: Code executed (no output)\n"
            result += "=" * 50

        return result

    except Exception as e:
        result += "=" * 50 + "\n"
        result += "ERROR:\n"
        result += "=" * 50 + "\n"
        result += f"Error: {str(e)}\n"
        if stderr_capture.getvalue():
            result += stderr_capture.getvalue()
        result += traceback.format_exc()
        return result


@mcp.tool()
async def list_variables() -> str:
    """List all variables in the current Python session."""
    # Get all variables except builtins
    variables = {
        k: v for k, v in global_namespace.items() if not k.startswith("_") and k != "__builtins__"
    }

    if not variables:
        return "No user variables defined in current session."

    output = "Current session variables:\n"
    output += "=" * 40 + "\n"

    for name, value in sorted(variables.items()):
        try:
            value_str = str(type(value).__name__)
            if hasattr(value, "__doc__") and callable(value):
                value_str += " (function)"
            elif hasattr(value, "shape"):
                value_str += f" {getattr(value, 'shape', '')}"

            output += f"{name:20} : {value_str}\n"
        except Exception:
            output += f"{name:20} : <repr error>\n"

    return output


if __name__ == "__main__":
    # Run the server
    mcp.run(transport="stdio")
