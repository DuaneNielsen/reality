#!/usr/bin/env python3
"""
Madrona-specific MCP Python REPL Server

Self-contained MCP server with built-in Madrona helper functions.
All utilities are pre-loaded and always available.
"""

import asyncio
import io
import subprocess
import re
import os
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from mcp.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions
import mcp.server.stdio
import mcp.types as types


class MadronaMCPServer:
    def __init__(self):
        self.server = Server("madrona-python-repl")
        
        # Pre-load Madrona environment in the global namespace
        self.global_namespace = {
            "__builtins__": __builtins__,
        }
        self._initialize_madrona_environment()
        
        # Set up handlers using decorators
        @self.server.list_tools()
        async def handle_list_tools() -> list[types.Tool]:
            return await self.handle_list_tools()
            
        @self.server.call_tool()
        async def handle_call_tool(name: str, arguments: dict | None) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
            return await self.handle_call_tool(name, arguments)

    def _initialize_madrona_environment(self, show_status=False):
        """Pre-load Madrona imports and helper functions into the namespace."""
        init_code = '''
import sys
import os
import numpy as np

# Add project to path - get current working directory
project_path = os.getcwd()
if project_path not in sys.path:
    sys.path.insert(0, project_path)

try:
    import madrona_escape_room
    from madrona_escape_room import SimManager, madrona
    import numpy as np
    _madrona_available = True
    if {show_status}:
        print("✓ Madrona Escape Room environment ready!")
        print("Use direct API: from madrona_escape_room import SimManager, madrona")
except ImportError as e:
    _madrona_available = False
    _import_error = str(e)
    if {show_status}:
        print(f"✗ Madrona not available: {{_import_error}}")
        print("Make sure you've built the project and installed the Python package.")
'''

        try:
            exec(init_code.format(show_status=show_status), self.global_namespace)
        except Exception as e:
            if show_status:
                print(f"Warning: Failed to initialize Madrona environment: {e}")

    async def handle_list_tools(self) -> list[types.Tool]:
        """List available tools"""
        return [
            types.Tool(
                name="execute_python",
                description=(
                    "Use this tool for Madrona simulation tasks. Execute Python code with "
                    "pre-loaded Madrona environment (SimManager, madrona, numpy). "
                    "Use when you need to:\n"
                    "• Run Madrona escape room simulations\n"
                    "• Test environment behavior and agent actions\n"
                    "• Benchmark simulation performance\n"
                    "• Analyze simulation data or trajectories\n"
                    "• Experiment with different simulation parameters\n\n"
                    "Variables persist between executions. Standard workflow:\n"
                    "1. Create manager: mgr = SimManager(exec_mode=madrona.ExecMode.CPU, num_worlds=2)\n"
                    "2. Get tensors: obs = mgr.self_observation_tensor().to_numpy()\n"
                    "3. Set actions: actions[0, :] = [move_amount, move_angle, rotate]\n"
                    "4. Step simulation: mgr.step()\n"
                    "5. Read results: pos = obs[0, 0, :3]  # world 0, agent 0 position\n\n"
                    "See ENVIRONMENT.md for complete action/observation space details."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python code to execute",
                        },
                        "reset": {
                            "type": "boolean",
                            "description": "Reset the Python session (clear all variables)",
                            "default": False
                        }
                    },
                    "required": ["code"],
                },
            ),
            types.Tool(
                name="list_variables",
                description="Use this tool to inspect the current Python session state. Lists all variables in the current session with their types and shapes (for arrays). Use when you need to see what variables are available or debug session state.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                }
            )
        ]

    async def handle_call_tool(self, name: str, arguments: dict | None) -> list[types.TextContent]:
        """Handle tool calls"""
        if arguments is None:
            arguments = {}

        if name == "execute_python":
            return await self._execute_python(arguments.get("code", ""), arguments.get("reset", False))
        elif name == "list_variables":
            return await self._list_variables()
        else:
            return [types.TextContent(
                type="text",
                text=f"Unknown tool: {name}"
            )]

    async def _execute_python(self, code: str, reset: bool = False) -> list[types.TextContent]:
        """Execute Python code and return output"""
        if reset:
            # Clear namespace but keep builtins and re-initialize
            self.global_namespace.clear()
            self.global_namespace["__builtins__"] = __builtins__
            self._initialize_madrona_environment(show_status=True)

        if not code.strip():
            return [types.TextContent(type="text", text="No code provided")]

        # Capture stdout and stderr
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                # Execute the code
                exec(code, self.global_namespace)

            # Get the captured output
            output = stdout_capture.getvalue()
            error_output = stderr_capture.getvalue()

            result = ""
            if output:
                result += output
            if error_output:
                result += error_output

            if not result:
                result = "Code executed successfully (no output)"

            return [types.TextContent(type="text", text=result)]

        except Exception as e:
            error_msg = f"Error: {str(e)}\\n"
            if stderr_capture.getvalue():
                error_msg += stderr_capture.getvalue()
            error_msg += traceback.format_exc()
            return [types.TextContent(type="text", text=error_msg)]

    async def _list_variables(self) -> list[types.TextContent]:
        """List all variables in the current session"""
        try:
            # Get all variables except builtins
            variables = {k: v for k, v in self.global_namespace.items() 
                        if not k.startswith('_') and k != '__builtins__'}

            if not variables:
                return [types.TextContent(type="text", text="No user variables defined in current session.")]

            output = "Current session variables:\\n"
            output += "=" * 40 + "\\n"

            for name, value in sorted(variables.items()):
                try:
                    value_str = str(type(value).__name__)
                    if hasattr(value, '__doc__') and callable(value):
                        value_str += " (function)"
                    elif hasattr(value, 'shape'):
                        value_str += f" {getattr(value, 'shape', '')}"

                    output += f"{name:20} : {value_str}\\n"
                except:
                    output += f"{name:20} : <repr error>\\n"

            return [types.TextContent(type="text", text=output)]

        except Exception as e:
            return [types.TextContent(
                type="text",
                text=f"Error listing variables: {str(e)}"
            )]

    async def run(self):
        """Run the MCP server"""
        async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="madrona-python-repl",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=NotificationOptions(),
                        experimental_capabilities={},
                    ),
                ),
            )


async def main():
    """Main entry point"""
    server = MadronaMCPServer()
    await server.run()


if __name__ == "__main__":
    asyncio.run(main())