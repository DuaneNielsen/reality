#!/usr/bin/env python3
"""
Claude Code SessionStart Hook: Check Python REPL MCP Server

This hook runs when a Claude Code session starts and checks if the
python-repl MCP server is configured. It only prints a warning if
the server is missing.

Part of: Madrona Escape Room Development Tools
See: docs/development/tools/PYTHON_REPL_MCP_SETUP.md
"""

import subprocess
import sys

def check_mcp_server():
    """Check if python-repl MCP server is configured."""
    try:
        # Run claude mcp list and check output
        result = subprocess.run(
            ['claude', 'mcp', 'list'], 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        # Only warn if madrona_repl is NOT configured
        if 'madrona_repl' not in result.stdout:
            print(".")
            print("⚠ Madrona Python REPL MCP server not configured for interactive development")
            print("To set up: claude mcp add madrona_repl uv run python -- scripts/madrona_mcp_server.py")
            print("See: docs/development/tools/PYTHON_REPL_MCP_SETUP.md")
            print(".")
            
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
        # Silently fail if claude command not available or times out
        pass

if __name__ == "__main__":
    check_mcp_server()
    # Always exit successfully to not block session start
    sys.exit(0)