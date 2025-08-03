#!/bin/bash
# 
# Claude Code SessionStart Hook: Check Python REPL MCP Server
#
# This hook runs when a Claude Code session starts and checks if the
# python-repl MCP server is configured. It only prints a warning if
# the server is missing.
#
# Part of: Madrona Escape Room Development Tools
# See: docs/development/tools/PYTHON_REPL_MCP_SETUP.md

# Only warn if python-repl is NOT configured
if ! claude mcp list 2>/dev/null | grep -q "python-repl"; then
    echo "⚠ Python REPL MCP server not configured for interactive development"
    echo "To set up: claude mcp add python-repl uv --directory external/mcp-python run mcp_python"
    echo "See: docs/development/tools/PYTHON_REPL_MCP_SETUP.md"
fi

# Exit successfully regardless (don't block session start)
exit 0