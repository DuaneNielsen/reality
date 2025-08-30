#!/bin/bash

# Repository status analysis script for commit workflow
# Checks main repo and submodule status for commit operations

set -e  # Exit on any error

# Ensure we're in the project root
cd /home/duane/madrona_escape_room

# Print current directory for verification
echo "Working directory: $(pwd)"
echo

# Main repository status
echo "=== Main Repository Status ==="
git status --porcelain
echo

# Current branch
echo "=== Current Branch ==="
git branch --show-current
echo

# Madrona submodule status (if it exists)
echo "=== Madrona Submodule Status ==="
if [ -d "external/madrona" ]; then
    git -C external/madrona status --porcelain 2>/dev/null || echo "Error checking madrona submodule status"
else
    echo "No madrona submodule found"
fi
echo

# Recent commits
echo "=== Recent Commits ==="
git log --oneline -5