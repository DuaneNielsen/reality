#!/bin/bash

# Repository status analysis script for commit workflow
# Checks main repo and submodule status for commit operations

set -e  # Exit on any error

# Find the git repository root (works with worktrees)
PROJECT_ROOT="$(git rev-parse --show-toplevel)"
cd "$PROJECT_ROOT"

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

# Submodule status check
echo "=== Submodule Status ==="
if git submodule status | grep -q .; then
    echo "Active submodules:"
    git submodule status
    echo
    # Check status of each submodule
    git submodule foreach --quiet 'echo "Submodule $name:"; git status --porcelain || echo "  Error checking status"; echo'
else
    echo "No submodules found"
fi
echo

# Recent commits
echo "=== Recent Commits ==="
git log --oneline -5