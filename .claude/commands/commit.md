# Commit and Push Command

When you're ready to commit and push changes:

1. **IMPORTANT**: First verify you're in the project root directory:
   ```bash
   pwd  # Should show /home/duane/madrona_escape_room
   ```
   If not in project root, navigate there first:
   ```bash
   cd /home/duane/madrona_escape_room
   ```

2. Check for all changes including submodules:
   ```bash
   git status
   ```
   Look for "modified: external/madrona" to identify submodule changes.

3. If there are submodule changes (files within external/madrona), commit them first:
   ```bash
   # Ensure git config is set in submodule (one-time setup, safe to run multiple times)
   git -C external/madrona config user.name "$(git config user.name)" 2>/dev/null || true
   git -C external/madrona config user.email "$(git config user.email)" 2>/dev/null || true
   
   # Stage and commit changes in the submodule
   git -C external/madrona add -u
   git -C external/madrona commit -m "[Your submodule change description]"
   git -C external/madrona push origin main
   ```
   
   Note: Ignore warnings about nested submodules (external/SPIRV-Reflect, etc.) unless specifically working on those dependencies.

4. Add all modified files in the main project:
   ```bash
   git add -u
   ```

5. If the submodule pointer changed (after committing in step 3), also add it:
   ```bash
   git add external/madrona 2>/dev/null || true
   ```

6. Create a commit in the main project:
   ```bash
   git commit -m "[Your change description]"
   ```

7. Push to remote:
   ```bash
   git push
   ```

8. Verify the commit:
   ```bash
   git log --oneline -1
   ```

**Key Rules:**
- Always use `git -C <path>` for submodule operations instead of `cd`
- Never use `cd` without chaining commands with `&&` in the same bash invocation
- Always verify your current directory with `pwd` when uncertain
- All commands assume you're in the project root directory

That's it! The changes are now pushed to the remote branch.