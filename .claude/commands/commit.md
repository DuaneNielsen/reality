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

3. **IMPORTANT**: You should stage all files that git status reflects as changed. If you think a modified file does not belong in the commit, prompt the user to request clarification on if the file should be staged.
   ```bash
   git diff <file>  # Review changes for any file you're unsure about
   ```

4. If there are submodule changes (files within external/madrona), commit them first:
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

5. Add all modified files in the main project:
   ```bash
   git add -u
   ```

6. If the submodule pointer changed (after committing in step 4), also add it:
   ```bash
   git add external/madrona 2>/dev/null || true
   ```

7. Create a commit in the main project:
   ```bash
   git commit -m "[Your change description]"
   ```

8. Push to remote:
   ```bash
   git push
   ```

9. Verify the commit:
   ```bash
   git log --oneline -1
   ```

**Key Rules:**
- Always stage all files shown as modified in `git status` unless explicitly told otherwise
- If unsure about a file, ask the user for clarification before proceeding
- Always use `git -C <path>` for submodule operations instead of `cd`
- Never use `cd` without chaining commands with `&&` in the same bash invocation
- Always verify your current directory with `pwd` when uncertain
- All commands assume you're in the project root directory

That's it! The changes are now pushed to the remote branch.