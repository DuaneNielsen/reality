# Commit and Push Command

When you're ready to commit and push changes:

1. Run `git status` to see what files have changed

2. Add the modified files with `git add -u` (stages all modified files)

3. Create a commit with this message format:
   ```
   git commit -m "$(cat <<'EOF'
   [Your change description here]
   
   🤖 Generated with [Claude Code](https://claude.ai/code)
   
   Co-Authored-By: Claude <noreply@anthropic.com>
   EOF
   )"
   ```

4. Push to remote: `git push`

5. Verify with `git log --oneline -1` to see your commit

That's it! The changes are now pushed to the remote branch.