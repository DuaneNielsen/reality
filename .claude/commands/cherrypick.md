---
command: cherrypick
description: Cherry-pick commits from feature/revenge_of_codegen branch safely
---

# Cherry-Pick Command

This command helps safely cherry-pick commits from the feature/revenge_of_codegen branch to the current stable branch (feature/the_dark_codegen).

## Background

- **Stable base**: Commit 66b73ff (passes 1000+ stress test iterations)
- **Problem branch**: feature/revenge_of_codegen has a non-deterministic crash after commit 62b92dc
- **Current branch**: feature/the_dark_codegen - clean rebuild from stable base

## Workflow

1. **Review the commit**
   ```bash
   git show --stat <commit-hash>
   ```

2. **Check what files it modifies**
   ```bash
   git diff <commit-hash>^..<commit-hash> --name-only
   ```

3. **Attempt cherry-pick**
   ```bash
   git cherry-pick <commit-hash>
   ```

4. **If conflicts occur**:
   - For .claude files: Usually keep ours (`git checkout --ours .claude/...`)
   - **For build system files (CMakeLists.txt, setup.py)**: Keep ours to preserve stable build configuration
     ```bash
     git checkout --ours CMakeLists.txt src/CMakeLists.txt setup.py
     ```
   - For code files: Review carefully
   - Stage resolved files: `git add <files>`
   - Continue: `git cherry-pick --continue`

5. **Test stability after cherry-pick**
   ```bash
   # Quick test
   uv run python tests/stress_test.py 100
   
   # Thorough test if suspicious
   uv run python tests/stress_test.py 1000
   ```

## Safe Commits Already Applied

- ✅ 66b73ff - Base commit (stable)
- ✅ 467faeb - Stress test script
- ✅ c1f0227 - Build system unification (CMake + DLPack)
- ✅ 4d3fecc - Clean setup.py
- ✅ df88970 - .claude configuration updates
- ✅ a76b419 - Isolated Python build environment
- ✅ 988e4e7 - Meshoptimizer documentation (72501c3)

## Potentially Safe Commits to Consider

These are likely safe as they're mostly documentation/refactoring:
- 855b19b - revert: remove ineffective meshoptimizer Makefile entry
- e124333 - feat: add verbose flag to suppress parse errors
- e03514c - chore: ignore generated Python binding files
- 3d6ac7e - feat: add uv verification and improve build documentation

## Risky Commits to Avoid

These involve significant code changes that might introduce the crash:
- 62b92dc and after - Known to have non-deterministic crash
- Anything involving physics/quaternion changes
- Major refactoring of core systems

## Testing Protocol

After each cherry-pick:

1. **Build**:
   ```bash
   # Use project-builder agent or:
   cmake -B build && make -C build -j16
   ```

2. **Reinstall package**:
   ```bash
   uv pip uninstall madrona-escape-room
   uv pip install -e .
   ```

3. **Run stress test**:
   ```bash
   uv run python tests/stress_test.py 100  # Quick test
   uv run python tests/stress_test.py 1000 # Thorough test
   ```

4. **Check basic functionality**:
   ```bash
   uv run python -c "import madrona_escape_room; print('OK')"
   ```

## Abort Cherry-Pick

If something goes wrong:
```bash
git cherry-pick --abort
```

## Important: Build System Files

**ALWAYS keep the current branch's version of build system files:**
- `CMakeLists.txt` (root and `src/CMakeLists.txt`)
- `setup.py`
- Any other build configuration files

These files have been carefully configured in the stable branch. Changes from the problematic branch could reintroduce instability or break the build system improvements.

## Tips

- Cherry-pick documentation/build system improvements first
- Be very cautious with commits that touch sim.cpp, mgr.cpp, or physics code
- If unsure, test with 1000 iterations
- Keep a log of what commits have been tested
- Always preserve current branch's build system configuration