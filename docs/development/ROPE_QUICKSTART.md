# Rope Refactoring Quickstart

This guide covers using rope for Python refactoring in this codebase.

## Setup

Rope CLI uses `.clirope/` directory for configuration instead of the standard `.ropeproject/`.

### Configuration

Copy our rope configuration to the CLI location:
```bash
cp .ropeproject/config.py .clirope/config.py
```

The key configuration in `config.py`:
```python
prefs['ignored_resources'] = ['*.pyc', '*~', '.ropeproject',
                              '.hg', '.svn', '_svn', '.git', '.tox',
                              'external*', 'scratch*', 'build*', '.venv*']
```

## Basic Commands

### Move Classes/Functions
```bash
# Move a class from one module to another
uv run rope move source_file.py::ClassName target_file.py

# Example: Move constants from test to proper module
uv run rope move tests/python/test_file.py::ObsIndex train_src/module/file.py
```

### Rename Variables/Classes
```bash
# Rename using CLI (limited - doesn't work for class attributes)
uv run rope rename module.py::old_name new_name

# Better: Use rope library directly for complex renames
uv run python -c "
import rope.base.project
from rope.refactor.rename import Rename

proj = rope.base.project.Project('.')
target_file = proj.get_file('path/to/file.py')
content = target_file.read()
offset = content.find('old_name')
renamer = Rename(proj, target_file, offset)
changes = renamer.get_changes('new_name')
proj.do(changes)
"
```

### List Entities
```bash
# See what rope can refactor in a file
uv run rope list path/to/file.py
```

## What Works Well with Rope

### ✅ Moving Classes with CLI - RECOMMENDED APPROACH
**Use Case**: Moving entire classes between modules

**Command**: 
```bash
uv run rope move source_file.py::ClassName target_file.py
```

**Real Example that WORKED**:
```bash
# Successfully moved constants from test to proper module
uv run rope move tests/python/test_sim_interface_adapter.py::ObsIndex train_src/madrona_escape_room_learn/sim_interface_adapter.py
uv run rope move tests/python/test_sim_interface_adapter.py::SelfObsIndex train_src/madrona_escape_room_learn/sim_interface_adapter.py
uv run rope move tests/python/test_sim_interface_adapter.py::CompassIndex train_src/madrona_escape_room_learn/sim_interface_adapter.py
uv run rope move tests/python/test_sim_interface_adapter.py::DepthIndex train_src/madrona_escape_room_learn/sim_interface_adapter.py
```

**What Rope Automatically Does**:
- ✅ Moves the class definition to target file
- ✅ Adds necessary import statements
- ✅ Updates most direct references (e.g., `SelfObsIndex.PROGRESS` → `module.SelfObsIndex.PROGRESS`)
- ✅ Maintains code structure and comments

## Common Issues & Solutions

### Issue: External Dependencies Cause Syntax Errors
**Problem**: Rope scans all Python files and crashes on Python 2 code in `external/`

**Solution**: Configure `ignored_resources` patterns in `.clirope/config.py`
```python
prefs['ignored_resources'] = ['external*', 'build*', 'scratch*']
```

### ⚠️ Issue: Rope Misses Constant Reassignments
**Problem**: Code like `SomeConstant = SomeConstant` inside classes creates NameErrors after move

**Root Cause**: This is terrible code practice - rope correctly removes the class but can't update these idiotic assignments

**Solution**: Fix the bad code pattern FIRST, then use rope:
```python
# BAD - Will break during rope refactoring
class MyClass:
    SomeConstant = SomeConstant  # NEVER DO THIS

# GOOD - Works perfectly with rope
from some_module import SomeConstant
class MyClass:
    pass  # Just use SomeConstant directly in methods
```

### Issue: Long Qualified Names After Move
**Problem**: Rope creates long imports like `module.submodule.Class.CONSTANT`

**Solution**: Clean up with proper imports:
```python
# Rope generates this
import module.submodule
result = module.submodule.Class.CONSTANT

# Clean it up to this
from module.submodule import Class
result = Class.CONSTANT
```

## Debugging Rope Issues

### Check Configuration Loading
```python
import rope.base.project
proj = rope.base.project.Project('.')
print('Ignored resources:', proj.prefs.get('ignored_resources'))
```

### Verify File Discovery
```python
import rope.base.project
proj = rope.base.project.Project('.')
all_files = list(proj.get_files())
print(f'Total files: {len(all_files)}')
external_files = [f for f in all_files if f.path.startswith('external')]
print(f'External files found: {len(external_files)} (should be 0)')
```

### Test Pattern Matching
```python
import fnmatch
patterns = ['external*', 'scratch*', 'build*']
test_files = ['external/file.py', 'src/main.py', 'scratch/test.py']
for test_file in test_files:
    matches = [p for p in patterns if fnmatch.fnmatch(test_file, p)]
    print(f'{test_file}: {"IGNORED" if matches else "NOT IGNORED"}')
```

## Best Practices (Based on Real Experience)

1. **Use rope CLI for class moves** - It works great and is much simpler than library interface
2. **Fix bad code patterns FIRST** - Remove `SomeConstant = SomeConstant` assignments before refactoring
3. **Always test configuration first** - verify ignored patterns work
4. **Clean up imports after rope** - Replace long qualified names with proper `from module import` statements  
5. **Test after refactoring** - run tests to verify all references updated
6. **Commit frequently** - rope changes can be extensive

## Example: Moving Constants (What Actually Worked)

**Step 1**: Fix bad code patterns first
```python
# BEFORE (in test file) - This will break rope refactoring
class SimInterface:
    ObsIndex = ObsIndex          # BAD - will cause NameError
    SelfObsIndex = SelfObsIndex  # BAD - will cause NameError
```

**Step 2**: Use simple rope CLI commands (MUCH easier than library interface)
```bash
# These commands worked perfectly
uv run rope move tests/python/test_sim_interface_adapter.py::ObsIndex train_src/madrona_escape_room_learn/sim_interface_adapter.py
uv run rope move tests/python/test_sim_interface_adapter.py::SelfObsIndex train_src/madrona_escape_room_learn/sim_interface_adapter.py  
uv run rope move tests/python/test_sim_interface_adapter.py::CompassIndex train_src/madrona_escape_room_learn/sim_interface_adapter.py
uv run rope move tests/python/test_sim_interface_adapter.py::DepthIndex train_src/madrona_escape_room_learn/sim_interface_adapter.py
```

**Step 3**: Clean up the imports rope created
```python
# AFTER - Rope creates long imports, clean them up
# Replace this:
import madrona_escape_room_learn.sim_interface_adapter  
result = madrona_escape_room_learn.sim_interface_adapter.SelfObsIndex.PROGRESS

# With this:
from madrona_escape_room_learn.sim_interface_adapter import SelfObsIndex
result = SelfObsIndex.PROGRESS
```

**Total Time**: ~5 minutes with CLI vs hours debugging library interface

## Troubleshooting

### Configuration Not Loading
- Check `.clirope/config.py` exists (not `.ropeproject/`)
- Verify Python syntax in config file
- Test with `rope.base.project.Project('.')` in Python

### Rename Operations Fail
- Use `rope list file.py` to see what's renameable
- Try library interface instead of CLI
- Check for syntax errors in target files

### Cross-File References Broken
- Use single project instance for related operations
- Run comprehensive tests after refactoring
- Consider manual fixes for edge cases rope misses