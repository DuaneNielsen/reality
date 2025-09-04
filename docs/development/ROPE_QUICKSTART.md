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

## Common Issues & Solutions

### Issue: External Dependencies Cause Syntax Errors
**Problem**: Rope scans all Python files and crashes on Python 2 code in `external/`

**Solution**: Configure `ignored_resources` patterns in `.clirope/config.py`
```python
prefs['ignored_resources'] = ['external*', 'build*', 'scratch*']
```

### Issue: Class Attributes Not Renameable via CLI
**Problem**: `rope rename file.py::Class.attribute new_name` fails

**Solution**: Use rope library directly with offset-based renaming:
```python
import rope.base.project
from rope.refactor.rename import Rename

proj = rope.base.project.Project('.')
file_resource = proj.get_file('file.py')
content = file_resource.read()
attribute_offset = content.find('attribute_name')
renamer = Rename(proj, file_resource, attribute_offset)
changes = renamer.get_changes('new_attribute_name')
proj.do(changes)
```

### Issue: Cross-File References Not Updated
**Problem**: Renaming in one file doesn't update imports/references in other files

**Root Cause**: Each rope operation creates separate project instances, losing cross-file context

**Solution**: Use single rope project instance for related operations:
```python
import rope.base.project
from rope.refactor.rename import Rename

# Create ONE project instance for all operations
proj = rope.base.project.Project('.')

# Do multiple renames with same project instance
for old_name, new_name in [('COMPASS_0', 'FIRST'), ('COMPASS_64', 'NORTH')]:
    file_resource = proj.get_file('target_file.py')
    content = file_resource.read()
    offset = content.find(old_name)
    if offset != -1:
        renamer = Rename(proj, file_resource, offset)
        changes = renamer.get_changes(new_name)
        proj.do(changes)
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

## Best Practices

1. **Always test configuration first** - verify ignored patterns work
2. **Use library interface for complex operations** - CLI has limitations
3. **Keep single project instance** - for related refactoring operations
4. **Test after refactoring** - run tests to verify all references updated
5. **Commit frequently** - rope changes can be extensive

## Example: Moving Constants (Complete Workflow)

```python
import rope.base.project
from rope.refactor.move import create_move
from rope.refactor.rename import Rename

# Single project instance for all operations
proj = rope.base.project.Project('.')

# 1. Move classes from test to proper module
source_file = proj.get_file('tests/python/test_file.py')
target_file = proj.get_file('src/proper_module.py')

for class_name in ['ObsIndex', 'SelfObsIndex', 'CompassIndex']:
    source_mod = proj.get_pymodule(source_file)
    class_obj = source_mod.get_attribute(class_name)
    mover = create_move(proj, class_obj)
    changes = mover.get_changes(target_file)
    proj.do(changes)

# 2. Rename constants to semantic names
renames = [
    ('COMPASS_0', 'FIRST'),
    ('COMPASS_64', 'NORTH'),
    ('COMPASS_127', 'LAST'),
    ('BEAM_0', 'LEFTMOST'),
    ('BEAM_64', 'CENTER'),
    ('BEAM_127', 'RIGHTMOST')
]

for old_name, new_name in renames:
    content = target_file.read()
    offset = content.find(old_name)
    if offset != -1:
        renamer = Rename(proj, target_file, offset)
        changes = renamer.get_changes(new_name)
        proj.do(changes)
        # Refresh file content for next iteration
        target_file = proj.get_file(target_file.path)
```

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