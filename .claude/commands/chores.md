# Repository Maintenance Chores

This document provides step-by-step instructions for regular repository maintenance activities based on successful cleanup patterns.

## Table of Contents
- [Documentation Alignment](#documentation-alignment)
- [Binary File Cleanup](#binary-file-cleanup)
- [Asset Organization](#asset-organization)
- [GitIgnore Management](#gitignore-management)
- [Commit Strategy](#commit-strategy)

## Documentation Alignment

### Purpose
Keep documentation accurate and aligned with the current codebase state.

### When to Run
- After major feature changes or removals
- When adding new capabilities
- During code reviews
- Monthly maintenance

### Steps

1. **Audit Documentation vs Code Reality**
   ```bash
   # Check current feature flags in code
   grep -r "numAgents\|numRooms" src/
   grep -r "grab\|door\|button" src/types.hpp
   ```

2. **Update README.md**
   - Verify environment description matches actual implementation
   - Check action/observation space descriptions
   - Update feature lists and capabilities
   - Ensure Python examples use correct tensor shapes

3. **Update ENVIRONMENT.md**
   - Verify tensor shapes match current implementation
   - Check action constants and ranges
   - Update usage examples with correct API calls
   - Test example code for accuracy

4. **Update Technical Documentation**
   - Review architecture docs for outdated references
   - Update Python bindings guide with current API
   - Check deployment guides for accuracy

5. **Verification**
   ```bash
   # Test documentation examples
   python -c "import madrona_escape_room; print('API accessible')"
   # Check for broken internal links
   find docs/ -name "*.md" -exec grep -l "\[.*\](.*.md)" {} \;
   ```

## Binary File Cleanup

### Purpose
Remove build artifacts, temporary files, and binary data from version control.

### When to Run
- Before major releases
- When repository size grows significantly
- After development sprints with lots of testing
- Monthly maintenance

### File Types to Remove

**Always Remove from Root Directory:**
- `*.csv` (test results, recordings)
- `*.rec` (simulation recordings)
- `*.xml` (test output, results)
- `*.bin` (binary test data)
- `*.png` (temporary screenshots)
- `*.txt` (temporary notes, except CMakeLists.txt)
- `*.mp4` (video recordings)

**Keep These Files:**
- `CMakeLists.txt` (build system)
- `README.md`, `*.md` (documentation)
- `*.py`, `*.cpp`, `*.hpp` (source code)

### Cleanup Steps

1. **Find Files to Remove**
   ```bash
   # Find binary/media files in root
   find . -maxdepth 1 -name "*.csv" -o -name "*.rec" -o -name "*.xml" -o -name "*.bin" -o -name "*.png" -o -name "*.mp4"
   
   # Find large files anywhere
   find . -size +1M -type f | grep -v ".git" | grep -v "build/"
   ```

2. **Remove from Git Tracking**
   ```bash
   # Remove files from git but keep locally
   git rm --cached file1.csv file2.rec file3.xml
   
   # Or remove multiple files matching pattern
   git ls-files | grep -E "\.(csv|rec|xml|bin)$" | xargs git rm --cached
   ```

3. **Delete from Filesystem** (if not needed)
   ```bash
   rm *.csv *.rec *.xml  # Be careful with wildcards
   ```

4. **Update .gitignore** (see GitIgnore Management section)

## Asset Organization

### Purpose
Organize project assets in appropriate directories.

### Directory Structure
```
/
├── resources/          # Project media (images, icons)
├── docs/               # All documentation
│   ├── architecture/   # System design docs
│   ├── development/    # Developer guides  
│   ├── deployment/     # Production guides
│   └── tools/          # Tool documentation
├── src/                # Source code
├── tests/              # Test files
└── scratch/            # Temporary development files
```

### Steps

1. **Move Documentation**
   ```bash
   # Move docs to appropriate subdirectories
   mv SOME_GUIDE.md docs/development/
   mv ARCHITECTURE_DOC.md docs/architecture/
   ```

2. **Organize Media Files**
   ```bash
   # Create resources directory if needed
   mkdir -p resources/
   
   # Move images used in documentation
   mv important_screenshot.png resources/
   
   # Update references in documentation
   sed -i 's|important_screenshot.png|resources/important_screenshot.png|g' README.md
   ```

3. **Update Documentation Links**
   - Update README.md documentation index
   - Fix any broken internal links
   - Ensure relative paths are correct

## GitIgnore Management

### Purpose
Prevent unwanted files from being committed to version control.

### Current Patterns
The .gitignore should include these patterns:

```gitignore
# Root-level media and result files
/*.png
/*.rec
/*.csv
/*.xml
/*.bin
/*.txt    # Add if needed

# Binary files globally
*.bin

# Build artifacts
/build
/out
*.so

# IDE and cache files  
.idea/
.cache/
.clangd/
**/__pycache__
*.pyc

# Temporary/scratch files
scratch/
```

### Steps

1. **Review Current .gitignore**
   ```bash
   cat .gitignore
   ```

2. **Add Missing Patterns**
   ```bash
   # Add patterns for new file types as needed
   echo "/*.txt" >> .gitignore  # If removing TXT files from root
   echo "*.log" >> .gitignore   # Log files
   ```

3. **Test Patterns**
   ```bash
   # Check what files would be ignored
   git check-ignore -v some_file.csv
   
   # List all ignored files
   git ls-files --others --ignored --exclude-standard
   ```

4. **Clean Up Tracked Files**
   ```bash
   # Remove files that should now be ignored
   git ls-files | grep -E "pattern_to_remove" | xargs git rm --cached
   ```

## Commit Strategy

### Purpose
Organize changes into logical, reviewable commits.

### Commit Types

1. **Documentation Updates**
   ```bash
   git add README.md ENVIRONMENT.md docs/
   git commit -m "docs: update documentation to reflect single-agent environment"
   ```

2. **Repository Cleanup**
   ```bash
   git add .gitignore
   git rm --cached *.csv *.rec *.xml *.bin
   git commit -m "chore: remove binary files and update gitignore patterns"
   ```

3. **Asset Organization**
   ```bash
   git add resources/ docs/
   git commit -m "refactor: organize project assets and documentation structure"
   ```

### Best Practices

- **Separate logical changes** into different commits
- **Use conventional commit messages** (feat:, fix:, docs:, chore:, refactor:)
- **Test before committing** to ensure nothing is broken
- **Batch related changes** but keep commits focused

### Example Workflow

```bash
# 1. Documentation updates
git add README.md ENVIRONMENT.md docs/
git commit -m "docs: align documentation with current codebase features"

# 2. File cleanup  
git rm --cached *.csv *.rec *.xml
git add .gitignore
git commit -m "chore: remove binary files and update gitignore"

# 3. Asset organization
git add resources/
git commit -m "refactor: organize project media in resources directory"

# 4. Push all changes
git push origin feature-branch
```

## Automation Ideas

### Monthly Maintenance Script
Create a script to automate common tasks:

```bash
#!/bin/bash
# monthly_cleanup.sh

echo "=== Monthly Repository Maintenance ==="

echo "1. Finding binary files in root..."
find . -maxdepth 1 -name "*.csv" -o -name "*.rec" -o -name "*.xml" -o -name "*.bin"

echo "2. Checking large files..."
find . -size +5M -type f | grep -v ".git" | grep -v "build/"

echo "3. Checking documentation links..."
find docs/ -name "*.md" -exec grep -l "broken_link_pattern" {} \;

echo "4. Verifying gitignore effectiveness..."
git ls-files | grep -E "\.(csv|rec|xml|bin)$" || echo "No binary files tracked ✓"

echo "=== Maintenance check complete ==="
```

## Troubleshooting

### Common Issues

1. **Files keep being committed despite .gitignore**
   - Solution: Remove from tracking with `git rm --cached filename`

2. **Documentation links are broken**
   - Solution: Use relative paths, check with `find docs/ -name "*.md"`

3. **Large repository size**
   - Solution: Use `git filter-branch` to remove large files from history (destructive)

4. **Accidental binary file commits**
   - Solution: `git reset --soft HEAD~1` if not pushed yet

### Recovery Commands

```bash
# Undo last commit (keep changes)
git reset --soft HEAD~1

# Remove file from last commit
git reset HEAD~1 -- unwanted_file.bin
git commit --amend

# Force update gitignore for already tracked files
git rm -r --cached .
git add .
git commit -m "chore: apply gitignore to all files"
```

## Checklist

Use this checklist for regular maintenance:

### Weekly
- [ ] Remove temporary files from root directory
- [ ] Check for new binary files in `git status`
- [ ] Verify documentation examples still work

### Monthly  
- [ ] Run full binary file cleanup
- [ ] Review and update .gitignore patterns
- [ ] Check documentation accuracy vs code
- [ ] Organize any misplaced assets
- [ ] Update README documentation links

### Before Releases
- [ ] Complete documentation audit
- [ ] Remove all development artifacts
- [ ] Verify asset organization
- [ ] Test all documentation examples
- [ ] Clean commit history with logical grouping