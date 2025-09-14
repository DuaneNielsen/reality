# Unified Level Format Implementation Plan

**Date:** 2025-01-12
**Purpose:** Eliminate distinction between single and multi-level files by using one unified format

## Problem Statement

Currently we have two different file formats:
1. **Single-level**: Raw `CompiledLevel` struct dump (85,200 bytes)
2. **Multi-level**: "MLEVL" header + count + array of `CompiledLevel` structs

This causes:
- Duplicate functions (`save_compiled_level` vs `save_compiled_levels`)
- Special case handling in tools
- Confusion about file types
- Crashes when tools expect one format but get another

## Solution: Unified Format

All level files will use the same format:
```
[Magic: "LEVELS"] [Count: uint32] [Level1] [Level2] ... [LevelN]
```

- Single level file: Count = 1
- Multi level file: Count = N
- Every file is self-describing

## Detailed Implementation

### 1. C API Changes (`src/madrona_escape_room_c_api.cpp`)

#### Remove these functions entirely:
```cpp
// DELETE THESE:
MER_Result mer_write_compiled_level(const char* filepath, const void* level);
MER_Result mer_read_compiled_level(const char* filepath, void* out_level);
```

#### Update mer_write_compiled_levels:
```cpp
MER_Result mer_write_compiled_levels(
    const char* filepath,
    const void* compiled_levels,
    uint32_t num_levels
) {
    // Validation
    if (!filepath || !compiled_levels || num_levels == 0) {
        return MER_ERROR_INVALID_PARAMETER;
    }
    
    FILE* f = fopen(filepath, "wb");
    if (!f) {
        return MER_ERROR_FILE_NOT_FOUND;
    }
    
    // Write unified format header
    const char magic[] = "LEVELS";  // Changed from "MLEVL"
    size_t magic_written = fwrite(magic, sizeof(char), 6, f);
    if (magic_written != 6) {
        fclose(f);
        return MER_ERROR_FILE_IO;
    }
    
    // Write count
    size_t count_written = fwrite(&num_levels, sizeof(uint32_t), 1, f);
    if (count_written != 1) {
        fclose(f);
        return MER_ERROR_FILE_IO;
    }
    
    // Write all levels
    size_t levels_size = sizeof(CompiledLevel) * num_levels;
    size_t levels_written = fwrite(compiled_levels, 1, levels_size, f);
    if (levels_written != levels_size) {
        fclose(f);
        return MER_ERROR_FILE_IO;
    }
    
    fclose(f);
    return MER_SUCCESS;
}
```

#### Update mer_read_compiled_levels:
```cpp
MER_Result mer_read_compiled_levels(
    const char* filepath,
    void* out_levels,
    uint32_t* out_num_levels,
    uint32_t max_levels
) {
    if (!filepath || !out_num_levels) {
        return MER_ERROR_INVALID_PARAMETER;
    }
    
    FILE* f = fopen(filepath, "rb");
    if (!f) {
        return MER_ERROR_FILE_NOT_FOUND;
    }
    
    // Check file size for backward compatibility
    fseek(f, 0, SEEK_END);
    long file_size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    // Handle old single-level format (raw struct dump)
    if (file_size == sizeof(CompiledLevel)) {
        // Old format detected
        *out_num_levels = 1;
        
        if (out_levels && max_levels >= 1) {
            size_t read = fread(out_levels, sizeof(CompiledLevel), 1, f);
            if (read != 1) {
                fclose(f);
                return MER_ERROR_FILE_IO;
            }
        }
        
        fclose(f);
        return MER_SUCCESS;
    }
    
    // Read magic header
    char magic[7] = {0};
    size_t magic_read = fread(magic, sizeof(char), 6, f);
    
    // Handle both "LEVELS" (new) and "MLEVL" (old multi) for compatibility
    bool is_new_format = (magic_read == 6 && strcmp(magic, "LEVELS") == 0);
    bool is_old_multi = (magic_read == 5 && strncmp(magic, "MLEVL", 5) == 0);
    
    if (!is_new_format && !is_old_multi) {
        fclose(f);
        return MER_ERROR_INVALID_FILE;
    }
    
    // For old "MLEVL" format, we already read 5 bytes, need to reposition
    if (is_old_multi) {
        fseek(f, 5, SEEK_SET);
    }
    
    // Read number of levels
    uint32_t num_levels;
    size_t count_read = fread(&num_levels, sizeof(uint32_t), 1, f);
    if (count_read != 1) {
        fclose(f);
        return MER_ERROR_FILE_IO;
    }
    
    *out_num_levels = num_levels;
    
    // Read levels if buffer provided
    if (out_levels && max_levels > 0) {
        uint32_t levels_to_read = (num_levels < max_levels) ? num_levels : max_levels;
        size_t levels_size = sizeof(CompiledLevel) * levels_to_read;
        size_t read = fread(out_levels, 1, levels_size, f);
        if (read != levels_size) {
            fclose(f);
            return MER_ERROR_FILE_IO;
        }
    }
    
    fclose(f);
    return MER_SUCCESS;
}
```

### 2. Python Changes (`madrona_escape_room/level_io.py`)

#### Remove single-level functions:
```python
# DELETE THESE FUNCTIONS:
# def save_compiled_level(level: CompiledLevel, filepath: Union[str, Path]) -> None:
# def load_compiled_level(filepath: Union[str, Path]) -> CompiledLevel:
```

#### Keep only the plural versions (unchanged):
```python
def save_compiled_levels(levels: List[CompiledLevel], filepath: Union[str, Path]) -> None:
    """Save list of CompiledLevel structs to binary file.
    
    For single levels, pass a list of one: save_compiled_levels([level], path)
    """
    # ... existing implementation stays the same
    
def load_compiled_levels(filepath: Union[str, Path]) -> List[CompiledLevel]:
    """Load list of CompiledLevel structs from binary file.
    
    Returns list with one element for single-level files.
    """
    # ... existing implementation stays the same
```

### 3. Viewer Changes (`src/viewer.cpp`)

#### Update level loading (~line 380):
```cpp
// Replace the current level loading section with:

std::vector<CompiledLevel> loaded_levels;
std::vector<CompiledLevel> per_world_levels;

if (has_replay) {
    // Replay mode - use embedded level
    auto embedded_level = Manager::readEmbeddedLevel(replay_path);
    if (!embedded_level.has_value()) {
        std::cerr << "Error: Failed to read embedded level from replay\n";
        return 1;
    }
    loaded_levels.push_back(embedded_level.value());
    
} else if (use_embedded_default) {
    // Use default level
    CompiledLevel default_level = getDefaultLevel();
    loaded_levels.push_back(default_level);
    
} else if (!load_path.empty()) {
    // Load from file - now always expects a list
    uint32_t num_levels = 0;
    
    // First pass: get count
    MER_Result result = mer_read_compiled_levels(
        load_path.c_str(), 
        nullptr,  // Just getting count
        &num_levels,
        0
    );
    
    if (result != MER_SUCCESS) {
        std::cerr << "Error: Failed to read level file: " << load_path << "\n";
        return 1;
    }
    
    // Second pass: read levels
    loaded_levels.resize(num_levels);
    result = mer_read_compiled_levels(
        load_path.c_str(),
        loaded_levels.data(),
        &num_levels,
        num_levels
    );
    
    if (result != MER_SUCCESS) {
        std::cerr << "Error: Failed to load levels from: " << load_path << "\n";
        return 1;
    }
    
    printf("Loaded %u level(s) from %s\n", num_levels, load_path.c_str());
    if (num_levels > 1) {
        printf("  Distribution: %u worlds will use levels round-robin\n", num_worlds);
    }
}

// Distribute levels across worlds
per_world_levels.resize(num_worlds);
for (uint32_t i = 0; i < num_worlds; i++) {
    per_world_levels[i] = loaded_levels[i % loaded_levels.size()];
}

// Pass per_world_levels to Manager config
Manager::Config mgr_cfg = {
    // ... existing config ...
    .perWorldCompiledLevels = per_world_levels,
};
```

### 4. Headless Changes (`src/headless.cpp`)

Similar changes to viewer.cpp for level loading section.

### 5. File Inspector Changes (`src/file_inspector.cpp`)

#### Update validateLevelFile (~line 143):
```cpp
bool validateLevelFile(const FileInfo& info) {
    std::cout << "Level File: " << std::filesystem::path(info.filepath).filename().string() << "\n";
    
    // Try to read using new API
    uint32_t num_levels = 0;
    MER_Result result = mer_read_compiled_levels(
        info.filepath.c_str(),
        nullptr,
        &num_levels, 
        0
    );
    
    if (result != MER_SUCCESS) {
        std::cout << "✗ Failed to read level file (error: " << result << ")\n";
        return false;
    }
    
    std::cout << "✓ Valid level file format\n";
    std::cout << "  Contains " << num_levels << " level(s)\n";
    
    // Read and validate each level
    std::vector<CompiledLevel> levels(num_levels);
    result = mer_read_compiled_levels(
        info.filepath.c_str(),
        levels.data(),
        &num_levels,
        num_levels
    );
    
    if (result != MER_SUCCESS) {
        std::cout << "✗ Failed to load level data\n";
        return false;
    }
    
    // Validate each level
    for (uint32_t i = 0; i < num_levels; i++) {
        const auto& level = levels[i];
        std::cout << "\nLevel " << (i+1) << "/" << num_levels << ":\n";
        std::cout << "  Name: " << level.level_name << "\n";
        std::cout << "  Grid: " << level.width << "x" << level.height << "\n";
        std::cout << "  Scale: " << level.world_scale << "\n";
        std::cout << "  Tiles: " << level.num_tiles << "\n";
        std::cout << "  Spawns: " << level.num_spawns << "\n";
        
        // Validate ranges
        bool valid = true;
        if (level.width <= 0 || level.width > consts::limits::maxGridSize) {
            std::cout << "  ✗ Invalid width: " << level.width << "\n";
            valid = false;
        }
        if (level.height <= 0 || level.height > consts::limits::maxGridSize) {
            std::cout << "  ✗ Invalid height: " << level.height << "\n";
            valid = false;
        }
        // ... other validations ...
        
        if (valid) {
            std::cout << "  ✓ Level data valid\n";
        }
    }
    
    // Show distribution example
    if (num_levels > 1) {
        std::cout << "\nLevel Distribution Examples:\n";
        std::cout << "  10 worlds: ";
        for (int i = 0; i < std::min(10u, num_levels); i++) {
            std::cout << (i % num_levels + 1) << " ";
        }
        std::cout << "...\n";
        
        std::cout << "  100 worlds: each level used " 
                  << (100 / num_levels) << "-" << (100 / num_levels + 1) 
                  << " times\n";
    }
    
    return true;
}
```

### 6. Manager Changes (`src/mgr.cpp`)

The manager already accepts `perWorldCompiledLevels` vector in its config, so no changes needed there. The distribution logic happens in the tools before passing to Manager.

## Migration Path

1. **Phase 1**: Implement changes with backward compatibility
   - Old single-level raw files still work
   - Old "MLEVL" multi-level files still work
   - New "LEVELS" format is written

2. **Phase 2**: Update all level generators to use new format
   - Python scripts that create levels
   - Default level generator

3. **Phase 3**: Convert existing level files
   - Simple script to read old format and write new format

4. **Phase 4**: Remove backward compatibility (future)
   - After all files converted
   - Simplify read functions

## Testing Plan

1. Test backward compatibility:
   - Old single-level files load correctly
   - Old multi-level files load correctly

2. Test new format:
   - Single level (count=1) saves and loads
   - Multi-level (count=N) saves and loads

3. Test distribution:
   - 1 level, 100 worlds → all worlds get same level
   - 20 levels, 100 worlds → each level used 5 times
   - 100 levels, 20 worlds → only first 20 levels used

4. Test tools:
   - Viewer displays correct level per world
   - File inspector shows all levels
   - Headless runs with different world/level ratios

## Benefits

- **Simplicity**: One format, one set of functions
- **Consistency**: No special cases in code
- **Self-describing**: Files always specify their contents
- **Clean API**: No duplicate functions
- **Future-proof**: Easy to add metadata later