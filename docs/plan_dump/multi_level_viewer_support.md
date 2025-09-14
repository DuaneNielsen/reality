# Multi-Level Support for Viewer, Headless, and File Inspector

**Date:** 2025-01-12
**Issue:** Multi-level .lvl files crash viewer/headless/file_inspector
**Root Cause:** Tools expect single-level format but multi-level uses different format with header

## Problem Analysis

### Current File Formats
1. **Single-level .lvl**: Raw `CompiledLevel` struct (85200 bytes)
2. **Multi-level .lvl**: "MLEVL" header + count + array of `CompiledLevel` structs

### Current Behavior
- `file_inspector`: Expects exactly 85200 bytes, fails on multi-level files
- `viewer`: Reads raw struct, gets garbage data from "MLEVL" header, segfaults
- `headless`: Same issue as viewer
- Python `save_compiled_levels()`: Writes "MLEVL" format
- C++ tools: Only understand raw single-level format

## Implementation Plan

### 1. Create Shared Level Loading Utilities

Add to `src/level_loader.hpp` and `src/level_loader.cpp`:

```cpp
namespace madEscape {

struct LevelFileInfo {
    bool is_multi_level;
    uint32_t num_levels;
    std::string filepath;
};

// Detect file format by checking magic header
LevelFileInfo detectLevelFormat(const std::string& filepath);

// Load single level (works for both formats)
std::optional<CompiledLevel> loadLevelFromFile(
    const std::string& filepath, 
    uint32_t index = 0
);

// Load all levels from multi-level file
std::vector<CompiledLevel> loadAllLevels(const std::string& filepath);

}
```

### 2. Update file_inspector.cpp

```cpp
bool validateLevelFile(const FileInfo& info) {
    auto level_info = madEscape::detectLevelFormat(info.filepath);
    
    if (level_info.is_multi_level) {
        return validateMultiLevelFile(info, level_info);
    } else {
        return validateSingleLevelFile(info);
    }
}

bool validateMultiLevelFile(const FileInfo& info, const LevelFileInfo& level_info) {
    std::cout << "Multi-Level File: " << filename << "\n";
    std::cout << "✓ Valid MLEVL format\n";
    std::cout << "  Contains " << level_info.num_levels << " levels\n";
    
    // Validate expected size
    size_t expected_size = 5 + 4 + (level_info.num_levels * sizeof(CompiledLevel));
    if (info.file_size == expected_size) {
        std::cout << "✓ File size correct (" << info.file_size << " bytes)\n";
    } else {
        std::cout << "✗ Invalid size: " << info.file_size << " (expected " << expected_size << ")\n";
        return false;
    }
    
    // Load and validate each level
    auto levels = loadAllLevels(info.filepath);
    for (uint32_t i = 0; i < levels.size(); i++) {
        std::cout << "\nLevel " << (i+1) << "/" << levels.size() << ":\n";
        std::cout << "  Name: " << levels[i].level_name << "\n";
        std::cout << "  Grid: " << levels[i].width << "x" << levels[i].height << "\n";
        std::cout << "  Tiles: " << levels[i].num_tiles << "\n";
        // Validate each level's data...
    }
    
    return true;
}
```

### 3. Update viewer.cpp

Add command-line option:
```cpp
enum OptionIndex {
    // ... existing options ...
    LEVEL_INDEX,  // New option for multi-level files
};

const option::Descriptor usage[] = {
    // ... existing options ...
    {LEVEL_INDEX, 0, "", "level-index", ArgChecker::Numeric, 
     "  --level-index <n>  \tSelect level N from multi-level file (default: 0)"},
};
```

Update level loading:
```cpp
if (!load_path.empty()) {
    auto level_info = madEscape::detectLevelFormat(load_path);
    
    uint32_t level_index = 0;
    if (options[LEVEL_INDEX]) {
        level_index = strtoul(options[LEVEL_INDEX].arg, nullptr, 10);
    }
    
    if (level_info.is_multi_level) {
        std::cout << "Multi-level file detected: " << level_info.num_levels << " levels\n";
        
        if (level_index >= level_info.num_levels) {
            std::cerr << "Error: Level index " << level_index 
                      << " out of range (0-" << (level_info.num_levels-1) << ")\n";
            return 1;
        }
        
        std::cout << "Loading level " << level_index << " (use --level-index to select)\n";
    }
    
    auto level_opt = madEscape::loadLevelFromFile(load_path, level_index);
    if (!level_opt.has_value()) {
        std::cerr << "Error: Failed to load level from: " << load_path << "\n";
        return 1;
    }
    
    loaded_level = level_opt.value();
    printf("Loaded level '%s': %dx%d grid, %d tiles\n", 
           loaded_level.level_name, loaded_level.width, 
           loaded_level.height, loaded_level.num_tiles);
}
```

### 4. Update headless.cpp

Same changes as viewer:
- Add `--level-index` option
- Use shared loading utilities
- Display multi-level info when detected

### 5. Implementation of Level Loader

`src/level_loader.cpp`:
```cpp
namespace madEscape {

LevelFileInfo detectLevelFormat(const std::string& filepath) {
    LevelFileInfo info;
    info.filepath = filepath;
    info.is_multi_level = false;
    info.num_levels = 1;
    
    FILE* f = fopen(filepath.c_str(), "rb");
    if (!f) return info;
    
    // Check for MLEVL magic
    char magic[6] = {0};
    size_t read = fread(magic, 1, 5, f);
    
    if (read == 5 && strcmp(magic, "MLEVL") == 0) {
        info.is_multi_level = true;
        fread(&info.num_levels, sizeof(uint32_t), 1, f);
    }
    
    fclose(f);
    return info;
}

std::optional<CompiledLevel> loadLevelFromFile(
    const std::string& filepath, 
    uint32_t index
) {
    auto info = detectLevelFormat(filepath);
    
    FILE* f = fopen(filepath.c_str(), "rb");
    if (!f) return std::nullopt;
    
    CompiledLevel level;
    
    if (info.is_multi_level) {
        // Skip header and previous levels
        fseek(f, 5 + 4 + (index * sizeof(CompiledLevel)), SEEK_SET);
        
        if (index >= info.num_levels) {
            fclose(f);
            return std::nullopt;
        }
    }
    
    size_t read = fread(&level, sizeof(CompiledLevel), 1, f);
    fclose(f);
    
    if (read != 1) return std::nullopt;
    
    return level;
}

std::vector<CompiledLevel> loadAllLevels(const std::string& filepath) {
    std::vector<CompiledLevel> levels;
    
    auto info = detectLevelFormat(filepath);
    if (!info.is_multi_level) {
        // Single level - just load it
        auto level = loadLevelFromFile(filepath, 0);
        if (level.has_value()) {
            levels.push_back(level.value());
        }
        return levels;
    }
    
    // Multi-level - load all
    for (uint32_t i = 0; i < info.num_levels; i++) {
        auto level = loadLevelFromFile(filepath, i);
        if (level.has_value()) {
            levels.push_back(level.value());
        }
    }
    
    return levels;
}

}
```

### 6. Add C++ Tests

Create `tests/cpp/unit/test_multi_level_loading.cpp`:
```cpp
TEST(MultiLevelTest, DetectSingleLevelFormat) {
    // Create single-level file
    CompiledLevel level = createTestLevel();
    writeSingleLevel("test_single.lvl", level);
    
    auto info = detectLevelFormat("test_single.lvl");
    EXPECT_FALSE(info.is_multi_level);
    EXPECT_EQ(info.num_levels, 1);
}

TEST(MultiLevelTest, DetectMultiLevelFormat) {
    // Create multi-level file
    std::vector<CompiledLevel> levels = {
        createTestLevel("level1"),
        createTestLevel("level2"),
        createTestLevel("level3")
    };
    writeMultiLevel("test_multi.lvl", levels);
    
    auto info = detectLevelFormat("test_multi.lvl");
    EXPECT_TRUE(info.is_multi_level);
    EXPECT_EQ(info.num_levels, 3);
}

TEST(MultiLevelTest, LoadFromMultiLevel) {
    // Test loading specific index
    auto level = loadLevelFromFile("test_multi.lvl", 1);
    ASSERT_TRUE(level.has_value());
    EXPECT_STREQ(level->level_name, "level2");
}
```

## Benefits

1. **Backward Compatible**: Single-level files continue to work unchanged
2. **Auto-Detection**: Tools automatically detect format from file header
3. **User-Friendly**: Clear messages about multi-level files and how to select levels
4. **Curriculum Support**: Multi-level files can be viewed/tested/inspected
5. **Shared Code**: Common utilities prevent duplication

## Migration Notes

- No changes needed for existing single-level files
- Multi-level files will now work in all tools
- Default behavior is to load first level from multi-level files
- Use `--level-index N` to select specific levels

## Testing

1. Test single-level files still work in all tools
2. Test multi-level files load without crashes
3. Test level selection with `--level-index`
4. Test file_inspector shows all levels
5. Test invalid index handling
6. Test corrupted file handling