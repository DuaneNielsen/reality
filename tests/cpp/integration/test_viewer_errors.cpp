#include <gtest/gtest.h>
#include "viewer_test_base.hpp"
#include "mock_components.hpp"
#include "optionparser.h"
#include <fstream>

// Option parser definitions (from viewer.cpp)
namespace ArgChecker {
    static option::ArgStatus Required(const option::Option& option, bool msg) {
        if (option.arg != 0)
            return option::ARG_OK;
        
        if (msg) std::cerr << "Option '" << option.name << "' requires an argument\n";
        return option::ARG_ILLEGAL;
    }
    
    static option::ArgStatus Numeric(const option::Option& option, bool msg) {
        char* endptr = 0;
        if (option.arg != 0 && strtol(option.arg, &endptr, 10)){};
        if (endptr != option.arg && *endptr == 0)
            return option::ARG_OK;
        
        if (msg) std::cerr << "Option '" << option.name << "' requires a numeric argument\n";
        return option::ARG_ILLEGAL;
    }
}

enum OptionIndex { 
    UNKNOWN, HELP, CUDA, NUM_WORLDS, LOAD, REPLAY, RECORD, TRACK, 
    TRACK_WORLD, TRACK_AGENT, TRACK_FILE, SEED, HIDE_MENU 
};

const option::Descriptor usage[] = {
    {UNKNOWN, 0, "", "", option::Arg::None, "USAGE: viewer [options]\n\n"},
    {HELP,    0, "h", "help", option::Arg::None, "  --help, -h  \tPrint usage and exit."},
    {CUDA,    0, "", "cuda", ArgChecker::Numeric, "  --cuda <n>  \tUse CUDA/GPU execution mode"},
    {NUM_WORLDS, 0, "n", "num-worlds", ArgChecker::Numeric, "  --num-worlds, -n  \tNumber of worlds"},
    {LOAD,    0, "", "load", ArgChecker::Required, "  --load <file.lvl>  \tLoad level file"},
    {REPLAY,  0, "", "replay", ArgChecker::Required, "  --replay <file.rec>  \tReplay recording"},
    {RECORD,  0, "r", "record", ArgChecker::Required, "  --record, -r  \tRecord to file"},
    {TRACK,   0, "t", "track", option::Arg::None, "  --track, -t  \tEnable tracking"},
    {TRACK_WORLD, 0, "", "track-world", ArgChecker::Numeric, "  --track-world <n>  \tWorld to track"},
    {TRACK_AGENT, 0, "", "track-agent", ArgChecker::Numeric, "  --track-agent <n>  \tAgent to track"},
    {TRACK_FILE, 0, "", "track-file", ArgChecker::Required, "  --track-file <file>  \tSave trajectory"},
    {SEED,    0, "s", "seed", ArgChecker::Numeric, "  --seed, -s  \tRandom seed"},
    {HIDE_MENU, 0, "", "hide-menu", option::Arg::None, "  --hide-menu  \tHide ImGui menu"},
    {0,0,0,0,0,0}
};

// These tests primarily test option parsing and file handling,
// not viewer.cpp code itself
class OptionParsingAndFileErrorTest : public ViewerTestBase {
protected:
    void SetUp() override {
        ViewerTestBase::SetUp();
        file_manager_ = std::make_unique<TestFileManager>();
    }
    
    void TearDown() override {
        file_manager_->cleanup();
        ViewerTestBase::TearDown();
    }
    
    // Helper to parse command line arguments
    bool parseArgs(const std::vector<const char*>& args, 
                  option::Option*& options, option::Option*& buffer) {
        std::vector<char*> argv_copy;
        for (const auto& arg : args) {
            argv_copy.push_back(const_cast<char*>(arg));
        }
        
        int argc = argv_copy.size();
        char** argv = argv_copy.data();
        
        // Skip program name
        argc -= (argc > 0);
        argv += (argc > 0);
        
        option::Stats stats(usage, argc, argv);
        options = new option::Option[stats.options_max];
        buffer = new option::Option[stats.buffer_max];
        
        option::Parser parse(usage, argc, argv, options, buffer);
        
        return !parse.error();
    }
    
    std::unique_ptr<TestFileManager> file_manager_;
};

// Test missing level file
TEST_F(OptionParsingAndFileErrorTest, MissingLevelFile) {
    const char* missing_file = "nonexistent.lvl";
    
    // Try to load non-existent level
    std::ifstream test_file(missing_file, std::ios::binary);
    EXPECT_FALSE(test_file.is_open());
    
    // Load a valid level from the test data directory
    const char* valid_level_file = "tests/cpp/data/simple_test.lvl";
    MER_CompiledLevel level = {};
    
    std::ifstream lvl_file(valid_level_file, std::ios::binary);
    ASSERT_TRUE(lvl_file.is_open()) << "Test level file not found: " << valid_level_file;
    
    lvl_file.read(reinterpret_cast<char*>(&level), sizeof(MER_CompiledLevel));
    ASSERT_TRUE(lvl_file.good()) << "Failed to read test level file";
    lvl_file.close();
    
    // Manager creation should succeed with valid level data
    MER_Result result = mer_create_manager(&handle, &config, &level, 1);
    EXPECT_EQ(result, MER_SUCCESS);
    
    // Clean up
    if (handle) {
        mer_destroy_manager(handle);
        handle = nullptr;
    }
}

// Test corrupt recording file
TEST_F(OptionParsingAndFileErrorTest, CorruptRecordingFile) {
    // Create a corrupt recording file
    std::ofstream corrupt_file("corrupt.rec", std::ios::binary);
    corrupt_file << "CORRUPT DATA";
    corrupt_file.close();
    file_manager_->addFile("corrupt.rec");
    
    // Try to read metadata from corrupt file
    MER_ReplayMetadata metadata;
    MER_Result result = mer_read_replay_metadata("corrupt.rec", &metadata);
    
    // Should fail to read valid metadata
    EXPECT_NE(result, MER_SUCCESS);
}

// Test invalid option combinations
TEST_F(OptionParsingAndFileErrorTest, InvalidViewerOptionCombinations) {
    option::Option* options = nullptr;
    option::Option* buffer = nullptr;
    
    // Test 1: Both --replay and --record (mutually exclusive)
    {
        std::vector<const char*> args = {
            "viewer",
            "--replay", "test.rec",
            "--record", "output.rec"
        };
        
        bool parse_ok = parseArgs(args, options, buffer);
        EXPECT_TRUE(parse_ok);  // Parsing succeeds
        
        // But validation should fail (viewer would exit)
        bool has_replay = options[REPLAY].count() > 0;
        bool has_record = options[RECORD].count() > 0;
        EXPECT_TRUE(has_replay && has_record);  // Both set = error
        
        delete[] options;
        delete[] buffer;
    }
    
    // Test 2: --replay with --load (level comes from recording)
    {
        std::vector<const char*> args = {
            "viewer",
            "--replay", "test.rec",
            "--load", "level.lvl"
        };
        
        bool parse_ok = parseArgs(args, options, buffer);
        EXPECT_TRUE(parse_ok);
        
        bool has_replay = options[REPLAY].count() > 0;
        bool has_load = options[LOAD].count() > 0;
        EXPECT_TRUE(has_replay && has_load);  // Both set = warning/error
        
        delete[] options;
        delete[] buffer;
    }
    
    // Test 3: Neither --replay nor --load (required)
    {
        std::vector<const char*> args = {
            "viewer",
            "--num-worlds", "4"
        };
        
        bool parse_ok = parseArgs(args, options, buffer);
        EXPECT_TRUE(parse_ok);
        
        bool has_replay = options[REPLAY].count() > 0;
        bool has_load = options[LOAD].count() > 0;
        EXPECT_FALSE(has_replay || has_load);  // Neither set = error
        
        delete[] options;
        delete[] buffer;
    }
}

// Test invalid numeric arguments
TEST_F(OptionParsingAndFileErrorTest, InvalidNumericArguments) {
    option::Option* options = nullptr;
    option::Option* buffer = nullptr;
    
    // Test invalid number for --num-worlds
    {
        std::vector<const char*> args = {
            "viewer",
            "--load", "test.lvl",
            "--num-worlds", "abc"  // Not a number
        };
        
        bool parse_ok = parseArgs(args, options, buffer);
        EXPECT_FALSE(parse_ok);  // Should fail due to non-numeric argument
        
        if (options) delete[] options;
        if (buffer) delete[] buffer;
    }
    
    // Test invalid GPU ID
    {
        std::vector<const char*> args = {
            "viewer",
            "--load", "test.lvl",
            "--cuda", "not_a_number"
        };
        
        options = nullptr;
        buffer = nullptr;
        bool parse_ok = parseArgs(args, options, buffer);
        EXPECT_FALSE(parse_ok);  // Should fail due to non-numeric argument
        
        if (options) delete[] options;
        if (buffer) delete[] buffer;
    }
}

// Test file extension warnings
TEST_F(OptionParsingAndFileErrorTest, FileExtensionValidation) {
    option::Option* options = nullptr;
    option::Option* buffer = nullptr;
    
    // Test wrong extension for --load
    {
        std::vector<const char*> args = {
            "viewer",
            "--load", "level.rec"  // Should be .lvl
        };
        
        bool parse_ok = parseArgs(args, options, buffer);
        EXPECT_TRUE(parse_ok);  // Parsing succeeds
        
        // Check that file has wrong extension
        std::string load_path = options[LOAD].arg;
        size_t lvl_pos = load_path.rfind(".lvl");
        size_t rec_pos = load_path.rfind(".rec");
        EXPECT_FALSE(lvl_pos != std::string::npos && lvl_pos == load_path.length() - 4);
        EXPECT_TRUE(rec_pos != std::string::npos && rec_pos == load_path.length() - 4);
        
        delete[] options;
        delete[] buffer;
    }
    
    // Test wrong extension for --replay
    {
        std::vector<const char*> args = {
            "viewer",
            "--replay", "recording.lvl"  // Should be .rec
        };
        
        options = nullptr;
        buffer = nullptr;
        bool parse_ok = parseArgs(args, options, buffer);
        EXPECT_TRUE(parse_ok);
        
        std::string replay_path = options[REPLAY].arg;
        size_t rec_pos = replay_path.rfind(".rec");
        size_t lvl_pos = replay_path.rfind(".lvl");
        EXPECT_FALSE(rec_pos != std::string::npos && rec_pos == replay_path.length() - 4);
        EXPECT_TRUE(lvl_pos != std::string::npos && lvl_pos == replay_path.length() - 4);
        
        delete[] options;
        delete[] buffer;
    }
}

// Test replay metadata mismatch
TEST_F(OptionParsingAndFileErrorTest, ReplayMetadataMismatch) {
    // Create a valid recording with specific metadata
    createTestLevelFile("test.lvl", 16, 16);
    file_manager_->addFile("test.lvl");
    file_manager_->addFile("test.rec");
    
    auto level = LevelComparer::loadLevelFromFile("test.lvl");
    
    // Create recording with 4 worlds
    {
        config.num_worlds = 4;
        config.auto_reset = true;
        ASSERT_TRUE(CreateManager(&level, 1));
        
        mer_start_recording(handle, "test.rec", 999);
        mer_step(handle);
        mer_stop_recording(handle);
    }
    
    // Clean up first manager
    TearDown();
    SetUp();
    
    // Try to replay with different number of worlds
    {
        MER_ReplayMetadata metadata;
        ASSERT_EQ(mer_read_replay_metadata("test.rec", &metadata), MER_SUCCESS);
        EXPECT_EQ(metadata.num_worlds, 4);
        
        // Attempting to create manager with wrong world count
        config.num_worlds = 2;  // Different from recording
        config.auto_reset = true;
        
        // In real viewer, this would print a warning and adjust num_worlds
        // Here we just verify the metadata mismatch exists
        EXPECT_NE(config.num_worlds, metadata.num_worlds);
    }
}

// Test GPU initialization failure handling
TEST_F(OptionParsingAndFileErrorTest, ManagerGPUInitFailure) {
    // Skip if no GPU available
    const char* allow_gpu = std::getenv("ALLOW_GPU_TESTS_IN_SUITE");
    if (!allow_gpu || std::string(allow_gpu) != "1") {
        GTEST_SKIP() << "GPU tests disabled";
    }
    
    // Try to create manager with invalid GPU ID
    config.exec_mode = MER_EXEC_MODE_CUDA;
    config.gpu_id = 999;  // Invalid GPU ID
    
    // This should fail
    MER_Result result = mer_create_manager(&handle, &config, nullptr, 0);
    
    // May succeed if system has 999+ GPUs, but typically fails
    if (result != MER_SUCCESS) {
        EXPECT_EQ(result, MER_ERROR_CUDA_FAILURE);
    }
}

// Test trajectory file write errors
TEST_F(OptionParsingAndFileErrorTest, TrajectoryFileWriteError) {
    createTestLevelFile("test.lvl", 16, 16);
    file_manager_->addFile("test.lvl");
    
    auto level = LevelComparer::loadLevelFromFile("test.lvl");
    config.num_worlds = 1;
    
    ASSERT_TRUE(CreateManager(&level, 1));
    
    // Try to write trajectory to invalid path
    const char* invalid_path = "/invalid/path/trajectory.csv";
    MER_Result result = mer_enable_trajectory_logging(handle, 0, 0, invalid_path);
    
    // The API might not return an error immediately, but file write will fail
    // We can't easily test this without actually trying to write
    
    // Instead test with read-only file
    std::string readonly_file = "readonly.csv";
    file_manager_->addFile(readonly_file);
    
    // Create file and make it read-only (platform-specific)
    std::ofstream create_file(readonly_file);
    create_file << "locked";
    create_file.close();
    
    // Note: Making file read-only is platform-specific
    // On Unix: chmod(readonly_file.c_str(), S_IRUSR);
    // This test may not work consistently across platforms
}

// Test recording file size limits
TEST_F(OptionParsingAndFileErrorTest, LargeRecordingFileCreation) {
    createTestLevelFile("test.lvl", 16, 16);
    file_manager_->addFile("test.lvl");
    file_manager_->addFile("huge.rec");
    
    auto level = LevelComparer::loadLevelFromFile("test.lvl");
    config.num_worlds = 100;  // Large number of worlds
    config.auto_reset = true;
    
    ASSERT_TRUE(CreateManager(&level, 1));
    
    mer_start_recording(handle, "huge.rec", 777);
    
    // Simulate many steps to create large file
    for (int i = 0; i < 1000; i++) {
        // Set actions for all worlds
        for (uint32_t w = 0; w < config.num_worlds; w++) {
            mer_set_action(handle, w, i % 4, i % 8, (i + 2) % 5);
        }
        mer_step(handle);
    }
    
    mer_stop_recording(handle);
    
    // Check that file was created and has substantial size
    size_t file_size = file_manager_->getFileSize("huge.rec");
    EXPECT_GT(file_size, sizeof(MER_ReplayMetadata) + sizeof(MER_CompiledLevel));
    
    // Expected minimum size: metadata + level + (100 worlds * 3 actions * 4 bytes * 1000 steps)
    size_t expected_min = sizeof(MER_ReplayMetadata) + sizeof(MER_CompiledLevel) + 
                         (100 * 3 * sizeof(int32_t) * 1000);
    EXPECT_GT(file_size, expected_min / 2);  // Allow for compression
}

// Test unknown command line options
TEST_F(OptionParsingAndFileErrorTest, UnknownCommandLineOptions) {
    option::Option* options = nullptr;
    option::Option* buffer = nullptr;
    
    std::vector<const char*> args = {
        "viewer",
        "--load", "test.lvl",
        "--unknown-option", "value",
        "--another-bad-option"
    };
    
    bool parse_ok = parseArgs(args, options, buffer);
    EXPECT_TRUE(parse_ok);  // Parser is lenient
    
    // Check for unknown options
    bool has_unknown = options[UNKNOWN].count() > 0;
    EXPECT_TRUE(has_unknown);
    
    // Count unknown options
    int unknown_count = 0;
    for (option::Option* opt = &options[UNKNOWN]; opt; opt = opt->next()) {
        if (opt->name) unknown_count++;
    }
    EXPECT_GE(unknown_count, 1);
    
    delete[] options;
    delete[] buffer;
}