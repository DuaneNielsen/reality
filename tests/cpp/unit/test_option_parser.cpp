#include <gtest/gtest.h>
#include "viewer_test_base.hpp"
#include "optionparser.h"

// Include the option parser definitions from viewer.cpp
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
    UNKNOWN, HELP, CUDA, NUM_WORLDS, LOAD, REPLAY, RECORD, TRACK, TRACK_WORLD, TRACK_AGENT, TRACK_FILE, SEED, HIDE_MENU 
};

const option::Descriptor usage[] = {
    {UNKNOWN, 0, "", "", option::Arg::None, "USAGE: viewer [options]\n\n"
                                            "Madrona Escape Room - Interactive Viewer\n"
                                            "3D visualization and control of the simulation\n\n"
                                            "Options:"},
    {HELP,    0, "h", "help", option::Arg::None, "  --help, -h  \tPrint usage and exit."},
    {CUDA,    0, "", "cuda", ArgChecker::Numeric, "  --cuda <n>  \tUse CUDA/GPU execution mode on device n"},
    {NUM_WORLDS, 0, "n", "num-worlds", ArgChecker::Numeric, "  --num-worlds <value>, -n <value>  \tNumber of parallel worlds (default: 1)"},
    {LOAD,    0, "", "load", ArgChecker::Required, "  --load <file.lvl>  \tLoad binary level file"},
    {REPLAY,  0, "", "replay", ArgChecker::Required, "  --replay <file.rec>  \tReplay recording file"},
    {RECORD,  0, "r", "record", ArgChecker::Required, "  --record <path.rec>, -r <path.rec>  \tRecord actions to file (press SPACE to start)"},
    {TRACK,   0, "t", "track", option::Arg::None, "  --track, -t  \tEnable trajectory tracking (default: world 0, agent 0)"},
    {TRACK_WORLD, 0, "", "track-world", ArgChecker::Numeric, "  --track-world <n>  \tSpecify world to track (default: 0)"},
    {TRACK_AGENT, 0, "", "track-agent", ArgChecker::Numeric, "  --track-agent <n>  \tSpecify agent to track (default: 0)"},
    {TRACK_FILE, 0, "", "track-file", ArgChecker::Required, "  --track-file <file>  \tSave trajectory to file"},
    {SEED,    0, "s", "seed", ArgChecker::Numeric, "  --seed <value>, -s <value>  \tSet random seed (default: 5)"},
    {HIDE_MENU, 0, "", "hide-menu", option::Arg::None, "  --hide-menu  \tHide ImGui menu (useful for clean screenshots)"},
    {0,0,0,0,0,0}
};

// Test fixture for command-line option parsing
class OptionParserTest : public ViewerTestBase {
protected:
    option::Option* parseArgs(const std::vector<const char*>& args) {
        // Copy args to modifiable array since option parser modifies it
        std::vector<char*> argv_copy;
        for (const auto& arg : args) {
            argv_copy.push_back(const_cast<char*>(arg));
        }
        
        int argc = argv_copy.size();
        char** argv = argv_copy.data();
        
        // Skip program name like viewer.cpp does
        argc -= (argc > 0);
        argv += (argc > 0);
        
        option::Stats stats(usage, argc, argv);
        options_ = new option::Option[stats.options_max];
        buffer_ = new option::Option[stats.buffer_max];
        
        option::Parser parse(usage, argc, argv, options_, buffer_);
        
        if (parse.error()) {
            delete[] options_;
            delete[] buffer_;
            options_ = nullptr;
            buffer_ = nullptr;
            return nullptr;
        }
        
        return options_;
    }
    
    void TearDown() override {
        if (options_) {
            delete[] options_;
            options_ = nullptr;
        }
        if (buffer_) {
            delete[] buffer_;
            buffer_ = nullptr;
        }
        ViewerTestBase::TearDown();
    }
    
private:
    option::Option* options_ = nullptr;
    option::Option* buffer_ = nullptr;
};

// Test valid argument combinations
TEST_F(OptionParserTest, ValidLoadArgument) {
    auto args = buildViewerArgs({"viewer", "--load", "test.lvl"});
    auto options = parseArgs(args);
    
    ASSERT_NE(options, nullptr);
    EXPECT_TRUE(options[LOAD]);
    EXPECT_STREQ(options[LOAD].arg, "test.lvl");
}

TEST_F(OptionParserTest, ValidReplayArgument) {
    auto args = buildViewerArgs({"viewer", "--replay", "test.rec"});
    auto options = parseArgs(args);
    
    ASSERT_NE(options, nullptr);
    EXPECT_TRUE(options[REPLAY]);
    EXPECT_STREQ(options[REPLAY].arg, "test.rec");
}

TEST_F(OptionParserTest, ValidLoadWithRecord) {
    auto args = buildViewerArgs({"viewer", "--load", "test.lvl", "--record", "output.rec"});
    auto options = parseArgs(args);
    
    ASSERT_NE(options, nullptr);
    EXPECT_TRUE(options[LOAD]);
    EXPECT_TRUE(options[RECORD]);
    EXPECT_STREQ(options[LOAD].arg, "test.lvl");
    EXPECT_STREQ(options[RECORD].arg, "output.rec");
}

// Test invalid combinations
TEST_F(OptionParserTest, InvalidReplayWithRecord) {
    // This combination should be caught by the viewer logic, not the parser
    // The parser will accept it, but the viewer should reject it
    auto args = buildViewerArgs({"viewer", "--replay", "test.rec", "--record", "output.rec"});
    auto options = parseArgs(args);
    
    ASSERT_NE(options, nullptr);
    EXPECT_TRUE(options[REPLAY]);
    EXPECT_TRUE(options[RECORD]);
    // The viewer would check this and fail
}

TEST_F(OptionParserTest, InvalidReplayWithLoad) {
    // Replay should ignore load (level comes from recording)
    auto args = buildViewerArgs({"viewer", "--replay", "test.rec", "--load", "test.lvl"});
    auto options = parseArgs(args);
    
    ASSERT_NE(options, nullptr);
    EXPECT_TRUE(options[REPLAY]);
    EXPECT_TRUE(options[LOAD]);
    // The viewer would warn about this
}

// Test default values
TEST_F(OptionParserTest, DefaultNumWorlds) {
    auto args = buildViewerArgs({"viewer", "--load", "test.lvl"});
    auto options = parseArgs(args);
    
    ASSERT_NE(options, nullptr);
    EXPECT_FALSE(options[NUM_WORLDS]); // Not specified, will use default
}

TEST_F(OptionParserTest, DefaultSeed) {
    auto args = buildViewerArgs({"viewer", "--load", "test.lvl"});
    auto options = parseArgs(args);
    
    ASSERT_NE(options, nullptr);
    EXPECT_FALSE(options[SEED]); // Not specified, will use default
}

// Test tracking options
TEST_F(OptionParserTest, TrackingOptions) {
    auto args = buildViewerArgs({"viewer", "--load", "test.lvl", "--track"});
    auto options = parseArgs(args);
    
    ASSERT_NE(options, nullptr);
    EXPECT_TRUE(options[TRACK]);
    EXPECT_FALSE(options[TRACK_WORLD]); // Will use default 0
    EXPECT_FALSE(options[TRACK_AGENT]); // Will use default 0
}

TEST_F(OptionParserTest, TrackingWithFile) {
    auto args = buildViewerArgs({"viewer", "--load", "test.lvl", "--track", "--track-file", "trajectory.csv"});
    auto options = parseArgs(args);
    
    ASSERT_NE(options, nullptr);
    EXPECT_TRUE(options[TRACK]);
    EXPECT_TRUE(options[TRACK_FILE]);
    EXPECT_STREQ(options[TRACK_FILE].arg, "trajectory.csv");
}

TEST_F(OptionParserTest, TrackingWorldAgent) {
    auto args = buildViewerArgs({"viewer", "--load", "test.lvl", "--track-world", "2", "--track-agent", "1"});
    auto options = parseArgs(args);
    
    ASSERT_NE(options, nullptr);
    // track-world and track-agent should auto-enable tracking
    EXPECT_TRUE(options[TRACK_WORLD]);
    EXPECT_TRUE(options[TRACK_AGENT]);
    EXPECT_STREQ(options[TRACK_WORLD].arg, "2");
    EXPECT_STREQ(options[TRACK_AGENT].arg, "1");
}

// Test CUDA options
TEST_F(OptionParserTest, CudaOption) {
    auto args = buildViewerArgs({"viewer", "--load", "test.lvl", "--cuda", "0"});
    auto options = parseArgs(args);
    
    ASSERT_NE(options, nullptr);
    EXPECT_TRUE(options[CUDA]);
    EXPECT_STREQ(options[CUDA].arg, "0");
}

// Test numeric validation
TEST_F(OptionParserTest, NumericWorldCount) {
    auto args = buildViewerArgs({"viewer", "--load", "test.lvl", "--num-worlds", "64"});
    auto options = parseArgs(args);
    
    ASSERT_NE(options, nullptr);
    EXPECT_TRUE(options[NUM_WORLDS]);
    EXPECT_STREQ(options[NUM_WORLDS].arg, "64");
}

TEST_F(OptionParserTest, InvalidNumericArgument) {
    auto args = buildViewerArgs({"viewer", "--load", "test.lvl", "--num-worlds", "abc"});
    auto options = parseArgs(args);
    
    // Parser should catch this as invalid and return nullptr (error state)
    ASSERT_EQ(options, nullptr);
}

// Test short options
TEST_F(OptionParserTest, ShortOptions) {
    auto args = buildViewerArgs({"viewer", "--load", "test.lvl", "-n", "4", "-t", "-s", "42"});
    auto options = parseArgs(args);
    
    ASSERT_NE(options, nullptr);
    EXPECT_TRUE(options[NUM_WORLDS]);
    EXPECT_TRUE(options[TRACK]);
    EXPECT_TRUE(options[SEED]);
    EXPECT_STREQ(options[NUM_WORLDS].arg, "4");
    EXPECT_STREQ(options[SEED].arg, "42");
}

// Test help option
TEST_F(OptionParserTest, HelpOption) {
    auto args = buildViewerArgs({"viewer", "--help"});
    auto options = parseArgs(args);
    
    ASSERT_NE(options, nullptr);
    EXPECT_TRUE(options[HELP]);
}

// Test hide menu option
TEST_F(OptionParserTest, HideMenuOption) {
    auto args = buildViewerArgs({"viewer", "--load", "test.lvl", "--hide-menu"});
    auto options = parseArgs(args);
    
    ASSERT_NE(options, nullptr);
    EXPECT_TRUE(options[HIDE_MENU]);
}

// Test missing required arguments
TEST_F(OptionParserTest, MissingLoadArgument) {
    auto args = buildViewerArgs({"viewer", "--load"});
    auto options = parseArgs(args);
    
    // Parser should catch missing required argument and return nullptr
    ASSERT_EQ(options, nullptr);
}

TEST_F(OptionParserTest, MissingRecordArgument) {
    auto args = buildViewerArgs({"viewer", "--record"});
    auto options = parseArgs(args);
    
    // Parser should catch missing required argument and return nullptr
    ASSERT_EQ(options, nullptr);
}