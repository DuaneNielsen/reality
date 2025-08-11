#include "mgr.hpp"

#include <optionparser.h>

#include <cstdio>
#include <chrono>
#include <string>
#include <filesystem>
#include <fstream>
#include <random>
#include <iostream>
#include <algorithm>

#include <madrona/heap_array.hpp>

using namespace madrona;

[[maybe_unused]] static void saveWorldActions(
    const HeapArray<int32_t> &action_store,
    int32_t total_num_steps,
    int32_t world_idx)
{
    const int32_t *world_base = action_store.data() + world_idx * total_num_steps * 3;

    std::ofstream f("/tmp/actions", std::ios::binary);
    f.write((char *)world_base,
            sizeof(uint32_t) * total_num_steps * 3);
}

namespace ArgChecker {
    static option::ArgStatus Required(const option::Option& option, bool msg)
    {
        if (option.arg != 0)
            return option::ARG_OK;
        
        if (msg) std::cerr << "Option '" << option.name << "' requires an argument\n";
        return option::ARG_ILLEGAL;
    }
    
    static option::ArgStatus Numeric(const option::Option& option, bool msg)
    {
        char* endptr = 0;
        if (option.arg != 0 && strtol(option.arg, &endptr, 10)){};
        if (endptr != option.arg && *endptr == 0)
            return option::ARG_OK;
        
        if (msg) std::cerr << "Option '" << option.name << "' requires a numeric argument\n";
        return option::ARG_ILLEGAL;
    }
}

enum OptionIndex { 
    UNKNOWN, HELP, CUDA, NUM_WORLDS, NUM_STEPS, RAND_ACTIONS, LOAD, REPLAY, RECORD, SEED, TRACK, TRACK_WORLD, TRACK_AGENT, TRACK_FILE 
};

const option::Descriptor usage[] = {
    {UNKNOWN, 0, "", "", option::Arg::None, "USAGE: headless [options]\n\n"
                                            "Madrona Escape Room - Headless Mode\n"
                                            "Run simulation without graphics for benchmarking and testing\n\n"
                                            "Options:"},
    {HELP,         0, "h", "help", option::Arg::None, "  --help, -h  \tShow this help message"},
    {CUDA,         0, "", "cuda", ArgChecker::Numeric, "  --cuda <n>  \tUse CUDA/GPU execution mode on device n"},
    {NUM_WORLDS,   0, "n", "num-worlds", ArgChecker::Numeric, "  --num-worlds <value>, -n <value>  \tNumber of parallel worlds (required)"},
    {NUM_STEPS,    0, "s", "num-steps", ArgChecker::Numeric, "  --num-steps <value>, -s <value>  \tNumber of simulation steps (required)"},
    {RAND_ACTIONS, 0, "", "rand-actions", option::Arg::None, "  --rand-actions  \tGenerate random actions for benchmarking"},
    {LOAD,         0, "", "load", ArgChecker::Required, "  --load <file.lvl>  \tLoad binary level file"},
    {REPLAY,       0, "", "replay", ArgChecker::Required, "  --replay <file.rec>  \tReplay recording file"},
    {RECORD,       0, "", "record", ArgChecker::Required, "  --record <file.rec>  \tRecord actions to file"},
    {SEED,         0, "", "seed", ArgChecker::Numeric, "  --seed <value>  \tSet random seed (default: 5)"},
    {TRACK,        0, "t", "track", option::Arg::None, "  --track, -t  \tEnable trajectory tracking (default: world 0, agent 0)"},
    {TRACK_WORLD,  0, "", "track-world", ArgChecker::Numeric, "  --track-world <n>  \tSpecify world to track (default: 0)"},
    {TRACK_AGENT,  0, "", "track-agent", ArgChecker::Numeric, "  --track-agent <n>  \tSpecify agent to track (default: 0)"},
    {TRACK_FILE,   0, "", "track-file", ArgChecker::Required, "  --track-file <file>  \tSave trajectory to file"},
    {0,0,0,0,0,0}
};

int main(int argc, char *argv[])
{
    using namespace madEscape;

    // Parse arguments
    argc-=(argc>0); argv+=(argc>0); // skip program name argv[0] if present
    option::Stats stats(usage, argc, argv);
    option::Option* options = new option::Option[stats.options_max];
    option::Option* buffer = new option::Option[stats.buffer_max];
    option::Parser parse(usage, argc, argv, options, buffer);

    if (parse.error()) {
        delete[] options;
        delete[] buffer;
        return 1;
    }

    if (options[HELP]) {
        option::printUsage(std::cout, usage);
        std::cout << "\nExamples:\n";
        std::cout << "  headless --num-worlds 1 --num-steps 1000                # Basic CPU run\n";
        std::cout << "  headless --cuda 0 -n 8192 -s 1000 --rand-actions        # GPU benchmark on device 0\n";
        std::cout << "  headless -n 100 -s 1000 --track --track-world 5         # Track agent 0 in world 5\n";
        std::cout << "  headless -n 100 -s 1000 --track-world 5 --track-agent 1 # Track agent 1 in world 5\n";
        std::cout << "  headless -n 2 -s 1000 --replay demo.bin                 # Replay demo.bin\n";
        std::cout << "  headless --cuda 1 -n 1000 -s 500 --track-file traj.csv  # GPU 1 with trajectory to file\n";
        delete[] options;
        delete[] buffer;
        return 0;
    }

    // Check for any remaining non-option arguments
    if (parse.nonOptionsCount() > 0) {
        std::cerr << "Error: Unexpected arguments. All parameters must be specified as options.\n";
        std::cerr << "Use --help for usage information.\n";
        delete[] options;
        delete[] buffer;
        return 1;
    }
    
    // Required options
    if (!options[NUM_WORLDS] || !options[NUM_STEPS]) {
        std::cerr << "Error: Missing required options.\n";
        std::cerr << "Required: --num-worlds, --num-steps\n";
        std::cerr << "Use --help for usage information.\n";
        delete[] options;
        delete[] buffer;
        return 1;
    }
    
    uint64_t num_worlds = std::stoul(options[NUM_WORLDS].arg);
    uint64_t num_steps = std::stoul(options[NUM_STEPS].arg);

    // Parse execution mode - default to CPU, use CUDA if specified
    ExecMode exec_mode = ExecMode::CPU;
    uint32_t gpu_id = 0;
    if (options[CUDA]) {
        exec_mode = ExecMode::CUDA;
        gpu_id = strtoul(options[CUDA].arg, nullptr, 10);
    }

    // Parse three-flag interface
    std::string load_path = options[LOAD] ? options[LOAD].arg : "";
    std::string replay_file = options[REPLAY] ? options[REPLAY].arg : "";
    std::string record_path = options[RECORD] ? options[RECORD].arg : "";
    
    // Validate three-flag interface (similar to viewer)
    // Must have either replay OR (load with optional record)
    if (!replay_file.empty() && !record_path.empty()) {
        std::cerr << "Error: Cannot specify both --replay and --record\n";
        delete[] options;
        delete[] buffer;
        return 1;
    }
    
    if (!replay_file.empty() && !load_path.empty()) {
        std::cerr << "Error: --replay ignores --load (level comes from recording)\n";
        delete[] options;
        delete[] buffer;
        return 1;
    }
    
    if (replay_file.empty() && load_path.empty()) {
        std::cerr << "Error: Must provide either --replay <file.rec> OR --load <file.lvl>\n";
        std::cerr << "Usage patterns:\n";
        std::cerr << "  --load <file.lvl>                    # Load and run level\n";
        std::cerr << "  --load <file.lvl> --record <file.rec> # Load level and record session\n";
        std::cerr << "  --replay <file.rec>                  # Replay recording (level embedded)\n";
        delete[] options;
        delete[] buffer;
        return 1;
    }
    
    // Optional parameters
    bool rand_actions = options[RAND_ACTIONS];
    uint32_t rand_seed = options[SEED] ? strtoul(options[SEED].arg, nullptr, 10) : 5;
    std::string track_file = options[TRACK_FILE] ? options[TRACK_FILE].arg : "";
    
    // Parse tracking options
    bool track_trajectory = false;
    int32_t track_world_idx = 0;  // Default to world 0
    int32_t track_agent_idx = 0;  // Default to agent 0
    
    // Handle tracking options
    if (options[TRACK]) {
        track_trajectory = true;
    }
    
    if (options[TRACK_WORLD]) {
        track_world_idx = strtol(options[TRACK_WORLD].arg, nullptr, 10);
        // If track-world is specified without --track, enable tracking
        track_trajectory = true;
    }
    
    if (options[TRACK_AGENT]) {
        track_agent_idx = strtol(options[TRACK_AGENT].arg, nullptr, 10);
        // If track-agent is specified without --track, enable tracking
        track_trajectory = true;
    }
    
    if (options[TRACK_FILE]) {
        track_file = options[TRACK_FILE].arg;
        // If track-file is specified without --track, enable tracking
        track_trajectory = true;
    }
    
    // Validate world index if tracking is enabled
    if (track_trajectory) {
        if (track_world_idx < 0 || track_world_idx >= (int32_t)num_worlds) {
            std::cerr << "Error: track world index " << track_world_idx 
                      << " out of range [0, " << num_worlds - 1 << "]\n";
            delete[] options;
            delete[] buffer;
            return 1;
        }
    }
    
    // No need to check if tracking is enabled for track_file since we auto-enable it now
    
    // Validate conflicting options
    if (rand_actions && !replay_file.empty()) {
        std::cerr << "Error: Cannot specify both --rand-actions and --replay\n";
        delete[] options;
        delete[] buffer;
        return 1;
    }

    bool replay_mode = !replay_file.empty();
    uint32_t replay_seed = rand_seed;
    
    // Handle replay mode - load metadata first
    if (replay_mode) {
        auto metadata_opt = Manager::readReplayMetadata(replay_file);
        if (!metadata_opt.has_value()) {
            std::cerr << "Error: Failed to load replay file: " << replay_file << "\n";
            delete[] options;
            delete[] buffer;
            return 1;
        }
        
        const auto& metadata = metadata_opt.value();
        
        // Update parameters from replay metadata
        if (num_worlds != metadata.num_worlds) {
            std::cerr << "Warning: Replay was recorded with " << metadata.num_worlds 
                      << " worlds, but headless is using " << num_worlds << " worlds.\n";
            std::cerr << "Setting num_worlds to match replay file.\n";
            num_worlds = metadata.num_worlds;
        }
        
        if (num_steps != metadata.num_steps) {
            std::cerr << "Warning: Replay was recorded with " << metadata.num_steps
                      << " steps, but headless is using " << num_steps << " steps.\n";
            std::cerr << "Setting num_steps to match replay file.\n";
            num_steps = metadata.num_steps;
        }
        
        replay_seed = metadata.seed;
        std::cout << "Using seed " << replay_seed << " from replay file\n";
    }
    
    // Only needed for random actions now
    HeapArray<int32_t> action_store(rand_actions ? (num_worlds * num_steps * 3) : 0);

    // Load level based on three-flag interface
    CompiledLevel loaded_level = {};
    std::vector<std::optional<CompiledLevel>> headless_levels;
    
    if (!load_path.empty()) {
        // Load from .lvl file
        std::ifstream lvl_file(load_path, std::ios::binary);
        if (!lvl_file.is_open()) {
            std::cerr << "Error: Cannot open level file: " << load_path << "\n";
            delete[] options;
            delete[] buffer;
            return 1;
        }
        
        lvl_file.read(reinterpret_cast<char*>(&loaded_level), sizeof(CompiledLevel));
        if (!lvl_file.good()) {
            std::cerr << "Error: Failed to read level from: " << load_path << "\n";
            delete[] options;
            delete[] buffer;
            return 1;
        }
        
        printf("Loaded level from %s: %dx%d grid, %d tiles\n", 
               load_path.c_str(), loaded_level.width, loaded_level.height, loaded_level.num_tiles);
        
        // All worlds use the same loaded level
        headless_levels.resize(num_worlds, loaded_level);
        
    } else if (!replay_file.empty()) {
        // Replay mode - extract embedded level data from recording
        auto embedded_level = Manager::readEmbeddedLevel(replay_file);
        if (!embedded_level.has_value()) {
            std::cerr << "Error: Failed to read embedded level from replay file: " << replay_file << "\n";
            delete[] options;
            delete[] buffer;
            return 1;
        }
        
        loaded_level = embedded_level.value();
        printf("Extracted embedded level from %s: %dx%d grid, %d tiles\n", 
               replay_file.c_str(), loaded_level.width, loaded_level.height, loaded_level.num_tiles);
        
        // All worlds use the embedded level
        headless_levels.resize(num_worlds, loaded_level);
    }

    printf("Executing %lu Steps x %lu Worlds (%s)%s\n",
           num_steps, num_worlds,
           exec_mode == ExecMode::CPU ? "CPU" : "CUDA",
           replay_mode ? " [REPLAY MODE]" : "");

    Manager mgr({
        .execMode = exec_mode,
        .gpuID = (int)gpu_id,
        .numWorlds = (uint32_t)num_worlds,
        .randSeed = replay_mode ? replay_seed : rand_seed,
        .autoReset = replay_mode,
        .enableBatchRenderer = false,
        .perWorldCompiledLevels = headless_levels,
    });
    
    // Load replay if available
    if (replay_mode) {
        if (!mgr.loadReplay(replay_file)) {
            std::cerr << "Error: Failed to load replay into manager\n";
            delete[] options;
            delete[] buffer;
            return 1;
        }
    }

    if (rand_actions || replay_mode) {
        std::mt19937 rand_gen(rand_seed);
        std::uniform_int_distribution<int32_t> move_amount_dist(0, 3);
        std::uniform_int_distribution<int32_t> move_angle_dist(0, 7);
        std::uniform_int_distribution<int32_t> turn_dist(0, 4);

        if (rand_actions) {
            // Generate random actions
            for (uint64_t i = 0; i < num_steps; i++) {
                for (uint64_t j = 0; j < num_worlds; j++) {
                    uint64_t base_idx = 3 * (i * num_worlds + j);
                    action_store[base_idx] = move_amount_dist(rand_gen);
                    action_store[base_idx + 1] = move_angle_dist(rand_gen);
                    action_store[base_idx + 2] = turn_dist(rand_gen);
                }
            }
        }
    }

    // Enable trajectory tracking if requested
    if (track_trajectory) {
        mgr.enableTrajectoryLogging(track_world_idx, track_agent_idx, 
            track_file.empty() ? std::nullopt : std::make_optional(track_file.c_str()));
        printf("Tracking agent %d in world %d\n", track_agent_idx, track_world_idx);
        if (!track_file.empty()) {
            printf("Trajectory will be saved to: %s\n", track_file.c_str());
        }
    }
    
    // Start recording if requested
    bool is_recording = !record_path.empty();
    if (is_recording) {
        mgr.startRecording(record_path, rand_seed);
        printf("Recording actions to: %s\n", record_path.c_str());
    }

    auto start = std::chrono::system_clock::now();
    for (uint64_t i = 0; i < num_steps; i++) {
        if (replay_mode) {
            // Use Manager's replay functionality
            bool finished = mgr.replayStep();
            if (finished) {
                printf("Replay finished at step %lu\n", i);
                break;
            }
        } else if (rand_actions) {
            // Set random actions
            for (uint64_t j = 0; j < num_worlds; j++) {
                uint64_t base_idx = 3 * (i * num_worlds + j);
                mgr.setAction(j,
                              action_store[base_idx],
                              action_store[base_idx + 1],
                              action_store[base_idx + 2]);
            }
        }
        mgr.step();
    }
    auto end = std::chrono::system_clock::now();

    // Stop recording if it was started
    if (is_recording) {
        mgr.stopRecording();
    }

    std::chrono::duration<double> elapsed = end - start;

    printf("FPS: %.0f\n", double(num_worlds * num_steps) / elapsed.count());

    // Note: Trajectory tracking output is handled internally by the Manager
    // when enableTrajectoryLogging is called

    //saveWorldActions(action_store, num_steps, 0);
    
    delete[] options;
    delete[] buffer;
    
    return 0;
}