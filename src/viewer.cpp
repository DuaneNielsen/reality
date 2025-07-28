#include <madrona/viz/viewer.hpp>
#include <madrona/render/render_mgr.hpp>
#include <madrona/window.hpp>

#include "sim.hpp"
#include "mgr.hpp"
#include "types.hpp"

#include <optionparser.h>

#include <filesystem>
#include <string>
#include <cstring>
#include <iostream>
#include <vector>

using namespace madrona;
using namespace madrona::viz;

// Replay functionality moved to Manager class

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
    UNKNOWN, HELP, MODE, CPU, CUDA, NUM_WORLDS, REPLAY, RECORD, TRACK, TRACK_WORLD, TRACK_AGENT, SEED 
};

const option::Descriptor usage[] = {
    {UNKNOWN, 0, "", "", option::Arg::None, "USAGE: viewer [options]\n\n"
                                            "Madrona Escape Room - Interactive Viewer\n"
                                            "3D visualization and control of the simulation\n\n"
                                            "Options:"},
    {HELP,    0, "h", "help", option::Arg::None, "  --help, -h  \tPrint usage and exit."},
    {MODE,    0, "m", "mode", ArgChecker::Required, "  --mode, -m  \tExecution mode (cpu|cuda). Default: cpu"},
    {CPU,     0, "", "cpu", option::Arg::None, "  --cpu  \tUse CPU execution mode (same as --mode cpu)"},
    {CUDA,    0, "", "cuda", option::Arg::None, "  --cuda  \tUse CUDA/GPU execution mode (same as --mode cuda)"},
    {NUM_WORLDS, 0, "n", "num-worlds", ArgChecker::Numeric, "  --num-worlds, -n <value>  \tNumber of parallel worlds (default: 1)"},
    {REPLAY,  0, "", "replay", ArgChecker::Required, "  --replay <file>  \tReplay actions from file"},
    {RECORD,  0, "r", "record", ArgChecker::Required, "  --record, -r <path>  \tRecord actions to file (press SPACE to start)"},
    {TRACK,   0, "t", "track", option::Arg::None, "  --track, -t  \tEnable trajectory tracking (default: world 0, agent 0)"},
    {TRACK_WORLD, 0, "", "track-world", ArgChecker::Numeric, "  --track-world <n>  \tSpecify world to track (default: 0)"},
    {TRACK_AGENT, 0, "", "track-agent", ArgChecker::Numeric, "  --track-agent <n>  \tSpecify agent to track (default: 0)"},
    {SEED,    0, "s", "seed", ArgChecker::Numeric, "  --seed, -s <value>  \tSet random seed (default: 5)"},
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
        std::cerr << "Error parsing command line arguments\n";
        delete[] options;
        delete[] buffer;
        return 1;
    }
    
    // Debug: Check for unknown options
    if (options[UNKNOWN]) {
        std::cerr << "Warning: Unknown options detected:\n";
        for (option::Option* opt = options[UNKNOWN]; opt; opt = opt->next()) {
            std::cerr << "  Unknown option: ";
            fwrite(opt->name, opt->namelen, 1, stderr);
            std::cerr << "\n";
        }
    }

    if (options[HELP]) {
        option::printUsage(std::cout, usage);
        std::cout << "\nKeyboard Controls:\n";
        std::cout << "  R          Reset current world\n";
        std::cout << "  T          Toggle trajectory tracking for current world\n";
        std::cout << "  SPACE      Start recording (when --record is used)\n";
        std::cout << "  WASD       Move agent (when in agent view)\n";
        std::cout << "  Q/E        Rotate agent left/right\n";
        std::cout << "  Shift      Move faster\n";
        std::cout << "\nExamples:\n";
        std::cout << "  viewer                                      # Single world on CPU\n";
        std::cout << "  viewer --num-worlds 4 --cpu                 # 4 worlds on CPU\n";
        std::cout << "  viewer --cuda --track                       # Track world 0, agent 0 on GPU\n";
        std::cout << "  viewer --cuda --track-world 2 --track-agent 1  # Track world 2, agent 1\n";
        std::cout << "  viewer -n 2 --cpu --record demo.bin        # Record 2 worlds to demo.bin\n";
        std::cout << "  viewer -n 2 --cpu --replay demo.bin        # Replay demo.bin with 2 worlds\n";
        std::cout << "  viewer -n 4 --cpu --seed 42                # 4 worlds with seed 42\n";
        delete[] options;
        delete[] buffer;
        return 0;
    }

    // Parameters with defaults
    uint32_t num_worlds = 1;
    ExecMode exec_mode = ExecMode::CPU;
    std::string record_path;
    std::string replay_path;
    uint32_t rand_seed = 5;
    int32_t track_world_idx = 0;  // Default to world 0
    int32_t track_agent_idx = 0;  // Default to agent 0
    bool track_trajectory = false;

    // Process options
    if (options[MODE]) {
        std::string mode_str(options[MODE].arg);
        if (mode_str == "cuda" || mode_str == "CUDA") {
            exec_mode = ExecMode::CUDA;
        } else if (mode_str != "cpu" && mode_str != "CPU") {
            std::cerr << "Invalid mode: " << mode_str << ". Use 'cpu' or 'cuda'\n";
            delete[] options;
            delete[] buffer;
            return 1;
        }
    }
    
    if (options[CPU]) {
        exec_mode = ExecMode::CPU;
    }
    
    if (options[CUDA]) {
        exec_mode = ExecMode::CUDA;
    }
    
    if (options[NUM_WORLDS]) {
        num_worlds = strtoul(options[NUM_WORLDS].arg, nullptr, 10);
    }
    
    if (options[REPLAY]) {
        replay_path = options[REPLAY].arg;
    }
    
    if (options[RECORD]) {
        record_path = options[RECORD].arg;
    }
    
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
    
    if (options[SEED]) {
        rand_seed = strtoul(options[SEED].arg, nullptr, 10);
    }

    // Check for any remaining non-option arguments
    if (parse.nonOptionsCount() > 0) {
        std::cerr << "Error: Unexpected arguments. All parameters must be specified as options.\n";
        std::cerr << "Use --help for usage information.\n";
        delete[] options;
        delete[] buffer;
        return 1;
    }

    // Validate options
    if (!record_path.empty() && !replay_path.empty()) {
        std::cerr << "Error: Cannot specify both --record and --replay\n";
        delete[] options;
        delete[] buffer;
        return 1;
    }

    // Setup replay - temporarily load to get metadata
    bool has_replay = false;
    uint32_t replay_seed = rand_seed;
    if (!replay_path.empty()) {
        // Temporarily create a manager to load replay and get metadata
        Manager temp_mgr({
            .execMode = exec_mode,
            .gpuID = 0,
            .numWorlds = 1,  // Temporary value
            .randSeed = 5,
            .autoReset = false,
        });
        
        if (temp_mgr.loadReplay(replay_path)) {
            const auto* replay_data = temp_mgr.getReplayData();
            if (replay_data) {
                // Validate and update num_worlds
                if (num_worlds != replay_data->metadata.num_worlds) {
                    std::cerr << "Warning: Replay was recorded with " << replay_data->metadata.num_worlds 
                              << " worlds, but viewer is using " << num_worlds << " worlds.\n";
                    std::cerr << "Setting num_worlds to match replay file.\n";
                    num_worlds = replay_data->metadata.num_worlds;
                }
                replay_seed = replay_data->metadata.seed;
                has_replay = true;
            }
        }
    }
    
    // Setup recording
    std::vector<int32_t> frame_actions;
    bool is_recording = !record_path.empty();
    
    if (is_recording) {
        frame_actions.resize(num_worlds * 3);
        // Initialize with default actions
        for (uint32_t i = 0; i < num_worlds; i++) {
            frame_actions[i * 3] = 0;      // move_amount = 0 (stop)
            frame_actions[i * 3 + 1] = 0;  // move_angle = 0 (forward)
            frame_actions[i * 3 + 2] = 2;  // rotate = 2 (no rotation)
        }
    }

    bool enable_batch_renderer =
#ifdef MADRONA_MACOS
        false;
#else
        true;
#endif

    WindowManager wm {};
    WindowHandle window = wm.makeWindow("Escape Room", 2730, 1536);
    render::GPUHandle render_gpu = wm.initGPU(0, { window.get() });

    // Use seed from replay if available
    uint32_t sim_seed = has_replay ? replay_seed : rand_seed;
    if (has_replay) {
        std::cout << "Using seed " << sim_seed << " from replay file\n";
    }

    // Create the simulation manager
    Manager mgr({
        .execMode = exec_mode,
        .gpuID = 0,
        .numWorlds = num_worlds,
        .randSeed = sim_seed,
        .autoReset = has_replay || is_recording,
        .enableBatchRenderer = enable_batch_renderer,
        .extRenderAPI = wm.gpuAPIManager().backend(),
        .extRenderDev = render_gpu.device(),
    });
    
    // Load replay if available
    if (has_replay) {
        mgr.loadReplay(replay_path);
    }
    
    // Start recording if requested
    if (is_recording) {
        mgr.startRecording(record_path, rand_seed);
    }
    
    // Enable trajectory tracking if requested
    if (track_trajectory) {
        mgr.enableTrajectoryLogging(track_world_idx, track_agent_idx, std::nullopt);
    }
    
    // Print help for controls
    std::cout << "\nViewer Controls:\n";
    std::cout << "  R: Reset current world\n";
    std::cout << "  T: Toggle trajectory tracking for current world\n";
    std::cout << "\n";

    float camera_move_speed = 10.f;

    math::Vector3 initial_camera_position = { 0, consts::worldLength / 2.f, 30 };

    math::Quat initial_camera_rotation =
        (math::Quat::angleAxis(-math::pi / 2.f, math::up) *
        math::Quat::angleAxis(-math::pi / 2.f, math::right)).normalize();


    // Create the viewer viewer
    viz::Viewer viewer(mgr.getRenderManager(), window.get(), {
        .numWorlds = num_worlds,
        .simTickRate = 20,
        .cameraMoveSpeed = camera_move_speed,
        .cameraPosition = initial_camera_position,
        .cameraRotation = initial_camera_rotation,
    });
    

    // Replay step
    auto replayStep = [&mgr]() {
        if (!mgr.hasReplay()) {
            return true;
        }
        
        printf("Step: %u\n", mgr.getCurrentReplayStep());
        bool finished = mgr.replayStep();
        return finished;
    };

    // Printers
    auto self_printer = mgr.selfObservationTensor().makePrinter();
    // Room entity observations removed
    auto steps_remaining_printer = mgr.stepsRemainingTensor().makePrinter();
    auto reward_printer = mgr.rewardTensor().makePrinter();

    auto printObs = [&]() {
        printf("Self\n");
        self_printer.print();

        // Room entity observations removed


        printf("Steps Remaining\n");
        steps_remaining_printer.print();

        printf("Reward\n");
        reward_printer.print();

        printf("\n");
    };
    (void)printObs;

    // Main loop for the viewer viewer
    viewer.loop(
    [&mgr, &track_trajectory, &track_world_idx, &track_agent_idx](CountT world_idx, const Viewer::UserInput &input)
    {
        using Key = Viewer::KeyboardKey;
        
        if (input.keyHit(Key::R)) {
            mgr.triggerReset(world_idx);
        }
        
        if (input.keyHit(Key::T)) {
            // Toggle trajectory tracking for current world
            if (track_trajectory && track_world_idx == (int32_t)world_idx) {
                mgr.disableTrajectoryLogging();
                track_trajectory = false;
                printf("Trajectory logging disabled\n");
            } else {
                mgr.enableTrajectoryLogging(world_idx, 0, std::nullopt); // Log agent 0 by default
                track_trajectory = true;
                track_world_idx = world_idx;
                track_agent_idx = 0;
                printf("Trajectory logging enabled for World %d, Agent 0\n", (int)world_idx);
            }
        }
    },
    [&mgr, &is_recording, &frame_actions](CountT world_idx, CountT,
           const Viewer::UserInput &input)
    {
        using Key = Viewer::KeyboardKey;

        int32_t x = 0;
        int32_t y = 0;
        int32_t r = 2;

        bool shift_pressed = input.keyPressed(Key::Shift);

        if (input.keyPressed(Key::W)) {
            y += 1;
        }
        if (input.keyPressed(Key::S)) {
            y -= 1;
        }

        if (input.keyPressed(Key::D)) {
            x += 1;
        }
        if (input.keyPressed(Key::A)) {
            x -= 1;
        }

        if (input.keyPressed(Key::Q)) {
            r += shift_pressed ? 2 : 1;
        }
        if (input.keyPressed(Key::E)) {
            r -= shift_pressed ? 2 : 1;
        }

        int32_t move_amount;
        if (x == 0 && y == 0) {
            move_amount = 0;
        } else if (shift_pressed) {
            move_amount = consts::numMoveAmountBuckets - 1;
        } else {
            move_amount = 1;
        }

        int32_t move_angle;
        if (x == 0 && y == 1) {
            move_angle = 0;
        } else if (x == 1 && y == 1) {
            move_angle = 1;
        } else if (x == 1 && y == 0) {
            move_angle = 2;
        } else if (x == 1 && y == -1) {
            move_angle = 3;
        } else if (x == 0 && y == -1) {
            move_angle = 4;
        } else if (x == -1 && y == -1) {
            move_angle = 5;
        } else if (x == -1 && y == 0) {
            move_angle = 6;
        } else if (x == -1 && y == 1) {
            move_angle = 7;
        } else {
            move_angle = 0;
        }

        mgr.setAction(world_idx, move_amount, move_angle, r);
        
        // Record the action if recording
        if (is_recording) {
            uint32_t base_idx = world_idx * 3;
            frame_actions[base_idx] = move_amount;
            frame_actions[base_idx + 1] = move_angle;
            frame_actions[base_idx + 2] = r;
        }
    }, [&]() {
        if (mgr.hasReplay()) {
            bool replay_finished = replayStep();

            if (replay_finished) {
                viewer.stopLoop();
            }
        }
        
        // Write frame actions if recording
        if (is_recording) {
            mgr.recordActions(frame_actions);
            
            // Reset actions to defaults for next frame
            for (uint32_t i = 0; i < num_worlds; i++) {
                frame_actions[i * 3] = 0;      // move_amount
                frame_actions[i * 3 + 1] = 0;  // move_angle
                frame_actions[i * 3 + 2] = 2;  // rotate
            }
        }

        mgr.step();

        //printObs();
    }, []() {});
    
    // Cleanup recording
    if (is_recording) {
        mgr.stopRecording();
    }
    
    delete[] options;
    delete[] buffer;
    
    return 0;
}