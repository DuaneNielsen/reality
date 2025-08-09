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
#include <thread>
#include <chrono>

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
    UNKNOWN, HELP, CUDA, NUM_WORLDS, REPLAY, RECORD, TRACK, TRACK_WORLD, TRACK_AGENT, TRACK_FILE, SEED, HIDE_MENU 
};

const option::Descriptor usage[] = {
    {UNKNOWN, 0, "", "", option::Arg::None, "USAGE: viewer [options]\n\n"
                                            "Madrona Escape Room - Interactive Viewer\n"
                                            "3D visualization and control of the simulation\n\n"
                                            "Options:"},
    {HELP,    0, "h", "help", option::Arg::None, "  --help, -h  \tPrint usage and exit."},
    {CUDA,    0, "", "cuda", ArgChecker::Numeric, "  --cuda <n>  \tUse CUDA/GPU execution mode on device n"},
    {NUM_WORLDS, 0, "n", "num-worlds", ArgChecker::Numeric, "  --num-worlds <value>, -n <value>  \tNumber of parallel worlds (default: 1)"},
    {REPLAY,  0, "", "replay", ArgChecker::Required, "  --replay <file>  \tReplay actions from file"},
    {RECORD,  0, "r", "record", ArgChecker::Required, "  --record <path>, -r <path>  \tRecord actions to file (press SPACE to start)"},
    {TRACK,   0, "t", "track", option::Arg::None, "  --track, -t  \tEnable trajectory tracking (default: world 0, agent 0)"},
    {TRACK_WORLD, 0, "", "track-world", ArgChecker::Numeric, "  --track-world <n>  \tSpecify world to track (default: 0)"},
    {TRACK_AGENT, 0, "", "track-agent", ArgChecker::Numeric, "  --track-agent <n>  \tSpecify agent to track (default: 0)"},
    {TRACK_FILE, 0, "", "track-file", ArgChecker::Required, "  --track-file <file>  \tSave trajectory to file"},
    {SEED,    0, "s", "seed", ArgChecker::Numeric, "  --seed <value>, -s <value>  \tSet random seed (default: 5)"},
    {HIDE_MENU, 0, "", "hide-menu", option::Arg::None, "  --hide-menu  \tHide ImGui menu (useful for clean screenshots)"},
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
        std::cout << "  SPACE      Pause/Resume simulation\n";
        std::cout << "  WASD       Move agent (when in agent view)\n";
        std::cout << "  Q/E        Rotate agent left/right\n";
        std::cout << "  Shift      Move faster\n";
        std::cout << "\nExamples:\n";
        std::cout << "  viewer                                      # Single world on CPU\n";
        std::cout << "  viewer --num-worlds 4                       # 4 worlds on CPU\n";
        std::cout << "  viewer --cuda 0 --track                     # Track world 0, agent 0 on GPU 0\n";
        std::cout << "  viewer --cuda 1 --track-world 2             # Use GPU 1, track world 2\n";
        std::cout << "  viewer -n 2 --record demo.bin              # Record 2 worlds to demo.bin\n";
        std::cout << "  viewer -n 2 --replay demo.bin              # Replay demo.bin with 2 worlds\n";
        std::cout << "  viewer -n 4 --seed 42                      # 4 worlds with seed 42\n";
        std::cout << "  viewer --track --track-file trajectory.csv  # Track and save to file\n";
        delete[] options;
        delete[] buffer;
        return 0;
    }

    // Parameters with defaults
    uint32_t num_worlds = 1;
    ExecMode exec_mode = ExecMode::CPU;
    uint32_t gpu_id = 0;
    std::string record_path;
    std::string replay_path;
    uint32_t rand_seed = 5;
    int32_t track_world_idx = 0;  // Default to world 0
    int32_t track_agent_idx = 0;  // Default to agent 0
    bool track_trajectory = false;
    std::string track_file;
    bool is_paused = false;  // Pause state

    // Process options
    if (options[CUDA]) {
        exec_mode = ExecMode::CUDA;
        gpu_id = strtoul(options[CUDA].arg, nullptr, 10);
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
    
    if (options[TRACK_FILE]) {
        track_file = options[TRACK_FILE].arg;
        // If track-file is specified without --track, enable tracking
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
    
    // No need to check if tracking is enabled for track_file since we auto-enable it now

    // Setup replay - read metadata from file
    bool has_replay = false;
    uint32_t replay_seed = rand_seed;
    if (!replay_path.empty()) {
        auto metadata_opt = Manager::readReplayMetadata(replay_path);
        if (metadata_opt.has_value()) {
            const auto& metadata = metadata_opt.value();
            // Validate and update num_worlds
            if (num_worlds != metadata.num_worlds) {
                std::cerr << "Warning: Replay was recorded with " << metadata.num_worlds 
                          << " worlds, but viewer is using " << num_worlds << " worlds.\n";
                std::cerr << "Setting num_worlds to match replay file.\n";
                num_worlds = metadata.num_worlds;
            }
            replay_seed = metadata.seed;
            has_replay = true;
        } else {
            std::cerr << "Error: Failed to read replay metadata\n";
            delete[] options;
            delete[] buffer;
            return 1;
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
        // Start paused in recording mode
        is_paused = true;
        printf("Recording mode: Starting PAUSED (press SPACE to start recording)\n");
    }

    bool enable_batch_renderer =
#ifdef MADRONA_MACOS
        false;
#else
        true;
#endif

    WindowManager wm {};
    WindowHandle window = wm.makeWindow("Escape Room", 1920, 1080);
    render::GPUHandle render_gpu = wm.initGPU(gpu_id, { window.get() });

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
        mgr.enableTrajectoryLogging(track_world_idx, track_agent_idx, 
            track_file.empty() ? std::nullopt : std::make_optional(track_file.c_str()));
        if (!track_file.empty()) {
            printf("Trajectory will be saved to: %s\n", track_file.c_str());
        }
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


    // Check if menu should be hidden
    bool hide_menu = options[HIDE_MENU];
    
    // Create the viewer viewer
    viz::Viewer viewer(mgr.getRenderManager(), window.get(), {
        .numWorlds = num_worlds,
        .simTickRate = 20,
        .cameraMoveSpeed = camera_move_speed,
        .cameraPosition = initial_camera_position,
        .cameraRotation = initial_camera_rotation,
        .hideMenu = hide_menu,
    });
    

    // Replay step
    auto replayStep = [&mgr]() {
        if (!mgr.hasReplay()) {
            return true;
        }
        
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
    [&mgr, &track_trajectory, &track_world_idx, &track_agent_idx, &is_paused](CountT world_idx, const Viewer::UserInput &input)
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
        
        if (input.keyHit(Key::Space)) {
            // Toggle pause
            is_paused = !is_paused;
            printf("Simulation %s\n", is_paused ? "PAUSED" : "RESUMED");
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
        if (!is_paused) {
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
        } else {
            // Sleep for 16ms (roughly 60 FPS) when paused to avoid burning CPU
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }

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