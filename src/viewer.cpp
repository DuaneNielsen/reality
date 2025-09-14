#include <madrona/viz/viewer.hpp>
#include <madrona/render/render_mgr.hpp>
#include <madrona/window.hpp>

#include "mgr.hpp"
#include "types.hpp"
#include "viewer_core.hpp"
#include "camera_controller.hpp"

#include <optionparser.h>

#include <string>
#include <cstring>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <chrono>

using namespace madrona;
using namespace madrona::viz;

namespace ArgChecker {
    static option::ArgStatus Required(const option::Option& option, bool msg)
    {
        if (option.arg != nullptr)
            return option::ARG_OK;
        
        if (msg) std::cerr << "Option '" << option.name << "' requires an argument\n";
        return option::ARG_ILLEGAL;
    }
    
    static option::ArgStatus Numeric(const option::Option& option, bool msg)
    {
        char* endptr = nullptr;
        if (option.arg != nullptr && strtol(option.arg, &endptr, 10)){};
        if (endptr != option.arg && *endptr == '\0')
            return option::ARG_OK;
        
        if (msg) std::cerr << "Option '" << option.name << "' requires a numeric argument\n";
        return option::ARG_ILLEGAL;
    }
    
    static option::ArgStatus OptionalNumeric(const option::Option& option, bool msg)
    {
        // No argument is OK (will use default behavior)
        if (option.arg == nullptr)
            return option::ARG_OK;
        
        // If argument provided, must be a valid float
        char* endptr = nullptr;
        strtof(option.arg, &endptr);
        if (endptr != option.arg && *endptr == '\0')
            return option::ARG_OK;
        
        if (msg) std::cerr << "Option '" << option.name << "' requires a numeric argument if provided\n";
        return option::ARG_ILLEGAL;
    }
}

enum OptionIndex { 
    UNKNOWN, HELP, CUDA, NUM_WORLDS, LOAD, REPLAY, RECORD, TRACK, TRACK_WORLD, TRACK_AGENT, TRACK_FILE, SEED, HIDE_MENU, PAUSE, AUTO_RESET, FOLLOW
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
    {PAUSE,   0, "p", "pause", ArgChecker::OptionalNumeric, "  --pause [delay], -p [delay]  \tStart paused, optionally auto-resume after delay seconds"},
    {AUTO_RESET, 0, "", "auto-reset", option::Arg::None, "  --auto-reset  \tAutomatically reset episodes when agents complete them"},
    {FOLLOW,  0, "f", "follow", option::Arg::None, "  --follow, -f  \tStart in tracking camera mode (follow the agent)"},
    {0, 0, 0, 0, 0, 0}
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
    
    // Check for unknown options
    if (options[UNKNOWN]) {
        std::cerr << "Error: Unknown options detected:\n";
        for (option::Option* opt = options[UNKNOWN]; opt; opt = opt->next()) {
            std::cerr << "  Unknown option: ";
            fwrite(opt->name, opt->namelen, 1, stderr);
            std::cerr << "\n";
        }
        std::cerr << "\nUse --help to see available options.\n";
        delete[] options;
        delete[] buffer;
        return 1;
    }

    if (options[HELP]) {
        option::printUsage(std::cout, usage);
        std::cout << "\nKeyboard Controls:\n";
        std::cout << "  R          Reset current world\n";
        std::cout << "  T          Toggle trajectory tracking for current world\n";
        std::cout << "  L          Toggle lidar ray visualization\n";
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
        std::cout << "  viewer --auto-reset                          # Enable automatic episode reset\n";
        delete[] options;
        delete[] buffer;
        return 0;
    }

    // Parameters with defaults
    uint32_t num_worlds = 1;
    madrona::ExecMode exec_mode = madrona::ExecMode::CPU;
    uint32_t gpu_id = 0;
    std::string load_path;
    std::string record_path;
    std::string replay_path;
    uint32_t rand_seed = consts::fileFormat::defaultSeed;
    int32_t track_world_idx = 0;  // Default to world 0
    int32_t track_agent_idx = 0;  // Default to agent 0
    bool track_trajectory = false;
    std::string track_file;
    bool start_paused = false;
    float pause_delay_seconds = 0.0f;
    bool auto_reset = false;
    bool start_follow_mode = false;

    // Process options
    if (options[CUDA]) {
        exec_mode = madrona::ExecMode::CUDA;
        gpu_id = strtoul(options[CUDA].arg, nullptr, 10);
    }
    
    if (options[NUM_WORLDS]) {
        num_worlds = strtoul(options[NUM_WORLDS].arg, nullptr, 10);
    }
    
    if (options[LOAD]) {
        load_path = options[LOAD].arg;
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
    
    if (options[PAUSE]) {
        start_paused = true;
        if (options[PAUSE].arg != nullptr) {
            pause_delay_seconds = strtof(options[PAUSE].arg, nullptr);
            printf("Starting paused, will auto-resume after %.1f seconds\n", pause_delay_seconds);
        } else {
            printf("Starting paused (press SPACE to resume)\n");
        }
    }
    
    if (options[AUTO_RESET]) {
        auto_reset = true;
        printf("Auto-reset enabled: episodes will restart automatically when agents complete them\n");
    }
    
    if (options[FOLLOW]) {
        start_follow_mode = true;
        printf("Starting in tracking camera mode (following agent)\n");
    }

    // Check for any remaining non-option arguments
    if (parse.nonOptionsCount() > 0) {
        std::cerr << "Error: Unexpected arguments. All parameters must be specified as options.\n";
        std::cerr << "Use --help for usage information.\n";
        delete[] options;
        delete[] buffer;
        return 1;
    }

    // Validate three-flag interface
    // int mode_flag_count = 0;
    // if (!replay_path.empty()) mode_flag_count++;
    // if (!record_path.empty()) mode_flag_count++;
    
    // --load can be combined with --record, but --replay is mutually exclusive
    if (!replay_path.empty() && !record_path.empty()) {
        std::cerr << "Error: Cannot specify both --replay and --record\n";
        delete[] options;
        delete[] buffer;
        return 1;
    }
    
    if (!replay_path.empty() && !load_path.empty()) {
        std::cerr << "Error: --replay ignores --load (level comes from recording)\n";
        delete[] options;
        delete[] buffer;
        return 1;
    }
    
    // If no replay and no load path specified, use embedded default level
    bool use_embedded_default = replay_path.empty() && load_path.empty();
    if (use_embedded_default) {
        std::cout << "No level file specified, using embedded default level\n";
    }
    
    // Validate file extensions
    if (!load_path.empty() && !load_path.ends_with(".lvl")) {
        std::cerr << "Warning: --load expects .lvl file, got: " << load_path << "\n";
    }
    if (!replay_path.empty() && !replay_path.ends_with(".rec")) {
        std::cerr << "Warning: --replay expects .rec file, got: " << replay_path << "\n";
    }
    if (!record_path.empty() && !record_path.ends_with(".rec")) {
        std::cerr << "Warning: --record expects .rec file, got: " << record_path << "\n";
    }

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

    bool enable_batch_renderer =
#ifdef MADRONA_MACOS
        false;
#else
        true;
#endif

    WindowManager wm {};
    WindowHandle window = wm.makeWindow("Escape Room", consts::display::defaultWindowWidth, consts::display::defaultWindowHeight);
    render::GPUHandle render_gpu = wm.initGPU(gpu_id, { window.get() });

    // Use seed from replay if available
    uint32_t sim_seed = has_replay ? replay_seed : rand_seed;
    if (has_replay) {
        std::cout << "Using seed " << sim_seed << " from replay file\n";
    }

    // Load level based on three-flag interface
    CompiledLevel loaded_level = {};
    std::vector<std::optional<CompiledLevel>> per_world_levels;
    
    if (use_embedded_default) {
        // Load from embedded default level data
        #include "default_level_data.h"
        if (sizeof(CompiledLevel) != default_level_lvl_len) {
            std::cerr << "Error: Embedded level size mismatch: expected " << sizeof(CompiledLevel) 
                     << " bytes, got " << default_level_lvl_len << " bytes\n";
            delete[] options;
            delete[] buffer;
            return 1;
        }
        
        std::memcpy(&loaded_level, default_level_lvl, sizeof(CompiledLevel));
        printf("Loaded embedded default level: %dx%d grid, %d tiles\n", 
               loaded_level.width, loaded_level.height, loaded_level.num_tiles);
        
        // All worlds use the same loaded level
        per_world_levels.resize(num_worlds, loaded_level);
        
    } else if (!load_path.empty()) {
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
        per_world_levels.resize(num_worlds, loaded_level);
        
    } else if (!replay_path.empty()) {
        // Replay mode - extract embedded level data from recording
        auto embedded_level = Manager::readEmbeddedLevel(replay_path);
        if (!embedded_level.has_value()) {
            std::cerr << "Error: Failed to read embedded level from replay file: " << replay_path << "\n";
            delete[] options;
            delete[] buffer;
            return 1;
        }
        
        loaded_level = embedded_level.value();
        printf("Extracted embedded level from %s: %dx%d grid, %d tiles\n", 
               replay_path.c_str(), loaded_level.width, loaded_level.height, loaded_level.num_tiles);
        
        // All worlds use the embedded level
        per_world_levels.resize(num_worlds, loaded_level);
    }
    
    // Create the simulation manager
    Manager mgr({
        .execMode = exec_mode,
        .gpuID = 0,
        .numWorlds = num_worlds,
        .randSeed = sim_seed,
        .autoReset = auto_reset || has_replay || !record_path.empty(),
        .enableBatchRenderer = enable_batch_renderer,
        .extRenderAPI = wm.gpuAPIManager().backend(),
        .extRenderDev = render_gpu.device(),
        .perWorldCompiledLevels = per_world_levels,
    });
    
    // Create ViewerCore config
    ViewerCore::Config core_config {
        .num_worlds = num_worlds,
        .rand_seed = rand_seed,
        .auto_reset = auto_reset || has_replay || !record_path.empty(),
        .load_path = load_path,
        .record_path = record_path,
        .replay_path = replay_path,
        .start_paused = start_paused,
        .pause_delay_seconds = pause_delay_seconds,
        .grid_cols = static_cast<uint32_t>(std::ceil(std::sqrt(num_worlds))),
    };
    
    // Initialize ViewerCore
    ViewerCore viewer_core(core_config, &mgr);
    
    // Load replay if available
    if (has_replay) {
        viewer_core.loadReplay(replay_path);
    }
    
    // Start recording if requested
    if (!record_path.empty()) {
        viewer_core.startRecording(record_path);
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
    std::cout << "  L: Toggle lidar ray visualization\n";
    std::cout << "  F: Toggle camera mode (FreeFly/Tracking)\n";
    std::cout << "  WASD: Move camera\n";
    std::cout << "  ZC: Move forward/backward (in camera direction)\n";
    std::cout << "  QE: Rotate camera 90 degrees left/right\n";
    std::cout << "  Right Mouse + Move: Free look rotation\n";
    std::cout << "\n";

    // Initialize camera controllers
    std::unique_ptr<madEscape::FreeFlyCameraController> freeFlyCamera(new madEscape::FreeFlyCameraController());
    std::unique_ptr<madEscape::TrackingCameraController> trackingCamera(new madEscape::TrackingCameraController());
    
    // Get initial agent position for tracking camera
    math::Vector3 agentPos = mgr.getAgentPosition(0, 0);
    
    // Get level bounds for smart camera positioning and rotation
    const CompiledLevel* level = mgr.getCompiledLevel(0);
    float width = level->world_max_x - level->world_min_x;
    float length = level->world_max_y - level->world_min_y;
    float height = std::max(width, length) * 0.5f;
    
    // Position camera above origin
    math::Vector3 origin = {0.0f, 0.0f, 0.0f};
    math::Vector3 camPos = {0.0f, 0.0f, height};
    
    freeFlyCamera->setPosition(camPos);
    freeFlyCamera->setLookAt(origin);
    
    // Rotate camera 90° if level is longer than wide
    if (length > width * 1.2f) {
        // Rotate camera 90 degrees to align Y-axis with screen width
        freeFlyCamera->setYaw(90.0f * M_PI / 180.0f);  // Convert degrees to radians
    }
    
    // Initialize tracking camera with agent position
    trackingCamera->setTarget(agentPos);
    
    // Current camera controller - start in follow mode if requested
    madEscape::CameraController* currentCamera = start_follow_mode ? 
        static_cast<madEscape::CameraController*>(trackingCamera.get()) : 
        static_cast<madEscape::CameraController*>(freeFlyCamera.get());
    int cameraMode = start_follow_mode ? 1 : 0; // 0=FreeFly, 1=Tracking
    
    float camera_move_speed = consts::display::defaultCameraDist;
    math::Vector3 initial_camera_position = { 0.0f, -14.0f, 35.0f };

    // Look down at the room to see all four walls
    // math::Quat initial_camera_rotation =
    //     (math::Quat::angleAxis(-math::pi / 2.f, math::up) *
    //     math::Quat::angleAxis(-math::pi * 0.4f, math::right)).normalize();

    math::Quat initial_camera_rotation =
    (math::Quat::angleAxis(-math::pi * consts::display::viewerRotationFactor, math::right)).normalize();

    // Check if menu should be hidden
    bool hide_menu = options[HIDE_MENU];
    
    // Create the viewer
    viz::Viewer viewer(mgr.getRenderManager(), window.get(), {
        .numWorlds = num_worlds,
        .simTickRate = consts::display::defaultSimTickRate,
        .cameraMoveSpeed = camera_move_speed,
        .cameraPosition = initial_camera_position,
        .cameraRotation = initial_camera_rotation,
        .hideMenu = hide_menu,
    });

    // Printers for debugging (kept from original)
    auto self_printer = mgr.selfObservationTensor().makePrinter();
    auto lidar_printer = mgr.lidarTensor().makePrinter();
    auto steps_taken_printer = mgr.stepsTakenTensor().makePrinter();
    auto reward_printer = mgr.rewardTensor().makePrinter();

    auto printObs = [&]() {
        printf("Self\n");
        self_printer.print();

        printf("Lidar\n");
        lidar_printer.print();

        printf("Steps Taken\n");
        steps_taken_printer.print();

        printf("Reward\n");
        reward_printer.print();

        printf("\n");
    };

    // Apply initial camera state from our controller to the viewer
    // This overrides the default camera position set in the viewer constructor
    {
        madEscape::CameraState initialCamState = currentCamera->getState();
        viewer.setCameraVectors(initialCamState.position, initialCamState.forward, 
                               initialCamState.up, initialCamState.right);
    }
    
    // Track time for frame delta
    auto lastFrameTime = std::chrono::steady_clock::now();
    
    // Main loop for the viewer
    viewer.loop(
    [&viewer_core, &printObs, &mgr, &viewer, &currentCamera, &cameraMode, 
     &freeFlyCamera, &trackingCamera, &lastFrameTime, num_worlds]
    (CountT world_idx, const Viewer::UserInput &input)
    {
        using Key = Viewer::KeyboardKey;
        
        // Calculate frame delta time
        auto currentTime = std::chrono::steady_clock::now();
        float deltaTime = std::chrono::duration<float>(currentTime - lastFrameTime).count();
        lastFrameTime = currentTime;
        
        // Camera mode switching (toggle between FreeFly and Tracking)
        if (input.keyHit(Key::F)) {
            cameraMode = 1 - cameraMode;  // Toggle between 0 and 1
            if (cameraMode == 0) {
                currentCamera = freeFlyCamera.get();
            } else {
                currentCamera = trackingCamera.get();
                // Initialize tracking camera with current agent position
                math::Vector3 agentPos = mgr.getAgentPosition(0, 0);
                trackingCamera->setTarget(agentPos);
            }
        }
        
        // Check if we're in free camera mode (viewID == -1) or agent camera mode (viewID >= 0)
        // getCurrentViewID() returns viewIdx - 1, so Free Camera returns -1
        bool inFreeCameraMode = (viewer.getCurrentViewID() == -1);
        
        if (inFreeCameraMode) {
            // Update tracking camera target if in tracking mode
            if (cameraMode == 1) {
                // Get agent position from manager (tracks first agent in world 0)
                math::Vector3 agentPos = mgr.getAgentPosition(0, 0);
                trackingCamera->setTarget(agentPos);
                
            }
            
            // Build camera input state
            madEscape::CameraInputState camInput;
            camInput.forward = input.keyPressed(Key::W);
            camInput.backward = input.keyPressed(Key::S);
            camInput.left = input.keyPressed(Key::A);
            camInput.right = input.keyPressed(Key::D);
            camInput.up = input.keyPressed(Key::Z);      // Z = forward in camera direction
            camInput.down = input.keyPressed(Key::C);    // C = backward in camera direction
            camInput.rotateLeft = input.keyPressed(Key::Q);   // Q = rotate left 90 degrees
            camInput.rotateRight = input.keyPressed(Key::E);  // E = rotate right 90 degrees  
            camInput.boost = input.keyPressed(Key::Shift);
            
            // Handle camera input and update
            currentCamera->handleInput(camInput, deltaTime);
            currentCamera->update(deltaTime);
            
            // Apply camera state to viewer
            madEscape::CameraState camState = currentCamera->getState();
            
            
            viewer.setCameraVectors(camState.position, camState.forward, 
                                   camState.up, camState.right);
        }
        
        // Map Viewer keyboard input to ViewerCore events
        if (input.keyHit(Key::R)) {
            ViewerCore::InputEvent event{
                ViewerCore::InputEvent::KeyHit,
                ViewerCore::InputEvent::R
            };
            viewer_core.handleInput(world_idx, event);
        }
        
        if (input.keyHit(Key::T)) {
            ViewerCore::InputEvent event{
                ViewerCore::InputEvent::KeyHit,
                ViewerCore::InputEvent::T
            };
            viewer_core.handleInput(world_idx, event);
        }
        
        if (input.keyHit(Key::Space)) {
            ViewerCore::InputEvent event{
                ViewerCore::InputEvent::KeyHit,
                ViewerCore::InputEvent::Space
            };
            viewer_core.handleInput(world_idx, event);
        }
        
        if (input.keyHit(Key::M)) {
            ViewerCore::InputEvent event{
                ViewerCore::InputEvent::KeyHit,
                ViewerCore::InputEvent::M
            };
            viewer_core.handleInput(world_idx, event);
            
            // Update viewer grid settings from viewer_core config
            const auto& config = viewer_core.getConfig();
            const CompiledLevel* level = mgr.getCompiledLevel(0);
            // Calculate actual world dimensions from level boundaries
            float worldWidth = level->world_max_x - level->world_min_x;
            float worldHeight = level->world_max_y - level->world_min_y;
            
            if (config.multi_world_grid) {
                printf("Multi-world grid DEBUG:\n");
                printf("  Level dimensions: width=%f, height=%f, scale=%f\n", level->width, level->height, level->world_scale);
                printf("  Level boundaries: min_x=%f, max_x=%f, min_y=%f, max_y=%f\n", 
                       level->world_min_x, level->world_max_x, level->world_min_y, level->world_max_y);
                printf("  Calculated WorldWidth: %f, WorldHeight: %f\n", worldWidth, worldHeight);
                printf("  Spacing: %f, GridCols: %u\n", config.world_spacing, config.grid_cols);
                printf("  Num worlds: %u\n", num_worlds);
                
                // Stage 2: Use external parameters from config and level data
                viewer.setMultiWorldGrid(true, config.world_spacing, config.grid_cols, 
                                       worldWidth, worldHeight);
                viewer.setExploreMode(false);
                
            } else if (config.explore_mode) {
                viewer.setMultiWorldGrid(false, config.world_spacing, config.grid_cols, 
                                       worldWidth, worldHeight);
                viewer.setExploreMode(true);
            } else {
                viewer.setMultiWorldGrid(false, config.world_spacing, config.grid_cols, 
                                       worldWidth, worldHeight);
                viewer.setExploreMode(false);
            }
        }
        
        // Print observations when 'O' is pressed
        if (input.keyHit(Key::O)) {
            printObs();
        }
        
        // Toggle lidar visualization with 'L' key
        if (input.keyHit(Key::L)) {
            // Call the actual toggle method on the manager
            mgr.toggleLidarVisualization(world_idx);
        }
    },
    [&viewer_core](CountT world_idx, CountT,
           const Viewer::UserInput &input)
    {
        using Key = Viewer::KeyboardKey;

        // Map key press/release events to ViewerCore
        auto sendEvent = [&](Key /* key */, ViewerCore::InputEvent::Key core_key, bool pressed) {
            if (pressed) {
                ViewerCore::InputEvent event{
                    ViewerCore::InputEvent::KeyPress,
                    core_key
                };
                viewer_core.handleInput(world_idx, event);
            } else {
                ViewerCore::InputEvent event{
                    ViewerCore::InputEvent::KeyRelease,
                    core_key
                };
                viewer_core.handleInput(world_idx, event);
            }
        };

        // Track previous state to detect changes
        static bool prev_w = false, prev_a = false, prev_s = false, prev_d = false;
        static bool prev_q = false, prev_e = false, prev_shift = false;
        
        bool curr_w = input.keyPressed(Key::W);
        bool curr_a = input.keyPressed(Key::A);
        bool curr_s = input.keyPressed(Key::S);
        bool curr_d = input.keyPressed(Key::D);
        bool curr_q = input.keyPressed(Key::Q);
        bool curr_e = input.keyPressed(Key::E);
        bool curr_shift = input.keyPressed(Key::Shift);
        
        // Send events only on state change
        if (curr_w != prev_w) sendEvent(Key::W, ViewerCore::InputEvent::W, curr_w);
        if (curr_a != prev_a) sendEvent(Key::A, ViewerCore::InputEvent::A, curr_a);
        if (curr_s != prev_s) sendEvent(Key::S, ViewerCore::InputEvent::S, curr_s);
        if (curr_d != prev_d) sendEvent(Key::D, ViewerCore::InputEvent::D, curr_d);
        if (curr_q != prev_q) sendEvent(Key::Q, ViewerCore::InputEvent::Q, curr_q);
        if (curr_e != prev_e) sendEvent(Key::E, ViewerCore::InputEvent::E, curr_e);
        if (curr_shift != prev_shift) sendEvent(Key::Shift, ViewerCore::InputEvent::Shift, curr_shift);
        
        // Update previous state
        prev_w = curr_w;
        prev_a = curr_a;
        prev_s = curr_s;
        prev_d = curr_d;
        prev_q = curr_q;
        prev_e = curr_e;
        prev_shift = curr_shift;
        
        // Compute and apply actions
        viewer_core.updateFrameActions(world_idx, 0);
    }, [&viewer_core, &viewer]() {
        // Update frame (handles timing and conditionally steps simulation)
        viewer_core.updateFrame();
        
        // Check if we should exit
        auto frame_state = viewer_core.getFrameState();
        if (frame_state.should_exit) {
            viewer.stopLoop();
        }
        
        // If paused, sleep to avoid burning CPU
        if (frame_state.is_paused) {
            std::this_thread::sleep_for(std::chrono::milliseconds(16));
        }
        
        //printObs();
    }, []() {});
    
    // Cleanup is handled by ViewerCore destructor
    
    delete[] options;
    delete[] buffer;
    
    return 0;
}