#include <madrona/viz/viewer.hpp>
#include <madrona/render/render_mgr.hpp>
#include <madrona/window.hpp>

#include "sim.hpp"
#include "mgr.hpp"
#include "types.hpp"

#include <filesystem>
#include <fstream>
#include <string>
#include <cstring>

using namespace madrona;
using namespace madrona::viz;

static HeapArray<int32_t> readReplayLog(const char *path)
{
    std::ifstream replay_log(path, std::ios::binary);
    replay_log.seekg(0, std::ios::end);
    int64_t size = replay_log.tellg();
    replay_log.seekg(0, std::ios::beg);

    HeapArray<int32_t> log(size / sizeof(int32_t));

    replay_log.read((char *)log.data(), (size / sizeof(int32_t)) * sizeof(int32_t));

    return log;
}

int main(int argc, char *argv[])
{
    using namespace madEscape;

    constexpr int64_t num_views = consts::numAgents;

    // Read command line arguments
    uint32_t num_worlds = 1;
    if (argc >= 2) {
        num_worlds = (uint32_t)atoi(argv[1]);
    }

    ExecMode exec_mode = ExecMode::CPU;
    if (argc >= 3) {
        if (!strcmp("--cpu", argv[2])) {
            exec_mode = ExecMode::CPU;
        } else if (!strcmp("--cuda", argv[2])) {
            exec_mode = ExecMode::CUDA;
        }
    }

    // Setup replay/recording
    const char *replay_log_path = nullptr;
    const char *record_log_path = nullptr;
    bool is_recording = false;
    
    // Parse additional arguments
    if (argc >= 4) {
        if (strcmp("--record", argv[3]) == 0) {
            if (argc >= 5) {
                record_log_path = argv[4];
                is_recording = true;
            } else {
                fprintf(stderr, "Error: --record flag requires output path\n");
                return 1;
            }
        } else {
            replay_log_path = argv[3];
        }
    }

    auto replay_log = Optional<HeapArray<int32_t>>::none();
    uint32_t cur_replay_step = 0;
    uint32_t num_replay_steps = 0;
    if (replay_log_path != nullptr) {
        replay_log = readReplayLog(replay_log_path);
        num_replay_steps = replay_log->size() / (num_worlds * num_views * 3);
    }
    
    // Setup recording
    std::ofstream recording_file;
    std::vector<int32_t> frame_actions;
    uint32_t recorded_frames = 0;
    bool recording_started = false;  // Track if Space has been pressed to start recording
    
    if (is_recording) {
        recording_file.open(record_log_path, std::ios::binary);
        if (!recording_file.is_open()) {
            fprintf(stderr, "Error: Failed to open recording file: %s\n", record_log_path);
            return 1;
        }
        frame_actions.resize(num_worlds * 3);
        // Initialize with default actions
        for (uint32_t i = 0; i < num_worlds; i++) {
            frame_actions[i * 3] = 0;      // move_amount = 0 (stop)
            frame_actions[i * 3 + 1] = 0;  // move_angle = 0 (forward)
            frame_actions[i * 3 + 2] = 2;  // rotate = 2 (no rotation)
        }
        printf("Recording mode enabled. Press SPACE to start recording to: %s\n", record_log_path);
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

    // Create the simulation manager
    Manager mgr({
        .execMode = exec_mode,
        .gpuID = 0,
        .numWorlds = num_worlds,
        .randSeed = 5,
        .autoReset = replay_log.has_value() || is_recording,
        .enableBatchRenderer = enable_batch_renderer,
        .extRenderAPI = wm.gpuAPIManager().backend(),
        .extRenderDev = render_gpu.device(),
    });

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
    auto replayStep = [&]() {
        if (cur_replay_step == num_replay_steps - 1) {
            return true;
        }

        printf("Step: %u\n", cur_replay_step);

        for (uint32_t i = 0; i < num_worlds; i++) {
            uint32_t base_idx = 3 * (cur_replay_step * num_worlds + i);

            int32_t move_amount = (*replay_log)[base_idx];
            int32_t move_angle = (*replay_log)[base_idx + 1];
            int32_t turn = (*replay_log)[base_idx + 2];

            printf("%d: %d %d %d\n",
                   i, move_amount, move_angle, turn);
            mgr.setAction(i, move_amount, move_angle, turn);
        }

        cur_replay_step++;

        return false;
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
    [&mgr, &is_recording, &recording_started](CountT world_idx, const Viewer::UserInput &input)
    {
        using Key = Viewer::KeyboardKey;
        
        // Check for Space key to start recording
        if (is_recording && !recording_started && input.keyHit(Key::Space)) {
            recording_started = true;
            printf("Recording started!\n");
            // Optionally reset the world for a clean start
            mgr.triggerReset(world_idx);
        }
        
        if (input.keyHit(Key::R)) {
            mgr.triggerReset(world_idx);
        }
    },
    [&mgr, &is_recording, &recording_started, &frame_actions](CountT world_idx, CountT agent_idx,
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
        
        // Record the action if recording has started
        if (is_recording && recording_started) {
            uint32_t base_idx = world_idx * 3;
            frame_actions[base_idx] = move_amount;
            frame_actions[base_idx + 1] = move_angle;
            frame_actions[base_idx + 2] = r;
        }
    }, [&]() {
        if (replay_log.has_value()) {
            bool replay_finished = replayStep();

            if (replay_finished) {
                viewer.stopLoop();
            }
        }
        
        // Write frame actions if recording has started
        if (is_recording && recording_started) {
            recording_file.write(reinterpret_cast<const char*>(frame_actions.data()), 
                               frame_actions.size() * sizeof(int32_t));
            recorded_frames++;
            
            // Reset actions to defaults for next frame
            for (uint32_t i = 0; i < num_worlds; i++) {
                frame_actions[i * 3] = 0;      // move_amount
                frame_actions[i * 3 + 1] = 0;  // move_angle
                frame_actions[i * 3 + 2] = 2;  // rotate
            }
            
            if (recorded_frames % 100 == 0) {
                printf("Recorded %u frames...\n", recorded_frames);
            }
        }

        mgr.step();

        //printObs();
    }, []() {});
    
    // Cleanup recording
    if (is_recording) {
        recording_file.close();
        if (recorded_frames > 0) {
            printf("Recording complete: %u frames saved to %s\n", recorded_frames, record_log_path);
        } else {
            printf("Recording cancelled: No frames were recorded\n");
        }
    }
}
