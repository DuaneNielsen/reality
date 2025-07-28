#include "mgr.hpp"

#include <cstdio>
#include <chrono>
#include <string>
#include <filesystem>
#include <fstream>
#include <random>

#include <madrona/heap_array.hpp>

using namespace madrona;

[[maybe_unused]] static void saveWorldActions(
    const HeapArray<int32_t> &action_store,
    int32_t total_num_steps,
    int32_t world_idx)
{
    const int32_t *world_base = action_store.data() + world_idx * total_num_steps * 2 * 3;

    std::ofstream f("/tmp/actions", std::ios::binary);
    f.write((char *)world_base,
            sizeof(uint32_t) * total_num_steps * 2 * 3);
}

int main(int argc, char *argv[])
{
    using namespace madEscape;

    if (argc < 4) {
        fprintf(stderr, "%s TYPE NUM_WORLDS NUM_STEPS [--rand-actions] [--replay <file>] [--seed <value>] [--track-agent WORLD_ID AGENT_ID [--track-file <file>]]\n", argv[0]);
        return -1;
    }
    std::string type(argv[1]);

    ExecMode exec_mode;
    if (type == "CPU") {
        exec_mode = ExecMode::CPU;
    } else if (type == "CUDA") {
        exec_mode = ExecMode::CUDA;
    } else {
        fprintf(stderr, "Invalid ExecMode\n");
        return -1;
    }

    uint64_t num_worlds = std::stoul(argv[2]);
    uint64_t num_steps = std::stoul(argv[3]);

    HeapArray<int32_t> action_store(
        num_worlds * 2 * num_steps * 3);

    bool rand_actions = false;
    bool track_agent = false;
    int32_t track_world_idx = -1;
    int32_t track_agent_idx = -1;
    std::string replay_file;
    std::string track_file;
    bool replay_mode = false;
    uint32_t rand_seed = 5;
    
    // Parse optional arguments
    for (int i = 4; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--rand-actions") {
            rand_actions = true;
        } else if (arg == "--replay" && i + 1 < argc) {
            replay_mode = true;
            replay_file = argv[i + 1];
            i += 1; // Skip the next argument
        } else if (arg == "--track-agent" && i + 2 < argc) {
            track_agent = true;
            track_world_idx = std::stoi(argv[i + 1]);
            track_agent_idx = std::stoi(argv[i + 2]);
            i += 2; // Skip the next two arguments
        } else if (arg == "--track-file" && i + 1 < argc) {
            track_file = argv[i + 1];
            i += 1; // Skip the next argument
        } else if (arg == "--seed" && i + 1 < argc) {
            rand_seed = (uint32_t)std::stoi(argv[i + 1]);
            i += 1; // Skip the next argument
        }
    }
    
    // Validate options
    if (rand_actions && replay_mode) {
        fprintf(stderr, "Cannot use both --rand-actions and --replay\n");
        return -1;
    }

    Manager mgr({
        .execMode = exec_mode,
        .gpuID = 0,
        .numWorlds = (uint32_t)num_worlds,
        .randSeed = rand_seed,
        .autoReset = false,
        .enableBatchRenderer = false,
    });
    
    // Load actions from replay file if specified
    HeapArray<int32_t> replay_actions(0);
    uint64_t replay_num_steps = 0;
    
    if (replay_mode) {
        std::ifstream replay_stream(replay_file, std::ios::binary | std::ios::ate);
        if (!replay_stream.is_open()) {
            fprintf(stderr, "Failed to open replay file: %s\n", replay_file.c_str());
            return -1;
        }
        
        // Get file size
        size_t file_size = replay_stream.tellg();
        replay_stream.seekg(0, std::ios::beg);
        
        // Each action is 3 int32_t values (12 bytes)
        size_t action_size = 3 * sizeof(int32_t);
        size_t total_actions = file_size / action_size;
        
        // Calculate number of steps (total_actions = num_worlds * num_steps)
        if (total_actions % num_worlds != 0) {
            fprintf(stderr, "Replay file contains %zu actions, not divisible by %lu worlds\n", 
                    total_actions, num_worlds);
            return -1;
        }
        
        replay_num_steps = total_actions / num_worlds;
        
        // Update num_steps to match replay file
        if (replay_num_steps != num_steps) {
            fprintf(stderr, "Replay file contains %lu steps, using this instead of %lu\n",
                    replay_num_steps, num_steps);
            num_steps = replay_num_steps;
        }
        
        // Load all actions
        replay_actions = HeapArray<int32_t>(total_actions * 3);
        replay_stream.read((char*)replay_actions.data(), file_size);
        replay_stream.close();
        
        printf("Loaded %lu steps for %lu worlds from %s\n", 
               num_steps, num_worlds, replay_file.c_str());
    }
    
    // Enable trajectory tracking if requested
    if (track_agent) {
        if (!track_file.empty()) {
            mgr.enableTrajectoryLogging(track_world_idx, track_agent_idx, track_file.c_str());
        } else {
            mgr.enableTrajectoryLogging(track_world_idx, track_agent_idx, std::nullopt);
        }
    }

    std::random_device rd;
    std::mt19937 rand_gen(rd());
    std::uniform_int_distribution<int32_t> act_rand(0, 4);

    auto start = std::chrono::system_clock::now();

    for (CountT i = 0; i < (CountT)num_steps; i++) {
        if (replay_mode) {
            // Apply actions from replay file
            for (CountT j = 0; j < (CountT)num_worlds; j++) {
                // Actions are stored as: [step][world][3 components]
                // So for step i, world j: index = (i * num_worlds + j) * 3
                int64_t base_idx = (i * num_worlds + j) * 3;
                int32_t x = replay_actions[base_idx];
                int32_t y = replay_actions[base_idx + 1];
                int32_t r = replay_actions[base_idx + 2];
                
                mgr.setAction(j, x, y, r);
            }
        } else if (rand_actions) {
            for (CountT j = 0; j < (CountT)num_worlds; j++) {
                int32_t x = act_rand(rand_gen);
                int32_t y = act_rand(rand_gen);
                int32_t r = act_rand(rand_gen);

                mgr.setAction(j, x, y, r);
                
                int64_t base_idx = j * num_steps * 3 + i * 3;
                action_store[base_idx] = x;
                action_store[base_idx + 1] = y;
                action_store[base_idx + 2] = r;
            }
        }
        mgr.step();
    }

    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    float fps = (double)num_steps * (double)num_worlds / elapsed.count();
    printf("FPS %f\n", fps);
}
