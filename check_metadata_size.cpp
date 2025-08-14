#include <iostream>
#include "src/replay_metadata.hpp"

int main() {
    std::cout << "sizeof(ReplayMetadata) = " << sizeof(madrona::escape_room::ReplayMetadata) << std::endl;
    
    madrona::escape_room::ReplayMetadata meta;
    std::cout << "Offset of magic: " << offsetof(madrona::escape_room::ReplayMetadata, magic) << std::endl;
    std::cout << "Offset of version: " << offsetof(madrona::escape_room::ReplayMetadata, version) << std::endl;
    std::cout << "Offset of sim_name: " << offsetof(madrona::escape_room::ReplayMetadata, sim_name) << std::endl;
    std::cout << "Offset of num_worlds: " << offsetof(madrona::escape_room::ReplayMetadata, num_worlds) << std::endl;
    std::cout << "Offset of num_agents_per_world: " << offsetof(madrona::escape_room::ReplayMetadata, num_agents_per_world) << std::endl;
    std::cout << "Offset of num_steps: " << offsetof(madrona::escape_room::ReplayMetadata, num_steps) << std::endl;
    std::cout << "Offset of actions_per_step: " << offsetof(madrona::escape_room::ReplayMetadata, actions_per_step) << std::endl;
    std::cout << "Offset of timestamp: " << offsetof(madrona::escape_room::ReplayMetadata, timestamp) << std::endl;
    std::cout << "Offset of seed: " << offsetof(madrona::escape_room::ReplayMetadata, seed) << std::endl;
    std::cout << "Offset of reserved: " << offsetof(madrona::escape_room::ReplayMetadata, reserved) << std::endl;
    
    return 0;
}