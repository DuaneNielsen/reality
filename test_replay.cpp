#include "src/mgr.hpp"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <replay_file>\n";
        return 1;
    }
    
    std::string filepath = argv[1];
    
    // Use Manager's own method to read metadata
    auto metadata_opt = madEscape::Manager::readReplayMetadata(filepath);
    
    if (!metadata_opt.has_value()) {
        std::cerr << "Failed to read replay metadata\n";
        return 1;
    }
    
    const auto& metadata = metadata_opt.value();
    std::cout << "Successfully read metadata:\n";
    std::cout << "  Sim name: " << metadata.sim_name << "\n";
    std::cout << "  Worlds: " << metadata.num_worlds << "\n";
    std::cout << "  Steps: " << metadata.num_steps << "\n";
    std::cout << "  Seed: " << metadata.seed << "\n";
    
    return 0;
}