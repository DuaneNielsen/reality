#include <iostream>
#include <fstream>
#include <cstring>
#include "types.hpp"

using namespace madEscape;

int main(int argc, char* argv[]) {
    const char* output_file = "default_level.lvl";
    if (argc > 1) {
        output_file = argv[1];
    }
    
    // Create a simple hardcoded level
    CompiledLevel level = {};
    level.width = 3;
    level.height = 3;
    level.scale = 1.0f;
    level.num_tiles = 4;
    level.max_entities = 10;
    std::strcpy(level.level_name, "default_level");
    
    // Set spawn point
    level.num_spawns = 1;
    level.spawn_x[0] = 0.0f;
    level.spawn_y[0] = 0.0f;
    level.spawn_facing[0] = 0.0f;
    
    // Tile 0: Wall at (-1, -1)
    level.object_ids[0] = 2;  // WALL
    level.tile_x[0] = -1.0f;
    level.tile_y[0] = -1.0f;
    level.tile_persistent[0] = true;
    level.tile_render_only[0] = false;
    level.tile_entity_type[0] = 2;  // EntityType::Wall
    
    // Tile 1: Cube at (1, -1)
    level.object_ids[1] = 1;  // CUBE
    level.tile_x[1] = 1.0f;
    level.tile_y[1] = -1.0f;
    level.tile_persistent[1] = false;
    level.tile_render_only[1] = false;
    level.tile_entity_type[1] = 1;  // EntityType::Cube
    
    // Tile 2: Wall at (-1, 1)
    level.object_ids[2] = 2;  // WALL
    level.tile_x[2] = -1.0f;
    level.tile_y[2] = 1.0f;
    level.tile_persistent[2] = true;
    level.tile_render_only[2] = false;
    level.tile_entity_type[2] = 2;  // EntityType::Wall
    
    // Tile 3: Axis marker (render-only) at (1, 1)
    level.object_ids[3] = 5;  // AXIS_X
    level.tile_x[3] = 1.0f;
    level.tile_y[3] = 1.0f;
    level.tile_persistent[3] = true;
    level.tile_render_only[3] = true;
    level.tile_entity_type[3] = 0;  // EntityType::None
    
    // Write to file
    std::ofstream file(output_file, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open output file: " << output_file << std::endl;
        return 1;
    }
    
    file.write(reinterpret_cast<const char*>(&level), sizeof(CompiledLevel));
    file.close();
    
    std::cout << "Generated level file: " << output_file << " (" << sizeof(CompiledLevel) << " bytes)" << std::endl;
    return 0;
}