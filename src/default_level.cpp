#include <iostream>
#include <fstream>
#include <cstring>
#include "types.hpp"
#include "asset_ids.hpp"

using namespace madEscape;

int main(int argc, char* argv[]) {
    const char* output_file = "default_level.lvl";
    if (argc > 1) {
        output_file = argv[1];
    }
    
    // Create a 32x32 room with border walls
    CompiledLevel level = {};
    level.width = 32;
    level.height = 32;
    level.scale = 1.0f;
    level.max_entities = 150;  // Enough for walls and other objects
    std::strcpy(level.level_name, "default_32x32_room");
    
    // Initialize all transform data to defaults
    for (int i = 0; i < CompiledLevel::MAX_TILES; i++) {
        level.tile_z[i] = 0.0f;
        level.tile_scale_x[i] = 1.0f;
        level.tile_scale_y[i] = 1.0f;
        level.tile_scale_z[i] = 1.0f;
        level.tile_rot_w[i] = 1.0f;  // Identity quaternion
        level.tile_rot_x[i] = 0.0f;
        level.tile_rot_y[i] = 0.0f;
        level.tile_rot_z[i] = 0.0f;
    }
    
    // Set spawn point at center
    level.num_spawns = 1;
    level.spawn_x[0] = 0.0f;
    level.spawn_y[0] = 0.0f;
    level.spawn_facing[0] = 0.0f;
    
    // Generate border walls
    // The room spans from -16 to 16 in both X and Y
    // Place walls at the edges (-15.5 and 15.5) to create the border
    int tile_index = 0;
    const float half_size = 15.5f;  // 32/2 - 0.5 to place walls at edges
    
    // Top and bottom walls
    for (int i = 0; i < 32; i++) {
        float x = -half_size + i;
        
        // Top wall
        level.object_ids[tile_index] = AssetIDs::WALL;
        level.tile_x[tile_index] = x;
        level.tile_y[tile_index] = half_size;
        level.tile_persistent[tile_index] = true;
        level.tile_render_only[tile_index] = false;
        level.tile_entity_type[tile_index] = 2;  // EntityType::Wall
        tile_index++;
        
        // Bottom wall
        level.object_ids[tile_index] = AssetIDs::WALL;
        level.tile_x[tile_index] = x;
        level.tile_y[tile_index] = -half_size;
        level.tile_persistent[tile_index] = true;
        level.tile_render_only[tile_index] = false;
        level.tile_entity_type[tile_index] = 2;  // EntityType::Wall
        tile_index++;
    }
    
    // Left and right walls (skip corners to avoid duplicates)
    for (int i = 1; i < 31; i++) {
        float y = -half_size + i;
        
        // Left wall
        level.object_ids[tile_index] = AssetIDs::WALL;
        level.tile_x[tile_index] = -half_size;
        level.tile_y[tile_index] = y;
        level.tile_persistent[tile_index] = true;
        level.tile_render_only[tile_index] = false;
        level.tile_entity_type[tile_index] = 2;  // EntityType::Wall
        tile_index++;
        
        // Right wall
        level.object_ids[tile_index] = AssetIDs::WALL;
        level.tile_x[tile_index] = half_size;
        level.tile_y[tile_index] = y;
        level.tile_persistent[tile_index] = true;
        level.tile_render_only[tile_index] = false;
        level.tile_entity_type[tile_index] = 2;  // EntityType::Wall
        tile_index++;
    }
    
    // Add an axis marker at the origin for visual reference
    level.object_ids[tile_index] = AssetIDs::AXIS_X;
    level.tile_x[tile_index] = 0.0f;
    level.tile_y[tile_index] = 0.0f;
    level.tile_persistent[tile_index] = true;
    level.tile_render_only[tile_index] = true;
    level.tile_entity_type[tile_index] = 0;  // EntityType::None
    tile_index++;
    
    // Set the actual number of tiles used
    level.num_tiles = tile_index;
    
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