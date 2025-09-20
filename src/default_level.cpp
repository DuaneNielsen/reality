#include <iostream>
#include <fstream>
#include <cstring>
#include <vector>
#include "types.hpp"
#include "asset_ids.hpp"
#include "level_io.hpp"

using namespace madEscape;

// Helper function to initialize a CompiledLevel with common settings
CompiledLevel createBaseLevelTemplate() {
    CompiledLevel level = {};
    level.width = 16;
    level.height = 16;
    level.world_scale = 1.0f;
    level.max_entities = 150;  // Enough for walls and obstacles
    
    // World boundaries: 16 tiles × 2.5 spacing = 40 units, centered = ±20
    level.world_min_x = -20.0f;
    level.world_max_x = 20.0f;
    level.world_min_y = -20.0f;
    level.world_max_y = 20.0f;
    level.world_min_z = 0.0f;    // Floor level
    level.world_max_z = 25.0f;   // 10 * 2.5 (reasonable max height)
    
    // Initialize all transform data to defaults
    for (int i = 0; i < CompiledLevel::MAX_TILES; i++) {
        level.tile_z[i] = 0.0f;
        level.tile_scale_x[i] = 1.0f;
        level.tile_scale_y[i] = 1.0f;
        level.tile_scale_z[i] = 1.0f;
        level.tile_rotation[i] = Quat::id();  // Identity quaternion
        level.tile_response_type[i] = (int32_t)ResponseType::Static;  // Default to Static
        level.tile_done_on_collide[i] = false;  // Default to not ending episode on collision
        
        // Initialize randomization arrays to 0 (no randomization)
        level.tile_rand_x[i] = 0.0f;
        level.tile_rand_y[i] = 0.0f;
        level.tile_rand_z[i] = 0.0f;
        level.tile_rand_rot_z[i] = 0.0f;
        level.tile_rand_scale_x[i] = 0.0f;
        level.tile_rand_scale_y[i] = 0.0f;
        level.tile_rand_scale_z[i] = 0.0f;
    }
    
    // Set spawn point at x=0, y=-14.5 (moved forward from southern wall)
    level.num_spawns = 1;
    level.spawn_x[0] = 0.0f;
    level.spawn_y[0] = -14.5f;  // Moved forward by 2.5 units from -17.0
    level.spawn_facing[0] = 0.0f;

    // Initialize target fields (no targets in default level)
    level.num_targets = 0;
    for (int i = 0; i < CompiledLevel::MAX_TARGETS; i++) {
        level.target_x[i] = 0.0f;
        level.target_y[i] = 0.0f;
        level.target_z[i] = 0.0f;
        level.target_motion_type[i] = 0;
    }
    for (int i = 0; i < CompiledLevel::MAX_TARGETS * 8; i++) {
        level.target_params[i] = 0.0f;
    }

    return level;
}

// Generate border walls for a level
int generateWalls(CompiledLevel& level, int start_tile_index) {
    int tile_index = start_tile_index;
    const float wall_tile_size = 2.5f;
    const int walls_per_side = 16;  // 16 wall tiles per side
    const float room_size = walls_per_side * wall_tile_size;  // 40 units
    const float half_room = room_size / 2.0f;  // 20.0f
    
    // Calculate wall edge position (walls should be at the edge of the room)
    const float wall_edge = half_room - wall_tile_size * 0.5f;  // 18.75
    
    // Top and bottom walls
    for (int i = 0; i < walls_per_side; i++) {
        float x = -wall_edge + i * wall_tile_size;  // Start from left edge and increment
        
        // Top wall
        level.object_ids[tile_index] = AssetIDs::WALL;
        level.tile_x[tile_index] = x;
        level.tile_y[tile_index] = wall_edge;
        level.tile_scale_x[tile_index] = wall_tile_size;
        level.tile_scale_y[tile_index] = wall_tile_size;
        level.tile_scale_z[tile_index] = 1.0f;
        level.tile_persistent[tile_index] = true;
        level.tile_render_only[tile_index] = false;
        level.tile_done_on_collide[tile_index] = false;  // Walls don't trigger episode end
        level.tile_entity_type[tile_index] = (int32_t)EntityType::Wall;
        level.tile_response_type[tile_index] = (int32_t)ResponseType::Static;
        tile_index++;
        
        // Bottom wall
        level.object_ids[tile_index] = AssetIDs::WALL;
        level.tile_x[tile_index] = x;
        level.tile_y[tile_index] = -wall_edge;
        level.tile_scale_x[tile_index] = wall_tile_size;
        level.tile_scale_y[tile_index] = wall_tile_size;
        level.tile_scale_z[tile_index] = 1.0f;
        level.tile_persistent[tile_index] = true;
        level.tile_render_only[tile_index] = false;
        level.tile_done_on_collide[tile_index] = false;  // Walls don't trigger episode end
        level.tile_entity_type[tile_index] = (int32_t)EntityType::Wall;
        level.tile_response_type[tile_index] = (int32_t)ResponseType::Static;
        tile_index++;
    }
    
    // Left and right walls (skip corners to avoid overlaps)
    for (int i = 1; i < walls_per_side - 1; i++) {
        float y = -wall_edge + i * wall_tile_size;
        
        // Left wall
        level.object_ids[tile_index] = AssetIDs::WALL;
        level.tile_x[tile_index] = -wall_edge;
        level.tile_y[tile_index] = y;
        level.tile_scale_x[tile_index] = wall_tile_size;
        level.tile_scale_y[tile_index] = wall_tile_size;
        level.tile_scale_z[tile_index] = 1.0f;
        level.tile_persistent[tile_index] = true;
        level.tile_render_only[tile_index] = false;
        level.tile_done_on_collide[tile_index] = false;  // Walls don't trigger episode end
        level.tile_entity_type[tile_index] = (int32_t)EntityType::Wall;
        level.tile_response_type[tile_index] = (int32_t)ResponseType::Static;
        tile_index++;
        
        // Right wall
        level.object_ids[tile_index] = AssetIDs::WALL;
        level.tile_x[tile_index] = wall_edge;
        level.tile_y[tile_index] = y;
        level.tile_scale_x[tile_index] = wall_tile_size;
        level.tile_scale_y[tile_index] = wall_tile_size;
        level.tile_scale_z[tile_index] = 1.0f;
        level.tile_persistent[tile_index] = true;
        level.tile_render_only[tile_index] = false;
        level.tile_done_on_collide[tile_index] = false;  // Walls don't trigger episode end
        level.tile_entity_type[tile_index] = (int32_t)EntityType::Wall;
        level.tile_response_type[tile_index] = (int32_t)ResponseType::Static;
        tile_index++;
    }
    
    return tile_index;
}

// Add axis marker for visual reference
int generateAxisMarker(CompiledLevel& level, int start_tile_index) {
    int tile_index = start_tile_index;
    
    level.object_ids[tile_index] = AssetIDs::AXIS_X;
    level.tile_x[tile_index] = 0.0f;
    level.tile_y[tile_index] = 12.5f;
    level.tile_persistent[tile_index] = true;
    level.tile_render_only[tile_index] = true;
    level.tile_done_on_collide[tile_index] = false;  // Render-only, no collision
    level.tile_entity_type[tile_index] = (int32_t)EntityType::NoEntity;
    level.tile_response_type[tile_index] = (int32_t)ResponseType::Static;
    tile_index++;
    
    return tile_index;
}

// Generate cylinder obstacles
int generateCylinders(CompiledLevel& level, int start_tile_index) {
    int tile_index = start_tile_index;
    const float cylinder_z_offset = 2.55f;  // Adjusted for 1.7x scale cylinders
    const float variance_3m = 3.0f;  // 3-meter variance for XY positions

    // Cylinder positions (8 cylinders)
    float cylinder_positions[][2] = {
        {-10.0f, 10.0f},   // Near top-left corner
        {8.0f, 12.0f},     // Near top-right corner
        {-12.0f, -3.0f},   // Left side
        {11.0f, 3.0f},     // Right side
        {3.0f, -2.0f},     // Near center but offset
        {-7.0f, -10.0f},   // Bottom-left area
        {9.0f, -8.0f},     // Bottom-right area
        {-5.0f, 4.0f}      // Mid-left
    };
    
    for (int i = 0; i < 8; i++) {
        level.object_ids[tile_index] = AssetIDs::CYLINDER;
        level.tile_x[tile_index] = cylinder_positions[i][0];
        level.tile_y[tile_index] = cylinder_positions[i][1];
        level.tile_z[tile_index] = cylinder_z_offset;
        level.tile_scale_x[tile_index] = 1.7f;  // 1.7x base size
        level.tile_scale_y[tile_index] = 1.7f;
        level.tile_scale_z[tile_index] = 1.7f;
        level.tile_persistent[tile_index] = false;
        level.tile_render_only[tile_index] = false;
        level.tile_done_on_collide[tile_index] = true;  // Obstacles trigger episode end
        level.tile_entity_type[tile_index] = (int32_t)EntityType::Cube;  // Static obstacles
        level.tile_response_type[tile_index] = (int32_t)ResponseType::Static;
        level.tile_rand_x[tile_index] = variance_3m;  // 3m variance in X
        level.tile_rand_y[tile_index] = variance_3m;  // 3m variance in Y
        level.tile_rand_scale_x[tile_index] = 1.5f;  // ±150% scale variation in X
        level.tile_rand_scale_y[tile_index] = 1.5f;  // ±150% scale variation in Y
        level.tile_rand_rot_z[tile_index] = 2.0f * consts::math::pi;  // Full 360° rotation randomization
        tile_index++;
    }
    
    return tile_index;
}

// Generate cube obstacles
int generateCubes(CompiledLevel& level, int start_tile_index) {
    int tile_index = start_tile_index;
    const float cube_z_offset = 0.75f;  // Half of scaled cube height (1.5 * 1.0 / 2)
    const float variance_3m = 3.0f;  // 3-meter variance for XY positions
    const float rotation_range = 2.0f * consts::math::pi;  // Full rotation range (360 degrees)
    
    // Cube positions (5 cubes)
    float cube_positions[][2] = {
        {-8.0f, 6.0f},     // Upper-left quadrant
        {6.0f, 8.0f},      // Upper-right quadrant
        {-10.0f, -6.0f},   // Lower-left quadrant
        {7.0f, -5.0f},     // Lower-right quadrant
        {-2.0f, 1.0f}      // Near center
    };
    
    for (int i = 0; i < 5; i++) {
        level.object_ids[tile_index] = AssetIDs::CUBE;
        level.tile_x[tile_index] = cube_positions[i][0];
        level.tile_y[tile_index] = cube_positions[i][1];
        level.tile_z[tile_index] = cube_z_offset;
        level.tile_scale_x[tile_index] = 1.5f;  // 50% larger
        level.tile_scale_y[tile_index] = 1.5f;
        level.tile_scale_z[tile_index] = 1.5f;
        level.tile_persistent[tile_index] = false;
        level.tile_render_only[tile_index] = false;
        level.tile_done_on_collide[tile_index] = true;  // Obstacles trigger episode end
        level.tile_entity_type[tile_index] = (int32_t)EntityType::Cube;
        level.tile_response_type[tile_index] = (int32_t)ResponseType::Static;
        level.tile_rand_x[tile_index] = variance_3m;
        level.tile_rand_y[tile_index] = variance_3m;
        level.tile_rand_rot_z[tile_index] = rotation_range;
        level.tile_rand_scale_x[tile_index] = 0.4f;  // ±40% scale variation
        level.tile_rand_scale_y[tile_index] = 0.4f;
        tile_index++;
    }
    
    return tile_index;
}

int main(int argc, char* argv[]) {
    const char* output_file = "default_level.lvl";
    if (argc > 1) {
        output_file = argv[1];
    }
    
    std::vector<CompiledLevel> levels;
    
    // Create first level: Full obstacles (cubes + cylinders)
    CompiledLevel level1 = createBaseLevelTemplate();
    std::strcpy(level1.level_name, "default_full_obstacles");
    
    int tile_index = 0;
    tile_index = generateWalls(level1, tile_index);
    tile_index = generateAxisMarker(level1, tile_index);
    tile_index = generateCylinders(level1, tile_index);
    tile_index = generateCubes(level1, tile_index);
    level1.num_tiles = tile_index;
    levels.push_back(level1);
    
    // Create second level: Cubes only (no cylinders)
    CompiledLevel level2 = createBaseLevelTemplate();
    std::strcpy(level2.level_name, "default_cubes_only");
    
    tile_index = 0;
    tile_index = generateWalls(level2, tile_index);
    tile_index = generateAxisMarker(level2, tile_index);
    tile_index = generateCubes(level2, tile_index);  // No cylinders
    level2.num_tiles = tile_index;
    levels.push_back(level2);
    
    // Write unified level format (with header) containing both levels
    Result result = writeCompiledLevels(output_file, levels);
    
    if (result != Result::Success) {
        std::cerr << "Failed to write level data to: " << output_file << std::endl;
        return 1;
    }
    
    std::cout << "Generated level file: " << output_file << " (unified format with " << levels.size() << " levels)" << std::endl;
    return 0;
}
