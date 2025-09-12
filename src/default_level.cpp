#include <iostream>
#include <fstream>
#include <cstring>
#include "types.hpp"
#include "asset_ids.hpp"
#include "level_io.hpp"

using namespace madEscape;

int main(int argc, char* argv[]) {
    const char* output_file = "default_level.lvl";
    if (argc > 1) {
        output_file = argv[1];
    }
    
    // Create a 16x16 room with border walls
    CompiledLevel level = {};
    level.width = 16;
    level.height = 16;
    level.world_scale = 1.0f;
    level.max_entities = 150;  // Enough for walls (16*4 = 64) and other objects
    std::strcpy(level.level_name, "default_16x16_room");
    
    // World boundaries using constants from consts.hpp
    // Use proper worldWidth and worldLength constants
    level.world_min_x = -consts::worldWidth / 2.0f;   // -10.0f
    level.world_max_x = consts::worldWidth / 2.0f;    // +10.0f
    level.world_min_y = -consts::worldLength / 2.0f;  // -20.0f
    level.world_max_y = consts::worldLength / 2.0f;   // +20.0f
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
    
    // Set spawn point at x=0, y=-17.0 (near southern wall)
    level.num_spawns = 1;
    level.spawn_x[0] = 0.0f;
    level.spawn_y[0] = -17.0f;
    level.spawn_facing[0] = 0.0f;
    
    // Generate border walls with 2.5 unit tile spacing
    // 16x16 means 16 wall tiles by 16 wall tiles
    // Each wall tile is 2.5 units, so room is 40x40 units total
    // The room spans from -20 to 20 in both X and Y
    int tile_index = 0;
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
    
    // Add an axis marker at x=0, y=12.5 for visual reference
    level.object_ids[tile_index] = AssetIDs::AXIS_X;
    level.tile_x[tile_index] = 0.0f;
    level.tile_y[tile_index] = 12.5f;
    level.tile_persistent[tile_index] = true;
    level.tile_render_only[tile_index] = true;
    level.tile_done_on_collide[tile_index] = false;  // Render-only, no collision
    level.tile_entity_type[tile_index] = (int32_t)EntityType::NoEntity;
    level.tile_response_type[tile_index] = (int32_t)ResponseType::Static;  // ResponseType::Static (render-only)
    tile_index++;
    
    // Add cylinders scattered around the level with 3m XY variance
    const float cylinder_z_offset = 2.55f;  // Adjusted for 1.7x scale cylinders
    const float variance_3m = 3.0f;  // 3-meter variance for XY positions
    
    // Cylinder 1: Near top-left corner
    level.object_ids[tile_index] = AssetIDs::CYLINDER;
    level.tile_x[tile_index] = -10.0f;
    level.tile_y[tile_index] = 10.0f;
    level.tile_z[tile_index] = cylinder_z_offset;
    level.tile_scale_x[tile_index] = 1.7f;  // 1.7x base size
    level.tile_scale_y[tile_index] = 1.7f;  // 1.7x base size
    level.tile_scale_z[tile_index] = 1.7f;  // 1.7x base size
    level.tile_persistent[tile_index] = false;
    level.tile_render_only[tile_index] = false;
    level.tile_done_on_collide[tile_index] = true;  // Obstacles trigger episode end
    level.tile_entity_type[tile_index] = (int32_t)EntityType::Cube;  // Static obstacles (static obstacle)
    level.tile_response_type[tile_index] = (int32_t)ResponseType::Static;
    level.tile_rand_x[tile_index] = variance_3m;  // 3m variance in X
    level.tile_rand_y[tile_index] = variance_3m;  // 3m variance in Y
    level.tile_rand_scale_x[tile_index] = 1.5f;  // ±150% scale variation in X
    level.tile_rand_scale_y[tile_index] = 1.5f;  // ±150% scale variation in Y
    level.tile_rand_rot_z[tile_index] = 2.0f * consts::math::pi;  // Full 360° rotation randomization
    tile_index++;
    
    // Cylinder 2: Near top-right corner
    level.object_ids[tile_index] = AssetIDs::CYLINDER;
    level.tile_x[tile_index] = 8.0f;
    level.tile_y[tile_index] = 12.0f;
    level.tile_z[tile_index] = cylinder_z_offset;
    level.tile_scale_x[tile_index] = 1.7f;  // 1.7x base size
    level.tile_scale_y[tile_index] = 1.7f;  // 1.7x base size
    level.tile_scale_z[tile_index] = 1.7f;  // 1.7x base size
    level.tile_persistent[tile_index] = false;
    level.tile_render_only[tile_index] = false;
    level.tile_done_on_collide[tile_index] = true;  // Obstacles trigger episode end
    level.tile_entity_type[tile_index] = (int32_t)EntityType::Cube;  // Static obstacles
    level.tile_response_type[tile_index] = (int32_t)ResponseType::Static;
    level.tile_rand_x[tile_index] = variance_3m;
    level.tile_rand_y[tile_index] = variance_3m;
    level.tile_rand_scale_x[tile_index] = 1.5f;  // ±150% scale variation in X
    level.tile_rand_scale_y[tile_index] = 1.5f;  // ±150% scale variation in Y
    level.tile_rand_rot_z[tile_index] = 2.0f * consts::math::pi;  // Full 360° rotation randomization
    tile_index++;
    
    // Cylinder 3: Left side
    level.object_ids[tile_index] = AssetIDs::CYLINDER;
    level.tile_x[tile_index] = -12.0f;
    level.tile_y[tile_index] = -3.0f;
    level.tile_z[tile_index] = cylinder_z_offset;
    level.tile_scale_x[tile_index] = 1.7f;  // 1.7x base size
    level.tile_scale_y[tile_index] = 1.7f;  // 1.7x base size
    level.tile_scale_z[tile_index] = 1.7f;  // 1.7x base size
    level.tile_persistent[tile_index] = false;
    level.tile_render_only[tile_index] = false;
    level.tile_done_on_collide[tile_index] = true;  // Obstacles trigger episode end
    level.tile_entity_type[tile_index] = (int32_t)EntityType::Cube;  // Static obstacles
    level.tile_response_type[tile_index] = (int32_t)ResponseType::Static;
    level.tile_rand_x[tile_index] = variance_3m;
    level.tile_rand_y[tile_index] = variance_3m;
    level.tile_rand_scale_x[tile_index] = 1.5f;  // ±150% scale variation in X
    level.tile_rand_scale_y[tile_index] = 1.5f;  // ±150% scale variation in Y
    level.tile_rand_rot_z[tile_index] = 2.0f * consts::math::pi;  // Full 360° rotation randomization
    tile_index++;
    
    // Cylinder 4: Right side
    level.object_ids[tile_index] = AssetIDs::CYLINDER;
    level.tile_x[tile_index] = 11.0f;
    level.tile_y[tile_index] = 3.0f;
    level.tile_z[tile_index] = cylinder_z_offset;
    level.tile_scale_x[tile_index] = 1.7f;  // 1.7x base size
    level.tile_scale_y[tile_index] = 1.7f;  // 1.7x base size
    level.tile_scale_z[tile_index] = 1.7f;  // 1.7x base size
    level.tile_persistent[tile_index] = false;
    level.tile_render_only[tile_index] = false;
    level.tile_done_on_collide[tile_index] = true;  // Obstacles trigger episode end
    level.tile_entity_type[tile_index] = (int32_t)EntityType::Cube;  // Static obstacles
    level.tile_response_type[tile_index] = (int32_t)ResponseType::Static;
    level.tile_rand_x[tile_index] = variance_3m;
    level.tile_rand_y[tile_index] = variance_3m;
    level.tile_rand_scale_x[tile_index] = 1.5f;  // ±150% scale variation in X
    level.tile_rand_scale_y[tile_index] = 1.5f;  // ±150% scale variation in Y
    level.tile_rand_rot_z[tile_index] = 2.0f * consts::math::pi;  // Full 360° rotation randomization
    tile_index++;
    
    // Cylinder 5: Near center but offset
    level.object_ids[tile_index] = AssetIDs::CYLINDER;
    level.tile_x[tile_index] = 3.0f;
    level.tile_y[tile_index] = -2.0f;
    level.tile_z[tile_index] = cylinder_z_offset;
    level.tile_scale_x[tile_index] = 1.7f;  // 1.7x base size
    level.tile_scale_y[tile_index] = 1.7f;  // 1.7x base size
    level.tile_scale_z[tile_index] = 1.7f;  // 1.7x base size
    level.tile_persistent[tile_index] = false;
    level.tile_render_only[tile_index] = false;
    level.tile_done_on_collide[tile_index] = true;  // Obstacles trigger episode end
    level.tile_entity_type[tile_index] = (int32_t)EntityType::Cube;  // Static obstacles
    level.tile_response_type[tile_index] = (int32_t)ResponseType::Static;
    level.tile_rand_x[tile_index] = variance_3m;
    level.tile_rand_y[tile_index] = variance_3m;
    level.tile_rand_scale_x[tile_index] = 1.5f;  // ±150% scale variation in X
    level.tile_rand_scale_y[tile_index] = 1.5f;  // ±150% scale variation in Y
    level.tile_rand_rot_z[tile_index] = 2.0f * consts::math::pi;  // Full 360° rotation randomization
    tile_index++;
    
    // Cylinder 6: Bottom-left area
    level.object_ids[tile_index] = AssetIDs::CYLINDER;
    level.tile_x[tile_index] = -7.0f;
    level.tile_y[tile_index] = -10.0f;
    level.tile_z[tile_index] = cylinder_z_offset;
    level.tile_scale_x[tile_index] = 1.7f;  // 1.7x base size
    level.tile_scale_y[tile_index] = 1.7f;  // 1.7x base size
    level.tile_scale_z[tile_index] = 1.7f;  // 1.7x base size
    level.tile_persistent[tile_index] = false;
    level.tile_render_only[tile_index] = false;
    level.tile_done_on_collide[tile_index] = true;  // Obstacles trigger episode end
    level.tile_entity_type[tile_index] = (int32_t)EntityType::Cube;  // Static obstacles
    level.tile_response_type[tile_index] = (int32_t)ResponseType::Static;
    level.tile_rand_x[tile_index] = variance_3m;
    level.tile_rand_y[tile_index] = variance_3m;
    level.tile_rand_scale_x[tile_index] = 1.5f;  // ±150% scale variation in X
    level.tile_rand_scale_y[tile_index] = 1.5f;  // ±150% scale variation in Y
    level.tile_rand_rot_z[tile_index] = 2.0f * consts::math::pi;  // Full 360° rotation randomization
    tile_index++;
    
    // Cylinder 7: Bottom-right area
    level.object_ids[tile_index] = AssetIDs::CYLINDER;
    level.tile_x[tile_index] = 9.0f;
    level.tile_y[tile_index] = -8.0f;
    level.tile_z[tile_index] = cylinder_z_offset;
    level.tile_scale_x[tile_index] = 1.7f;  // 1.7x base size
    level.tile_scale_y[tile_index] = 1.7f;  // 1.7x base size
    level.tile_scale_z[tile_index] = 1.7f;  // 1.7x base size
    level.tile_persistent[tile_index] = false;
    level.tile_render_only[tile_index] = false;
    level.tile_done_on_collide[tile_index] = true;  // Obstacles trigger episode end
    level.tile_entity_type[tile_index] = (int32_t)EntityType::Cube;  // Static obstacles
    level.tile_response_type[tile_index] = (int32_t)ResponseType::Static;
    level.tile_rand_x[tile_index] = variance_3m;
    level.tile_rand_y[tile_index] = variance_3m;
    level.tile_rand_scale_x[tile_index] = 1.5f;  // ±150% scale variation in X
    level.tile_rand_scale_y[tile_index] = 1.5f;  // ±150% scale variation in Y
    level.tile_rand_rot_z[tile_index] = 2.0f * consts::math::pi;  // Full 360° rotation randomization
    tile_index++;
    
    // Cylinder 8: Mid-left
    level.object_ids[tile_index] = AssetIDs::CYLINDER;
    level.tile_x[tile_index] = -5.0f;
    level.tile_y[tile_index] = 4.0f;
    level.tile_z[tile_index] = cylinder_z_offset;
    level.tile_scale_x[tile_index] = 1.7f;  // 1.7x base size
    level.tile_scale_y[tile_index] = 1.7f;  // 1.7x base size
    level.tile_scale_z[tile_index] = 1.7f;  // 1.7x base size
    level.tile_persistent[tile_index] = false;
    level.tile_render_only[tile_index] = false;
    level.tile_done_on_collide[tile_index] = true;  // Obstacles trigger episode end
    level.tile_entity_type[tile_index] = (int32_t)EntityType::Cube;  // Static obstacles
    level.tile_response_type[tile_index] = (int32_t)ResponseType::Static;
    level.tile_rand_x[tile_index] = variance_3m;
    level.tile_rand_y[tile_index] = variance_3m;
    level.tile_rand_scale_x[tile_index] = 1.5f;  // ±150% scale variation in X
    level.tile_rand_scale_y[tile_index] = 1.5f;  // ±150% scale variation in Y
    level.tile_rand_rot_z[tile_index] = 2.0f * consts::math::pi;  // Full 360° rotation randomization
    tile_index++;
    
    // Add cubes with physics, XY variance, and random rotation
    // Cube base size is 1.0, scaled by 1.5, so half-height is 0.75
    const float cube_z_offset = 0.75f;  // Half of scaled cube height (1.5 * 1.0 / 2)
    const float rotation_range = 2.0f * consts::math::pi;  // Full rotation range (360 degrees)
    
    // Cube 1: Upper-left quadrant
    level.object_ids[tile_index] = AssetIDs::CUBE;
    level.tile_x[tile_index] = -8.0f;
    level.tile_y[tile_index] = 6.0f;
    level.tile_z[tile_index] = cube_z_offset;
    level.tile_scale_x[tile_index] = 1.5f;  // 50% larger
    level.tile_scale_y[tile_index] = 1.5f;
    level.tile_scale_z[tile_index] = 1.5f;
    level.tile_persistent[tile_index] = false;
    level.tile_render_only[tile_index] = false;  // Has physics
    level.tile_done_on_collide[tile_index] = true;  // Obstacles trigger episode end
    level.tile_entity_type[tile_index] = (int32_t)EntityType::Cube;
    level.tile_response_type[tile_index] = (int32_t)ResponseType::Static;  // ResponseType::Static (immovable)
    level.tile_rand_x[tile_index] = variance_3m;
    level.tile_rand_y[tile_index] = variance_3m;
    level.tile_rand_rot_z[tile_index] = rotation_range;  // Random Z-axis rotation
    level.tile_rand_scale_x[tile_index] = 0.4f;  // ±40% scale variation
    level.tile_rand_scale_y[tile_index] = 0.4f;
    // No Z randomization to keep cubes at consistent height
    tile_index++;
    
    // Cube 2: Upper-right quadrant
    level.object_ids[tile_index] = AssetIDs::CUBE;
    level.tile_x[tile_index] = 6.0f;
    level.tile_y[tile_index] = 8.0f;
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
    // No Z randomization to keep cubes at consistent height
    tile_index++;
    
    // Cube 3: Lower-left quadrant
    level.object_ids[tile_index] = AssetIDs::CUBE;
    level.tile_x[tile_index] = -10.0f;
    level.tile_y[tile_index] = -6.0f;
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
    // No Z randomization to keep cubes at consistent height
    tile_index++;
    
    // Cube 4: Lower-right quadrant
    level.object_ids[tile_index] = AssetIDs::CUBE;
    level.tile_x[tile_index] = 7.0f;
    level.tile_y[tile_index] = -5.0f;
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
    // No Z randomization to keep cubes at consistent height
    tile_index++;
    
    // Cube 5: Near center
    level.object_ids[tile_index] = AssetIDs::CUBE;
    level.tile_x[tile_index] = -2.0f;
    level.tile_y[tile_index] = 1.0f;
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
    // No Z randomization to keep cubes at consistent height
    tile_index++;
    
    // Set the actual number of tiles used
    level.num_tiles = tile_index;
    
    // Write to file using unified format
    std::vector<CompiledLevel> levels = {level};
    Result result = writeCompiledLevels(output_file, levels);
    
    if (result != Result::Success) {
        std::cerr << "Failed to write level file: " << output_file << " (error: " << static_cast<int>(result) << ")" << std::endl;
        return 1;
    }
    
    std::cout << "Generated level file: " << output_file << " (unified format with 1 level)" << std::endl;
    return 0;
}