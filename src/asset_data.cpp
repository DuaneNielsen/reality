#include "asset_registry.hpp"
#include <filesystem>

// Need DATA_DIR for asset paths
#ifndef DATA_DIR
static_assert(false, "DATA_DIR must be defined");
#endif

using namespace madrona;
using namespace madrona::imp;

namespace madEscape {

// Texture definitions
namespace AssetTextures {
    const TextureDescriptor TEXTURES[NUM_TEXTURES] = {
        { "tron_grid", "tron_grid.png" },
        { "smile", "smile.png" },
    };
}

// Material definitions - 80s Tron color palette
namespace AssetMaterials {
    const MaterialDescriptor MATERIALS[NUM_MATERIALS] = {
        { "cube", {1.0f, 102.f/255.f, 0.0f, 1.0f}, -1, 0.8f, 0.2f },  // Tron orange
        { "wall", {0.0f, 51.f/255.f, 102.f/255.f, 1.0f}, -1, 0.8f, 0.2f },  // Grid blue
        { "agent_body", {0.0f, 1.0f, 1.0f, 1.0f}, 1, 0.5f, 1.0f },  // Cyan, uses smile texture  
        { "agent_parts", {1.0f, 0.0f, 1.0f, 1.0f}, -1, 0.8f, 0.2f },  // Magenta
        { "floor", {0.3f, 0.3f, 0.3f, 1.0f}, 0, 0.8f, 0.2f },  // Dark gray to darken texture, uses tron_grid texture
        { "axis_x", {1.0f, 0.0f, 0.0f, 1.0f}, -1, 0.8f, 0.2f },  // Red
        { "button", {1.0f, 1.0f, 0.0f, 1.0f}, -1, 0.8f, 0.2f },  // Yellow
        { "axis_y", {0.0f, 1.0f, 0.0f, 1.0f}, -1, 0.8f, 0.2f },  // Green
        { "axis_z", {0.0f, 0.0f, 1.0f, 1.0f}, -1, 0.8f, 0.2f },  // Blue
        { "cylinder", {0.0f, 1.0f, 1.0f, 1.0f}, -1, 0.8f, 0.2f },  // Cyan
        { "lidar_ray", {0.0f, 1.0f, 0.0f, 0.5f}, -1, 0.8f, 0.2f },  // Semi-transparent green
        { "target", {1.0f, 0.0f, 0.0f, 1.0f}, -1, 0.8f, 0.2f },  // Red target
        { "compass_indicator", {0.3f, 0.7f, 1.0f, 0.8f}, -1, 0.8f, 0.2f },  // Light blue, semi-transparent
    };
}

namespace Assets {

// Static material index arrays for assets with multiple materials
static constexpr uint32_t CUBE_MATERIALS[] = { 0 };
static constexpr uint32_t WALL_MATERIALS[] = { 1 };
static constexpr uint32_t AGENT_MATERIALS[] = { 2, 3, 3 };  // body, parts, parts
static constexpr uint32_t PLANE_MATERIALS[] = { 4 };
static constexpr uint32_t AXIS_X_MATERIALS[] = { 5 };
static constexpr uint32_t AXIS_Y_MATERIALS[] = { 7 };
static constexpr uint32_t AXIS_Z_MATERIALS[] = { 8 };
static constexpr uint32_t CYLINDER_MATERIALS[] = { 9 };
static constexpr uint32_t LIDAR_RAY_MATERIALS[] = { 10 };
static constexpr uint32_t TARGET_MATERIALS[] = { 11 };
static constexpr uint32_t COMPASS_INDICATOR_MATERIALS[] = { 12 };

// Physics property constants
namespace PhysicsProps {
    constexpr float cubeInverseMass = 0.075f;     // Pushable cube with ~13kg mass
    constexpr float wallInverseMass = 0.f;        // Static walls (infinite mass)
    constexpr float agentInverseMass = 1.f;       // Agent unit mass for direct control
    constexpr float planeInverseMass = 0.f;       // Static plane (infinite mass)
    constexpr float cylinderInverseMass = 0.f;    // Static cylinder (infinite mass)
    
    constexpr float standardStaticFriction = 0.5f;
    constexpr float standardDynamicFriction = 0.5f;
    constexpr float cubeStaticFriction = 0.5f;
    constexpr float cubeDynamicFriction = 0.75f;
}

// The main asset table - sparse array indexed by AssetID
// Uninitialized entries will be zero-initialized (nullptr for pointers, 0 for numbers, false for bools)
const AssetInfo ASSET_TABLE[AssetIDs::MAX_ASSETS] = {
    // [0] = INVALID - leave as zero-initialized
    {},
    
    // [1] = CUBE
    {
        .name = "cube",
        .id = AssetIDs::CUBE,
        .hasPhysics = true,
        .hasRender = true,
        .assetType = AssetInfo::FILE_MESH,
        .filepath = "cube_collision.obj",
        .inverseMass = PhysicsProps::cubeInverseMass,
        .friction = { PhysicsProps::cubeStaticFriction, PhysicsProps::cubeDynamicFriction },
        .constrainRotationXY = false,
        .meshPath = "cube_render.obj",
        .materialIndices = CUBE_MATERIALS,
        .numMaterialIndices = 1,
        .numMeshes = 1,
    },
    
    // [2] = WALL
    {
        .name = "wall",
        .id = AssetIDs::WALL,
        .hasPhysics = true,
        .hasRender = true,
        .assetType = AssetInfo::FILE_MESH,
        .filepath = "wall_collision.obj",
        .inverseMass = PhysicsProps::wallInverseMass,
        .friction = { PhysicsProps::standardStaticFriction, PhysicsProps::standardDynamicFriction },
        .constrainRotationXY = false,
        .meshPath = "wall_render.obj",
        .materialIndices = WALL_MATERIALS,
        .numMaterialIndices = 1,
        .numMeshes = 1,
    },
    
    // [3] = AGENT
    {
        .name = "agent",
        .id = AssetIDs::AGENT,
        .hasPhysics = true,
        .hasRender = true,
        .assetType = AssetInfo::FILE_MESH,
        .filepath = "agent_collision_simplified.obj",
        .inverseMass = PhysicsProps::agentInverseMass,
        .friction = { PhysicsProps::standardStaticFriction, PhysicsProps::standardDynamicFriction },
        .constrainRotationXY = true,  // Prevent tipping over
        .meshPath = "agent_render.obj",
        .materialIndices = AGENT_MATERIALS,
        .numMaterialIndices = 3,
        .numMeshes = 3,
    },
    
    // [4] = PLANE
    {
        .name = "plane",
        .id = AssetIDs::PLANE,
        .hasPhysics = true,
        .hasRender = true,
        .assetType = AssetInfo::BUILTIN_PLANE,
        .filepath = nullptr,  // No file for built-in
        .inverseMass = PhysicsProps::planeInverseMass,
        .friction = { PhysicsProps::standardStaticFriction, PhysicsProps::standardDynamicFriction },
        .constrainRotationXY = false,
        .meshPath = "plane_render.obj",
        .materialIndices = PLANE_MATERIALS,
        .numMaterialIndices = 1,
        .numMeshes = 1,
    },
    
    // [5] = AXIS_X
    {
        .name = "axis_x",
        .id = AssetIDs::AXIS_X,
        .hasPhysics = false,
        .hasRender = true,
        .assetType = AssetInfo::FILE_MESH,
        .filepath = nullptr,
        .inverseMass = 0.f,
        .friction = { 0.f, 0.f },
        .constrainRotationXY = false,
        .meshPath = "cube_render.obj",
        .materialIndices = AXIS_X_MATERIALS,
        .numMaterialIndices = 1,
        .numMeshes = 1,
    },
    
    // [6] = AXIS_Y
    {
        .name = "axis_y",
        .id = AssetIDs::AXIS_Y,
        .hasPhysics = false,
        .hasRender = true,
        .assetType = AssetInfo::FILE_MESH,
        .filepath = nullptr,
        .inverseMass = 0.f,
        .friction = { 0.f, 0.f },
        .constrainRotationXY = false,
        .meshPath = "cube_render.obj",
        .materialIndices = AXIS_Y_MATERIALS,
        .numMaterialIndices = 1,
        .numMeshes = 1,
    },
    
    // [7] = AXIS_Z
    {
        .name = "axis_z",
        .id = AssetIDs::AXIS_Z,
        .hasPhysics = false,
        .hasRender = true,
        .assetType = AssetInfo::FILE_MESH,
        .filepath = nullptr,
        .inverseMass = 0.f,
        .friction = { 0.f, 0.f },
        .constrainRotationXY = false,
        .meshPath = "cube_render.obj",
        .materialIndices = AXIS_Z_MATERIALS,
        .numMaterialIndices = 1,
        .numMeshes = 1,
    },
    
    // [8] = CYLINDER
    {
        .name = "cylinder",
        .id = AssetIDs::CYLINDER,
        .hasPhysics = true,
        .hasRender = true,
        .assetType = AssetInfo::FILE_MESH,
        .filepath = "cylinder_collision.obj",
        .inverseMass = PhysicsProps::cylinderInverseMass,
        .friction = { PhysicsProps::standardStaticFriction, PhysicsProps::standardDynamicFriction },
        .constrainRotationXY = false,
        .meshPath = "cylinder_render.obj",
        .materialIndices = CYLINDER_MATERIALS,
        .numMaterialIndices = 1,
        .numMeshes = 1,
    },
    
    // [9] = LIDAR_RAY (thin cylinder for visualization)
    {
        .name = "lidar_ray",
        .id = AssetIDs::LIDAR_RAY,
        .hasPhysics = false,
        .hasRender = true,
        .assetType = AssetInfo::FILE_MESH,
        .filepath = nullptr,
        .inverseMass = 0.f,
        .friction = { 0.f, 0.f },
        .constrainRotationXY = false,
        .meshPath = "cylinder_render.obj",  // Reuse cylinder mesh, will scale it thin
        .materialIndices = LIDAR_RAY_MATERIALS,
        .numMaterialIndices = 1,
        .numMeshes = 1,
    },

    // [10] = TARGET (small red sphere for compass tracking)
    {
        .name = "target",
        .id = AssetIDs::TARGET,
        .hasPhysics = false,  // NOT registered with physics - uses custom motion
        .hasRender = true,
        .assetType = AssetInfo::FILE_MESH,
        .filepath = nullptr,  // No physics mesh needed
        .inverseMass = 0.f,   // Not used since hasPhysics = false
        .friction = { 0.f, 0.f },
        .constrainRotationXY = false,
        .meshPath = "cube_render.obj",  // Reuse cube mesh, will scale it small and spherical
        .materialIndices = TARGET_MATERIALS,
        .numMaterialIndices = 1,
        .numMeshes = 1,
    },

    // [11] = COMPASS_INDICATOR (light blue cylinder pointing to target)
    {
        .name = "compass_indicator",
        .id = AssetIDs::COMPASS_INDICATOR,
        .hasPhysics = false,
        .hasRender = true,
        .assetType = AssetInfo::FILE_MESH,
        .filepath = nullptr,
        .inverseMass = 0.f,
        .friction = { 0.f, 0.f },
        .constrainRotationXY = false,
        .meshPath = "cylinder_render.obj",  // Reuse cylinder mesh, will scale appropriately
        .materialIndices = COMPASS_INDICATOR_MATERIALS,
        .numMaterialIndices = 1,
        .numMeshes = 1,
    },

    // Rest of the array will be zero-initialized
};

Span<SourceTexture> loadTextures(StackAlloc& alloc)
{
    ImageImporter img_importer;
    
    // Load textures using the static texture descriptors
    return img_importer.importImages(
        alloc, {
            (std::filesystem::path(DATA_DIR) / 
                AssetTextures::TEXTURES[0].filepath).string().c_str(),  // Tron grid texture
            (std::filesystem::path(DATA_DIR) / 
                AssetTextures::TEXTURES[1].filepath).string().c_str(),  // Smile texture
        });
}

std::array<SourceMaterial, AssetMaterials::NUM_MATERIALS> createMaterials()
{
    std::array<SourceMaterial, AssetMaterials::NUM_MATERIALS> materials;
    
    for (size_t i = 0; i < AssetMaterials::NUM_MATERIALS; i++) {
        const auto& desc = AssetMaterials::MATERIALS[i];
        materials[i] = SourceMaterial{
            desc.color,
            desc.textureIdx,
            desc.roughness,
            desc.metallic,
        };
    }
    
    return materials;
}

} // namespace Assets

} // namespace madEscape