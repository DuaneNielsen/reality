#include "asset_registry.hpp"
#include "asset_ids.hpp"
#include <cassert>
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
        { "green_grid", "green_grid.png" },
        { "smile", "smile.png" },
    };
}

// Material definitions
namespace AssetMaterials {
    const MaterialDescriptor MATERIALS[NUM_MATERIALS] = {
        { "cube", {139.f/255.f, 69.f/255.f, 19.f/255.f, 1.0f}, -1, 0.8f, 0.2f },
        { "wall", {0.5f, 0.5f, 0.5f, 1.0f}, -1, 0.8f, 0.2f },  // No texture
        { "agent_body", {1.0f, 1.0f, 1.0f, 1.0f}, 1, 0.5f, 1.0f },  // Uses smile texture  
        { "agent_parts", {0.7f, 0.7f, 0.7f, 1.0f}, -1, 0.8f, 0.2f },
        { "floor", {0.2f, 0.6f, 0.2f, 1.0f}, 0, 0.8f, 0.2f },  // Uses green_grid texture
        { "axis_x", {1.0f, 0.0f, 0.0f, 1.0f}, -1, 0.8f, 0.2f },
        { "button", {1.0f, 1.0f, 0.0f, 1.0f}, -1, 0.8f, 0.2f },
        { "axis_y", {0.0f, 1.0f, 0.0f, 1.0f}, -1, 0.8f, 0.2f },
        { "axis_z", {0.0f, 0.0f, 1.0f, 1.0f}, -1, 0.8f, 0.2f },
        { "cylinder", {0.3f, 0.7f, 0.9f, 1.0f}, -1, 0.8f, 0.2f },  // Light blue cylinder
    };
}

// Asset constants for physics and rendering properties
namespace {
    // Physics properties
    constexpr float cubeInverseMass = 0.075f;     // Pushable cube with ~13kg mass
    constexpr float wallInverseMass = 0.f;        // Static walls (infinite mass)
    constexpr float agentInverseMass = 1.f;       // Agent unit mass for direct control
    constexpr float planeInverseMass = 0.f;       // Static plane (infinite mass)
    constexpr float cylinderInverseMass = 0.1f;    // Pushable cylinder with ~10kg mass
    
    constexpr float standardStaticFriction = 0.5f;     // Standard static friction
    constexpr float standardDynamicFriction = 0.5f;    // Standard dynamic friction
    constexpr float cubeStaticFriction = 0.5f;         // Cube static friction
    constexpr float cubeDynamicFriction = 0.75f;       // Cube dynamic friction
    
    // Material indices
    constexpr uint32_t materialCube = 0;
    constexpr uint32_t materialWall = 1;
    constexpr uint32_t materialAgentBody = 2;
    constexpr uint32_t materialAgentParts = 3;
    constexpr uint32_t materialFloor = 4;
    constexpr uint32_t materialAxisX = 5;
    constexpr uint32_t materialAxisY = 7;
    constexpr uint32_t materialAxisZ = 8;
    constexpr uint32_t materialCylinder = 9;
    
    // Mesh counts
    constexpr uint32_t agentMeshCount = 3;
}

AssetRegistry::AssetRegistry() : nextId(AssetIDs::DYNAMIC_START) {
    idToAsset.resize(AssetIDs::MAX_ASSETS);
    
    // Register cube asset
    {
        AssetInfo cube;
        cube.name = "cube";
        cube.id = AssetIDs::CUBE;
        cube.hasPhysics = true;
        cube.hasRender = true;
        cube.assetType = FILE_MESH;
        cube.filepath = "cube_collision.obj";
        cube.inverseMass = cubeInverseMass;
        cube.friction = { cubeStaticFriction, cubeDynamicFriction };
        cube.constrainRotationXY = false;
        cube.meshPath = "cube_render.obj";
        cube.materialIndices = { materialCube };
        cube.numMeshes = 1;
        registerFullAsset(cube);
    }
    
    // Register wall asset
    {
        AssetInfo wall;
        wall.name = "wall";
        wall.id = AssetIDs::WALL;
        wall.hasPhysics = true;
        wall.hasRender = true;
        wall.assetType = FILE_MESH;
        wall.filepath = "wall_collision.obj";
        wall.inverseMass = wallInverseMass;
        wall.friction = { standardStaticFriction, standardDynamicFriction };
        wall.constrainRotationXY = false;
        wall.meshPath = "wall_render.obj";
        wall.materialIndices = { materialWall };
        wall.numMeshes = 1;
        registerFullAsset(wall);
    }
    
    // Register agent asset
    {
        AssetInfo agent;
        agent.name = "agent";
        agent.id = AssetIDs::AGENT;
        agent.hasPhysics = true;
        agent.hasRender = true;
        agent.assetType = FILE_MESH;
        agent.filepath = "agent_collision_simplified.obj";
        agent.inverseMass = agentInverseMass;
        agent.friction = { standardStaticFriction, standardDynamicFriction };
        agent.constrainRotationXY = true;  // Prevent tipping over
        agent.meshPath = "agent_render.obj";
        agent.materialIndices = { materialAgentBody, materialAgentParts, materialAgentParts };
        agent.numMeshes = agentMeshCount;
        registerFullAsset(agent);
    }
    
    // Register plane asset
    {
        AssetInfo plane;
        plane.name = "plane";
        plane.id = AssetIDs::PLANE;
        plane.hasPhysics = true;
        plane.hasRender = true;
        plane.assetType = BUILTIN_PLANE;
        plane.filepath = "";  // No file for built-in
        plane.inverseMass = planeInverseMass;
        plane.friction = { standardStaticFriction, standardDynamicFriction };
        plane.constrainRotationXY = false;
        plane.meshPath = "plane_render.obj";
        plane.materialIndices = { materialFloor };
        plane.numMeshes = 1;
        registerFullAsset(plane);
    }
    
    // Register axis visualization assets (render only)
    {
        AssetInfo axis_x;
        axis_x.name = "axis_x";
        axis_x.id = AssetIDs::AXIS_X;
        axis_x.hasPhysics = false;
        axis_x.hasRender = true;
        axis_x.assetType = FILE_MESH;
        axis_x.meshPath = "cube_render.obj";
        axis_x.materialIndices = { materialAxisX };
        axis_x.numMeshes = 1;
        registerFullAsset(axis_x);
    }
    
    {
        AssetInfo axis_y;
        axis_y.name = "axis_y";
        axis_y.id = AssetIDs::AXIS_Y;
        axis_y.hasPhysics = false;
        axis_y.hasRender = true;
        axis_y.assetType = FILE_MESH;
        axis_y.meshPath = "cube_render.obj";
        axis_y.materialIndices = { materialAxisY };
        axis_y.numMeshes = 1;
        registerFullAsset(axis_y);
    }
    
    {
        AssetInfo axis_z;
        axis_z.name = "axis_z";
        axis_z.id = AssetIDs::AXIS_Z;
        axis_z.hasPhysics = false;
        axis_z.hasRender = true;
        axis_z.assetType = FILE_MESH;
        axis_z.meshPath = "cube_render.obj";
        axis_z.materialIndices = { materialAxisZ };
        axis_z.numMeshes = 1;
        registerFullAsset(axis_z);
    }
    
    // Register cylinder asset
    {
        AssetInfo cylinder;
        cylinder.name = "cylinder";
        cylinder.id = AssetIDs::CYLINDER;
        cylinder.hasPhysics = true;
        cylinder.hasRender = true;
        cylinder.assetType = FILE_MESH;
        cylinder.filepath = "cylinder_collision.obj";
        cylinder.inverseMass = cylinderInverseMass;
        cylinder.friction = { standardStaticFriction, standardDynamicFriction };
        cylinder.constrainRotationXY = false;
        cylinder.meshPath = "cylinder_render.obj";
        cylinder.materialIndices = { materialCylinder };
        cylinder.numMeshes = 1;
        registerFullAsset(cylinder);
    }
}

uint32_t AssetRegistry::registerFullAsset(const AssetInfo& info) {
    if (info.name.empty()) {
        return AssetIDs::INVALID;
    }
    
    auto it = nameToAsset.find(info.name);
    if (it != nameToAsset.end()) {
        return it->second.id;
    }
    
    if (info.id >= AssetIDs::MAX_ASSETS) {
        return AssetIDs::INVALID;
    }
    
    nameToAsset[info.name] = info;
    idToAsset[info.id] = info;
    
    return info.id;
}

uint32_t AssetRegistry::registerAsset(const std::string& name, 
                                     bool hasPhysics, bool hasRender) {
    if (name.empty()) {
        return AssetIDs::INVALID;
    }
    
    auto it = nameToAsset.find(name);
    if (it != nameToAsset.end()) {
        return it->second.id;
    }
    
    if (nextId >= AssetIDs::MAX_ASSETS) {
        return AssetIDs::INVALID;
    }
    
    uint32_t id = nextId++;
    return registerAssetWithId(name, id, hasPhysics, hasRender);
}

uint32_t AssetRegistry::registerAssetWithId(const std::string& name, uint32_t id,
                                           bool hasPhysics, bool hasRender) {
    if (name.empty()) {
        return AssetIDs::INVALID;
    }
    
    if (id >= AssetIDs::MAX_ASSETS) {
        return AssetIDs::INVALID;
    }
    
    auto it = nameToAsset.find(name);
    if (it != nameToAsset.end()) {
        return it->second.id;
    }
    
    AssetInfo info;
    info.name = name;
    info.id = id;
    info.hasPhysics = hasPhysics;
    info.hasRender = hasRender;
    // Set default values for other fields
    info.assetType = FILE_MESH;
    info.filepath = "";
    info.inverseMass = 0.f;
    info.friction = {0.f, 0.f};
    info.constrainRotationXY = false;
    info.meshPath = "";
    info.materialIndices = {};
    info.numMeshes = 0;
    
    nameToAsset[name] = info;
    idToAsset[id] = info;
    
    return id;
}

uint32_t AssetRegistry::getAssetId(const std::string& name) const {
    uint32_t id;
    if (tryGetAssetId(name, id)) {
        return id;
    }
    assert(false && "Asset not found");
    return AssetIDs::INVALID;
}

bool AssetRegistry::tryGetAssetId(const std::string& name, uint32_t& outId) const {
    auto it = nameToAsset.find(name);
    if (it == nameToAsset.end()) {
        return false;
    }
    outId = it->second.id;
    return true;
}

bool AssetRegistry::assetHasPhysics(uint32_t id) const {
    if (id >= AssetIDs::MAX_ASSETS || idToAsset[id].name.empty()) {
        return false;
    }
    return idToAsset[id].hasPhysics;
}

bool AssetRegistry::assetHasRender(uint32_t id) const {
    if (id >= AssetIDs::MAX_ASSETS || idToAsset[id].name.empty()) {
        return false;
    }
    return idToAsset[id].hasRender;
}

const std::string& AssetRegistry::getAssetName(uint32_t id) const {
    static const std::string empty;
    if (id >= AssetIDs::MAX_ASSETS || idToAsset[id].name.empty()) {
        return empty;
    }
    return idToAsset[id].name;
}

const AssetRegistry::AssetInfo* AssetRegistry::getAssetInfo(uint32_t id) const {
    if (id >= AssetIDs::MAX_ASSETS || idToAsset[id].name.empty()) {
        return nullptr;
    }
    return &idToAsset[id];
}

const AssetRegistry::AssetInfo* AssetRegistry::getAssetInfoByName(const std::string& name) const {
    auto it = nameToAsset.find(name);
    if (it == nameToAsset.end()) {
        return nullptr;
    }
    return &it->second;
}

std::vector<AssetRegistry::AssetInfo> AssetRegistry::getAllAssets() const {
    std::vector<AssetInfo> result;
    for (const auto& [name, info] : nameToAsset) {
        result.push_back(info);
    }
    return result;
}

std::vector<AssetRegistry::AssetInfo> AssetRegistry::getPhysicsAssets() const {
    std::vector<AssetInfo> result;
    for (const auto& [name, info] : nameToAsset) {
        if (info.hasPhysics) {
            result.push_back(info);
        }
    }
    return result;
}

std::vector<AssetRegistry::AssetInfo> AssetRegistry::getRenderAssets() const {
    std::vector<AssetInfo> result;
    for (const auto& [name, info] : nameToAsset) {
        if (info.hasRender) {
            result.push_back(info);
        }
    }
    return result;
}

uint32_t AssetRegistry::getPhysicsAssetCount() const {
    uint32_t count = 0;
    for (const auto& [name, info] : nameToAsset) {
        if (info.hasPhysics) {
            count++;
        }
    }
    return count;
}

uint32_t AssetRegistry::getRenderAssetCount() const {
    uint32_t count = 0;
    for (const auto& [name, info] : nameToAsset) {
        if (info.hasRender) {
            count++;
        }
    }
    return count;
}

AssetRegistry& AssetRegistry::getInstance() {
    static AssetRegistry instance;
    return instance;
}

Span<SourceTexture> AssetRegistry::loadTextures(StackAlloc& alloc)
{
    ImageImporter img_importer;
    
    // Load textures using the static texture descriptors
    return img_importer.importImages(
        alloc, {
            (std::filesystem::path(DATA_DIR) / 
                AssetTextures::TEXTURES[0].filepath).string().c_str(),  // Green grid texture
            (std::filesystem::path(DATA_DIR) / 
                AssetTextures::TEXTURES[1].filepath).string().c_str(),  // Smile texture
        });
}

std::array<SourceMaterial, AssetMaterials::NUM_MATERIALS> AssetRegistry::createMaterials()
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

std::vector<std::string> AssetRegistry::getRenderAssetPaths() const
{
    std::vector<std::string> paths;
    for (const auto& [name, info] : nameToAsset) {
        if (info.hasRender && !info.meshPath.empty()) {
            paths.push_back((std::filesystem::path(DATA_DIR) / info.meshPath).string());
        }
    }
    return paths;
}

std::vector<std::string> AssetRegistry::getPhysicsAssetPaths() const
{
    std::vector<std::string> paths;
    for (const auto& [name, info] : nameToAsset) {
        if (info.hasPhysics && info.assetType == FILE_MESH && !info.filepath.empty()) {
            paths.push_back((std::filesystem::path(DATA_DIR) / info.filepath).string());
        }
    }
    return paths;
}

}