#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <array>
#include <madrona/physics.hpp>
#include <madrona/math.hpp>
#include <madrona/heap_array.hpp>
#include <madrona/importer.hpp>
#include <madrona/stack_alloc.hpp>

namespace madEscape {

// Texture and Material definitions
namespace AssetTextures {
    struct TextureDescriptor {
        const char* name;
        const char* filepath;
    };
    
    constexpr size_t NUM_TEXTURES = 2;
    extern const TextureDescriptor TEXTURES[NUM_TEXTURES];
}

namespace AssetMaterials {
    struct MaterialDescriptor {
        const char* name;
        madrona::math::Vector4 color;
        int32_t textureIdx;
        float roughness;
        float metallic;
    };
    
    constexpr size_t NUM_MATERIALS = 10;
    extern const MaterialDescriptor MATERIALS[NUM_MATERIALS];
}

class AssetRegistry {
public:
    enum AssetType {
        FILE_MESH,      // Load from .obj file
        BUILTIN_PLANE,  // Use built-in infinite plane
    };
    
    struct AssetInfo {
        std::string name;
        uint32_t id;
        bool hasPhysics;
        bool hasRender;
        
        // Physics properties
        AssetType assetType;
        std::string filepath;  // Relative to DATA_DIR
        float inverseMass;
        madrona::phys::RigidBodyFrictionData friction;
        bool constrainRotationXY;
        
        // Render properties
        std::string meshPath;  // Relative to DATA_DIR
        std::vector<uint32_t> materialIndices;
        uint32_t numMeshes;
    };
    
private:
    std::unordered_map<std::string, AssetInfo> nameToAsset;
    std::vector<AssetInfo> idToAsset;
    uint32_t nextId;
    
public:
    AssetRegistry();
    
    // Register a complete asset with all properties
    uint32_t registerFullAsset(const AssetInfo& info);
    
    // Legacy methods for backward compatibility
    uint32_t registerAsset(const std::string& name, 
                          bool hasPhysics, bool hasRender);
    uint32_t registerAssetWithId(const std::string& name, uint32_t id,
                                bool hasPhysics, bool hasRender);
    
    uint32_t getAssetId(const std::string& name) const;
    bool tryGetAssetId(const std::string& name, uint32_t& outId) const;
    bool assetHasPhysics(uint32_t id) const;
    bool assetHasRender(uint32_t id) const;
    const std::string& getAssetName(uint32_t id) const;
    const AssetInfo* getAssetInfo(uint32_t id) const;
    const AssetInfo* getAssetInfoByName(const std::string& name) const;
    
    // Iteration methods
    std::vector<AssetInfo> getAllAssets() const;
    std::vector<AssetInfo> getPhysicsAssets() const;
    std::vector<AssetInfo> getRenderAssets() const;
    uint32_t getPhysicsAssetCount() const;
    uint32_t getRenderAssetCount() const;
    
    // Asset loading helpers
    static madrona::Span<madrona::imp::SourceTexture> loadTextures(madrona::StackAlloc& alloc);
    static std::array<madrona::imp::SourceMaterial, AssetMaterials::NUM_MATERIALS> createMaterials();
    std::vector<std::string> getRenderAssetPaths() const;
    std::vector<std::string> getPhysicsAssetPaths() const;
    
    // Global instance accessor
    static AssetRegistry& getInstance();
};

}