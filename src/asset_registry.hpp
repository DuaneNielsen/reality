#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <madrona/physics.hpp>

namespace madEscape {

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
    
    // Global instance accessor
    static AssetRegistry& getInstance();
};

}