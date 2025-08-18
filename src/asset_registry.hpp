#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace madEscape {

class AssetRegistry {
public:
    struct AssetInfo {
        std::string name;
        uint32_t id;
        bool hasPhysics;
        bool hasRender;
    };
    
private:
    std::unordered_map<std::string, AssetInfo> nameToAsset;
    std::vector<AssetInfo> idToAsset;
    uint32_t nextId;
    
public:
    AssetRegistry();
    
    uint32_t registerAsset(const std::string& name, 
                          bool hasPhysics, bool hasRender);
    uint32_t registerAssetWithId(const std::string& name, uint32_t id,
                                bool hasPhysics, bool hasRender);
    
    uint32_t getAssetId(const std::string& name) const;
    bool tryGetAssetId(const std::string& name, uint32_t& outId) const;
    bool assetHasPhysics(uint32_t id) const;
    bool assetHasRender(uint32_t id) const;
    const std::string& getAssetName(uint32_t id) const;
};

}