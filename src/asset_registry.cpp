#include "asset_registry.hpp"
#include "asset_ids.hpp"
#include <cassert>

namespace madEscape {

AssetRegistry::AssetRegistry() : nextId(AssetIDs::DYNAMIC_START) {
    idToAsset.resize(AssetIDs::MAX_ASSETS);
    
    registerAssetWithId("cube", AssetIDs::CUBE, true, true);
    registerAssetWithId("wall", AssetIDs::WALL, true, true);
    registerAssetWithId("agent", AssetIDs::AGENT, true, true);
    registerAssetWithId("plane", AssetIDs::PLANE, true, true);
    
    registerAssetWithId("axis_x", AssetIDs::AXIS_X, false, true);
    registerAssetWithId("axis_y", AssetIDs::AXIS_Y, false, true);
    registerAssetWithId("axis_z", AssetIDs::AXIS_Z, false, true);
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
    
    AssetInfo info{name, id, hasPhysics, hasRender};
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

}