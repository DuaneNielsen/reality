#pragma once

#include <array>
#include <madrona/physics.hpp>
#include <madrona/math.hpp>
#include <madrona/heap_array.hpp>
#include <madrona/importer.hpp>
#include <madrona/stack_alloc.hpp>
#include "asset_ids.hpp"

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
    
    constexpr size_t NUM_MATERIALS = 12;
    extern const MaterialDescriptor MATERIALS[NUM_MATERIALS];
}

// Asset information structure
struct AssetInfo {
    enum AssetType {
        FILE_MESH,      // Load from .obj file
        BUILTIN_PLANE,  // Use built-in infinite plane
    };
    
    const char* name;
    uint32_t id;
    bool hasPhysics;
    bool hasRender;
    
    // Physics properties
    AssetType assetType;
    const char* filepath;  // Relative to DATA_DIR
    float inverseMass;
    madrona::phys::RigidBodyFrictionData friction;
    bool constrainRotationXY;
    
    // Render properties
    const char* meshPath;  // Relative to DATA_DIR
    const uint32_t* materialIndices;  // Pointer to array of material indices
    uint32_t numMaterialIndices;      // Number of material indices
    uint32_t numMeshes;
};

// Static asset data - all assets defined at compile time
namespace Assets {
    // Number of statically defined assets
    constexpr size_t NUM_ASSETS = 8;
    
    // The main asset table indexed by AssetID
    extern const AssetInfo ASSET_TABLE[AssetIDs::MAX_ASSETS];
    
    // Helper functions for asset queries
    constexpr const AssetInfo* getAssetInfo(uint32_t id) {
        if (id >= AssetIDs::MAX_ASSETS || id == AssetIDs::INVALID) {
            return nullptr;
        }
        // Check if the entry is valid by checking the name
        if (ASSET_TABLE[id].name == nullptr) {
            return nullptr;
        }
        return &ASSET_TABLE[id];
    }
    
    constexpr bool assetHasPhysics(uint32_t id) {
        const auto* info = getAssetInfo(id);
        return info ? info->hasPhysics : false;
    }
    
    constexpr bool assetHasRender(uint32_t id) {
        const auto* info = getAssetInfo(id);
        return info ? info->hasRender : false;
    }
    
    constexpr const char* getAssetName(uint32_t id) {
        const auto* info = getAssetInfo(id);
        return info ? info->name : "";
    }
    
    // Get asset ID by name (compile-time for known names)
    inline uint32_t getAssetId(const char* name) {
        // Linear search through the table - acceptable for small fixed set
        for (uint32_t i = 0; i < AssetIDs::MAX_ASSETS; ++i) {
            if (ASSET_TABLE[i].name != nullptr) {
                // Simple string comparison
                const char* tableName = ASSET_TABLE[i].name;
                const char* searchName = name;
                bool match = true;
                while (*tableName && *searchName) {
                    if (*tableName != *searchName) {
                        match = false;
                        break;
                    }
                    tableName++;
                    searchName++;
                }
                if (match && *tableName == '\0' && *searchName == '\0') {
                    return i;
                }
            }
        }
        return AssetIDs::INVALID;
    }
    
    // Count functions
    inline uint32_t getPhysicsAssetCount() {
        uint32_t count = 0;
        for (uint32_t i = 0; i < AssetIDs::MAX_ASSETS; ++i) {
            if (ASSET_TABLE[i].name != nullptr && ASSET_TABLE[i].hasPhysics) {
                count++;
            }
        }
        return count;
    }
    
    inline uint32_t getRenderAssetCount() {
        uint32_t count = 0;
        for (uint32_t i = 0; i < AssetIDs::MAX_ASSETS; ++i) {
            if (ASSET_TABLE[i].name != nullptr && ASSET_TABLE[i].hasRender) {
                count++;
            }
        }
        return count;
    }
    
    // Asset loading helpers
    madrona::Span<madrona::imp::SourceTexture> loadTextures(madrona::StackAlloc& alloc);
    std::array<madrona::imp::SourceMaterial, AssetMaterials::NUM_MATERIALS> createMaterials();
}

}