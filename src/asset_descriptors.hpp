#pragma once

#include <madrona/physics.hpp>
#include <madrona/render/ecs.hpp>
#include <madrona/render/common.hpp>
#include <madrona/math.hpp>

namespace madrona::escape_room {

// Asset-specific constants moved from consts.hpp
// These are only used for asset loading and descriptor tables

namespace asset_constants {

// Physics properties
namespace physics {
    inline constexpr float cubeInverseMass = 0.075f;     // Pushable cube with ~13kg mass
    inline constexpr float wallInverseMass = 0.f;        // Static walls (infinite mass)
    inline constexpr float agentInverseMass = 1.f;       // Agent unit mass for direct control
    inline constexpr float planeInverseMass = 0.f;       // Static plane (infinite mass)
    
    inline constexpr float standardStaticFriction = 0.5f;     // Standard static friction
    inline constexpr float standardDynamicFriction = 0.5f;    // Standard dynamic friction
    inline constexpr float cubeStaticFriction = 0.5f;         // Cube static friction
    inline constexpr float cubeDynamicFriction = 0.75f;       // Cube dynamic friction
}

// Rendering properties
namespace rendering {
    // Material properties
    namespace material {
        inline constexpr float defaultRoughness = 0.8f;
        inline constexpr float defaultMetallic = 0.2f;
        inline constexpr float agentBodyMetallic = 1.0f;
        inline constexpr float floorAlpha = 1.0f;  // Opaque floor
        inline constexpr float agentBodyRoughness = 0.5f;
    }
    
    // Material indices for object rendering
    namespace materialIndex {
        inline constexpr int32_t cube = 0;         // Brown cube material
        inline constexpr int32_t wall = 1;         // Gray wall material  
        inline constexpr int32_t agentBody = 2;    // White agent body material
        inline constexpr int32_t agentParts = 3;   // Light gray agent parts material
        inline constexpr int32_t floor = 4;        // Brown floor material
        inline constexpr int32_t axisX = 5;        // Red X-axis material
        inline constexpr int32_t button = 6;       // Yellow button material
        inline constexpr int32_t axisY = 7;        // Green Y-axis material
        inline constexpr int32_t axisZ = 8;        // Blue Z-axis material
    }
    
    // Color values (RGB 0-255)
    namespace colors {
        inline constexpr uint8_t cubeRed = 139;
        inline constexpr uint8_t cubeGreen = 69;
        inline constexpr uint8_t cubeBlue = 19;
        
        inline constexpr uint8_t agentGray = 180;
        
        inline constexpr uint8_t axisRed = 255;
        inline constexpr uint8_t redGreen = 0;
        inline constexpr uint8_t redBlue = 0;
        
        inline constexpr uint8_t yellowRed = 255;
        inline constexpr uint8_t yellowGreen = 255;
        inline constexpr uint8_t yellowBlue = 0;
        
        inline constexpr uint8_t greenRed = 0;
        inline constexpr uint8_t axisGreen = 255;
        inline constexpr uint8_t greenBlue = 0;
        
        inline constexpr uint8_t blueRed = 0;
        inline constexpr uint8_t blueGreen = 0;
        inline constexpr uint8_t axisBlue = 255;
    }
    
    // Normalized color values (0.0-1.0)
    namespace normalizedColors {
        inline constexpr float wallGray = 0.5f;
        inline constexpr float white = 1.0f;
        inline constexpr float floorBrownRed = 0.4f;
        inline constexpr float floorBrownGreen = 0.3f;
        inline constexpr float floorBrownBlue = 0.2f;
    }
}

} // namespace asset_constants

// Physics asset descriptor - defines how to load physics collision data
struct PhysicsAssetDescriptor {
    // Human-readable name for debugging/logging
    const char* name;
    
    // Which asset ID this physics asset is for (from AssetIDs namespace)
    uint32_t objectId;
    
    // Type of physics asset
    enum Type {
        FILE_MESH,      // Load from .obj file
        BUILTIN_PLANE,  // Use built-in infinite plane
    } type;
    
    // File path relative to DATA_DIR (nullptr for built-ins)
    const char* filepath;
    
    // Inverse mass (0 = infinite mass/static)
    float inverseMass;
    
    // Friction coefficients
    phys::RigidBodyFrictionData friction;
    
    // Constrain rotation to Z-axis only (for agents)
    bool constrainRotationXY;
};

// Render asset descriptor - defines visual mesh and material assignments
struct RenderAssetDescriptor {
    // Human-readable name for debugging/logging
    const char* name;
    
    // Which asset ID this render asset is for (from AssetIDs namespace)
    uint32_t objectId;
    
    // Mesh file path relative to DATA_DIR
    const char* meshPath;
    
    // Material index for each mesh in the file
    // (some .obj files contain multiple meshes)
    const uint32_t* materialIndices;
    uint32_t numMeshes;
};

// Material descriptor - defines surface appearance properties
struct MaterialDescriptor {
    // Human-readable name for debugging/logging
    const char* name;
    
    // RGBA color (alpha used for transparency)
    math::Vector4 color;
    
    // Texture index (-1 for no texture)
    int32_t textureIdx;
    
    // PBR material properties
    float roughness;
    float metallic;
};

// Texture descriptor - defines texture file to load
struct TextureDescriptor {
    // Human-readable name
    const char* name;
    
    // File path relative to DATA_DIR
    const char* filepath;
};

// Static asset tables - defined in asset_descriptors.cpp
extern const PhysicsAssetDescriptor PHYSICS_ASSETS[];
constexpr size_t NUM_PHYSICS_ASSETS = 4;  // Cube, Wall, Agent, Plane

extern const RenderAssetDescriptor RENDER_ASSETS[];
constexpr size_t NUM_RENDER_ASSETS = 7;  // All SimObjects

extern const MaterialDescriptor MATERIALS[];
constexpr size_t NUM_MATERIALS = 9;  // All material types

extern const TextureDescriptor TEXTURES[];
constexpr size_t NUM_TEXTURES = 2;  // green_grid.png, smile.png

// Helper function to get physics asset by asset ID
const PhysicsAssetDescriptor* getPhysicsAsset(uint32_t objectId);

// Helper function to get render asset by asset ID
const RenderAssetDescriptor* getRenderAsset(uint32_t objectId);

}