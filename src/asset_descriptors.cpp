#include "asset_descriptors.hpp"

using namespace madrona;
using namespace madrona::phys;

namespace madrona::escape_room {

using namespace asset_constants;

// Physics asset definitions
const PhysicsAssetDescriptor PHYSICS_ASSETS[] = {
    {
        .name = "cube",
        .objectId = madEscape::SimObject::Cube,
        .type = madrona::escape_room::PhysicsAssetDescriptor::FILE_MESH,
        .filepath = "cube_collision.obj",
        .inverseMass = physics::cubeInverseMass,
        .friction = {
            .muS = physics::cubeStaticFriction,
            .muD = physics::cubeDynamicFriction,
        },
        .constrainRotationXY = false,
    },
    {
        .name = "wall",
        .objectId = madEscape::SimObject::Wall,
        .type = madrona::escape_room::PhysicsAssetDescriptor::FILE_MESH,
        .filepath = "wall_collision.obj",
        .inverseMass = physics::wallInverseMass,
        .friction = {
            .muS = physics::standardStaticFriction,
            .muD = physics::standardDynamicFriction,
        },
        .constrainRotationXY = false,
    },
    {
        .name = "agent",
        .objectId = madEscape::SimObject::Agent,
        .type = madrona::escape_room::PhysicsAssetDescriptor::FILE_MESH,
        .filepath = "agent_collision_simplified.obj",
        .inverseMass = physics::agentInverseMass,
        .friction = {
            .muS = physics::standardStaticFriction,
            .muD = physics::standardDynamicFriction,
        },
        .constrainRotationXY = true,  // Prevent tipping over
    },
    {
        .name = "plane",
        .objectId = madEscape::SimObject::Plane,
        .type = madrona::escape_room::PhysicsAssetDescriptor::BUILTIN_PLANE,
        .filepath = nullptr,
        .inverseMass = physics::planeInverseMass,
        .friction = {
            .muS = physics::standardStaticFriction,
            .muD = physics::standardDynamicFriction,
        },
        .constrainRotationXY = false,
    },
};
// NUM_PHYSICS_ASSETS is defined as constexpr in header

// Material assignments for render assets
static const uint32_t CUBE_MATERIALS[] = { rendering::materialIndex::cube };
static const uint32_t WALL_MATERIALS[] = { rendering::materialIndex::wall };
static const uint32_t AGENT_MATERIALS[] = { 
    rendering::materialIndex::agentBody,   // Body mesh
    rendering::materialIndex::agentParts,  // Eyes mesh
    rendering::materialIndex::agentParts   // Other parts mesh
};
static const uint32_t PLANE_MATERIALS[] = { rendering::materialIndex::floor };
static const uint32_t AXIS_X_MATERIALS[] = { rendering::materialIndex::axisX };
static const uint32_t AXIS_Y_MATERIALS[] = { rendering::materialIndex::axisY };
static const uint32_t AXIS_Z_MATERIALS[] = { rendering::materialIndex::axisZ };

// Render asset definitions
const RenderAssetDescriptor RENDER_ASSETS[] = {
    {
        .name = "cube",
        .objectId = madEscape::SimObject::Cube,
        .meshPath = "cube_render.obj",
        .materialIndices = CUBE_MATERIALS,
        .numMeshes = 1,
    },
    {
        .name = "wall",
        .objectId = madEscape::SimObject::Wall,
        .meshPath = "wall_render.obj",
        .materialIndices = WALL_MATERIALS,
        .numMeshes = 1,
    },
    {
        .name = "agent",
        .objectId = madEscape::SimObject::Agent,
        .meshPath = "agent_render.obj",
        .materialIndices = AGENT_MATERIALS,
        .numMeshes = 3,
    },
    {
        .name = "plane",
        .objectId = madEscape::SimObject::Plane,
        .meshPath = "plane.obj",
        .materialIndices = PLANE_MATERIALS,
        .numMeshes = 1,
    },
    {
        .name = "axis_x",
        .objectId = madEscape::SimObject::AxisX,
        .meshPath = "cube_render.obj",  // Reuse cube mesh
        .materialIndices = AXIS_X_MATERIALS,
        .numMeshes = 1,
    },
    {
        .name = "axis_y",
        .objectId = madEscape::SimObject::AxisY,
        .meshPath = "cube_render.obj",  // Reuse cube mesh
        .materialIndices = AXIS_Y_MATERIALS,
        .numMeshes = 1,
    },
    {
        .name = "axis_z",
        .objectId = madEscape::SimObject::AxisZ,
        .meshPath = "cube_render.obj",  // Reuse cube mesh
        .materialIndices = AXIS_Z_MATERIALS,
        .numMeshes = 1,
    },
};
// NUM_RENDER_ASSETS is defined as constexpr in header

// Material definitions
const MaterialDescriptor MATERIALS[] = {
    {
        .name = "cube_brown",
        .color = render::rgb8ToFloat(
            rendering::colors::cubeRed,
            rendering::colors::cubeGreen,
            rendering::colors::cubeBlue
        ),
        .textureIdx = -1,
        .roughness = rendering::material::defaultRoughness,
        .metallic = rendering::material::defaultMetallic,
    },
    {
        .name = "wall_gray",
        .color = math::Vector4{
            rendering::normalizedColors::wallGray,
            rendering::normalizedColors::wallGray,
            rendering::normalizedColors::wallGray,
            1.0f  // Full opacity
        },
        .textureIdx = -1,
        .roughness = rendering::material::defaultRoughness,
        .metallic = rendering::material::defaultMetallic,
    },
    {
        .name = "agent_body",
        .color = math::Vector4{
            rendering::normalizedColors::white,
            rendering::normalizedColors::white,
            rendering::normalizedColors::white,
            1.0f  // Full opacity
        },
        .textureIdx = 1,  // Smile texture
        .roughness = rendering::material::agentBodyRoughness,
        .metallic = rendering::material::agentBodyMetallic,
    },
    {
        .name = "agent_parts",
        .color = render::rgb8ToFloat(
            rendering::colors::agentGray,
            rendering::colors::agentGray,
            rendering::colors::agentGray
        ),
        .textureIdx = -1,
        .roughness = rendering::material::defaultRoughness,
        .metallic = rendering::material::agentBodyMetallic,
    },
    {
        .name = "floor_brown",
        .color = math::Vector4{
            rendering::normalizedColors::floorBrownRed,
            rendering::normalizedColors::floorBrownGreen,
            rendering::normalizedColors::floorBrownBlue,
            1.0f  // Full opacity
        },
        .textureIdx = 0,  // Green grid texture
        .roughness = rendering::material::defaultRoughness,
        .metallic = rendering::material::defaultMetallic,
    },
    {
        .name = "axis_x_red",
        .color = render::rgb8ToFloat(
            rendering::colors::axisRed,
            rendering::colors::redGreen,
            rendering::colors::redBlue
        ),
        .textureIdx = -1,
        .roughness = rendering::material::defaultRoughness,
        .metallic = rendering::material::agentBodyMetallic,
    },
    {
        .name = "button_yellow",
        .color = render::rgb8ToFloat(
            rendering::colors::yellowRed,
            rendering::colors::yellowGreen,
            rendering::colors::yellowBlue
        ),
        .textureIdx = -1,
        .roughness = rendering::material::defaultRoughness,
        .metallic = rendering::material::agentBodyMetallic,
    },
    {
        .name = "axis_y_green",
        .color = render::rgb8ToFloat(
            rendering::colors::greenRed,
            rendering::colors::axisGreen,
            rendering::colors::greenBlue
        ),
        .textureIdx = -1,
        .roughness = rendering::material::defaultRoughness,
        .metallic = rendering::material::agentBodyMetallic,
    },
    {
        .name = "axis_z_blue",
        .color = render::rgb8ToFloat(
            rendering::colors::blueRed,
            rendering::colors::blueGreen,
            rendering::colors::axisBlue
        ),
        .textureIdx = -1,
        .roughness = rendering::material::defaultRoughness,
        .metallic = rendering::material::agentBodyMetallic,
    },
};
// NUM_MATERIALS is defined as constexpr in header

// Texture definitions
const TextureDescriptor TEXTURES[] = {
    {
        .name = "green_grid",
        .filepath = "green_grid.png",
    },
    {
        .name = "smile",
        .filepath = "smile.png",
    },
};
// NUM_TEXTURES is defined as constexpr in header

// Helper function implementations
const madrona::escape_room::PhysicsAssetDescriptor* getPhysicsAsset(madEscape::SimObject obj) {
    for (size_t i = 0; i < madrona::escape_room::NUM_PHYSICS_ASSETS; i++) {
        if (madrona::escape_room::PHYSICS_ASSETS[i].objectId == obj) {
            return &madrona::escape_room::PHYSICS_ASSETS[i];
        }
    }
    return nullptr;
}

const RenderAssetDescriptor* getRenderAsset(madEscape::SimObject obj) {
    for (size_t i = 0; i < madrona::escape_room::NUM_RENDER_ASSETS; i++) {
        if (madrona::escape_room::RENDER_ASSETS[i].objectId == obj) {
            return &madrona::escape_room::RENDER_ASSETS[i];
        }
    }
    return nullptr;
}

}