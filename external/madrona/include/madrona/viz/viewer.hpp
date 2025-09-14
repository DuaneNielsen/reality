#pragma once

#include <madrona/importer.hpp>
#include <madrona/render/api.hpp>
#include <madrona/render/render_mgr.hpp>
#include <madrona/window.hpp>

#include <memory>

namespace madrona::viz {

// Forward declaration
struct ViewerCam;

// The viewer app simply provides UI overlay over the rendering output
// of the render context and presents the whole rendering output
// to the screen.
class Viewer {
public:
    struct Config {
        uint32_t numWorlds;
        uint32_t simTickRate;
        // Initial camera position
        float cameraMoveSpeed;
        math::Vector3 cameraPosition;
        math::Quat cameraRotation;
        bool hideMenu = false;  // Hide ImGui menu for clean screenshots
        void *manager = nullptr; // Manager pointer for camera tracking
    };

    enum class KeyboardKey : uint32_t {
        A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X,
        Y, Z, K1, K2, K3, K4, K5, K6, K7, K8, K9, K0,
        Shift, Space, NumKeys,
    };

    class UserInput {
    public:
        inline UserInput(bool *keys_state, bool *press_state);

        inline bool keyPressed(KeyboardKey key) const;
        inline bool keyHit(KeyboardKey key) const;

    private:
        bool *keys_state_;
        bool *press_state_;
    };

    Viewer(const render::RenderManager &render_mgr,
           const Window *window,
           const Config &cfg);
    Viewer(Viewer &&);
    ~Viewer();

    // Viewer app can also load objects (this would be used if the
    // batch renderer isn't used).
    CountT loadObjects(Span<const imp::SourceObject> objs,
                       Span<const imp::SourceMaterial> mats,
                       Span<const imp::SourceTexture> textures,
                       bool override_materials = false);

    void configureLighting(Span<const render::LightConfig> lights);

    // Run the viewer
    template <typename WorldInputFn, typename AgentInputFn,
              typename StepFn, typename UIFn>
    void loop(WorldInputFn &&world_input_fn, AgentInputFn &&agent_input_fn,
              StepFn &&step_fn, UIFn &&ui_fn);

    void stopLoop();

    CountT getCurrentWorldID() const;
    CountT getCurrentViewID() const;
    CountT getCurrentControlID() const;
    
    // Camera tracking controls
    void toggleCameraTracking();
    
    // Direct camera control methods
    void setCameraPosition(const math::Vector3& pos);
    void setCameraOrientation(const math::Quat& rot);
    void setCameraLookAt(const math::Vector3& pos, const math::Vector3& target);
    void setCameraVectors(const math::Vector3& pos, const math::Vector3& fwd, 
                         const math::Vector3& up, const math::Vector3& right);
    ViewerCam& getCamera();
    const ViewerCam& getCamera() const;
    
    // Multi-world grid layout control
    void setMultiWorldGrid(bool enabled, float spacing = 10.0f, uint32_t gridCols = 4, 
                          float worldScaleX = 40.0f, float worldScaleY = 40.0f);
    void setMultiWorldGrid(bool enabled);
    
    // Explore mode control - overlay all agents in same coordinate space
    void setExploreMode(bool enabled);

private:
    void loop(
        void (*world_input_fn)(void *, CountT, const UserInput &),
        void *world_input_data,
        void (*agent_input_fn)(void *, CountT, CountT, const UserInput &),
        void *agent_input_data,
        void (*step_fn)(void *), void *step_data,
        void (*ui_fn)(void *), void *ui_data);

    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}

#include "viewer.inl"
