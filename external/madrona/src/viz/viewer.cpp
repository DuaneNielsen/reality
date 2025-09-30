#include <madrona/viz/viewer.hpp>
#include <madrona/stack_alloc.hpp>
#include <madrona/utils.hpp>
#include <madrona/math.hpp>

#include "viewer_common.hpp"
#include "viewer_renderer.hpp"

#include <imgui.h>

#include <cstdio>
#include <cstdlib>
#include <thread>
#include <algorithm>

// Include manager header for camera tracking
#include "../../../../src/mgr.hpp"

using namespace std;

namespace madrona::viz {

namespace InternalConfig {
inline constexpr float mouseSpeed = 2e-4f;

inline constexpr auto nsPerFrame = chrono::nanoseconds(8333333);
inline constexpr auto nsPerFrameLongWait =
    chrono::nanoseconds(7000000);
inline constexpr float secondsPerFrame =
    chrono::duration<float>(nsPerFrame).count();
}

struct SceneProperties {
    math::AABB aabb;
    uint32_t totalTriangles; // Post transform
};

struct Viewer::Impl {
    ViewerRenderer renderer;
    ViewerControl vizCtrl;

    uint32_t numWorlds;
    uint32_t maxNumAgents;
    int32_t simTickRate;
    float cameraMoveSpeed;
    bool shouldExit;
    bool hideMenu;
    void *manager; // Manager pointer for camera tracking

    inline Impl(const render::RenderManager &render_mgr,
                const Window *window,
                const Viewer::Config &cfg);

    inline bool startFrame();

    // This is going to render all the views which were registered
    // by the ECS
    inline void renderViews();

    // This is going to render the viewer window itself (and support
    // the fly camera)
    inline void render(float frame_duration);

    inline void loop(
        void (*world_input_fn)(void *, CountT, const UserInput &),
        void *world_input_data,
        void (*agent_input_fn)(void *, CountT, CountT, const UserInput &),
        void *agent_input_data,
        void (*step_fn)(void *), void *step_data,
        void (*ui_fn)(void *), void *ui_data);
};

static void handleCamera(GLFWwindow *window,
                         ViewerCam &cam,
                         float cam_move_speed,
                         void *mgr_ptr = nullptr)
{
    static int call_counter = 0;
    if (call_counter++ % 300 == 0) { // Print every 5 seconds at 60fps
        printf("handleCamera called (frame %d), tracking=%s, mgr_ptr=%p\n", 
               call_counter, cam.isTracking ? "true" : "false", mgr_ptr);
    }
    
    auto keyPressed = [&](uint32_t key) {
        return glfwGetKey(window, key) == GLFW_PRESS;
    };

    math::Vector3 translate = math::Vector3::zero();

    auto cursorPosition = [window]() {
        double mouse_x, mouse_y;
        glfwGetCursorPos(window, &mouse_x, &mouse_y);

        return math::Vector2 { float(mouse_x), float(-mouse_y) };
    };


    if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_2) == GLFW_PRESS ||
            glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) {

        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

        math::Vector2 mouse_cur = cursorPosition();
        math::Vector2 mouse_delta = mouse_cur - cam.mousePrev;

        auto around_right = math::Quat::angleAxis(
            mouse_delta.y * InternalConfig::mouseSpeed, cam.right);

        auto around_up = math::Quat::angleAxis(
            -mouse_delta.x * InternalConfig::mouseSpeed, math::up);

        auto rotation = (around_up * around_right).normalize();

        cam.up = rotation.rotateVec(cam.up);
        cam.fwd = rotation.rotateVec(cam.fwd);
        cam.right = rotation.rotateVec(cam.right);

        if (keyPressed(GLFW_KEY_W)) {
            translate += cam.fwd;
        }

        if (keyPressed(GLFW_KEY_A)) {
            translate -= cam.right;
        }

        if (keyPressed(GLFW_KEY_S)) {
            translate -= cam.fwd;
        }

        if (keyPressed(GLFW_KEY_D)) {
            translate += cam.right;
        }

        cam.mousePrev = mouse_cur;
    } else {
        glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

        if (keyPressed(GLFW_KEY_W)) {
            translate += cam.up;
        }

        if (keyPressed(GLFW_KEY_A)) {
            translate -= cam.right;
        }

        if (keyPressed(GLFW_KEY_S)) {
            translate -= cam.up;
        }

        if (keyPressed(GLFW_KEY_D)) {
            translate += cam.right;
        }

        cam.mousePrev = cursorPosition();
    }

    // Apply camera movement normally - tracking will override position
    cam.position += translate * cam_move_speed *
        InternalConfig::secondsPerFrame;
    
    // Camera tracking removed - now handled by application layer
}

static float throttleFPS(chrono::time_point<chrono::steady_clock> start) {
    using namespace chrono;
    using namespace chrono_literals;

    auto end = steady_clock::now();
    while (end - start <
           InternalConfig::nsPerFrameLongWait) {
        this_thread::sleep_for(1ms);

        end = steady_clock::now();
    }

    while (end - start < InternalConfig::nsPerFrame) {
        this_thread::yield();

        end = steady_clock::now();
    }

    return duration<float>(end - start).count();
}

bool Viewer::Impl::startFrame()
{
    glfwPollEvents();

    renderer.waitUntilFrameReady();

#if 1
    // Handle the window resize if needed before anything else
    if (renderer.needResize()) {
        // Handle resize!
        renderer.handleResize();

        return false;
    }
#endif

    renderer.startFrame();
    ImGui::NewFrame();

    return true;
}

// https://lemire.me/blog/2021/06/03/computing-the-number-of-digits-of-an-integer-even-faster/
static int32_t numDigits(uint32_t x)
{
  static uint64_t table[] = {
      4294967296,  8589934582,  8589934582,  8589934582,  12884901788,
      12884901788, 12884901788, 17179868184, 17179868184, 17179868184,
      21474826480, 21474826480, 21474826480, 21474826480, 25769703776,
      25769703776, 25769703776, 30063771072, 30063771072, 30063771072,
      34349738368, 34349738368, 34349738368, 34349738368, 38554705664,
      38554705664, 38554705664, 41949672960, 41949672960, 41949672960,
      42949672960, 42949672960};

  uint32_t idx = 31 - __builtin_clz(x | 1);
  return (x + table[idx]) >> 32;
}

static void flyCamUI(ViewerCam &cam)
{
    auto side_size = ImGui::CalcTextSize(" Bottom " );
    side_size.y *= 1.4f;
    ImGui::PushStyleVar(ImGuiStyleVar_ButtonTextAlign,
                        ImVec2(0.5f, 0.f));

    if (ImGui::Button("Top", side_size)) {
        cam.position = 10.f * math::up;
        cam.fwd = -math::up;
        cam.up = -math::fwd;
        cam.right = cross(cam.fwd, cam.up);
    }

    ImGui::SameLine();

    if (ImGui::Button("Left", side_size)) {
        cam.position = -10.f * math::right;
        cam.fwd = math::right;
        cam.up = math::up;
        cam.right = cross(cam.fwd, cam.up);
    }

    ImGui::SameLine();

    if (ImGui::Button("Right", side_size)) {
        cam.position = 10.f * math::right;
        cam.fwd = -math::right;
        cam.up = math::up;
        cam.right = cross(cam.fwd, cam.up);
    }

    ImGui::SameLine();

    if (ImGui::Button("Bottom", side_size)) {
        cam.position = -10.f * math::up;
        cam.fwd = math::up;
        cam.up = math::fwd;
        cam.right = cross(cam.fwd, cam.up);
    }

    ImGui::PopStyleVar();

#if 0
    auto ortho_size = ImGui::CalcTextSize(" Orthographic ");
    ImGui::PushStyleVar(ImGuiStyleVar_SelectableTextAlign,
                        ImVec2(0.5f, 0.f));
    if (ImGui::Selectable("Perspective", cam.perspective, 0,
                          ortho_size)) {
        cam.perspective = true;
    }
    ImGui::SameLine();

    if (ImGui::Selectable("Orthographic", !cam.perspective, 0,
                          ortho_size)) {
        cam.perspective = false;
    }

    ImGui::SameLine();

    ImGui::PopStyleVar();

    ImGui::TextUnformatted("Projection");
#endif

    // Camera Tracking Controls
    ImGui::Spacing();
    ImGui::TextUnformatted("Agent Tracking");
    ImGui::Separator();
    
    ImGui::Checkbox("Enable Tracking", &cam.isTracking);
    
    if (cam.isTracking) {
        ImGui::SetNextItemWidth(ImGui::CalcTextSize("000").x * 2);
        ImGui::DragInt("World", (int*)&cam.trackedWorld, 1.0f, 0, 99, "%d");
        
        ImGui::SameLine();
        ImGui::SetNextItemWidth(ImGui::CalcTextSize("000").x * 2);
        ImGui::DragInt("Agent", (int*)&cam.trackedAgent, 1.0f, 0, 9, "%d");
        
        ImGui::SetNextItemWidth(ImGui::CalcTextSize("000.0").x * 2);
        ImGui::DragFloat("X Offset", &cam.trackingOffset, 0.5f, 0.0f, 20.0f, "%.1f");
        
        ImGui::Spacing();
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "WASD controls agent");
        ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "Camera follows automatically");
    }
    
    ImGui::Spacing();

    float digit_width = ImGui::CalcTextSize("0").x;
    ImGui::SetNextItemWidth(digit_width * 6);
    if (cam.perspective) {
        ImGui::DragFloat("FOV", &cam.fov, 1.f, 1.f, 179.f, "%.0f");
    } else {
        ImGui::DragFloat("View Size", &cam.orthoHeight,
                          0.5f, 0.f, 100.f, "%0.1f");
    }
}

static void cfgUI(ViewerControl &ctrl,
                  CountT num_agents,
                  CountT num_worlds,
                  int32_t *tick_rate,
                  void *manager)
{
    ctrl.batchRenderOffsetY -=
        ImGui::GetIO().MouseWheel * InternalConfig::secondsPerFrame * 120;
    ctrl.batchRenderOffsetX -=
        ImGui::GetIO().MouseWheelH * InternalConfig::secondsPerFrame * 120;

    ImGui::Begin("Controls");

    const char *viewer_type_ops[] = {
        "Visualizer", "Batch Renderer"
    };

    ImGui::PushItemWidth(
        ImGui::CalcTextSize(" ").x * (float)(strlen(viewer_type_ops[1]) + 3));
    if (ImGui::BeginCombo("Viewer Mode",
                          viewer_type_ops[(uint32_t)ctrl.viewerType])) {
        for (uint32_t i = 0; i < 2; i++) {
            const bool is_selected = (uint32_t)ctrl.viewerType == i;
            if (ImGui::Selectable(viewer_type_ops[i], is_selected)) {
                ctrl.viewerType = (ViewerType)i;
            }

            if (is_selected) {
                ImGui::SetItemDefaultFocus();
            }
        }

        ImGui::EndCombo();
    }

    ImGui::PopItemWidth();

    {
        int world_idx = ctrl.worldIdx;
        float worldbox_width =
            ImGui::CalcTextSize(" ").x * (max(numDigits(num_worlds) + 2, 7_i32));
        if (num_worlds == 1) {
            ImGui::BeginDisabled();
        }
        ImGui::PushItemWidth(worldbox_width);
        ImGui::DragInt("Current World ID", &world_idx, 1.f, 0, num_worlds - 1,
                       "%d", ImGuiSliderFlags_AlwaysClamp);
        ImGui::PopItemWidth();

        if (num_worlds == 1) {
            ImGui::EndDisabled();
        }

        ctrl.worldIdx = world_idx;
    }

    // Display checksum status indicator if manager is available
    if (manager != nullptr) {
        madEscape::Manager* mgr = static_cast<madEscape::Manager*>(manager);

        ImGui::Text("Checksum:");
        ImGui::SameLine();

        // Check checksum status for the currently selected world
        if (mgr->hasChecksumFailed(ctrl.worldIdx)) {
            // Red LED indicator for failed checksum
            ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "[FAIL]");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Checksum verification FAILED - replay diverged from recording");
            }
        } else {
            // Green LED indicator for passed checksum
            ImGui::TextColored(ImVec4(0.0f, 1.0f, 0.0f, 1.0f), "[PASS]");
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Checksum verification passed");
            }
        }
    }

    ImGui::Checkbox("Control Current View", &ctrl.linkViewControl);

    {
        StackAlloc str_alloc;
        const char **cam_opts = str_alloc.allocN<const char *>(num_agents + 1);
        cam_opts[0] = "Free Camera";

        ImVec2 combo_size = ImGui::CalcTextSize(" Free Camera ");

        for (CountT i = 0; i < num_agents; i++) {
            const char *agent_prefix = "Agent ";

            CountT num_bytes = strlen(agent_prefix) + numDigits(i) + 1;
            cam_opts[i + 1] = str_alloc.allocN<char>(num_bytes);
            snprintf((char *)cam_opts[i + 1], num_bytes, "%s%u",
                     agent_prefix, (uint32_t)i);
        }

        ImGui::PushItemWidth(combo_size.x * 1.25f);
        if (ImGui::BeginCombo("Current View", cam_opts[ctrl.viewIdx])) {
            for (CountT i = 0; i < num_agents + 1; i++) {
                const bool is_selected = ctrl.viewIdx == (uint32_t)i;
                if (ImGui::Selectable(cam_opts[i], is_selected)) {
                    ctrl.viewIdx = (uint32_t)i;
                }

                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }

            ImGui::EndCombo();
        }
        ImGui::PopItemWidth();

        if (ctrl.linkViewControl) {
            ctrl.controlIdx = ctrl.viewIdx;
            ImGui::BeginDisabled();
        }

        ImGui::PushItemWidth(combo_size.x * 1.25f);
        if (ImGui::BeginCombo("Input Control", cam_opts[ctrl.controlIdx])) {
            for (CountT i = 0; i < num_agents + 1; i++) {
                const bool is_selected = ctrl.controlIdx == (uint32_t)i;
                if (ImGui::Selectable(cam_opts[i], is_selected)) {
                    ctrl.controlIdx = (uint32_t)i;
                }

                if (is_selected) {
                    ImGui::SetItemDefaultFocus();
                }
            }

            ImGui::EndCombo();
        }
        ImGui::PopItemWidth();

        if (ctrl.linkViewControl) {
            ImGui::EndDisabled();
        }
    }

    ImGui::Spacing();
    ImGui::TextUnformatted("Simulation Settings");
    ImGui::Separator();

#if 0
    {
        static bool override_sun_dir = false;
        ImGui::Checkbox("Override Sun Direction", &override_sun_dir);

        ctrl.overrideLightDir = override_sun_dir;

        if (override_sun_dir) {
            static float theta = 0.0f;
            ImGui::SliderFloat("Theta", &theta, 0.0f, 360.0f);

            static float phi = 0.0f;
            ImGui::SliderFloat("Phi", &phi, 0.0f, 90.0f);

            float x = std::cos(math::toRadians(theta)) * std::sin(math::toRadians(phi));
            float y = std::sin(math::toRadians(theta)) * std::sin(math::toRadians(phi));
            float z = std::cos(math::toRadians(phi));

            math::Vector3 dir = {x, y, z};
            cfg.newLightDir = dir;
        }
    }
#endif

    ImGui::PushItemWidth(ImGui::CalcTextSize(" ").x * 7);
    ImGui::DragInt("Tick Rate (Hz)", (int *)tick_rate, 5.f, 0, 1000);
    if (*tick_rate < 0) {
        *tick_rate = 0;
    }
    ImGui::PopItemWidth();

    ImGui::Spacing();

    if (ctrl.viewIdx != 0 || ctrl.viewerType != ViewerType::Flycam) {
        ImGui::BeginDisabled();
    }

    ImGui::TextUnformatted("Free Camera Config");
    ImGui::Separator();

    flyCamUI(ctrl.flyCam);

    if (ctrl.viewIdx != 0 || ctrl.viewerType != ViewerType::Flycam) {
        ImGui::EndDisabled();
    }

    if (ctrl.viewerType != ViewerType::Grid) {
        ImGui::BeginDisabled();
    }

    ImGui::Spacing();
    ImGui::TextUnformatted("Batch Renderer Visualization");
    ImGui::Separator();

    ImGui::PushItemWidth(ImGui::CalcTextSize(" ").x * 8.0f);

    int grid_img_size = ctrl.gridImageSize;
    ImGui::DragInt("View Width", &grid_img_size, 
                   1.0f, 16, 1024, "%d", ImGuiSliderFlags_AlwaysClamp);
    ctrl.gridImageSize = (uint32_t)grid_img_size;

    int grid_width = ctrl.gridWidth;
    ImGui::DragInt("Grid Width", &grid_width, 
                   1.0f, 1, 1024, "%d", ImGuiSliderFlags_AlwaysClamp);
    ctrl.gridWidth = grid_width;

    ImGui::PopItemWidth();

    ImGui::Checkbox("Render Depth", &ctrl.batchRenderShowDepth);

    if (ctrl.viewerType != ViewerType::Grid) {
        ImGui::EndDisabled();
    }

    ImGui::Spacing();
    ImGui::TextUnformatted("Utilities");
    ImGui::Separator();
    ctrl.requestedScreenshot = ImGui::Button("Take Screenshot");

    ImGui::PushItemWidth(ImGui::CalcTextSize(" ").x * float(
        strlen("Take Screenshot")) + 4.f);
    ImGui::InputText("Output File (.bmp)", ctrl.screenshotFilePath, 256);
    ImGui::PopItemWidth();

    ImGui::End();
}

static void fpsCounterUI(float frame_duration)
{
    auto viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(ImVec2(viewport->WorkSize.x, 0.f),
                            0, ImVec2(1.f, 0.f));
    ImGui::PushStyleVar(ImGuiStyleVar_Alpha, 0.5f);
    ImGui::Begin("FPS Counter", nullptr,
                 ImGuiWindowFlags_NoMove |
                 ImGuiWindowFlags_NoInputs |
                 ImGuiWindowFlags_NoTitleBar |
                 ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::PopStyleVar();
    ImGui::Text("%.3f ms per frame (%.1f FPS)",
                1000.f * frame_duration, 1.f / frame_duration);

    ImGui::End();
}

static ViewerCam initCam(math::Vector3 pos, math::Quat rot)
{
    math::Vector3 fwd = normalize(rot.rotateVec(math::fwd));
    math::Vector3 up = normalize(rot.rotateVec(math::up));
    math::Vector3 right = normalize(cross(fwd, up));

    return ViewerCam {
        .position = pos,
        .fwd = fwd,
        .up = up,
        .right = right,
    };
}

Viewer::Impl::Impl(
        const render::RenderManager &render_mgr,
        const Window *window,
        const Viewer::Config &cfg)
    : renderer(render_mgr, window),
      vizCtrl {
          .worldIdx = 0,
          .viewIdx = 0,
          .controlIdx = 0,
          .linkViewControl = true,
          .flyCam = initCam(cfg.cameraPosition, cfg.cameraRotation),
          .requestedScreenshot = false,
          .screenshotFilePath = "screenshot.bmp",
          .viewerType = ViewerType::Flycam,
          .gridWidth = 10,
          .gridImageSize = 256,
          .batchRenderOffsetX = 0,
          .batchRenderOffsetY = 0,
          .batchRenderShowDepth = false,
          // Initialize multi-world grid parameters with defaults
          .multiWorldGrid = false,
          .gridCols = 4,
          .worldSpacing = 10.0f,
          .worldScaleX = 40.0f,
          .worldScaleY = 40.0f,
      },
      numWorlds(cfg.numWorlds),
      maxNumAgents(
          render_mgr.renderContext().engine_interop_.maxViewsPerWorld),
      simTickRate(cfg.simTickRate),
      cameraMoveSpeed(cfg.cameraMoveSpeed),
      shouldExit(false),
      hideMenu(cfg.hideMenu),
      manager(cfg.manager)
{}

void Viewer::Impl::render(float frame_duration)
{
    // Only render UI if not hidden
    if (!hideMenu) {
        // FIXME: pass actual active agents, not max
        cfgUI(vizCtrl, maxNumAgents, numWorlds, &simTickRate, manager);

        fpsCounterUI(frame_duration);
    }

    ImGui::Render();

    renderer.render(vizCtrl);
}

Viewer::UserInput::UserInput(bool *keys_state, bool *press_state)
    : keys_state_(keys_state), press_state_(press_state)
{}

void Viewer::Impl::loop(
    void (*world_input_fn)(void *, CountT, const UserInput &),
    void *world_input_data,
    void (*agent_input_fn)(void *, CountT, CountT, const UserInput &),
    void *agent_input_data,
    void (*step_fn)(void *), void *step_data,
    void (*ui_fn)(void *), void *ui_data)
{
    GLFWwindow *window = renderer.osWindow();

    std::array<bool, (size_t)KeyboardKey::NumKeys> key_state;
    std::array<bool, (size_t)KeyboardKey::NumKeys> press_state;
    std::array<bool, (size_t)KeyboardKey::NumKeys> prev_key_state;
    utils::zeroN<bool>(prev_key_state.data(), prev_key_state.size());

    std::array<int, (size_t)KeyboardKey::NumKeys> glfw_keys;
#define SETGLFWKEY(KB) \
    glfw_keys[(size_t)KeyboardKey::KB] = GLFW_KEY_##KB

    SETGLFWKEY(A);
    SETGLFWKEY(B);
    SETGLFWKEY(C);
    SETGLFWKEY(D);
    SETGLFWKEY(E);
    SETGLFWKEY(F);
    SETGLFWKEY(G);
    SETGLFWKEY(H);
    SETGLFWKEY(I);
    SETGLFWKEY(J);
    SETGLFWKEY(K);
    SETGLFWKEY(L);
    SETGLFWKEY(M);
    SETGLFWKEY(N);
    SETGLFWKEY(O);
    SETGLFWKEY(P);
    SETGLFWKEY(Q);
    SETGLFWKEY(R);
    SETGLFWKEY(S);
    SETGLFWKEY(T);
    SETGLFWKEY(U);
    SETGLFWKEY(V);
    SETGLFWKEY(W);
    SETGLFWKEY(X);
    SETGLFWKEY(Y);
    SETGLFWKEY(Z);

#undef SETGLFWKEY

    glfw_keys[(size_t)KeyboardKey::K1] = GLFW_KEY_1;
    glfw_keys[(size_t)KeyboardKey::K2] = GLFW_KEY_2;
    glfw_keys[(size_t)KeyboardKey::K3] = GLFW_KEY_3;
    glfw_keys[(size_t)KeyboardKey::K4] = GLFW_KEY_4;
    glfw_keys[(size_t)KeyboardKey::K5] = GLFW_KEY_5;
    glfw_keys[(size_t)KeyboardKey::K6] = GLFW_KEY_6;
    glfw_keys[(size_t)KeyboardKey::K7] = GLFW_KEY_7;
    glfw_keys[(size_t)KeyboardKey::K8] = GLFW_KEY_8;
    glfw_keys[(size_t)KeyboardKey::K9] = GLFW_KEY_9;
    glfw_keys[(size_t)KeyboardKey::K0] = GLFW_KEY_0;

    glfw_keys[(size_t)KeyboardKey::Shift] = GLFW_KEY_LEFT_SHIFT;
    glfw_keys[(size_t)KeyboardKey::Space] = GLFW_KEY_SPACE;

    float frame_duration = InternalConfig::secondsPerFrame;
    auto last_sim_tick_time = chrono::steady_clock::now();
    while (!glfwWindowShouldClose(window) && !shouldExit) {
        utils::zeroN<bool>(key_state.data(), key_state.size());
        utils::zeroN<bool>(press_state.data(), press_state.size());

        for (CountT i = 0; i < (CountT)KeyboardKey::NumKeys; i++) {
            key_state[i] = glfwGetKey(window, glfw_keys[i]) == GLFW_PRESS;
            press_state[i] = !prev_key_state[i] && key_state[i];
        }

        
        // Camera handling moved to application layer
        // if (vizCtrl.controlIdx == 0) {
        //     handleCamera(window, vizCtrl.flyCam, cameraMoveSpeed, manager);
        // }

        auto cur_frame_start_time = chrono::steady_clock::now();

        auto sim_delta_t = chrono::duration<float>(1.f / (float)simTickRate);

        bool success = startFrame();

        if (success) {
            if (cur_frame_start_time - last_sim_tick_time >= sim_delta_t) {
                prev_key_state = key_state;
                UserInput user_input(key_state.data(), press_state.data());

                world_input_fn(world_input_data, vizCtrl.worldIdx, user_input);

                if (vizCtrl.controlIdx != 0) {
                    agent_input_fn(agent_input_data, vizCtrl.worldIdx,
                             vizCtrl.controlIdx - 1, user_input);
                }

                step_fn(step_data);

                last_sim_tick_time = cur_frame_start_time;
            }

            ui_fn(ui_data);

            render(frame_duration);

            frame_duration = throttleFPS(cur_frame_start_time);
        }
    }

    renderer.waitForIdle();
}

Viewer::Viewer(const render::RenderManager &render_mgr,
               const Window *window,
               const Config &cfg)
    : impl_(new Impl(render_mgr, window, cfg))
{}

Viewer::Viewer(Viewer &&o) = default;
Viewer::~Viewer() = default;

CountT Viewer::loadObjects(Span<const imp::SourceObject> objs,
                           Span<const imp::SourceMaterial> mats,
                           Span<const imp::SourceTexture> textures,
                           bool override_materials)
{
    return impl_->renderer.loadObjects(objs, mats, textures,
            override_materials);
}

void Viewer::configureLighting(Span<const render::LightConfig> lights)
{
    impl_->renderer.configureLighting(lights);
}

void Viewer::loop(
    void (*world_input_fn)(void *, CountT, const UserInput &),
    void *world_input_data,
    void (*agent_input_fn)(void *, CountT, CountT, const UserInput &),
    void *agent_input_data,
    void (*step_fn)(void *), void *step_data,
    void (*ui_fn)(void *), void *ui_data)
{
    impl_->loop(world_input_fn, world_input_data,
                agent_input_fn, agent_input_data,
                step_fn, step_data, ui_fn, ui_data);
}

void Viewer::stopLoop()
{
    impl_->shouldExit = true;
}

CountT Viewer::getCurrentWorldID() const
{
    return (CountT)impl_->vizCtrl.worldIdx;
}

CountT Viewer::getCurrentViewID() const
{
    return (CountT)impl_->vizCtrl.viewIdx - 1;
}

CountT Viewer::getCurrentControlID() const
{
    return (CountT)impl_->vizCtrl.viewIdx - 1;
}

void Viewer::toggleCameraTracking()
{
    impl_->vizCtrl.flyCam.isTracking = !impl_->vizCtrl.flyCam.isTracking;
    printf("Camera tracking %s (F key pressed)\n", impl_->vizCtrl.flyCam.isTracking ? "enabled" : "disabled");
}

void Viewer::setCameraPosition(const math::Vector3& pos)
{
    impl_->vizCtrl.flyCam.position = pos;
}

void Viewer::setCameraOrientation(const math::Quat& rot)
{
    // Convert quaternion to direction vectors
    impl_->vizCtrl.flyCam.fwd = rot.rotateVec(math::fwd);
    impl_->vizCtrl.flyCam.up = rot.rotateVec(math::up);
    impl_->vizCtrl.flyCam.right = rot.rotateVec(math::right);
}

void Viewer::setCameraLookAt(const math::Vector3& pos, const math::Vector3& target)
{
    impl_->vizCtrl.flyCam.position = pos;
    impl_->vizCtrl.flyCam.fwd = (target - pos).normalize();
    impl_->vizCtrl.flyCam.right = cross(impl_->vizCtrl.flyCam.fwd, math::up).normalize();
    impl_->vizCtrl.flyCam.up = cross(impl_->vizCtrl.flyCam.right, impl_->vizCtrl.flyCam.fwd).normalize();
}

void Viewer::setCameraVectors(const math::Vector3& pos, const math::Vector3& fwd, 
                              const math::Vector3& up, const math::Vector3& right)
{
    impl_->vizCtrl.flyCam.position = pos;
    impl_->vizCtrl.flyCam.fwd = fwd;
    impl_->vizCtrl.flyCam.up = up;
    impl_->vizCtrl.flyCam.right = right;
}

ViewerCam& Viewer::getCamera()
{
    return impl_->vizCtrl.flyCam;
}

const ViewerCam& Viewer::getCamera() const
{
    return impl_->vizCtrl.flyCam;
}

void Viewer::setMultiWorldGrid(bool enabled, float spacing, uint32_t gridCols, 
                              float worldScaleX, float worldScaleY)
{
    impl_->vizCtrl.multiWorldGrid = enabled;
    impl_->vizCtrl.worldSpacing = spacing;
    impl_->vizCtrl.gridCols = gridCols;
    impl_->vizCtrl.worldScaleX = worldScaleX;
    impl_->vizCtrl.worldScaleY = worldScaleY;
}

void Viewer::setMultiWorldGrid(bool enabled)
{
    impl_->vizCtrl.multiWorldGrid = enabled;
    // Keep existing values for other parameters if they were set before
}

void Viewer::setExploreMode(bool enabled)
{
    impl_->vizCtrl.exploreMode = enabled;
}

}
