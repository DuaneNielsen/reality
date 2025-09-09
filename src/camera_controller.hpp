#pragma once

#include <madrona/math.hpp>
#include <madrona/viz/viewer.hpp>

namespace madEscape {

// Input state for camera controllers
struct CameraInputState {
    bool forward = false;
    bool backward = false;
    bool left = false;
    bool right = false;
    bool up = false;
    bool down = false;
    bool rotateLeft = false;
    bool rotateRight = false;
    bool boost = false;
    float mouseX = 0.0f;
    float mouseY = 0.0f;
    float mouseDeltaX = 0.0f;
    float mouseDeltaY = 0.0f;
    bool mouseRightButton = false;
};

// Camera state that gets applied to the viewer
struct CameraState {
    madrona::math::Vector3 position;
    madrona::math::Vector3 forward;
    madrona::math::Vector3 up;
    madrona::math::Vector3 right;
    float fov = 60.0f;
    bool perspective = true;
    float orthoHeight = 5.0f;
};

// Base interface for all camera controllers
class CameraController {
public:
    virtual ~CameraController() = default;
    
    // Handle input and update camera state
    virtual void handleInput(const CameraInputState& input, float deltaTime) = 0;
    
    // Update camera (for animations, tracking, etc.)
    virtual void update(float deltaTime) = 0;
    
    // Get current camera state
    virtual CameraState getState() const = 0;
    
    // Reset camera to default position
    virtual void reset() = 0;
    
    // Get controller type name for UI
    virtual const char* getName() const = 0;
};

// Free-fly camera controller (traditional FPS-style)
class FreeFlyCameraController : public CameraController {
public:
    FreeFlyCameraController();
    
    void handleInput(const CameraInputState& input, float deltaTime) override;
    void update(float deltaTime) override;
    CameraState getState() const override;
    void reset() override;
    const char* getName() const override { return "Free Fly"; }
    
    // Configuration
    void setSpeed(float speed) { moveSpeed_ = speed; }
    void setMouseSensitivity(float sensitivity) { mouseSensitivity_ = sensitivity; }
    void setPosition(const madrona::math::Vector3& pos);
    void setLookAt(const madrona::math::Vector3& target);
    
private:
    CameraState state_;
    float moveSpeed_ = 10.0f;
    float boostMultiplier_ = 3.0f;
    float mouseSensitivity_ = 0.002f;
    float pitch_ = 0.0f;
    float yaw_ = 0.0f;
    
    void updateVectors();
};

// Tracking camera controller (follows an entity)
class TrackingCameraController : public CameraController {
public:
    TrackingCameraController();
    
    void handleInput(const CameraInputState& input, float deltaTime) override;
    void update(float deltaTime) override;
    CameraState getState() const override;
    void reset() override;
    const char* getName() const override { return "Tracking"; }
    
    // Set the target to track
    void setTarget(const madrona::math::Vector3& targetPos);
    
    // Configuration
    void setOffset(const madrona::math::Vector3& offset) { offset_ = offset; }
    void setSmoothingFactor(float factor) { smoothingFactor_ = factor; }
    void setDistance(float distance) { distance_ = distance; }
    void setHeight(float height) { height_ = height; }
    
private:
    CameraState state_;
    madrona::math::Vector3 targetPosition_;
    madrona::math::Vector3 offset_{0.0f, -10.0f, 15.0f};  // Default offset
    float smoothingFactor_ = 0.1f;
    float distance_ = 15.0f;
    float height_ = 10.0f;
    float orbitAngle_ = 0.0f;
    
    void updateCameraPosition();
};

// Orbit camera controller (orbits around a point)
class OrbitCameraController : public CameraController {
public:
    OrbitCameraController();
    
    void handleInput(const CameraInputState& input, float deltaTime) override;
    void update(float deltaTime) override;
    CameraState getState() const override;
    void reset() override;
    const char* getName() const override { return "Orbit"; }
    
    // Set the center point to orbit around
    void setCenter(const madrona::math::Vector3& center) { center_ = center; }
    void setDistance(float distance) { distance_ = distance; }
    void setAngles(float azimuth, float elevation);
    
private:
    CameraState state_;
    madrona::math::Vector3 center_{0.0f, 0.0f, 0.0f};
    float distance_ = 20.0f;
    float azimuth_ = 0.0f;      // Horizontal angle
    float elevation_ = 45.0f;    // Vertical angle
    float orbitSpeed_ = 1.0f;
    float zoomSpeed_ = 2.0f;
    
    void updateCameraFromSpherical();
};

// Fixed camera controller (static positions)
class FixedCameraController : public CameraController {
public:
    enum Preset {
        TOP_DOWN,
        FRONT,
        SIDE,
        ISOMETRIC,
        CUSTOM
    };
    
    FixedCameraController(Preset preset = TOP_DOWN);
    
    void handleInput(const CameraInputState& input, float deltaTime) override;
    void update(float deltaTime) override;
    CameraState getState() const override;
    void reset() override;
    const char* getName() const override { return "Fixed"; }
    
    // Set preset or custom position
    void setPreset(Preset preset);
    void setCustomPosition(const madrona::math::Vector3& pos, const madrona::math::Vector3& target);
    
private:
    CameraState state_;
    Preset currentPreset_;
    
    void applyPreset();
};

} // namespace madEscape