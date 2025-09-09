#include "camera_controller.hpp"
#include "consts.hpp"
#include <cmath>
#include <algorithm>

namespace madEscape {

using namespace madrona::math;

// Helper functions
static Vector3 sphericalToCartesian(float azimuth, float elevation, float radius) {
    float azimuthRad = azimuth * consts::math::pi / 180.0f;
    float elevationRad = elevation * consts::math::pi / 180.0f;
    
    float cosElev = std::cos(elevationRad);
    return Vector3{
        radius * cosElev * std::sin(azimuthRad),
        radius * cosElev * std::cos(azimuthRad),
        radius * std::sin(elevationRad)
    };
}

// FreeFlyCameraController implementation
FreeFlyCameraController::FreeFlyCameraController() {
    reset();
}

void FreeFlyCameraController::handleInput(const CameraInputState& input, float deltaTime) {
    // Handle mouse rotation when right button is held
    if (input.mouseRightButton) {
        yaw_ -= input.mouseDeltaX * mouseSensitivity_;
        pitch_ -= input.mouseDeltaY * mouseSensitivity_;
        
        // Clamp pitch to prevent flipping
        pitch_ = std::clamp(pitch_, -89.0f * consts::math::pi / 180.0f, 
                           89.0f * consts::math::pi / 180.0f);
        
        updateVectors();
    }
    
    // Handle movement
    float speed = moveSpeed_;
    if (input.boost) {
        speed *= boostMultiplier_;
    }
    
    Vector3 movement = Vector3::zero();
    
    // W/S should move forward/backward in Y direction (into/out of screen)
    if (input.forward) {
        movement.y += 1.0f;  // Move +Y (forward into the scene)
    }
    if (input.backward) {
        movement.y -= 1.0f;  // Move -Y (backward)
    }
    
    // A/D should move left/right in X direction
    if (input.left) {
        movement.x -= 1.0f;  // Move -X (left)
    }
    if (input.right) {
        movement.x += 1.0f;  // Move +X (right)
    }
    
    // Q/E or other keys for up/down in Z direction
    if (input.up) {
        movement.z += 1.0f;  // Move +Z (up)
    }
    if (input.down) {
        movement.z -= 1.0f;  // Move -Z (down)
    }
    
    // Normalize diagonal movement
    float moveLength = movement.length();
    if (moveLength > 0.001f) {
        movement = movement / moveLength;
        state_.position += movement * speed * deltaTime;
    }
}

void FreeFlyCameraController::update(float /* deltaTime */) {
    // No automatic updates for free-fly camera
}

CameraState FreeFlyCameraController::getState() const {
    return state_;
}

void FreeFlyCameraController::reset() {
    state_.position = Vector3{0.0f, -14.0f, 35.0f};
    state_.forward = Vector3{0.0f, 0.0f, -1.0f};
    state_.up = Vector3{0.0f, 0.0f, 1.0f};
    state_.right = Vector3{1.0f, 0.0f, 0.0f};
    state_.fov = 60.0f;
    state_.perspective = true;
    
    // Calculate initial pitch and yaw from forward vector
    pitch_ = -40.0f * consts::math::pi / 180.0f;
    yaw_ = 0.0f;
    updateVectors();
}

void FreeFlyCameraController::setPosition(const Vector3& pos) {
    state_.position = pos;
}

void FreeFlyCameraController::setLookAt(const Vector3& target) {
    Vector3 direction = (target - state_.position).normalize();
    
    // Calculate pitch and yaw from direction
    pitch_ = std::asin(direction.z);
    yaw_ = std::atan2(direction.x, direction.y);
    
    updateVectors();
}

void FreeFlyCameraController::updateVectors() {
    // Calculate forward vector from pitch and yaw
    float cosPitch = std::cos(pitch_);
    state_.forward = Vector3{
        cosPitch * std::sin(yaw_),
        cosPitch * std::cos(yaw_),
        std::sin(pitch_)
    }.normalize();
    
    // Calculate right and up vectors
    Vector3 worldUp{0.0f, 0.0f, 1.0f};
    state_.right = cross(state_.forward, worldUp).normalize();
    state_.up = cross(state_.right, state_.forward).normalize();
}

// TrackingCameraController implementation
TrackingCameraController::TrackingCameraController() {
    reset();
}

void TrackingCameraController::handleInput(const CameraInputState& input, float deltaTime) {
    // Allow user to adjust viewing angle and distance
    if (input.mouseRightButton) {
        orbitAngle_ -= input.mouseDeltaX * 0.01f;
        height_ += input.mouseDeltaY * 0.1f;
        height_ = std::clamp(height_, 5.0f, 50.0f);
    }
    
    // Zoom with scroll or up/down
    if (input.up) {
        distance_ -= 10.0f * deltaTime;
    }
    if (input.down) {
        distance_ += 10.0f * deltaTime;
    }
    distance_ = std::clamp(distance_, 5.0f, 50.0f);
    
    // Adjust offset with movement keys
    if (input.forward) {
        offset_.y += 5.0f * deltaTime;
    }
    if (input.backward) {
        offset_.y -= 5.0f * deltaTime;
    }
    if (input.left) {
        offset_.x -= 5.0f * deltaTime;
    }
    if (input.right) {
        offset_.x += 5.0f * deltaTime;
    }
}

void TrackingCameraController::update(float /* deltaTime */) {
    updateCameraPosition();
}

CameraState TrackingCameraController::getState() const {
    return state_;
}

void TrackingCameraController::reset() {
    state_.position = Vector3{0.0f, -10.0f, 15.0f};
    state_.forward = Vector3{0.0f, 1.0f, -0.5f}.normalize();
    state_.up = Vector3{0.0f, 0.0f, 1.0f};
    state_.right = cross(state_.forward, state_.up).normalize();
    state_.fov = 60.0f;
    state_.perspective = true;
    
    targetPosition_ = Vector3::zero();
    offset_ = Vector3{0.0f, -10.0f, 15.0f};
    distance_ = 15.0f;
    height_ = 10.0f;
    orbitAngle_ = 0.0f;
}

void TrackingCameraController::setTarget(const Vector3& targetPos) {
    targetPosition_ = targetPos;
    updateCameraPosition();
}

void TrackingCameraController::updateCameraPosition() {
    // Calculate desired position based on target and offset
    Vector3 desiredPos = targetPosition_ + offset_;
    
    // Add orbit rotation
    if (std::abs(orbitAngle_) > 0.001f) {
        float cosAngle = std::cos(orbitAngle_);
        float sinAngle = std::sin(orbitAngle_);
        Vector3 relativePos = desiredPos - targetPosition_;
        desiredPos = targetPosition_ + Vector3{
            relativePos.x * cosAngle - relativePos.y * sinAngle,
            relativePos.x * sinAngle + relativePos.y * cosAngle,
            relativePos.z
        };
    }
    
    // Smooth camera movement
    state_.position = state_.position + (desiredPos - state_.position) * smoothingFactor_;
    
    // Look at target
    state_.forward = (targetPosition_ - state_.position).normalize();
    Vector3 worldUp{0.0f, 0.0f, 1.0f};
    state_.right = cross(state_.forward, worldUp).normalize();
    state_.up = cross(state_.right, state_.forward).normalize();
}

// OrbitCameraController implementation
OrbitCameraController::OrbitCameraController() {
    reset();
}

void OrbitCameraController::handleInput(const CameraInputState& input, float deltaTime) {
    // Orbit with mouse drag
    if (input.mouseRightButton) {
        azimuth_ -= input.mouseDeltaX * orbitSpeed_;
        elevation_ -= input.mouseDeltaY * orbitSpeed_;
        elevation_ = std::clamp(elevation_, -89.0f, 89.0f);
    }
    
    // Zoom with scroll or up/down
    if (input.up) {
        distance_ -= zoomSpeed_ * deltaTime * 5.0f;
    }
    if (input.down) {
        distance_ += zoomSpeed_ * deltaTime * 5.0f;
    }
    distance_ = std::clamp(distance_, 2.0f, 100.0f);
    
    // Pan with WASD
    Vector3 panMovement = Vector3::zero();
    if (input.forward) {
        panMovement += state_.forward;
    }
    if (input.backward) {
        panMovement -= state_.forward;
    }
    if (input.left) {
        panMovement -= state_.right;
    }
    if (input.right) {
        panMovement += state_.right;
    }
    
    if (panMovement.length() > 0.001f) {
        center_ += panMovement.normalize() * deltaTime * 10.0f;
    }
    
    updateCameraFromSpherical();
}

void OrbitCameraController::update(float /* deltaTime */) {
    // Could add automatic orbiting here if desired
}

CameraState OrbitCameraController::getState() const {
    return state_;
}

void OrbitCameraController::reset() {
    center_ = Vector3::zero();
    distance_ = 30.0f;
    azimuth_ = 45.0f;
    elevation_ = 30.0f;
    updateCameraFromSpherical();
}

void OrbitCameraController::setAngles(float azimuth, float elevation) {
    azimuth_ = azimuth;
    elevation_ = std::clamp(elevation, -89.0f, 89.0f);
    updateCameraFromSpherical();
}

void OrbitCameraController::updateCameraFromSpherical() {
    // Convert spherical coordinates to cartesian
    Vector3 offset = sphericalToCartesian(azimuth_, elevation_, distance_);
    state_.position = center_ + offset;
    
    // Look at center
    state_.forward = (center_ - state_.position).normalize();
    Vector3 worldUp{0.0f, 0.0f, 1.0f};
    state_.right = cross(state_.forward, worldUp).normalize();
    state_.up = cross(state_.right, state_.forward).normalize();
    
    state_.fov = 60.0f;
    state_.perspective = true;
}

// FixedCameraController implementation
FixedCameraController::FixedCameraController(Preset preset) 
    : currentPreset_(preset) {
    reset();
}

void FixedCameraController::handleInput(const CameraInputState& /* input */, float /* deltaTime */) {
    // Fixed camera doesn't respond to input
}

void FixedCameraController::update(float /* deltaTime */) {
    // No updates for fixed camera
}

CameraState FixedCameraController::getState() const {
    return state_;
}

void FixedCameraController::reset() {
    applyPreset();
}

void FixedCameraController::setPreset(Preset preset) {
    currentPreset_ = preset;
    applyPreset();
}

void FixedCameraController::setCustomPosition(const Vector3& pos, const Vector3& target) {
    currentPreset_ = CUSTOM;
    state_.position = pos;
    state_.forward = (target - pos).normalize();
    
    Vector3 worldUp{0.0f, 0.0f, 1.0f};
    state_.right = cross(state_.forward, worldUp).normalize();
    state_.up = cross(state_.right, state_.forward).normalize();
    
    state_.fov = 60.0f;
    state_.perspective = true;
}

void FixedCameraController::applyPreset() {
    switch (currentPreset_) {
        case TOP_DOWN:
            state_.position = Vector3{0.0f, 0.0f, 50.0f};
            state_.forward = Vector3{0.0f, 0.0f, -1.0f};
            state_.up = Vector3{0.0f, 1.0f, 0.0f};
            state_.right = Vector3{1.0f, 0.0f, 0.0f};
            state_.perspective = false;
            state_.orthoHeight = 40.0f;
            break;
            
        case FRONT:
            state_.position = Vector3{0.0f, -30.0f, 10.0f};
            state_.forward = Vector3{0.0f, 1.0f, 0.0f};
            state_.up = Vector3{0.0f, 0.0f, 1.0f};
            state_.right = Vector3{1.0f, 0.0f, 0.0f};
            state_.perspective = true;
            break;
            
        case SIDE:
            state_.position = Vector3{-30.0f, 0.0f, 10.0f};
            state_.forward = Vector3{1.0f, 0.0f, 0.0f};
            state_.up = Vector3{0.0f, 0.0f, 1.0f};
            state_.right = Vector3{0.0f, 1.0f, 0.0f};
            state_.perspective = true;
            break;
            
        case ISOMETRIC: {
            state_.position = Vector3{-20.0f, -20.0f, 30.0f};
            state_.forward = (Vector3::zero() - state_.position).normalize();
            Vector3 worldUp{0.0f, 0.0f, 1.0f};
            state_.right = cross(state_.forward, worldUp).normalize();
            state_.up = cross(state_.right, state_.forward).normalize();
            state_.perspective = true;
            break;
        }
            
        case CUSTOM:
            // Keep current custom settings
            break;
    }
    
    state_.fov = 60.0f;
}

} // namespace madEscape