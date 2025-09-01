#pragma once

#include <gtest/gtest.h>
#include <string>
#include <cstdlib>
#include <cstring>
#include <iostream>

// For capturing stdout/stderr output in tests - but with debug support
using testing::internal::CaptureStdout;
using testing::internal::GetCapturedStdout;

namespace DebugCapture {

// Custom flag to disable capture for debugging
extern bool g_disable_capture;

// Wrapper that can be disabled for debugging
inline void CaptureStdoutDebug() {
    // Check environment variable directly each time
    const char* disable_env = std::getenv("GTEST_DISABLE_CAPTURE");
    if (disable_env && (strcmp(disable_env, "1") == 0 || strcmp(disable_env, "true") == 0)) {
        std::cout << "DEBUG: Stdout capture disabled via GTEST_DISABLE_CAPTURE\n";
        return; // Skip capture - output goes to terminal
    }
    CaptureStdout(); // Normal capture
}

// Wrapper that handles the case when capture was disabled
inline std::string GetCapturedStdoutDebug() {
    const char* disable_env = std::getenv("GTEST_DISABLE_CAPTURE");
    if (disable_env && (strcmp(disable_env, "1") == 0 || strcmp(disable_env, "true") == 0)) {
        return "[Capture was disabled for debugging]";
    }
    return GetCapturedStdout();
}

} // namespace DebugCapture