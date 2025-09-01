#include "debug_capture.hpp"
#include <gtest/gtest.h>
#include <iostream>
#include <cstdlib>
#include <cstring>

namespace DebugCapture {

// Global flag to disable capture - defaults to false (capture enabled)
bool g_disable_capture = false;

// NOTE: Static initialization removed - we now check environment variable directly in header

} // namespace DebugCapture