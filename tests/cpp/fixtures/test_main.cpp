#include <gtest/gtest.h>
#include <iostream>
#include <cstring>
#include "debug_capture.hpp"

int main(int argc, char** argv) {
    // Initialize GoogleTest first - this removes GoogleTest's own flags from argv
    testing::InitGoogleTest(&argc, argv);
    
    // Process remaining custom flags
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--disable-capture") == 0) {
            DebugCapture::g_disable_capture = true;
            std::cout << "INFO: Stdout capture disabled via --disable-capture flag\n";
        } else if (strcmp(argv[i], "--help") == 0) {
            std::cout << "\nCustom flags:\n";
            std::cout << "  --disable-capture    Disable stdout capture for debug output visibility\n";
            std::cout << "\nEnvironment variables:\n";
            std::cout << "  GTEST_DISABLE_CAPTURE=1    Alternative way to disable stdout capture\n\n";
        }
    }
    
    return RUN_ALL_TESTS();
}