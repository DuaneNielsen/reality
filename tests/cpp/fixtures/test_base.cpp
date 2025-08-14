#include "test_base.hpp"

// Static member definitions for GPU test fixture
bool MadronaGPUTest::cuda_available_checked = false;
bool MadronaGPUTest::cuda_available = false;
std::mutex MadronaGPUTest::gpu_test_mutex;