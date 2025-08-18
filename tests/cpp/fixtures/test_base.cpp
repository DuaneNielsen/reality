#include "test_base.hpp"
#include "cpp_test_base.hpp"

// Static member definitions for GPU test fixture
bool MadronaGPUTest::cuda_available_checked = false;
bool MadronaGPUTest::cuda_available = false;
std::mutex MadronaGPUTest::gpu_test_mutex;

// Static member definition for MadronaCppGPUTest
namespace madEscape {
std::mutex MadronaCppGPUTest::gpu_test_mutex;
}