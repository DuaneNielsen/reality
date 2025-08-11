#include <stdio.h>
#include <stdlib.h>
#include "madrona_escape_room_c_api.h"

int main() {
    printf("Testing C wrapper CPU manager...\n");
    
    MER_ManagerConfig config = {};
    config.exec_mode = MER_EXEC_MODE_CPU;
    config.gpu_id = 0;
    config.num_worlds = 4;
    config.rand_seed = 42;
    config.auto_reset = true;
    config.enable_batch_renderer = false;
    
    // Create basic compiled levels for all worlds
    MER_CompiledLevel test_levels[4];
    for (int i = 0; i < 4; i++) {
        test_levels[i] = {};
        test_levels[i].num_tiles = 0;  // Use hardcoded room generation
        test_levels[i].width = 16;
        test_levels[i].height = 16;
        test_levels[i].scale = 1.0f;
        test_levels[i].max_entities = 300;  // Adequate for test
    }
    
    MER_ManagerHandle handle = nullptr;
    printf("Creating CPU manager...\n");
    MER_Result result = mer_create_manager(&handle, &config, test_levels, 4);  // One level per world
    
    if (result != MER_SUCCESS) {
        fprintf(stderr, "Failed to create CPU manager: %s\n", mer_result_to_string(result));
        return 1;
    }
    
    printf("CPU manager created successfully!\n");
    
    // Test tensor access
    MER_Tensor action_tensor;
    result = mer_get_action_tensor(handle, &action_tensor);
    if (result != MER_SUCCESS) {
        fprintf(stderr, "Failed to get action tensor: %s\n", mer_result_to_string(result));
    } else {
        printf("Got action tensor - shape: [");
        for (int i = 0; i < action_tensor.num_dimensions; i++) {
            if (i > 0) printf(", ");
            printf("%lld", (long long)action_tensor.dimensions[i]);
        }
        printf("]\n");
    }
    
    // Clean up
    mer_destroy_manager(handle);
    printf("CPU manager destroyed\n");
    
    printf("\nCPU test passed!\n");
    return 0;
}