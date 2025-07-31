#include <stdio.h>
#include <stdlib.h>
#include "madrona_escape_room_c_api.h"

int main() {
    printf("Testing C wrapper directly...\n");
    
    // Test 1: CPU Manager
    printf("\n1. Creating CPU manager...\n");
    {
        MER_ManagerConfig config = {};
        config.exec_mode = MER_EXEC_MODE_CPU;
        config.gpu_id = 0;
        config.num_worlds = 4;
        config.rand_seed = 42;
        config.auto_reset = true;
        config.enable_batch_renderer = false;
        
        MER_ManagerHandle handle = nullptr;
        MER_Result result = mer_create_manager(&handle, &config);
        
        if (result != MER_SUCCESS) {
            fprintf(stderr, "Failed to create CPU manager: %s\n", mer_result_to_string(result));
            return 1;
        }
        
        printf("  CPU manager created successfully!\n");
        
        // Test tensor access
        MER_Tensor action_tensor;
        result = mer_get_action_tensor(handle, &action_tensor);
        if (result != MER_SUCCESS) {
            fprintf(stderr, "Failed to get action tensor: %s\n", mer_result_to_string(result));
        } else {
            printf("  Got action tensor - shape: [");
            for (int i = 0; i < action_tensor.num_dimensions; i++) {
                if (i > 0) printf(", ");
                printf("%lld", (long long)action_tensor.dimensions[i]);
            }
            printf("]\n");
        }
        
        mer_destroy_manager(handle);
        printf("  CPU manager destroyed\n");
    }
    
    // Test 2: GPU Manager
    printf("\n2. Creating GPU manager...\n");
    {
        MER_ManagerConfig config = {};
        config.exec_mode = MER_EXEC_MODE_CUDA;
        config.gpu_id = 0;
        config.num_worlds = 4;
        config.rand_seed = 42;
        config.auto_reset = true;
        config.enable_batch_renderer = false;
        
        MER_ManagerHandle handle = nullptr;
        MER_Result result = mer_create_manager(&handle, &config);
        
        if (result != MER_SUCCESS) {
            fprintf(stderr, "Failed to create GPU manager: %s\n", mer_result_to_string(result));
            return 1;
        }
        
        printf("  GPU manager created successfully!\n");
        
        // Test tensor access
        MER_Tensor action_tensor;
        result = mer_get_action_tensor(handle, &action_tensor);
        if (result != MER_SUCCESS) {
            fprintf(stderr, "Failed to get action tensor: %s\n", mer_result_to_string(result));
        } else {
            printf("  Got action tensor - GPU ID: %d, shape: [", action_tensor.gpu_id);
            for (int i = 0; i < action_tensor.num_dimensions; i++) {
                if (i > 0) printf(", ");
                printf("%lld", (long long)action_tensor.dimensions[i]);
            }
            printf("]\n");
        }
        
        mer_destroy_manager(handle);
        printf("  GPU manager destroyed\n");
    }
    
    printf("\n3. All tests passed!\n");
    return 0;
}