#include <iostream>
#include <chrono>

#include <madrona/mw_cpu.hpp>
#include <madrona/taskgraph_builder.hpp>
#include <madrona/custom_context.hpp>
#include <madrona/ecs.hpp>

#ifdef MADRONA_ECS_DEBUG_TRACKING
extern "C" {
    void simple_tracker_print_statistics();
    void simple_tracker_print_memory_map();
    uint32_t simple_tracker_get_range_count();
    void simple_tracker_dump_ranges();

    // Address lookup functionality
    typedef struct {
        uint32_t component_id;
        uint32_t archetype_id;
        uint32_t world_id;
        uint32_t column_idx;
        uint32_t row;
        uint32_t component_size;
        void* column_base;
        char component_name[64];
        char archetype_name[64];
        char formatted_value[256];  // Formatted component value
    } address_info_t;

    int simple_tracker_lookup(void* address, address_info_t* info);
    void simple_tracker_register_component_type(
        uint32_t component_id, const char* type_name, uint32_t size, uint32_t alignment);
    void simple_tracker_register_archetype_type(
        uint32_t archetype_id, const char* archetype_name);
    const char* simple_tracker_format_component_value(void* address);
    void simple_tracker_print_component_value(void* address);
    void simple_tracker_print_memory_map();  // Add this to see detailed component info
}
#endif

namespace TestECS {

// Task graph IDs
enum class TaskGraphID : uint32_t {
    Step,
    NumTaskGraphs,
};

// Export IDs for tensor exports
enum class ExportID : uint32_t {
    HealthComponent,
    PositionComponent,
    NumExports,
};

// Health component for game entities
struct HealthComponent {
    float currentHealth;
};

// Position component for testing default formatter (no specialization)
struct Position {
    float x, y, z;
};

// Game entity with both health and position components
struct GameEntity : public madrona::Archetype<HealthComponent, Position> {};

// Minimal config (like Sim::Config)
struct Config {
    bool enableDebug;
};

// Minimal world init (like Sim::WorldInit)
struct WorldInit {
    uint32_t worldID;
};

// Forward declare Engine for Sim
class Engine;

// The Sim class equivalent - must inherit from WorldBase
class Sim : public madrona::WorldBase {
public:
    // Required static registration function
    static void registerTypes(madrona::ECSRegistry &registry,
                             const Config &cfg);

    // Required static task setup function
    static void setupTasks(madrona::TaskGraphManager &mgr,
                          const Config &cfg);

    // Constructor called for each world
    Sim(Engine &ctx, const Config &cfg, const WorldInit &init);

    bool enableDebug;
    uint32_t worldID;
};

// The Engine class - context for ECS operations
class Engine : public madrona::CustomContext<Engine, Sim> {
public:
    using CustomContext::CustomContext;

    // Health regeneration system that operates on HealthComponent
    static void healthRegenTask(Engine &ctx, HealthComponent &health) {
        (void)ctx; // Suppress unused parameter warning
        // The ParallelForNode will iterate over all entities with HealthComponent
        // and regenerate health by 1.0 per frame
        health.currentHealth += 1.0f;
    }
};

// Implementation of static methods
void Sim::registerTypes(madrona::ECSRegistry &registry, const Config &cfg) {
    (void)cfg; // Suppress unused parameter warning

    // Register both component types
    registry.registerComponent<HealthComponent>();
    registry.registerComponent<Position>();

    // Then register the archetype that uses them
    registry.registerArchetype<GameEntity>();

    // Export components as tensors for testing
    registry.exportColumn<GameEntity, HealthComponent>(
        (uint32_t)TestECS::ExportID::HealthComponent);
    registry.exportColumn<GameEntity, Position>(
        (uint32_t)TestECS::ExportID::PositionComponent);
}

void Sim::setupTasks(madrona::TaskGraphManager &mgr, const Config &cfg) {
    (void)cfg; // Suppress unused parameter warning
    madrona::TaskGraphBuilder &builder = mgr.init(TaskGraphID::Step);

    // Health regeneration task that processes HealthComponent
    (void)builder.addToGraph<madrona::ParallelForNode<Engine,
        Engine::healthRegenTask, HealthComponent>>({});
}

Sim::Sim(Engine &ctx, const Config &cfg, const WorldInit &init)
    : madrona::WorldBase(ctx), enableDebug(cfg.enableDebug), worldID(init.worldID) {
    // Initialize world state
    // Create game entities with health and position - this will trigger table allocation
    auto player = ctx.makeEntity<GameEntity>();
    auto &playerHealth = ctx.get<HealthComponent>(player);
    playerHealth.currentHealth = 100.0f;
    auto &playerPos = ctx.get<Position>(player);
    playerPos.x = 5.0f; playerPos.y = 10.0f; playerPos.z = 15.0f;

    auto enemy = ctx.makeEntity<GameEntity>();
    auto &enemyHealth = ctx.get<HealthComponent>(enemy);
    enemyHealth.currentHealth = 75.0f;
    auto &enemyPos = ctx.get<Position>(enemy);
    enemyPos.x = -3.5f; enemyPos.y = 7.2f; enemyPos.z = -1.0f;

    // Component addresses will be tracked automatically by the debug system
    // Reverse lookup will be demonstrated after table setup is complete
}

}

// Specialized formatter for HealthComponent - must be outside namespace
#ifdef MADRONA_ECS_DEBUG_TRACKING
template<>
const char* format_component_default<TestECS::HealthComponent>(const void* ptr, const void* info_ptr) {
    static thread_local char buffer[128];
    const auto& health = *static_cast<const TestECS::HealthComponent*>(ptr);
    const auto* info = static_cast<const address_info_t*>(info_ptr);
    snprintf(buffer, sizeof(buffer), "%s(%u)-%s(%u) : HealthComponent{currentHealth=%.1f}",
             info->archetype_name, info->archetype_id,
             info->component_name, info->component_id, health.currentHealth);
    return buffer;
}
#endif

int main() {
    std::cout << "=== Simple ECS Table Test ===" << std::endl;

#ifdef MADRONA_ECS_DEBUG_TRACKING
    std::cout << "Debug tracking is ENABLED" << std::endl;

    // Check initial state
    std::cout << "\n=== Initial State ===" << std::endl;
    simple_tracker_print_statistics();
    uint32_t initial_ranges = simple_tracker_get_range_count();

    auto start = std::chrono::steady_clock::now();

    {
        std::cout << "\n=== Creating TaskGraphExecutor ===" << std::endl;

        // Setup world initialization data
        TestECS::WorldInit world_init;
        world_init.worldID = 0;

        // Create the minimal config
        TestECS::Config cfg;
        cfg.enableDebug = true;

        // Create executor following the real pattern
        using TestTaskGraph = madrona::TaskGraphExecutor<
            TestECS::Engine,       // Context type
            TestECS::Sim,          // World data type
            TestECS::Config,       // Config type
            TestECS::WorldInit     // World init type
        >;

        TestTaskGraph exec({
            .numWorlds = 1,          // Single world for simplicity
            .numExportedBuffers = (uint32_t)TestECS::ExportID::NumExports, // Export health and position tensors
            .numWorkers = 1          // Single worker thread
        }, cfg, &world_init, 1);     // 1 taskgraph

        std::cout << "TaskGraphExecutor created successfully!" << std::endl;

        // Run one iteration to trigger ECS table creation
        std::cout << "\n=== Running Task Graph ===" << std::endl;
        exec.runTaskGraph(0);
        std::cout << "Task graph execution completed!" << std::endl;

        // Show what component types the ECS system actually registered
        std::cout << "\n=== ECS System Component Registration ===" << std::endl;
        simple_tracker_print_memory_map();

        // Type names are now automatically registered during ECS setup!
        std::cout << "Type names now registered automatically during ECS setup!" << std::endl;

        // Check what we captured after executor creation and execution
        std::cout << "\n=== After Task Execution ===" << std::endl;
        simple_tracker_print_statistics();
        uint32_t after_execution_ranges = simple_tracker_get_range_count();

        std::cout << "Ranges before: " << initial_ranges << std::endl;
        std::cout << "Ranges after: " << after_execution_ranges << std::endl;

        if (after_execution_ranges > initial_ranges) {
            uint32_t new_ranges = after_execution_ranges - initial_ranges;
            std::cout << "SUCCESS: Captured " << new_ranges << " new table allocations!" << std::endl;

            // Show the captured ranges
            std::cout << "\n=== Captured Table Memory Layout ===" << std::endl;
            simple_tracker_dump_ranges();

            // Now demonstrate ACTUAL reverse lookup using REAL component queries
            std::cout << "\n=== LIVE Reverse Lookup with Component Query ===" << std::endl;
            std::cout << "Querying TestComponents and testing reverse lookup on actual addresses...\n" << std::endl;

            // Get the world context to run queries
            auto& world_ctx = exec.getWorldContext(0);

            // Query all HealthComponents in the world
            auto query = world_ctx.query<TestECS::HealthComponent>();

            std::cout << "Running query for HealthComponent..." << std::endl;

            int component_count = 0;
            world_ctx.iterateQuery(query, [&](TestECS::HealthComponent &health) {
                component_count++;

                // Get the actual address of this component
                void* comp_addr = &health;

                std::cout << "\n🔍 Testing component #" << component_count << ":" << std::endl;
                std::cout << "  Address: " << comp_addr << std::endl;
                std::cout << "  Health: " << health.currentHealth << std::endl;
                std::cout << "  Formatted Value: " << simple_tracker_format_component_value(comp_addr) << std::endl;

                // Perform reverse lookup on this real component address
                address_info_t lookup_info;
                int found = simple_tracker_lookup(comp_addr, &lookup_info);

                if (found) {
                    std::cout << "  ✅ REVERSE LOOKUP SUCCESS:" << std::endl;
                    std::cout << "    Component: " << lookup_info.component_name
                              << " (ID: " << lookup_info.component_id << ")" << std::endl;
                    std::cout << "    Value: " << lookup_info.formatted_value << std::endl;  // ✨ NEW!
                    std::cout << "    Archetype: " << lookup_info.archetype_name
                              << " (ID: " << lookup_info.archetype_id << ")" << std::endl;
                    std::cout << "    World ID: " << lookup_info.world_id << std::endl;
                    std::cout << "    Column Index: " << lookup_info.column_idx << std::endl;
                    std::cout << "    Row: " << lookup_info.row << std::endl;
                    std::cout << "    Component Size: " << lookup_info.component_size << " bytes" << std::endl;
                    std::cout << "    Column Base: " << lookup_info.column_base << std::endl;
                } else {
                    std::cout << "  ❌ REVERSE LOOKUP FAILED (code: " << found << ")" << std::endl;
                }
            });

            std::cout << "\nQuery completed. Found " << component_count << " HealthComponent instances." << std::endl;

            // Now test Position components (using default formatter)
            std::cout << "\n=== Testing Default Formatter with Position Components ===" << std::endl;
            auto position_query = world_ctx.query<TestECS::Position>();

            std::cout << "Running query for Position..." << std::endl;

            int position_count = 0;
            world_ctx.iterateQuery(position_query, [&](TestECS::Position &pos) {
                position_count++;

                // Get the actual address of this component
                void* comp_addr = &pos;

                std::cout << "\n🔍 Testing Position #" << position_count << ":" << std::endl;
                std::cout << "  Address: " << comp_addr << std::endl;
                std::cout << "  Position: x=" << pos.x << ", y=" << pos.y << ", z=" << pos.z << std::endl;
                std::cout << "  Formatted Value: " << simple_tracker_format_component_value(comp_addr) << std::endl;

                // Perform reverse lookup on this real component address
                address_info_t lookup_info;
                int found = simple_tracker_lookup(comp_addr, &lookup_info);

                if (found) {
                    std::cout << "  ✅ REVERSE LOOKUP SUCCESS:" << std::endl;
                    std::cout << "    Component: " << lookup_info.component_name
                              << " (ID: " << lookup_info.component_id << ")" << std::endl;
                    std::cout << "    Value: " << lookup_info.formatted_value << std::endl;
                    std::cout << "    Archetype: " << lookup_info.archetype_name
                              << " (ID: " << lookup_info.archetype_id << ")" << std::endl;
                } else {
                    std::cout << "  ❌ REVERSE LOOKUP FAILED (code: " << found << ")" << std::endl;
                }
            });

            std::cout << "\nPosition query completed. Found " << position_count << " Position instances." << std::endl;

            // Test tensor exports
            std::cout << "\n=== Testing Tensor Exports ===" << std::endl;

            // Get exported tensor data
            void* health_data = exec.getExported((uint32_t)TestECS::ExportID::HealthComponent);
            void* position_data = exec.getExported((uint32_t)TestECS::ExportID::PositionComponent);

            std::cout << "Health tensor data pointer: " << health_data << std::endl;
            std::cout << "Position tensor data pointer: " << position_data << std::endl;

            if (health_data && position_data) {
                // Cast to appropriate types and verify data
                float* health_tensor = static_cast<float*>(health_data);
                float* position_tensor = static_cast<float*>(position_data);

                std::cout << "\n📊 Health Tensor Data:" << std::endl;
                for (int i = 0; i < 2; i++) {  // We have 2 entities
                    std::cout << "  Entity " << i << " Health: " << health_tensor[i] << std::endl;
                }

                std::cout << "\n📊 Position Tensor Data:" << std::endl;
                for (int i = 0; i < 2; i++) {  // We have 2 entities
                    std::cout << "  Entity " << i << " Position: x=" << position_tensor[i*3]
                              << ", y=" << position_tensor[i*3+1]
                              << ", z=" << position_tensor[i*3+2] << std::endl;
                }

                // Verify the values match what we set
                bool health_correct = (health_tensor[0] == 101.0f && health_tensor[1] == 76.0f);
                bool position_correct = (position_tensor[0] == 5.0f && position_tensor[1] == 10.0f && position_tensor[2] == 15.0f &&
                                       position_tensor[3] == -3.5f && position_tensor[4] == 7.2f && position_tensor[5] == -1.0f);

                if (health_correct && position_correct) {
                    std::cout << "\n✅ TENSOR VERIFICATION SUCCESS: All values match expected data!" << std::endl;
                } else {
                    std::cout << "\n❌ TENSOR VERIFICATION FAILED: Values don't match expected data!" << std::endl;
                    if (!health_correct) {
                        std::cout << "  Health mismatch: expected [101.0, 76.0], got [" << health_tensor[0] << ", " << health_tensor[1] << "]" << std::endl;
                    }
                    if (!position_correct) {
                        std::cout << "  Position mismatch detected" << std::endl;
                    }
                }
            } else {
                std::cout << "\n❌ TENSOR EXPORT FAILED: Null pointers returned!" << std::endl;
            }

            // Also test an invalid address for comparison
            std::cout << "\n🔍 Testing invalid address for comparison:" << std::endl;
            void* invalid_addr = (void*)0x12345678;
            std::cout << "  Address: " << invalid_addr << std::endl;

            address_info_t invalid_lookup;
            int found = simple_tracker_lookup(invalid_addr, &invalid_lookup);

            if (found) {
                std::cout << "  ⚠️  UNEXPECTED SUCCESS on invalid address!" << std::endl;
            } else {
                std::cout << "  ✅ CORRECTLY FAILED on invalid address (code: " << found << ")" << std::endl;
            }

        } else {
            std::cout << "WARNING: No new ranges captured!" << std::endl;
        }

        std::cout << "\n=== Executor Going Out of Scope ===" << std::endl;
    }

    // Check final state
    std::cout << "\n=== After Executor Destruction ===" << std::endl;
    simple_tracker_print_statistics();
    uint32_t final_ranges = simple_tracker_get_range_count();
    std::cout << "Final ranges: " << final_ranges << std::endl;

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "\nTotal test time: " << duration.count() << "ms" << std::endl;

    if (duration.count() > 500) {
        std::cout << "WARNING: Test took too long" << std::endl;
        return 1;
    } else {
        std::cout << "SUCCESS: Performance is good" << std::endl;
    }

#else
    std::cout << "Debug tracking is DISABLED" << std::endl;
#endif

    return 0;
}