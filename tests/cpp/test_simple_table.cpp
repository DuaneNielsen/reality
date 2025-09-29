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
    } address_info_t;

    int simple_tracker_lookup(void* address, address_info_t* info);
    void simple_tracker_register_component_type(
        uint32_t component_id, const char* type_name, uint32_t size, uint32_t alignment);
    void simple_tracker_register_archetype_type(
        uint32_t archetype_id, const char* archetype_name);
    void simple_tracker_print_memory_map();  // Add this to see detailed component info
}
#endif

namespace TestECS {

// Task graph IDs
enum class TaskGraphID : uint32_t {
    Step,
    NumTaskGraphs,
};

// Health component for game entities
struct HealthComponent {
    float currentHealth;
};

// Simple game entity with health component
struct GameEntity : public madrona::Archetype<HealthComponent> {};

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

    // First register the component type
    registry.registerComponent<HealthComponent>();

    // Then register the archetype that uses it
    registry.registerArchetype<GameEntity>();
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
    // Create game entities with health - this will trigger table allocation
    auto player = ctx.makeEntity<GameEntity>();
    auto &playerHealth = ctx.get<HealthComponent>(player);
    playerHealth.currentHealth = 100.0f;

    auto enemy = ctx.makeEntity<GameEntity>();
    auto &enemyHealth = ctx.get<HealthComponent>(enemy);
    enemyHealth.currentHealth = 75.0f;

    // Component addresses will be tracked automatically by the debug system
    // Reverse lookup will be demonstrated after table setup is complete
}

}

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
            .numExportedBuffers = 0, // No exports needed for test
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

                // Perform reverse lookup on this real component address
                address_info_t lookup_info;
                int found = simple_tracker_lookup(comp_addr, &lookup_info);

                if (found) {
                    std::cout << "  ✅ REVERSE LOOKUP SUCCESS:" << std::endl;
                    std::cout << "    Component: " << lookup_info.component_name
                              << " (ID: " << lookup_info.component_id << ")" << std::endl;
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