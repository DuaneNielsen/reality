#include "mgr.hpp"
#include "sim.hpp"
#include "asset_ids.hpp"
#include "asset_registry.hpp"

#include <chrono>

/*
 * This file uses a three-tier classification system for code:
 * 
 * [BOILERPLATE] - Pure Madrona framework code that never changes
 * [REQUIRED_INTERFACE] - Methods/structures every environment must implement
 * [GAME_SPECIFIC] - Implementation details unique to this escape room game
 * 
 * When creating a new environment, focus on:
 * - Implementing all [REQUIRED_INTERFACE] methods with your game's content
 * - Replacing all [GAME_SPECIFIC] code with your game's logic
 * - Leave all [BOILERPLATE] code unchanged
 */

#include <madrona/utils.hpp>
#include <madrona/importer.hpp>
#include <madrona/physics_loader.hpp>
#include <madrona/tracing.hpp>
#include <madrona/mw_cpu.hpp>
#include <madrona/render/api.hpp>

#include <array>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>

#ifdef MADRONA_CUDA_SUPPORT
#include <madrona/mw_gpu.hpp>
#include <madrona/cuda_utils.hpp>
#endif

using namespace madrona;
using namespace madrona::math;
using namespace madrona::phys;
using namespace madrona::py;


namespace madEscape {

// ============================================================================
// BOILERPLATE: Rendering Infrastructure (Required by Madrona)
// ============================================================================

// [BOILERPLATE] GPU rendering state container required by Madrona
struct RenderGPUState {
    render::APILibHandle apiLib;
    render::APIManager apiMgr;
    render::GPUHandle gpu;
};

// [BOILERPLATE] Initialize GPU state for rendering - standard Madrona pattern
static inline Optional<RenderGPUState> initRenderGPUState(
    const Manager::Config &mgr_cfg)
{
    if (mgr_cfg.extRenderDev || !mgr_cfg.enableBatchRenderer) {
        return Optional<RenderGPUState>::none();
    }

    auto render_api_lib = render::APIManager::loadDefaultLib();
    render::APIManager render_api_mgr(render_api_lib.lib());
    render::GPUHandle gpu = render_api_mgr.initGPU(mgr_cfg.gpuID);

    return RenderGPUState {
        .apiLib = std::move(render_api_lib),
        .apiMgr = std::move(render_api_mgr),
        .gpu = std::move(gpu),
    };
}

// [BOILERPLATE] Initialize render manager - uses game constants: numAgents, numRows, numCols
static inline Optional<render::RenderManager> initRenderManager(
    const Manager::Config &mgr_cfg,
    const Optional<RenderGPUState> &render_gpu_state)
{
    if (!mgr_cfg.extRenderDev && !mgr_cfg.enableBatchRenderer) {
        return Optional<render::RenderManager>::none();
    }

    render::APIBackend *render_api;
    render::GPUDevice *render_dev;

    if (render_gpu_state.has_value()) {
        render_api = render_gpu_state->apiMgr.backend();
        render_dev = render_gpu_state->gpu.device();
    } else {
        render_api = mgr_cfg.extRenderAPI;
        render_dev = mgr_cfg.extRenderDev;
    }

    return render::RenderManager(render_api, render_dev, {
        .enableBatchRenderer = mgr_cfg.enableBatchRenderer,
        .renderMode = static_cast<render::RenderManager::Config::RenderMode>(mgr_cfg.renderMode),
        .agentViewWidth = mgr_cfg.batchRenderViewWidth,
        .agentViewHeight = mgr_cfg.batchRenderViewHeight,
        .numWorlds = mgr_cfg.numWorlds,
        .maxViewsPerWorld = madEscape::consts::numAgents,
        .maxInstancesPerWorld = madEscape::consts::performance::maxProgressEntries,           // [GAME_SPECIFIC] Max entities to render
        .execMode = mgr_cfg.execMode,
        .voxelCfg = {},
    });
}

// ============================================================================
// BOILERPLATE: Manager Implementation Base Class
// ============================================================================

// [BOILERPLATE] Base implementation class
struct Manager::Impl {
    Config cfg;
    PhysicsLoader physicsLoader;
    WorldReset *worldResetBuffer;    // [REQUIRED_INTERFACE] Reset control buffer
    Action *agentActionsBuffer;      // [REQUIRED_INTERFACE] Agent action buffer
    Optional<RenderGPUState> renderGPUState;
    Optional<render::RenderManager> renderMgr;
    
    // Trajectory logging state
    bool enableTrajectoryLogging = false;
    int32_t trackWorldIdx = -1;
    int32_t trackAgentIdx = -1;
    FILE* trajectoryLogFile = nullptr;
    
    // Replay state
    Optional<ReplayData> replayData = Optional<ReplayData>::none();
    uint32_t currentReplayStep = 0;
    
    // Recording state
    std::ofstream recordingFile;
    madEscape::ReplayMetadata recordingMetadata;
    uint32_t recordedFrames = 0;
    bool isRecordingActive = false;

    inline Impl(const Manager::Config &mgr_cfg,
                PhysicsLoader &&phys_loader,
                WorldReset *reset_buffer,
                Action *action_buffer,
                Optional<RenderGPUState> &&render_gpu_state,
                Optional<render::RenderManager> &&render_mgr)
        : cfg(mgr_cfg),
          physicsLoader(std::move(phys_loader)),
          worldResetBuffer(reset_buffer),
          agentActionsBuffer(action_buffer),
          renderGPUState(std::move(render_gpu_state)),
          renderMgr(std::move(render_mgr))
    {}

    inline virtual ~Impl() {}

    virtual void run() = 0;

    virtual Tensor exportTensor(ExportID slot,
        TensorElementType type,
        madrona::Span<const int64_t> dimensions) const = 0;

    static inline Impl * init(const Config &cfg);
};

// [BOILERPLATE] CPU-specific implementation
struct Manager::CPUImpl final : Manager::Impl {
    using TaskGraphT =
        TaskGraphExecutor<Engine, Sim, Sim::Config, Sim::WorldInit>;

    TaskGraphT cpuExec;

    inline CPUImpl(const Manager::Config &mgr_cfg,
                   PhysicsLoader &&phys_loader,
                   WorldReset *reset_buffer,
                   Action *action_buffer,
                   Optional<RenderGPUState> &&render_gpu_state,
                   Optional<render::RenderManager> &&render_mgr,
                   TaskGraphT &&cpu_exec)
        : Impl(mgr_cfg, std::move(phys_loader),
               reset_buffer, action_buffer,
               std::move(render_gpu_state), std::move(render_mgr)),
          cpuExec(std::move(cpu_exec))
    {}

    ~CPUImpl() final = default;

    void run() final
    {
        cpuExec.run();
    }

    Tensor exportTensor(ExportID slot,
        TensorElementType type,
        madrona::Span<const int64_t> dims) const final
    {
        void *dev_ptr = cpuExec.getExported((uint32_t)slot);
        return Tensor(dev_ptr, type, dims, Optional<int>::none());
    }
};

#ifdef MADRONA_CUDA_SUPPORT
// [BOILERPLATE] CUDA-specific implementation
struct Manager::CUDAImpl final : Manager::Impl {
    MWCudaExecutor gpuExec;
    MWCudaLaunchGraph stepGraph;

    inline CUDAImpl(const Manager::Config &mgr_cfg,
                   PhysicsLoader &&phys_loader,
                   WorldReset *reset_buffer,
                   Action *action_buffer,
                   Optional<RenderGPUState> &&render_gpu_state,
                   Optional<render::RenderManager> &&render_mgr,
                   MWCudaExecutor &&gpu_exec)
        : Impl(mgr_cfg, std::move(phys_loader),
               reset_buffer, action_buffer,
               std::move(render_gpu_state), std::move(render_mgr)),
          gpuExec(std::move(gpu_exec)),
          stepGraph(gpuExec.buildLaunchGraphAllTaskGraphs())
    {}

    ~CUDAImpl() final = default;

    void run() final
    {
        gpuExec.run(stepGraph);
    }

    Tensor exportTensor(ExportID slot,
        TensorElementType type,
        madrona::Span<const int64_t> dims) const final
    {
        void *dev_ptr = gpuExec.getExported((uint32_t)slot);
        return Tensor(dev_ptr, type, dims, cfg.gpuID);
    }
};
#endif

// ============================================================================
// REQUIRED_INTERFACE: Asset Loading Functions
// Every Madrona environment must implement these methods
// ============================================================================

// [REQUIRED_INTERFACE] Load visual assets - must be implemented by every environment
static void loadRenderObjects(render::RenderManager &render_mgr)
{
    StackAlloc tmp_alloc;

    // Get render assets from static asset table
    std::vector<const madEscape::AssetInfo*> render_assets_list;
    std::vector<std::string> dense_paths;
    std::vector<uint32_t> asset_id_map;
    
    // Collect all assets that have render data
    for (uint32_t i = 0; i < madEscape::AssetIDs::MAX_ASSETS; ++i) {
        const auto* asset = madEscape::Assets::getAssetInfo(i);
        if (asset && asset->hasRender) {
            render_assets_list.push_back(asset);
            dense_paths.push_back((std::filesystem::path(DATA_DIR) / asset->meshPath).string());
            asset_id_map.push_back(asset->id);
        }
    }
    
    // Convert to C-strings for importer
    std::vector<const char*> render_asset_cstrs;
    for (const auto& path : dense_paths) {
        render_asset_cstrs.push_back(path.c_str());
    }

    // [BOILERPLATE]
    imp::AssetImporter importer;

    // [BOILERPLATE]
    std::array<char, madEscape::consts::performance::importErrorBufferSize> import_err;
    auto render_assets = importer.importFromDisk(
        render_asset_cstrs, Span<char>(import_err.data(), import_err.size()));

    // [BOILERPLATE]
    if (!render_assets.has_value()) {
        FATAL("Failed to load render assets: %s", import_err.data());
    }

    // Load textures using static Assets functions
    auto imported_textures = madEscape::Assets::loadTextures(tmp_alloc);
    
    // Create materials using static Assets functions
    auto materials = madEscape::Assets::createMaterials();
    
    // Now we need to create a properly indexed sparse array of objects
    HeapArray<imp::SourceObject> indexed_objects(madEscape::AssetIDs::MAX_ASSETS);
    
    // Initialize all entries to prevent accessing uninitialized memory
    // Assets without render data will have empty objects
    for (CountT i = 0; i < indexed_objects.size(); i++) {
        indexed_objects[i] = imp::SourceObject{};
    }
    
    // Copy imported objects to their proper positions based on object IDs
    for (size_t i = 0; i < asset_id_map.size(); i++) {
        uint32_t obj_id = asset_id_map[i];
        indexed_objects[obj_id] = render_assets->objects[i];
    }
    
    // Assign materials to each object's meshes based on static asset data
    for (const auto* asset : render_assets_list) {
        auto& obj_meshes = indexed_objects[(CountT)asset->id].meshes;
        
        for (uint32_t mesh_idx = 0; mesh_idx < asset->numMeshes && mesh_idx < obj_meshes.size(); mesh_idx++) {
            if (mesh_idx < asset->numMaterialIndices) {
                obj_meshes[mesh_idx].materialIDX = asset->materialIndices[mesh_idx];
            }
        }
    }

    // [BOILERPLATE]
    render_mgr.loadObjects(
        indexed_objects, materials, imported_textures);

    // [GAME_SPECIFIC] Configure scene lighting
    render_mgr.configureLighting({
        { true, math::Vector3{1.0f, 1.0f, madEscape::consts::rendering::lightPositionZ}, math::Vector3{1.0f, 1.0f, 1.0f} }
    });
}

// [REQUIRED_INTERFACE] Load physics assets - must be implemented by every environment
static void loadPhysicsObjects(PhysicsLoader &loader)
{
    // Get physics assets from static asset table
    std::vector<const madEscape::AssetInfo*> physics_assets;
    std::vector<std::string> dense_paths;
    std::vector<uint32_t> asset_id_map;
    
    // Collect all assets that have physics data
    for (uint32_t i = 0; i < madEscape::AssetIDs::MAX_ASSETS; ++i) {
        const auto* asset = madEscape::Assets::getAssetInfo(i);
        if (asset && asset->hasPhysics) {
            physics_assets.push_back(asset);
            if (asset->assetType == madEscape::AssetInfo::FILE_MESH && asset->filepath) {
                dense_paths.push_back((std::filesystem::path(DATA_DIR) / asset->filepath).string());
                asset_id_map.push_back(asset->id);
            }
        }
    }
    
    // Convert to C-strings for importer
    std::vector<const char*> asset_cstrs;
    for (const auto& path : dense_paths) {
        asset_cstrs.push_back(path.c_str());
    }

    // [BOILERPLATE]
    imp::AssetImporter importer;
    
    // [BOILERPLATE]
    char import_err_buffer[madEscape::consts::performance::defaultBufferSize];
    auto imported_src_hulls = importer.importFromDisk(
        asset_cstrs, import_err_buffer, true);
    
    // [BOILERPLATE]
    if (!imported_src_hulls.has_value()) {
        FATAL("%s", import_err_buffer);
    }
    
    // [BOILERPLATE]
    DynArray<imp::SourceMesh> src_convex_hulls(
        imported_src_hulls->objects.size());

    // [BOILERPLATE]
    DynArray<DynArray<SourceCollisionPrimitive>> prim_arrays(0);
    // Only allocate space for the built-in assets we actually have (IDs 0-8)
    constexpr CountT NUM_BUILTIN_ASSETS = 9;  // Updated to include cylinder (ID 8)
    HeapArray<SourceCollisionObject> src_objs(NUM_BUILTIN_ASSETS);
    
    // Initialize all entries to prevent accessing uninitialized memory
    // Render-only objects will have empty collision data
    for (CountT i = 0; i < src_objs.size(); i++) {
        src_objs[i] = {
            .prims = Span<const SourceCollisionPrimitive>(nullptr, 0),
            .invMass = 0.f,  // Infinite mass (static)
            .friction = {0.f, 0.f},
        };
    }

    // Map asset IDs to physics array indices
    auto getPhysicsIdx = [](uint32_t obj_id) -> CountT {
        return static_cast<CountT>(obj_id);
    };

    // Helper lambda to setup hull-based collision objects
    auto setupHull = [&](uint32_t obj_id, size_t import_idx,
                        float inv_mass,
                        RigidBodyFrictionData friction) {
        auto meshes = imported_src_hulls->objects[(CountT)import_idx].meshes;
        DynArray<SourceCollisionPrimitive> prims(meshes.size());

        for (const imp::SourceMesh &mesh : meshes) {
            src_convex_hulls.push_back(mesh);
            prims.push_back({
                .type = CollisionPrimitive::Type::Hull,
                .hullInput = {
                    .hullIDX = uint32_t(src_convex_hulls.size() - 1),
                },
            });
        }

        prim_arrays.emplace_back(std::move(prims));

        src_objs[getPhysicsIdx(obj_id)] = SourceCollisionObject {
            .prims = Span<const SourceCollisionPrimitive>(prim_arrays.back()),
            .invMass = inv_mass,
            .friction = friction,
        };
    };
    
    // Process file-based assets using static asset data
    size_t import_idx = 0;
    for (const auto* asset : physics_assets) {
        if (asset->assetType == madEscape::AssetInfo::FILE_MESH) {
            setupHull(asset->id, import_idx, asset->inverseMass, asset->friction);
            import_idx++;
        }
    }
    
    // Process built-in assets (plane)
    for (const auto* asset : physics_assets) {
        if (asset->assetType == madEscape::AssetInfo::BUILTIN_PLANE) {
            static SourceCollisionPrimitive plane_prim {
                .type = CollisionPrimitive::Type::Plane,
                .plane = {},
            };
            
            src_objs[getPhysicsIdx(asset->id)] = {
                .prims = Span<const SourceCollisionPrimitive>(&plane_prim, 1),
                .invMass = asset->inverseMass,
                .friction = asset->friction,
            };
        }
    }


    // [BOILERPLATE] process the rigid body assets
    StackAlloc tmp_alloc;
    RigidBodyAssets rigid_body_assets;
    CountT num_rigid_body_data_bytes;
    void *rigid_body_data = RigidBodyAssets::processRigidBodyAssets(
        src_convex_hulls,
        src_objs,
        false,
        tmp_alloc,
        &rigid_body_assets,
        &num_rigid_body_data_bytes);

    if (rigid_body_data == nullptr) {
        FATAL("Invalid collision hull input");
    }

    // Apply rotation constraints using static asset data
    for (const auto* asset : physics_assets) {
        if (asset->constrainRotationXY) {
            // Setting inverse inertia to 0 makes rotation impossible around that axis
            rigid_body_assets.metadatas[
                static_cast<CountT>(asset->id)].mass.invInertiaTensor.x = 0.f;
            rigid_body_assets.metadatas[
                static_cast<CountT>(asset->id)].mass.invInertiaTensor.y = 0.f;
        }
    }

    // [BOILERPLATE]
    loader.loadRigidBodies(rigid_body_assets);
    free(rigid_body_data);
}

// ============================================================================
// BOILERPLATE: Manager Initialization
// ============================================================================

// [BOILERPLATE] Factory method for CPU/GPU setup
Manager::Impl * Manager::Impl::init(
    const Manager::Config &mgr_cfg)
{
    // [BOILERPLATE] Create sim config
    Sim::Config sim_cfg;
    sim_cfg.autoReset = mgr_cfg.autoReset;
    sim_cfg.initRandKey = rand::initKey(mgr_cfg.randSeed);
    sim_cfg.customVerticalFov = mgr_cfg.customVerticalFov;
    // REQUIRE per-world compiled levels - no defaults allowed
    if (mgr_cfg.perWorldCompiledLevels.empty() || 
        !std::any_of(mgr_cfg.perWorldCompiledLevels.begin(), mgr_cfg.perWorldCompiledLevels.end(),
                     [](const std::optional<CompiledLevel>& level) { return level.has_value(); })) {
        std::cerr << "ERROR: No compiled level provided! All simulations must use ASCII levels.\n";
        std::cerr << "Use level_ascii parameter in SimManager constructor.\n";
        std::abort();
    }
    
    // Use first valid per-world level as sim config default
    for (const auto& level : mgr_cfg.perWorldCompiledLevels) {
        if (level.has_value()) {
            sim_cfg.compiledLevel = level.value();
            break;
        }
    }

    switch (mgr_cfg.execMode) {
    case ExecMode::CUDA: {
#ifdef MADRONA_CUDA_SUPPORT
        CUcontext cu_ctx = MWCudaExecutor::initCUDA(mgr_cfg.gpuID);

        PhysicsLoader phys_loader(ExecMode::CUDA, 10);
        loadPhysicsObjects(phys_loader);

        ObjectManager *phys_obj_mgr = &phys_loader.getObjectManager();
        sim_cfg.rigidBodyObjMgr = phys_obj_mgr;

        Optional<RenderGPUState> render_gpu_state =
            initRenderGPUState(mgr_cfg);

        Optional<render::RenderManager> render_mgr =
            initRenderManager(mgr_cfg, render_gpu_state);

        if (render_mgr.has_value()) {
            loadRenderObjects(*render_mgr);
            sim_cfg.renderBridge = render_mgr->bridge();
        } else {
            sim_cfg.renderBridge = nullptr;
        }

        HeapArray<Sim::WorldInit> world_inits(mgr_cfg.numWorlds);
        
        // Populate per-world compiled levels
        for (uint32_t i = 0; i < mgr_cfg.numWorlds; i++) {
            if (i < mgr_cfg.perWorldCompiledLevels.size() && mgr_cfg.perWorldCompiledLevels[i].has_value()) {
                world_inits[i].compiledLevel = mgr_cfg.perWorldCompiledLevels[i].value();
            } else {
                // Use sim_cfg.compiledLevel as fallback (first valid level from array)
                world_inits[i].compiledLevel = sim_cfg.compiledLevel;
            }
        }

        // [BOILERPLATE] Create GPU executor with configuration
        MWCudaExecutor gpu_exec({
            .worldInitPtr = world_inits.data(),
            .numWorldInitBytes = sizeof(Sim::WorldInit),
            .userConfigPtr = (void *)&sim_cfg,
            .numUserConfigBytes = sizeof(Sim::Config),
            .numWorldDataBytes = sizeof(Sim),
            .worldDataAlignment = alignof(Sim),
            .numWorlds = mgr_cfg.numWorlds,
            .numTaskGraphs = 1,
            .numExportedBuffers = (uint32_t)ExportID::NumExports,
        }, {
            { GPU_HIDESEEK_SRC_LIST },
            { GPU_HIDESEEK_COMPILE_FLAGS },
            CompileConfig::OptMode::LTO,
        }, cu_ctx);

        WorldReset *world_reset_buffer = 
            (WorldReset *)gpu_exec.getExported((uint32_t)ExportID::Reset);

        Action *agent_actions_buffer = 
            (Action *)gpu_exec.getExported((uint32_t)ExportID::Action);

        return new CUDAImpl {
            mgr_cfg,
            std::move(phys_loader),
            world_reset_buffer,
            agent_actions_buffer,
            std::move(render_gpu_state),
            std::move(render_mgr),
            std::move(gpu_exec),
        };
#else
        FATAL("Madrona was not compiled with CUDA support");
#endif
    } break;
    case ExecMode::CPU: {
        // [BOILERPLATE] Create physics loader
        PhysicsLoader phys_loader(ExecMode::CPU, 10);
        // [REQUIRED_INTERFACE] Call game's physics asset loader
        loadPhysicsObjects(phys_loader);

        ObjectManager *phys_obj_mgr = &phys_loader.getObjectManager();
        sim_cfg.rigidBodyObjMgr = phys_obj_mgr;

        Optional<RenderGPUState> render_gpu_state =
            initRenderGPUState(mgr_cfg);

        Optional<render::RenderManager> render_mgr =
            initRenderManager(mgr_cfg, render_gpu_state);

        if (render_mgr.has_value()) {
            loadRenderObjects(*render_mgr);
            // [BOILERPLATE] Connect render bridge to sim
            sim_cfg.renderBridge = render_mgr->bridge();
        } else {
            sim_cfg.renderBridge = nullptr;
        }

        // [BOILERPLATE] Allocate per-world initialization data
        HeapArray<Sim::WorldInit> world_inits(mgr_cfg.numWorlds);
        
        // Populate per-world compiled levels
        for (uint32_t i = 0; i < mgr_cfg.numWorlds; i++) {
            if (i < mgr_cfg.perWorldCompiledLevels.size() && mgr_cfg.perWorldCompiledLevels[i].has_value()) {
                world_inits[i].compiledLevel = mgr_cfg.perWorldCompiledLevels[i].value();
            } else {
                // Use sim_cfg.compiledLevel as fallback (first valid level from array)
                world_inits[i].compiledLevel = sim_cfg.compiledLevel;
            }
        }

        // [BOILERPLATE] Create CPU executor with configuration
        CPUImpl::TaskGraphT cpu_exec {
            ThreadPoolExecutor::Config {
                .numWorlds = mgr_cfg.numWorlds,
                .numExportedBuffers = (uint32_t)ExportID::NumExports,
            },
            sim_cfg,
            world_inits.data(),
            (uint32_t)TaskGraphID::NumTaskGraphs,
        };

        // [BOILERPLATE] Get pointers to exported buffers
        WorldReset *world_reset_buffer = 
            (WorldReset *)cpu_exec.getExported((uint32_t)ExportID::Reset);

        Action *agent_actions_buffer = 
            (Action *)cpu_exec.getExported((uint32_t)ExportID::Action);

        auto cpu_impl = new CPUImpl {
            mgr_cfg,
            std::move(phys_loader),
            world_reset_buffer,
            agent_actions_buffer,
            std::move(render_gpu_state),
            std::move(render_mgr),
            std::move(cpu_exec),
        };

        return cpu_impl;
    } break;
    default: MADRONA_UNREACHABLE();
    }
}

// ============================================================================
// BOILERPLATE: Manager Public Interface
// ============================================================================

// [BOILERPLATE] Manager constructor following Madrona initialization pattern
Manager::Manager(const Config &cfg)
    : impl_(Impl::init(cfg))
{
    // [BOILERPLATE] Required initialization sequence for Madrona
    // Currently, there is no way to populate the initial set of observations
    // without stepping the simulations in order to execute the taskgraph.
    // Therefore, after setup, we step all the simulations with a forced reset
    // that ensures the first real step will have valid observations at the
    // start of a fresh episode in order to compute actions.
    //
    // This will be improved in the future with support for multiple task
    // graphs, allowing a small task graph to be executed after initialization.
    
    for (int32_t i = 0; i < (int32_t)cfg.numWorlds; i++) {
        triggerReset(i);
    }

    step();
}

Manager::~Manager() {}

// [BOILERPLATE] Execute one simulation step - standard Madrona pattern
void Manager::step()
{
    // static uint32_t step_counter = 0;
    // step_counter++;
    // Record actions if recording is active
    if (impl_->isRecordingActive) {
        // Get current actions from the action tensor
        auto action_tensor = actionTensor();
        std::vector<int32_t> frame_actions;
        
        // Calculate total size needed: num_worlds * 3 actions per world
        uint32_t num_worlds = impl_->cfg.numWorlds;
        uint32_t total_actions = num_worlds * madEscape::consts::numActionComponents;
        frame_actions.resize(total_actions);
        
        // Handle GPU vs CPU memory access
        if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
            // For GPU, we need to copy the data from device to host
            cudaMemcpy(frame_actions.data(), action_tensor.devicePtr(), 
                      total_actions * sizeof(int32_t), cudaMemcpyDeviceToHost);
#endif
        } else {
            // For CPU, we can directly access the memory
            const int32_t* action_data = (const int32_t*)action_tensor.devicePtr();
            for (uint32_t i = 0; i < total_actions; i++) {
                frame_actions[i] = action_data[i];
            }
        }
        
        // Record the actions
        recordActions(frame_actions);
    }
    
    impl_->run();

    if (impl_->renderMgr.has_value()) {
        impl_->renderMgr->readECS();
    }

    if (impl_->cfg.enableBatchRenderer) {
        impl_->renderMgr->batchRender();
    }
    
    // Log trajectory if enabled
    logCurrentTrajectoryState();
    
}

// ============================================================================
// REQUIRED_INTERFACE: Tensor Accessors for Python Integration
// Every environment must export tensors for observations, actions, rewards, etc.
// ============================================================================

// [REQUIRED_INTERFACE] All tensor methods below define the observation/action space
// [GAME_SPECIFIC] The specific tensors exported and their shapes

// [BOILERPLATE]
Tensor Manager::resetTensor() const
{
    return impl_->exportTensor(ExportID::Reset,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   1,
                               });
}

// [BOILERPLATE]
Tensor Manager::actionTensor() const
{
    // [GAME_SPECIFIC] Discrete actions per world: move amount/angle, rotate
    return impl_->exportTensor(ExportID::Action, TensorElementType::Int32,
        {
            impl_->cfg.numWorlds,
            madEscape::consts::numActionComponents,
        });
}

// [BOILERPLATE]
Tensor Manager::rewardTensor() const
{
    return impl_->exportTensor(ExportID::Reward, TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   madEscape::consts::numAgents,
                                   1,
                               });
}

//[BOILERPLATE]
Tensor Manager::doneTensor() const
{
    return impl_->exportTensor(ExportID::Done, TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   madEscape::consts::numAgents,
                                   1,
                               });
}

//[GAME SPECIFIC]
Tensor Manager::selfObservationTensor() const
{
    return impl_->exportTensor(ExportID::SelfObservation,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   madEscape::consts::numAgents,
                                   SelfObservationFloatCount,
                               });
}

//[GAME SPECIFIC]
Tensor Manager::compassTensor() const
{
    return impl_->exportTensor(ExportID::CompassObservation,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   madEscape::consts::numAgents,
                                   CompassObservationFloatCount,
                               });
}


// Removed roomEntityObservationsTensor - no longer tracking room entities


//[BOILERPLATE]
Tensor Manager::stepsTakenTensor() const
{
    return impl_->exportTensor(ExportID::StepsTaken,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   madEscape::consts::numAgents,
                                   1,
                               });
}

Tensor Manager::progressTensor() const
{
    return impl_->exportTensor(ExportID::Progress,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   madEscape::consts::numAgents,
                                   1,
                               });
}

//[GAME_SPECIFIC]
Tensor Manager::rgbTensor() const
{
    // [BOILERPLATE] Get raw RGB buffer from renderer
    const uint8_t *rgb_ptr = impl_->renderMgr->batchRendererRGBOut();

    // [GAME_SPECIFIC] Return tensor with escape room view dimensions
    return Tensor((void*)rgb_ptr, TensorElementType::UInt8, {
        impl_->cfg.numWorlds,
        madEscape::consts::numAgents,
        impl_->cfg.batchRenderViewHeight,
        impl_->cfg.batchRenderViewWidth,
        4,
    }, impl_->cfg.gpuID);
}

//[GAME_SPECIFIC]
Tensor Manager::depthTensor() const
{
    const float *depth_ptr = impl_->renderMgr->batchRendererDepthOut();

    return Tensor((void *)depth_ptr, TensorElementType::Float32, {
        impl_->cfg.numWorlds,
        madEscape::consts::numAgents,
        impl_->cfg.batchRenderViewHeight,
        impl_->cfg.batchRenderViewWidth,
        1,
    }, impl_->cfg.gpuID);
}

// [REQUIRED_INTERFACE] Trigger episode reset for a specific world
void Manager::triggerReset(int32_t world_idx)
{
    WorldReset reset {
        1,
    };

    auto *reset_ptr = impl_->worldResetBuffer + world_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(reset_ptr, &reset, sizeof(WorldReset),
                   cudaMemcpyHostToDevice);
#endif
    }  else {
        *reset_ptr = reset;
    }
}


// [REQUIRED_INTERFACE] Set agent actions for the escape room
void Manager::setAction(int32_t world_idx,
                        int32_t move_amount,
                        int32_t move_angle,
                        int32_t rotate)
{
    // [GAME_SPECIFIC] Pack discrete actions into struct
    Action action { 
        .moveAmount = move_amount,
        .moveAngle = move_angle,
        .rotate = rotate,
    };

    // [GAME_SPECIFIC] Calculate buffer offset for world
    auto *action_ptr = impl_->agentActionsBuffer + world_idx;

    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        cudaMemcpy(action_ptr, &action, sizeof(Action),
                   cudaMemcpyHostToDevice);
#endif
    } else {
        *action_ptr = action;
    }
}

// Helper function to log current trajectory state
void Manager::logCurrentTrajectoryState()
{
    if (!impl_->enableTrajectoryLogging || 
        impl_->trackWorldIdx < 0 || 
        impl_->trackAgentIdx < 0) {
        return;
    }
    
    // Get tensor data
    auto self_obs = selfObservationTensor();
    auto compass = compassTensor();
    auto progress = progressTensor();
    auto done = doneTensor();
    auto steps_taken = stepsTakenTensor();
    
    // Calculate index for the specific agent
    int32_t idx = impl_->trackWorldIdx * madEscape::consts::numAgents + impl_->trackAgentIdx;
    
    // Get data pointers based on execution mode
    const SelfObservation* obs_data;
    const float* compass_data;
    const float* progress_data;
    const int32_t* done_data;
    const uint32_t* steps_taken_data;
    
    if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
        // For CUDA, we need to copy data to host
        static SelfObservation host_obs;
        static float host_compass[128];
        static float host_progress;
        static int32_t host_done;
        static uint32_t host_steps_taken;
        
        cudaMemcpy(&host_obs, 
                  ((const SelfObservation*)self_obs.devicePtr()) + idx,
                  sizeof(SelfObservation), 
                  cudaMemcpyDeviceToHost);
        cudaMemcpy(host_compass,
                  ((const float*)compass.devicePtr()) + (idx * 128),
                  128 * sizeof(float),
                  cudaMemcpyDeviceToHost);
        cudaMemcpy(&host_progress,
                  ((const float*)progress.devicePtr()) + idx,
                  sizeof(float),
                  cudaMemcpyDeviceToHost);
        cudaMemcpy(&host_done,
                  ((const int32_t*)done.devicePtr()) + idx,
                  sizeof(int32_t),
                  cudaMemcpyDeviceToHost);
        cudaMemcpy(&host_steps_taken,
                  ((const uint32_t*)steps_taken.devicePtr()) + idx,
                  sizeof(uint32_t),
                  cudaMemcpyDeviceToHost);
        
        obs_data = &host_obs;
        compass_data = host_compass;
        progress_data = &host_progress;
        done_data = &host_done;
        steps_taken_data = &host_steps_taken;
#endif
    } else {
        // For CPU, direct access
        obs_data = ((const SelfObservation*)self_obs.devicePtr()) + idx;
        compass_data = ((const float*)compass.devicePtr()) + (idx * 128);
        progress_data = ((const float*)progress.devicePtr()) + idx;
        done_data = ((const int32_t*)done.devicePtr()) + idx;
        steps_taken_data = ((const uint32_t*)steps_taken.devicePtr()) + idx;
    }
    
    // Find the active compass index (one-hot encoding)
    int compass_index = -1;
    for (int i = 0; i < 128; i++) {
        if (compass_data[i] > 0.5f) {
            compass_index = i;
            break;
        }
    }
    
    // Log trajectory to file or stdout  
    FILE* output = impl_->trajectoryLogFile ? impl_->trajectoryLogFile : stdout;
    uint32_t remaining_display = (*steps_taken_data >= madEscape::consts::episodeLen) ? 0 : (madEscape::consts::episodeLen - *steps_taken_data);
    
    fprintf(output, "Episode step %3u (%3u remaining): World %d Agent %d: pos=(%.2f,%.2f,%.2f) rot=%.1f° compass=%d progress=%.2f done=%d\n",
            *steps_taken_data,
            remaining_display,
            impl_->trackWorldIdx,
            impl_->trackAgentIdx,
            obs_data->globalX,
            obs_data->globalY,
            obs_data->globalZ,
            obs_data->theta * madEscape::consts::math::degreesInHalfCircle / M_PI,
            compass_index,
            *progress_data,
            *done_data);
    fflush(output);
}

void Manager::enableTrajectoryLogging(int32_t world_idx, int32_t agent_idx, std::optional<const char*> filename)
{
    // Validate world index
    if (world_idx < 0 || world_idx >= (int32_t)impl_->cfg.numWorlds) {
        fprintf(stderr, "ERROR: Invalid world_idx: %d. Must be between 0 and %u\n", 
                world_idx, impl_->cfg.numWorlds - 1);
        return;
    }
    
    // Validate agent index
    if (agent_idx < 0 || agent_idx >= madEscape::consts::numAgents) {
        fprintf(stderr, "ERROR: Invalid agent_idx: %d. Must be between 0 and %d\n", 
                agent_idx, madEscape::consts::numAgents - 1);
        return;
    }
    
    // Close existing file if open
    if (impl_->trajectoryLogFile != nullptr && impl_->trajectoryLogFile != stdout) {
        fclose(impl_->trajectoryLogFile);
        impl_->trajectoryLogFile = nullptr;
    }
    
    // Open new file if filename provided
    if (filename.has_value() && filename.value() != nullptr) {
        impl_->trajectoryLogFile = fopen(filename.value(), "w");
        if (impl_->trajectoryLogFile == nullptr) {
            fprintf(stderr, "ERROR: Could not open file '%s' for trajectory logging\n", filename.value());
            return;
        }
        printf("Trajectory logging enabled for World %d, Agent %d to file: %s\n", 
               world_idx, agent_idx, filename.value());
    } else {
        impl_->trajectoryLogFile = stdout;
        printf("Trajectory logging enabled for World %d, Agent %d\n", world_idx, agent_idx);
    }
    
    impl_->enableTrajectoryLogging = true;
    impl_->trackWorldIdx = world_idx;
    impl_->trackAgentIdx = agent_idx;
    
    // Log initial state (step 0) immediately
    logCurrentTrajectoryState();
}

void Manager::disableTrajectoryLogging()
{
    impl_->enableTrajectoryLogging = false;
    impl_->trackWorldIdx = -1;
    impl_->trackAgentIdx = -1;
    
    // Close file if it's not stdout
    if (impl_->trajectoryLogFile != nullptr && impl_->trajectoryLogFile != stdout) {
        fclose(impl_->trajectoryLogFile);
    }
    impl_->trajectoryLogFile = nullptr;
    
    printf("Trajectory logging disabled\n");
}

// [BOILERPLATE] Expose render manager for visualization tools
render::RenderManager & Manager::getRenderManager()
{
    return *impl_->renderMgr;
}

// Recording functionality
void Manager::startRecording(const std::string& filepath, uint32_t seed)
{
    if (impl_->isRecordingActive) {
        std::cerr << "Recording already in progress\n";
        return;
    }
    
    impl_->recordingFile.open(filepath, std::ios::binary);
    if (!impl_->recordingFile.is_open()) {
        std::cerr << "Failed to open recording file: " << filepath << "\n";
        return;
    }
    
    // Always embed compiled level data after metadata
    // Format: [ReplayMetadata] [CompiledLevel] [ActionFrames...]
    // Use first valid per-world level for embedding
    CompiledLevel levelToEmbed;
    bool foundLevel = false;
    for (const auto& level : impl_->cfg.perWorldCompiledLevels) {
        if (level.has_value()) {
            levelToEmbed = level.value();
            foundLevel = true;
            break;
        }
    }
    if (!foundLevel) {
        std::cerr << "ERROR: Cannot start recording - no compiled level available\n";
        return;
    }
    
    // Prepare metadata
    impl_->recordingMetadata = madEscape::ReplayMetadata::createDefault();
    impl_->recordingMetadata.num_worlds = impl_->cfg.numWorlds;
    impl_->recordingMetadata.num_agents_per_world = 1; // Single agent per world
    impl_->recordingMetadata.seed = seed;
    impl_->recordingMetadata.timestamp = std::chrono::system_clock::now().time_since_epoch().count();
    // Copy level name from the CompiledLevel to the ReplayMetadata
    std::strncpy(impl_->recordingMetadata.level_name, levelToEmbed.level_name, sizeof(impl_->recordingMetadata.level_name) - 1);
    impl_->recordingMetadata.level_name[sizeof(impl_->recordingMetadata.level_name) - 1] = '\0';
    
    // Write metadata header (will update num_steps when closing)
    impl_->recordingFile.write(reinterpret_cast<const char*>(&impl_->recordingMetadata), 
                              sizeof(impl_->recordingMetadata));
    
    // Write compiled level data
    impl_->recordingFile.write(reinterpret_cast<const char*>(&levelToEmbed), 
                              sizeof(CompiledLevel));
    
    impl_->recordedFrames = 0;
    impl_->isRecordingActive = true;
}

void Manager::stopRecording()
{
    if (!impl_->isRecordingActive) {
        return;
    }
    
    if (impl_->recordedFrames > 0) {
        // Update metadata with actual number of steps
        impl_->recordingMetadata.num_steps = impl_->recordedFrames;
        
        // Seek back to beginning and rewrite the metadata
        impl_->recordingFile.seekp(0, std::ios::beg);
        impl_->recordingFile.write(reinterpret_cast<const char*>(&impl_->recordingMetadata), 
                                  sizeof(impl_->recordingMetadata));
        
    } else {
    }
    
    impl_->recordingFile.close();
    impl_->isRecordingActive = false;
}

bool Manager::isRecording() const
{
    return impl_->isRecordingActive;
}

void Manager::recordActions(const std::vector<int32_t>& frame_actions)
{
    if (!impl_->isRecordingActive) {
        return;
    }
    
    impl_->recordingFile.write(reinterpret_cast<const char*>(frame_actions.data()), 
                              frame_actions.size() * sizeof(int32_t));
    impl_->recordedFrames++;
    
    // Tracking frame count without logging
}

// Replay functionality
std::optional<madEscape::ReplayMetadata> Manager::readReplayMetadata(const std::string& filepath)
{
    std::ifstream replay_file(filepath, std::ios::binary);
    if (!replay_file.is_open()) {
        std::cerr << "Error: Failed to open replay file: " << filepath << "\n";
        return std::nullopt;
    }
    
    // Read metadata header
    madEscape::ReplayMetadata metadata;
    replay_file.read(reinterpret_cast<char*>(&metadata), sizeof(metadata));
    
    // Check if we actually read any data
    if (replay_file.gcount() == 0) {
        std::cerr << "Error: Replay file is empty: " << filepath << "\n";
        return std::nullopt;
    }
    
    // Check if we read the full metadata structure
    if (replay_file.gcount() < static_cast<std::streamsize>(sizeof(metadata))) {
        std::cerr << "Error: Replay file is too small. Expected " << sizeof(metadata) 
                  << " bytes for metadata, got " << replay_file.gcount() << " bytes.\n";
        return std::nullopt;
    }
    
    // Validate metadata
    if (!metadata.isValid()) {
        std::cerr << "Error: Invalid replay file format. Expected magic: 0x" 
                  << std::hex << REPLAY_MAGIC << ", got: 0x" 
                  << metadata.magic << std::dec << "\n";
        return std::nullopt;
    }
    
    // Show replay information
    
    return metadata;
}

bool Manager::loadReplay(const std::string& filepath)
{
    std::ifstream replay_file(filepath, std::ios::binary);
    if (!replay_file.is_open()) {
        std::cerr << "Error: Failed to open replay file: " << filepath << "\n";
        return false;
    }
    
    // Read metadata header
    madEscape::ReplayMetadata metadata;
    replay_file.read(reinterpret_cast<char*>(&metadata), sizeof(metadata));
    
    // Validate metadata
    if (!metadata.isValid()) {
        std::cerr << "Error: Invalid replay file format. Expected magic: 0x" 
                  << std::hex << REPLAY_MAGIC << ", got: 0x" 
                  << metadata.magic << std::dec << "\n";
        return false;
    }
    
    // NEW: Read embedded level data after metadata
    // Format: [ReplayMetadata] [CompiledLevel] [ActionFrames...]
    CompiledLevel embeddedLevel;
    replay_file.read(reinterpret_cast<char*>(&embeddedLevel), sizeof(CompiledLevel));
    
    if (replay_file.fail()) {
        std::cerr << "Error: Failed to read embedded level data from replay file\n";
        return false;
    }
    
    // TODO: Apply the embedded level to the manager/simulation
    // For now, just log that we read level data
    
    // Read actions after embedded level data
    int64_t actions_size = metadata.num_steps * metadata.num_worlds * metadata.actions_per_step * sizeof(int32_t);
    HeapArray<int32_t> actions(actions_size / sizeof(int32_t));
    replay_file.read((char *)actions.data(), actions_size);
    
    
    impl_->replayData = ReplayData{metadata, std::move(actions)};
    impl_->currentReplayStep = 0;
    
    return true;
}

bool Manager::hasReplay() const
{
    return impl_->replayData.has_value();
}

bool Manager::replayStep()
{
    if (!impl_->replayData.has_value() || 
        impl_->currentReplayStep >= impl_->replayData->metadata.num_steps) {
        return true; // Replay finished
    }
    
    const auto& actions = impl_->replayData->actions;
    const auto& metadata = impl_->replayData->metadata;
    
    for (uint32_t i = 0; i < metadata.num_worlds; i++) {
        uint32_t base_idx = (impl_->currentReplayStep * metadata.num_worlds * madEscape::consts::numActionComponents) + (i * madEscape::consts::numActionComponents);
        
        int32_t move_amount = actions[base_idx];
        int32_t move_angle = actions[base_idx + 1];
        int32_t turn = actions[base_idx + 2];
        
        setAction(i, move_amount, move_angle, turn);
    }
    
    impl_->currentReplayStep++;
    
    // Check if we just consumed the last step
    return impl_->currentReplayStep >= impl_->replayData->metadata.num_steps;
}

uint32_t Manager::getCurrentReplayStep() const
{
    return impl_->currentReplayStep;
}

uint32_t Manager::getTotalReplaySteps() const
{
    if (!impl_->replayData.has_value()) {
        return 0;
    }
    return impl_->replayData->metadata.num_steps;
}

const Manager::ReplayData* Manager::getReplayData() const
{
    if (!impl_->replayData.has_value()) {
        return nullptr;
    }
    return &(*impl_->replayData);
}

std::optional<CompiledLevel> Manager::readEmbeddedLevel(const std::string& filepath)
{
    std::ifstream replay_file(filepath, std::ios::binary);
    if (!replay_file.is_open()) {
        return std::nullopt;
    }
    
    // Read and skip metadata header
    madEscape::ReplayMetadata metadata;
    replay_file.read(reinterpret_cast<char*>(&metadata), sizeof(metadata));
    
    // Validate metadata
    if (!metadata.isValid()) {
        return std::nullopt;
    }
    
    // Read embedded level data after metadata
    CompiledLevel embeddedLevel;
    replay_file.read(reinterpret_cast<char*>(&embeddedLevel), sizeof(CompiledLevel));
    
    if (replay_file.fail()) {
        return std::nullopt;
    }
    
    return embeddedLevel;
}

}
