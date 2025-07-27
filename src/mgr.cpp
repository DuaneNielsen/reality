#include "mgr.hpp"
#include "sim.hpp"

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
#include <charconv>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>
#include <cmath>

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
        .renderMode = render::RenderManager::Config::RenderMode::RGBD,
        .agentViewWidth = mgr_cfg.batchRenderViewWidth,
        .agentViewHeight = mgr_cfg.batchRenderViewHeight,
        .numWorlds = mgr_cfg.numWorlds,
        .maxViewsPerWorld = consts::numAgents,
        .maxInstancesPerWorld = 1000,           // [GAME_SPECIFIC] Max entities to render
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
    
    // Trajectory tracking state
    bool enableTrajectoryTracking = false;
    int32_t trackWorldIdx = -1;
    int32_t trackAgentIdx = -1;
    uint32_t stepCount = 0;

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

    inline virtual ~CPUImpl() final {}

    inline virtual void run()
    {
        cpuExec.run();
    }

    virtual inline Tensor exportTensor(ExportID slot,
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

    inline virtual ~CUDAImpl() final {}

    inline virtual void run()
    {
        gpuExec.run(stepGraph);
    }

    virtual inline Tensor exportTensor(ExportID slot,
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

    // [GAME_SPECIFIC] Map each game object type to its visual mesh
    std::array<std::string, (size_t)SimObject::NumObjects> render_asset_paths;
    render_asset_paths[(size_t)SimObject::Cube] =
        (std::filesystem::path(DATA_DIR) / "cube_render.obj").string();
    render_asset_paths[(size_t)SimObject::Wall] =
        (std::filesystem::path(DATA_DIR) / "wall_render.obj").string();
    render_asset_paths[(size_t)SimObject::Agent] =
        (std::filesystem::path(DATA_DIR) / "agent_render.obj").string();
    render_asset_paths[(size_t)SimObject::Plane] =
        (std::filesystem::path(DATA_DIR) / "plane.obj").string();
    render_asset_paths[(size_t)SimObject::AxisX] =
        (std::filesystem::path(DATA_DIR) / "cube_render.obj").string();  // Reuse cube mesh
    render_asset_paths[(size_t)SimObject::AxisY] =
        (std::filesystem::path(DATA_DIR) / "cube_render.obj").string();  // Reuse cube mesh
    render_asset_paths[(size_t)SimObject::AxisZ] =
        (std::filesystem::path(DATA_DIR) / "cube_render.obj").string();  // Reuse cube mesh

    // [BOILERPLATE]
    std::array<const char *, (size_t)SimObject::NumObjects> render_asset_cstrs;
    for (size_t i = 0; i < render_asset_paths.size(); i++) {
        render_asset_cstrs[i] = render_asset_paths[i].c_str();
    }

    // [BOILERPLATE]
    imp::AssetImporter importer;

    // [BOILERPLATE]
    std::array<char, 1024> import_err;
    auto render_assets = importer.importFromDisk(
        render_asset_cstrs, Span<char>(import_err.data(), import_err.size()));

    // [BOILERPLATE]
    if (!render_assets.has_value()) {
        FATAL("Failed to load render assets: %s", import_err);
    }

    // [GAME_SPECIFIC] Define materials for each object type
    auto materials = std::to_array<imp::SourceMaterial>({
        { render::rgb8ToFloat(191, 108, 10), -1, 0.8f, 0.2f },      // Brown (cube)
        { math::Vector4{0.4f, 0.4f, 0.4f, 0.0f}, -1, 0.8f, 0.2f,},  // Gray (wall)
        { math::Vector4{1.f, 1.f, 1.f, 0.0f}, 1, 0.5f, 1.0f,},      // White (agent body)
        { render::rgb8ToFloat(230, 230, 230),   -1, 0.8f, 1.0f },   // Light gray (agent parts)
        { math::Vector4{0.5f, 0.3f, 0.3f, 0.0f},  0, 0.8f, 0.2f,},  // Brown (floor)
        { render::rgb8ToFloat(230, 20, 20),   -1, 0.8f, 1.0f },     // Red (door/X-axis)
        { render::rgb8ToFloat(230, 230, 20),   -1, 0.8f, 1.0f },    // Yellow (button)
        { render::rgb8ToFloat(20, 230, 20),   -1, 0.8f, 1.0f },     // Green (Y-axis)
        { render::rgb8ToFloat(20, 20, 230),   -1, 0.8f, 1.0f },     // Blue (Z-axis)
    });

    // [GAME_SPECIFIC] Assign materials to each object's meshes
    render_assets->objects[(CountT)SimObject::Cube].meshes[0].materialIDX = 0;
    render_assets->objects[(CountT)SimObject::Wall].meshes[0].materialIDX = 1;
    render_assets->objects[(CountT)SimObject::Agent].meshes[0].materialIDX = 2;  // Body
    render_assets->objects[(CountT)SimObject::Agent].meshes[1].materialIDX = 3;  // Eyes
    render_assets->objects[(CountT)SimObject::Agent].meshes[2].materialIDX = 3;  // Other parts
    render_assets->objects[(CountT)SimObject::Plane].meshes[0].materialIDX = 4;
    render_assets->objects[(CountT)SimObject::AxisX].meshes[0].materialIDX = 5;  // Red
    render_assets->objects[(CountT)SimObject::AxisY].meshes[0].materialIDX = 7;  // Green
    render_assets->objects[(CountT)SimObject::AxisZ].meshes[0].materialIDX = 8;  // Blue

    // [GAME_SPECIFIC] Load textures for materials
    imp::ImageImporter img_importer;
    Span<imp::SourceTexture> imported_textures = img_importer.importImages(
        tmp_alloc, {
            (std::filesystem::path(DATA_DIR) /
               "green_grid.png").string().c_str(),   // Floor texture
            (std::filesystem::path(DATA_DIR) /
               "smile.png").string().c_str(),        // Agent texture
        });

    // [BOILERPLATE]
    render_mgr.loadObjects(
        render_assets->objects, materials, imported_textures);

    // [GAME_SPECIFIC] Configure scene lighting
    render_mgr.configureLighting({
        { true, math::Vector3{1.0f, 1.0f, -2.0f}, math::Vector3{1.0f, 1.0f, 1.0f} }
    });
}

// [REQUIRED_INTERFACE] Load physics assets - must be implemented by every environment
static void loadPhysicsObjects(PhysicsLoader &loader)
{

    // [GAME_SPECIFIC]
    // Only Cube, Wall, and Agent need physics assets (3 objects)
    // Plane uses built-in plane collision, AxisX/Y/Z are render-only
    constexpr size_t numPhysicsAssets = 3;
    std::array<std::string, numPhysicsAssets> asset_paths;
    asset_paths[(size_t)SimObject::Cube] =
        (std::filesystem::path(DATA_DIR) / "cube_collision.obj").string();
    asset_paths[(size_t)SimObject::Wall] =
        (std::filesystem::path(DATA_DIR) / "wall_collision.obj").string();
    asset_paths[(size_t)SimObject::Agent] =
        (std::filesystem::path(DATA_DIR) / "agent_collision_simplified.obj").string();


    // [BOILERPLATE]
    std::array<const char *, numPhysicsAssets> asset_cstrs;
    for (size_t i = 0; i < asset_paths.size(); i++) {
        asset_cstrs[i] = asset_paths[i].c_str();
    }

    // [BOILERPLATE]
    imp::AssetImporter importer;

    // [BOILERPLATE]
    char import_err_buffer[4096];
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
    // Only allocate physics for objects that need it (Cube, Wall, Agent, Plane)
    constexpr CountT numPhysicsObjects = 4;  // Cube, Wall, Agent, Plane
    HeapArray<SourceCollisionObject> src_objs(numPhysicsObjects);

    // Map SimObject IDs to physics array indices
    auto getPhysicsIdx = [](SimObject obj_id) -> CountT {
        switch(obj_id) {
            case SimObject::Cube: return 0;
            case SimObject::Wall: return 1;
            case SimObject::Agent: return 2;
            case SimObject::Plane: return 3;
            default: FATAL("Object type has no physics");
        }
    };

    // [BOILERPLATE]
    auto setupHull = [&](SimObject obj_id,
                         float inv_mass,
                         RigidBodyFrictionData friction) {
        auto meshes = imported_src_hulls->objects[(CountT)obj_id].meshes;
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

    // [GAME_SPECIFIC] Configure physics for each object type
    setupHull(SimObject::Cube, 0.075f, {    // Pushable with ~13kg mass
        .muS = 0.5f,
        .muD = 0.75f,
    });

    setupHull(SimObject::Wall, 0.f, {       // Static (infinite mass)
        .muS = 0.5f,
        .muD = 0.5f,
    });


    setupHull(SimObject::Agent, 1.f, {      // Unit mass for direct control
        .muS = 0.5f,
        .muD = 0.5f,
    });

    SourceCollisionPrimitive plane_prim {
        .type = CollisionPrimitive::Type::Plane,
        .plane = {},
    };

    src_objs[getPhysicsIdx(SimObject::Plane)] = {
        .prims = Span<const SourceCollisionPrimitive>(&plane_prim, 1),
        .invMass = 0.f,
        .friction = {
            .muS = 0.5f,
            .muD = 0.5f,
        },
    };


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

    // [GAME_SPECIFIC] Constrain agent rotation to Z-axis only
    // This prevents agents from tipping over and ensures controllability
    // Setting inverse inertia to 0 makes rotation impossible around that axis
    rigid_body_assets.metadatas[
        getPhysicsIdx(SimObject::Agent)].mass.invInertiaTensor.x = 0.f;
    rigid_body_assets.metadatas[
        getPhysicsIdx(SimObject::Agent)].mass.invInertiaTensor.y = 0.f;

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
    impl_->run();

    if (impl_->renderMgr.has_value()) {
        impl_->renderMgr->readECS();
    }

    if (impl_->cfg.enableBatchRenderer) {
        impl_->renderMgr->batchRender();
    }
    
    // Print trajectory if tracking is enabled
    if (impl_->enableTrajectoryTracking && 
        impl_->trackWorldIdx >= 0 && 
        impl_->trackAgentIdx >= 0) {
        
        // Get tensor data
        auto self_obs = selfObservationTensor();
        auto progress = progressTensor();
        
        // Calculate index for the specific agent
        int32_t idx = impl_->trackWorldIdx * consts::numAgents + impl_->trackAgentIdx;
        
        // Get data pointers based on execution mode
        const SelfObservation* obs_data;
        const float* progress_data;
        
        if (impl_->cfg.execMode == ExecMode::CUDA) {
#ifdef MADRONA_CUDA_SUPPORT
            // For CUDA, we need to copy data to host
            SelfObservation host_obs;
            float host_progress;
            
            cudaMemcpy(&host_obs, 
                      ((const SelfObservation*)self_obs.devicePtr()) + idx,
                      sizeof(SelfObservation), 
                      cudaMemcpyDeviceToHost);
            cudaMemcpy(&host_progress,
                      ((const float*)progress.devicePtr()) + idx,
                      sizeof(float),
                      cudaMemcpyDeviceToHost);
            
            obs_data = &host_obs;
            progress_data = &host_progress;
#endif
        } else {
            // For CPU, direct access
            obs_data = ((const SelfObservation*)self_obs.devicePtr()) + idx;
            progress_data = ((const float*)progress.devicePtr()) + idx;
        }
        
        // Print trajectory
        printf("Step %4u: World %d Agent %d: pos=(%.2f,%.2f,%.2f) rot=%.1f° progress=%.2f\n",
               impl_->stepCount++,
               impl_->trackWorldIdx,
               impl_->trackAgentIdx,
               obs_data->globalX,
               obs_data->globalY,
               obs_data->globalZ,
               obs_data->theta * 180.0f / M_PI,
               *progress_data);
    }
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
            consts::numActionComponents,
        });
}

// [BOILERPLATE]
Tensor Manager::rewardTensor() const
{
    return impl_->exportTensor(ExportID::Reward, TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   consts::numAgents,
                                   1,
                               });
}

//[BOILERPLATE]
Tensor Manager::doneTensor() const
{
    return impl_->exportTensor(ExportID::Done, TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   consts::numAgents,
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
                                   consts::numAgents,
                                   SelfObservationFloatCount,
                               });
}


// Removed roomEntityObservationsTensor - no longer tracking room entities


//[BOILERPLATE]
Tensor Manager::stepsRemainingTensor() const
{
    return impl_->exportTensor(ExportID::StepsRemaining,
                               TensorElementType::Int32,
                               {
                                   impl_->cfg.numWorlds,
                                   consts::numAgents,
                                   1,
                               });
}

Tensor Manager::progressTensor() const
{
    return impl_->exportTensor(ExportID::Progress,
                               TensorElementType::Float32,
                               {
                                   impl_->cfg.numWorlds,
                                   consts::numAgents,
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
        consts::numAgents,
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
        consts::numAgents,
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

void Manager::enableAgentTrajectory(int32_t world_idx, int32_t agent_idx)
{
    impl_->enableTrajectoryTracking = true;
    impl_->trackWorldIdx = world_idx;
    impl_->trackAgentIdx = agent_idx;
    impl_->stepCount = 0;
    printf("Trajectory tracking enabled for World %d, Agent %d\n", world_idx, agent_idx);
}

void Manager::disableAgentTrajectory()
{
    impl_->enableTrajectoryTracking = false;
    impl_->trackWorldIdx = -1;
    impl_->trackAgentIdx = -1;
    printf("Trajectory tracking disabled\n");
}

// [BOILERPLATE] Expose render manager for visualization tools
render::RenderManager & Manager::getRenderManager()
{
    return *impl_->renderMgr;
}

}
