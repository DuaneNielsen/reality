# Step-by-Step Guide to Remove Doors from Madrona Escape Room

This guide documents the complete process of removing the door system from the Madrona Escape Room simulation. The door system was a major gameplay element that blocked passages between rooms.

## Overview

The door system consisted of:
- Physical door entities that blocked room exits
- Animation system to move doors up/down
- Observation system for agents to see door state
- Component system tracking door open/closed state
- Asset loading for door meshes and materials

## Step-by-Step Removal Process

### Step 1: Remove Door Components from types.hpp

**File**: `src/types.hpp`

1. **Remove DoorObservation struct** (lines 119-125):
   ```cpp
   // DELETE THIS:
   struct DoorObservation {
       PolarObservation polar;
       float isOpen; // 1.0 when open, 0.0 when closed.
   };
   ```

2. **Remove OpenState struct** (lines 185-188):
   ```cpp
   // DELETE THIS:
   struct OpenState {
       bool isOpen;
   };
   ```

3. **Remove DoorProperties struct** (lines 191-195):
   ```cpp
   // DELETE THIS:
   struct DoorProperties {
       bool isPersistent;
   };
   ```

4. **Remove Door from EntityType enum** (line 169):
   ```cpp
   enum class EntityType : uint32_t {
       None,
       Cube,
       Wall,
       Agent,
       Door,  // DELETE THIS LINE
       NumTypes,
   };
   ```

5. **Remove door field from Room struct** (line 211):
   ```cpp
   struct Room {
       Entity entities[consts::maxEntitiesPerRoom];
       Entity walls[2];
       Entity door;  // DELETE THIS LINE
   };
   ```

6. **Remove DoorObservation from Agent archetype** (line 237):
   ```cpp
   // Change from:
   SelfObservation, PartnerObservations, RoomEntityObservations, DoorObservation, Lidar, StepsRemaining,
   // To:
   SelfObservation, PartnerObservations, RoomEntityObservations, Lidar, StepsRemaining,
   ```

7. **Remove DoorEntity archetype** (lines 251-254):
   ```cpp
   // DELETE THIS ENTIRE ARCHETYPE:
   struct DoorEntity
       : public madrona::Archetype<RigidBody, OpenState, DoorProperties, EntityType, madrona::render::Renderable> {};
   ```

### Step 2: Remove Door from Simulation Enums (sim.hpp)

**File**: `src/sim.hpp`

1. **Remove DoorObservation from ExportID enum** (line 31):
   ```cpp
   enum class ExportID : uint32_t {
       Reset,
       Action,
       Reward,
       Done,
       SelfObservation,
       PartnerObservations,
       RoomEntityObservations,
       DoorObservation,  // DELETE THIS LINE
       Lidar,
       StepsRemaining,
       NumExports,
   };
   ```

2. **Remove Door from SimObject enum** (line 42):
   ```cpp
   enum class SimObject : uint32_t {
       Cube,
       Wall,
       Door,  // DELETE THIS LINE
       Agent,
       Plane,
       NumObjects,
   };
   ```

### Step 3: Update Component Registration (sim.cpp)

**File**: `src/sim.cpp`

1. **Remove component registrations** in `registerTypes()` (lines 52-55):
   ```cpp
   // DELETE THESE LINES:
   registry.registerComponent<DoorObservation>();
   registry.registerComponent<OpenState>();
   registry.registerComponent<DoorProperties>();
   ```

2. **Remove DoorEntity archetype registration** (line 69):
   ```cpp
   // DELETE THIS LINE:
   registry.registerArchetype<DoorEntity>();
   ```

3. **Remove DoorObservation export** (lines 88-89):
   ```cpp
   // DELETE THESE LINES:
   registry.exportColumn<Agent, DoorObservation>(
       (uint32_t)ExportID::DoorObservation);
   ```

### Step 4: Remove Door Systems (sim.cpp)

**File**: `src/sim.cpp`

1. **Remove setDoorPositionSystem function** (lines 256-276):
   ```cpp
   // DELETE THIS ENTIRE FUNCTION:
   inline void setDoorPositionSystem(Engine &,
                                     Position &pos,
                                     OpenState &open_state)
   {
       if (open_state.isOpen) {
           // Put underground
           if (pos.z > -4.5f) {
               pos.z += -consts::doorSpeed * consts::deltaT;
           }
       }
       else if (pos.z < 0.0f) {
           // Put back on surface
           pos.z += consts::doorSpeed * consts::deltaT;
       }
       
       if (pos.z >= 0.0f) {
           pos.z = 0.0f;
       }
   }
   ```

2. **Remove doorOpenSystem function** (lines 279-285):
   ```cpp
   // DELETE THIS ENTIRE FUNCTION:
   inline void doorOpenSystem(Engine &ctx,
                              OpenState &open_state,
                              const DoorProperties &props)
   {
       // Doors always remain open
       open_state.isOpen = true;
   }
   ```

3. **Remove door observation collection** from `collectObservationsSystem()`:
   - Remove `DoorObservation &door_obs` parameter from function signature (line 354)
   - Remove door observation logic (lines 410-415):
   ```cpp
   // DELETE THESE LINES:
   Entity cur_door = room.door;
   Vector3 door_pos = ctx.get<Position>(cur_door);
   OpenState door_open_state = ctx.get<OpenState>(cur_door);

   door_obs.polar = xyToPolar(to_view.rotateVec(door_pos - pos));
   door_obs.isOpen = door_open_state.isOpen ? 1.f : 0.f;
   ```

4. **Remove door entity destruction** from `cleanupWorld()` (line 112):
   ```cpp
   // DELETE THIS LINE:
   ctx.destroyRenderableEntity(room.door);
   ```

### Step 5: Update Task Graph (sim.cpp)

**File**: `src/sim.cpp` in `setupTasks()` function

1. **Remove door position system node** (lines 581-588):
   ```cpp
   // DELETE THIS:
   auto set_door_pos_sys = builder.addToGraph<ParallelForNode<Engine,
       setDoorPositionSystem,
           Position,
           OpenState
       >>({move_sys});
   ```

2. **Update broadphase dependency** (line 591):
   ```cpp
   // Change from:
   auto broadphase_setup_sys = phys::PhysicsSystem::setupBroadphaseTasks(
       builder, {set_door_pos_sys});
   // To:
   auto broadphase_setup_sys = phys::PhysicsSystem::setupBroadphaseTasks(
       builder, {move_sys});
   ```

3. **Remove door open system node** (lines 616-623):
   ```cpp
   // DELETE THIS:
   auto door_open_sys = builder.addToGraph<ParallelForNode<Engine,
       doorOpenSystem,
           OpenState,
           DoorProperties
       >>({phys_done});
   ```

4. **Update reward system dependency** (line 632):
   ```cpp
   // Change from:
   >>({door_open_sys});
   // To:
   >>({phys_done});
   ```

5. **Remove DoorObservation from collect observations** (line 680):
   ```cpp
   // Change from:
   SelfObservation,
   PartnerObservations,
   RoomEntityObservations,
   DoorObservation
   // To:
   SelfObservation,
   PartnerObservations,
   RoomEntityObservations
   ```

6. **Remove door GPU sorting** (lines 712-714):
   ```cpp
   // DELETE THESE LINES:
   auto sort_walls = queueSortByWorld<DoorEntity>(
       builder, {sort_phys_objects});
   (void)sort_walls;
   ```

### Step 6: Update Level Generation (level_gen.cpp)

**File**: `src/level_gen.cpp`

1. **Remove door entity creation** from `makeEndWall()` (lines 305-324):
   ```cpp
   // DELETE THIS ENTIRE BLOCK:
   Entity door = ctx.makeRenderableEntity<DoorEntity>();
   setupRigidBodyEntity(
       ctx,
       door,
       Vector3 {
           door_center - consts::worldWidth / 2.f,
           y_pos,
           0,
       },
       Quat { 1, 0, 0, 0 },
       SimObject::Door,
       EntityType::Door,
       ResponseType::Static,
       Diag3x3 {
           consts::doorWidth * 0.8f,
           consts::wallWidth,
           1.75f,
       });
   registerRigidBodyEntity(ctx, door, SimObject::Door);
   ctx.get<OpenState>(door).isOpen = false;
   ```

2. **Remove door assignment to room** (line 328):
   ```cpp
   // DELETE THIS LINE:
   room.door = door;
   ```

3. **Remove setupDoor function** (lines 359-367):
   ```cpp
   // DELETE THIS ENTIRE FUNCTION:
   static void setupDoor(Engine &ctx,
                         Entity door)
   {
       // Doors are always open
       ctx.get<OpenState>(door).isOpen = true;
       
       DoorProperties &props = ctx.get<DoorProperties>(door);
       props.isPersistent = true;
   }
   ```

4. **Remove setupDoor calls** from all room generation functions:
   - `makeEmptyRoom()` - line 377
   - `makeEmptyRoomVariant()` - line 385
   - `makeCubeObstacleRoom()` - line 395
   - `makeCubeRoom()` - line 427

5. **Update cube positioning** in `makeCubeObstacleRoom()`:
   ```cpp
   // Replace door position reference (lines 397-398):
   Vector3 door_pos = ctx.get<Position>(room.door);
   
   // With fixed coordinates:
   float door_x = 0.f;  // Center of room
   float door_y = y_max;
   
   // Update all cube positions to use door_x, door_y instead of door_pos.x, door_pos.y
   ```

### Step 7: Remove Door Assets (mgr.cpp)

**File**: `src/mgr.cpp`

1. **Remove door render asset path** (lines 233-234):
   ```cpp
   // DELETE THESE LINES:
   render_asset_paths[(size_t)SimObject::Door] =
       (std::filesystem::path(DATA_DIR) / "wall_render.obj").string();  // Reuses wall mesh
   ```

2. **Remove door material assignment** (line 273):
   ```cpp
   // DELETE THIS LINE:
   render_assets->objects[(CountT)SimObject::Door].meshes[0].materialIDX = 5;   // Red
   ```

3. **Remove door physics asset path** (lines 310-311):
   ```cpp
   // DELETE THESE LINES:
   asset_paths[(size_t)SimObject::Door] =
       (std::filesystem::path(DATA_DIR) / "wall_collision.obj").string();
   ```

4. **Remove door physics setup** (lines 381-385):
   ```cpp
   // DELETE THIS:
   setupHull(SimObject::Door, 0.f, {       // Static (infinite mass)
       .muS = 0.5f,
       .muD = 0.5f,
   });
   ```

5. **Remove doorObservationTensor function** (lines 702-713):
   ```cpp
   // DELETE THIS ENTIRE FUNCTION:
   Tensor Manager::doorObservationTensor() const
   {
       return impl_->exportTensor(ExportID::DoorObservation,
                                  TensorElementType::Float32,
                                  {
                                      impl_->cfg.numWorlds,
                                      consts::numAgents,
                                      3,
                                  });
   }
   ```

### Step 8: Update Manager Interface (mgr.hpp)

**File**: `src/mgr.hpp`

1. **Remove doorObservationTensor declaration** (line 45):
   ```cpp
   // DELETE THIS LINE:
   madrona::py::Tensor doorObservationTensor() const;
   ```

### Step 9: Update Python Bindings (bindings.cpp)

**File**: `src/bindings.cpp`

1. **Remove door_observation_tensor binding** (lines 48-49):
   ```cpp
   // DELETE THESE LINES:
   .def("door_observation_tensor",
        &Manager::doorObservationTensor)
   ```

### Step 10: Update Viewer (viewer.cpp)

**File**: `src/viewer.cpp`

1. **Remove door printer creation** (line 139):
   ```cpp
   // DELETE THIS LINE:
   auto door_printer = mgr.doorObservationTensor().makePrinter();
   ```

2. **Remove door observation printing** (lines 154-155):
   ```cpp
   // DELETE THESE LINES:
   printf("Door\n");
   door_printer.print();
   ```

## Verification Steps

After completing all changes:

1. **Build the project** to ensure no compilation errors
2. **Run the simulation** to verify doors are removed
3. **Check performance** - should see ~24% improvement (501 FPS → 624 FPS)
4. **Test training** to ensure observation spaces are correct
5. **Verify physics** - agents should be able to move freely between rooms

## Summary

This removal eliminates:
- 3 component structs (OpenState, DoorProperties, DoorObservation)
- 1 archetype (DoorEntity)
- 2 enum values (SimObject::Door, EntityType::Door)
- 2 systems (setDoorPositionSystem, doorOpenSystem)
- 1 export tensor (doorObservationTensor)
- All associated asset loading and entity creation

The result is a simpler simulation where rooms have open passages instead of doors, improving both performance and code complexity.