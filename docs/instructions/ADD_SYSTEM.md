### Adding a System

To add a new system to process components:

1. **Write the System Function** in `src/sim.cpp`:
   ```cpp
   inline void myNewSystem(Engine &ctx,
                          Position &pos,
                          MyNewComponent &my_comp)
   {
       // System logic here
       my_comp.value1 += pos.x;
   }
   ```

2. **Register System** in `Sim::setupTasks()`:
   ```cpp
   TaskGraphNodeID my_new_sys = builder.addToGraph<ParallelForNode<Engine,
       myNewSystem,
       Position,
       MyNewComponent
   >>({optional_dependencies});
   ```

3. **Define Dependencies**:
    - Systems execute in dependency order
    - Add node ID to dependency array of later systems:
      ```cpp
      TaskGraphNodeID later_sys = builder.addToGraph<...>({
          my_new_sys,  // This system depends on myNewSystem
          other_dep
      });
      ```

4. **Considerations**:
    - **C++ code limitations**: see
    - **Query Scope**: Systems automatically iterate over all entities with required components
    - **Context Access**: Use `ctx` to access world state, entity references
    - **Performance**: Keep systems focused, avoid random memory access
    - **GPU Compatibility**: Use `#ifdef MADRONA_GPU_MODE` for GPU-specific code
    - **Parallelism**: Systems run in parallel across worlds and entities