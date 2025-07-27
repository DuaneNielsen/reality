### Adding a Component

To add a new component to the Madrona ECS:

1. **Define the Component** in `src/types.hpp`:
   ```cpp
   struct MyNewComponent {
       float value1;
       int32_t value2;
   };
   ```

2. **Register the Component** in `Sim::registerTypes()`:
   ```cpp
   registry.registerComponent<MyNewComponent>();
   ```

3. **Add to Archetype** if needed:
   ```cpp
   struct MyEntity : public madrona::Archetype<
       Position,
       Rotation,
       MyNewComponent  // Add here
   > {};
   ```

4. **Export for Python Access** (optional):
    - In `Sim::registerTypes()`, add export column:
      ```cpp
      registry.exportColumn<MyEntity, MyNewComponent>(
          (uint32_t)ExportID::MyNewComponent);
      ```
    - Add to `ExportID` enum in `src/types.hpp`
    - Map tensor in `src/mgr.cpp`:
      ```cpp
      exported.myNewComponent = gpu_exec.getExported((uint32_t)ExportID::MyNewComponent);
      ```

5. **Initialize Component Values**:
    - Set initial values when creating entities
    - Update in reset systems if component should reset
