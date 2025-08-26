# Code Generation Tools

This directory contains Python scripts that generate code from C++ structures and constants.

## Scripts

### generate_dataclass_structs.py
Generates Python dataclass wrappers from compiled C++ structs using `pahole` to extract memory layout.

### generate_python_constants.py
Generates Python constants from C++ headers using libclang to parse the AST.

## Adding Support for New C++ Types

When you need to add support for a new C++ type (like we did for `Quat`), follow this pattern:

### 1. Type Detection
The generator uses `pahole` to extract type information from compiled binaries. Pahole shows the actual C++ type, e.g., `struct Quat` for quaternion types.

### 2. Update Type Mapping
In `generate_dataclass_structs.py`, update the type mapping functions:

```python
def map_c_to_python_type(c_type: str, size: int, is_array: bool = False) -> str:
    # Check for struct types first
    if c_type.startswith("struct "):
        struct_name = c_type[7:]  # Remove "struct " prefix
        if struct_name == "YourNewType":
            # Map to appropriate Python type
            if is_array:
                return "List[YourPythonType]"
            return "YourPythonType"
    # ... existing mappings
```

### 3. Update ctypes Mapping
Map the C++ type to its ctypes equivalent:

```python
def map_c_to_ctypes(c_type: str, size: int) -> str:
    if c_type.startswith("struct "):
        struct_name = c_type[7:]
        if struct_name == "YourNewType":
            # Return ctypes structure
            # For example, Quat is 4 floats: w, x, y, z
            return "ctypes.c_float * 4"
    # ... existing mappings
```

### 4. Handle Special Initialization
For types that need special initialization (like quaternions needing identity values), detect them in the field generation:

```python
# In generate_dataclass_struct()
if field.type_str.startswith("struct YourType"):
    factory_name = f"_make_yourtype_array_{field.array_size}"
    factories_needed.add((factory_name, field.array_size, "yourtype"))
    meta_str = f"meta((your_ctypes_structure) * {field.array_size})"
```

### 5. Generate Factory Functions
Add factory generation for your type:

```python
# In generate_python_bindings() factory generation section
if type_name == "yourtype":
    output_lines.append(f"def {factory_name}():")
    output_lines.append(f'    """Factory for {size}-element YourType array"""')
    output_lines.append(f"    return [your_default_value] * {size}")
```

## Example: Quaternion Support

We added quaternion support following this pattern:

1. **Detection**: Pahole shows `struct Quat` for quaternion fields
2. **Python Type**: Maps to `Tuple[float, float, float, float]` (w, x, y, z)
3. **ctypes**: Maps to `(ctypes.c_float * 4)` for array of 4 floats
4. **Factory**: Creates identity quaternions `(1.0, 0.0, 0.0, 0.0)`

This ensures quaternions are properly initialized to valid identity values rather than invalid (0,0,0,0) which would cause NaN in physics calculations.

## Running the Generators

### Generate Dataclass Structures
```bash
uv run python codegen/generate_dataclass_structs.py \
    libmadrona_escape_room_c_api.so \
    madrona_escape_room/dataclass_structs.py
```

### Generate Python Constants
```bash
uv run python codegen/generate_python_constants.py \
    -I external/madrona/include \
    -I src \
    -o madrona_escape_room/generated_constants.py
```

## Dependencies

- **pahole**: For extracting struct memory layout from compiled binaries
- **libclang**: For parsing C++ headers and extracting constants
- **cdataclass**: For creating C-compatible Python dataclasses

## Integration with CMake

The generators are integrated into the CMake build process. See `src/CMakeLists.txt` for the custom commands that run these generators automatically during the build.