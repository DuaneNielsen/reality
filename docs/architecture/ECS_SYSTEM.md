# Madrona ECS System Architecture

This document provides a comprehensive overview of the Madrona Entity Component System (ECS) internals, including memory layout, data structures, and access patterns.

## Core Type Definitions

```cpp
// Entity: Unique identifier with generation for safe recycling
struct Entity {
    uint32_t gen;  // Generation counter
    int32_t id;    // Entity ID (-1 = invalid, 0xFFFFFFFF = none)
};

// Loc: Direct table position
struct Loc {
    uint32_t archetype;  // Archetype table ID (0xFFFFFFFF = invalid)
    int32_t row;         // Row within table
};

// Type identifiers
struct ComponentID { uint32_t id; };
struct ArchetypeID { uint32_t id; };
struct WorldID { int32_t idx; };

// Component metadata
struct TypeInfo {
    uint32_t alignment;  // Memory alignment requirement
    uint32_t numBytes;   // Size in bytes
};
```

## Memory Architecture Overview

```
┌─────────────── StateManager ──────────────┐
│                                            │
│  ┌──────────────────────────────────┐     │
│  │       EntityStore                 │     │
│  │  ┌─────────────────────────┐     │     │
│  │  │  IDMap<Entity, Loc>     │     │     │
│  │  │  Entity ID → Location   │     │     │
│  │  └─────────────────────────┘     │     │
│  └──────────────────────────────────┘     │
│                                            │
│  ┌──────────────────────────────────┐     │
│  │    archetype_stores_             │     │
│  │    [ArchetypeStore array]        │     │
│  │                                  │     │
│  │    [0]: Agent Archetype ────────►│────►│──┐
│  │    [1]: Wall Archetype           │     │  │
│  │    [2]: Cube Archetype           │     │  │
│  │    ...                           │     │  │
│  └──────────────────────────────────┘     │  │
│                                            │  │
│  ┌──────────────────────────────────┐     │  │
│  │    component_infos_              │     │  │
│  │    [Component metadata]          │     │  │
│  │                                  │     │  │
│  │    [0]: Entity {8 bytes, 8 align}│     │  │
│  │    [1]: Position {12B, 4 align}  │     │  │
│  │    [2]: Velocity {12B, 4 align}  │     │  │
│  │    [3]: Action {16B, 4 align}    │     │  │
│  └──────────────────────────────────┘     │  │
└────────────────────────────────────────────┘  │
                                                 │
         ┌───────────────────────────────────────┘
         ▼
┌─────── ArchetypeStore (e.g., Agent) ──────┐
│                                            │
│  componentOffset: 12                      │
│  numComponents: 4                         │
│                                            │
│  ┌──────── ColumnMap ──────────┐          │
│  │ ComponentID → Column Index  │          │
│  │                             │          │
│  │  Entity(0) → 0              │          │
│  │  Position(1) → 1            │          │
│  │  Velocity(2) → 2            │          │
│  │  Action(3) → 3              │          │
│  └─────────────────────────────┘          │
│                                            │
│  ┌──────── TableStorage ────────┐         │
│  │                               │         │
│  │  Table (or Tables array)  ───┼────────►│──┐
│  │                               │         │  │
│  └───────────────────────────────┘         │  │
└─────────────────────────────────────────────┘  │
                                                 │
         ┌───────────────────────────────────────┘
         ▼
┌──────────────── Table ─────────────────┐
│                                         │
│  num_rows_: 3                          │
│  num_allocated_rows_: 10               │
│  num_components_: 4                    │
│                                         │
│  columns_: [InlineArray<void*, 128>]   │
│    [0] → Entity column ─────────┐      │
│    [1] → Position column ───┐   │      │
│    [2] → Velocity column ─┐ │   │      │
│    [3] → Action column ─┐ │ │   │      │
│                         │ │ │   │      │
│  bytes_per_column_:     │ │ │   │      │
│    [0]: 8 bytes         │ │ │   │      │
│    [1]: 12 bytes        │ │ │   │      │
│    [2]: 12 bytes        │ │ │   │      │
│    [3]: 16 bytes        │ │ │   │      │
└─────────────────────────┼─┼─┼───┼──────┘
                          │ │ │   │
     ┌────────────────────┘ │ │   │
     │  ┌───────────────────┘ │   │
     │  │  ┌──────────────────┘   │
     │  │  │  ┌───────────────────┘
     ▼  ▼  ▼  ▼
┌────────────────── Column Storage ──────────────┐
│                                                 │
│  Action    Velocity   Position    Entity       │
│  Column    Column     Column      Column       │
│  ┌──────┐  ┌──────┐   ┌──────┐   ┌──────────┐ │
│  │16B   │  │12B   │   │12B   │   │8B        │ │ Row 0
│  ├──────┤  ├──────┤   ├──────┤   ├──────────┤ │
│  │16B   │  │12B   │   │12B   │   │8B        │ │ Row 1
│  ├──────┤  ├──────┤   ├──────┤   ├──────────┤ │
│  │16B   │  │12B   │   │12B   │   │8B        │ │ Row 2
│  ├──────┤  ├──────┤   ├──────┤   ├──────────┤ │
│  │...   │  │...   │   │...   │   │...       │ │ (allocated)
│  └──────┘  └──────┘   └──────┘   └──────────┘ │
│                                                 │
│  malloc'd  malloc'd   malloc'd   malloc'd      │
│  memory    memory     memory     memory         │
└─────────────────────────────────────────────────┘
```

## Access Path Examples

### 1. Getting Component by Entity

```
ctx.get<Position>(entity)
         │
         ▼
┌──── Step 1: Entity → Loc ────┐
│ EntityStore::getLoc(entity)  │
│                               │
│ IDMap lookup:                 │
│   entity.id = 42             │
│   → Loc{archetype=0, row=1}  │
└───────────────────────────────┘
         │
         ▼
┌──── Step 2: Get Archetype ────┐
│ archetype_stores_[0]          │
│ → ArchetypeStore for Agent    │
└────────────────────────────────┘
         │
         ▼
┌─── Step 3: Component → Column ─┐
│ ColumnMap lookup:              │
│   Position ID = 1              │
│   → Column index = 1           │
└─────────────────────────────────┘
         │
         ▼
┌──── Step 4: Access Table ──────┐
│ TableStorage.column<Position>( │
│     column_idx=1)              │
│ → Position* base_ptr           │
└─────────────────────────────────┘
         │
         ▼
┌──── Step 5: Index by Row ──────┐
│ return base_ptr[row]           │
│ = base_ptr[1]                  │
│ = Position at row 1            │
└─────────────────────────────────┘
```

### 2. Query Iteration

```
Query<Position, Velocity> q = ctx.query<Position, Velocity>();
ctx.iterateQuery(q, lambda);
         │
         ▼
┌──── Query Structure ────────────┐
│ QueryRef:                       │
│   offset: 1000                  │
│   numMatchingArchetypes: 2      │
│   numComponents: 2              │
└──────────────────────────────────┘
         │
         ▼
┌──── Query Data Array ───────────┐
│ query_state_.queryData[1000...] │
│                                  │
│ [1000]: archetype_id = 0 (Agent)│
│ [1001]: Position column = 1     │
│ [1002]: Velocity column = 2     │
│                                  │
│ [1003]: archetype_id = 2 (Cube) │
│ [1004]: Position column = 1     │
│ [1005]: Velocity column = 2     │
└──────────────────────────────────┘
         │
         ▼
┌──── Iteration Process ──────────┐
│ For each archetype:             │
│   1. Get Table from archetype   │
│   2. Get column pointers:       │
│      - Entity* entities         │
│      - Position* positions      │
│      - Velocity* velocities     │
│   3. For i = 0 to num_rows:     │
│      if (entities[i] != none)   │
│        lambda(positions[i],     │
│               velocities[i])    │
└──────────────────────────────────┘
```

## Table Memory Layout Detail

### Table Structure

```cpp
class Table {
    uint32_t num_rows_;              // Current active rows
    uint32_t num_allocated_rows_;    // Allocated capacity
    uint32_t num_components_;        // Number of columns
    InlineArray<void*, 128> columns_;           // Column pointers
    InlineArray<uint32_t, 128> bytes_per_column_; // Bytes per element
};
```

### Physical Memory Layout

```
Physical Memory Layout for Agent Table:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Entity Column (8 bytes/row):
┌────────┬────────┬────────┬────────┐
│gen: 0  │gen: 0  │gen: 0  │  ...   │
│id: 42  │id: 43  │id: 44  │        │
└────────┴────────┴────────┴────────┘
0x1000   0x1008   0x1010   0x1018

Position Column (12 bytes/row):
┌──────────────┬──────────────┬──────────────┐
│x:1.0 y:2.0   │x:4.0 y:5.0   │x:7.0 y:8.0   │
│z:3.0         │z:6.0         │z:9.0         │
└──────────────┴──────────────┴──────────────┘
0x2000        0x200C        0x2018

Velocity Column (12 bytes/row):
┌──────────────┬──────────────┬──────────────┐
│x:0.0 y:1.0   │x:1.0 y:0.0   │x:0.0 y:0.0   │
│z:0.0         │z:0.0         │z:1.0         │
└──────────────┴──────────────┴──────────────┘
0x3000        0x300C        0x3018

Action Column (16 bytes/row):
┌────────────────┬────────────────┬────────────────┐
│moveAmount: 1.0 │moveAmount: 0.5 │moveAmount: 0.0 │
│moveAngle: 0.0  │moveAngle: 90.0 │moveAngle: 0.0  │
│rotate: 0.0     │rotate: 0.0     │rotate: 45.0    │
│grab: 0         │grab: 1         │grab: 0         │
└────────────────┴────────────────┴────────────────┘
0x4000          0x4010          0x4020
```

### Component Access Calculation

```cpp
// For component access at Table[row][column]:
void* Table::getValue(uint32_t column_idx, uint32_t row) {
    return (char*)columns_[column_idx] +
           (row * bytes_per_column_[column_idx]);
}

// Example: Get Position at row 2, column 1
// columns_[1] = 0x2000 (Position column base)
// bytes_per_column_[1] = 12
// Address = 0x2000 + (2 * 12) = 0x2018
```

## Multi-World Mode Structure

```
┌────── TableStorage (Multi-World) ──────┐
│                                         │
│  union {                                │
│    HeapArray<Table> tbls;  ◄────┐      │
│    Fixed fixed;                  │      │
│  };                              │      │
│  maxNumPerWorld: 1000            │      │
└───────────────────────────────┼─────────┘
                                 │
    ┌────────────────────────────┘
    ▼
┌────── HeapArray<Table> ────────┐
│                                 │
│  [0]: Table for World 0        │
│  [1]: Table for World 1        │
│  [2]: Table for World 2        │
│  ...                           │
│  [N-1]: Table for World N-1    │
│                                 │
│  Each table is independent     │
│  with its own columns          │
└─────────────────────────────────┘
```

## Query Cache Structure

```
┌────── QueryState ───────────────┐
│                                  │
│  VirtualArray<uint32_t> data    │
│                                  │
│  Query A: [offset 0]            │
│    [0]: archetype_id            │
│    [1]: col_idx for Component1  │
│    [2]: col_idx for Component2  │
│    [3]: archetype_id            │
│    [4]: col_idx for Component1  │
│    [5]: col_idx for Component2  │
│                                  │
│  Query B: [offset 6]            │
│    [6]: archetype_id            │
│    [7]: col_idx for Component1  │
│    ...                          │
└──────────────────────────────────┘
```

## Key Data Structures

### StateManager - Central ECS Coordinator

```cpp
class StateManager {
private:
    EntityStore entity_store_;                    // Entity → Loc mapping
    DynArray<Optional<TypeInfo>> component_infos_; // Component metadata
    DynArray<ComponentID> archetype_components_;  // Component lists
    DynArray<Optional<ArchetypeStore>> archetype_stores_; // All tables
    DynArray<Optional<BundleInfo>> bundle_infos_; // Component groups

#ifdef MADRONA_MW_MODE
    uint32_t num_worlds_;
    SpinLock register_lock_;
#endif
};
```

### EntityStore - Entity to Location Mapping

```cpp
class EntityStore {
private:
    using Map = IDMap<Entity, Loc, LockedMapStore>;
    Map map_;
public:
    using Cache = Map::Cache;

    Loc getLoc(Entity e) const;
    void setLoc(Entity e, Loc loc);
    Entity newEntity(Cache &cache);
    void freeEntity(Cache &cache, Entity e);
};
```

### ArchetypeStore - Table + Metadata

```cpp
struct ArchetypeStore {
    uint32_t componentOffset;    // Offset in component array
    uint32_t numComponents;      // Number of columns
    TableStorage tblStorage;     // The actual table(s)
    ColumnMap columnLookup;      // ComponentID → column index
};
```

## Access Pattern Performance Characteristics

| Access Method | Lookup Cost | Use Case |
|---------------|-------------|----------|
| `get<T>(Entity)` | Hash lookup + column lookup | General entity access |
| `get<T>(Loc)` | Column lookup only | When location is cached |
| `getDirect<T>(col, Loc)` | Direct indexing | Hot paths with known columns |
| `iterateQuery()` | Cached archetype scan | Bulk processing |
| `exportColumn<T>()` | Direct column pointer | External system integration |

## Memory Management

### Table Growth Strategy
- Initial allocation: `max(init_rows, 1)`
- Growth factor: `max(10, current_size * 2)`
- Memory reallocation: `realloc()` for each column independently
- Max table size: 2^28 rows (268 million entities)

### Entity ID Recycling
- Generation counter prevents ABA problems
- Free entity IDs are recycled with incremented generation
- `Entity::none()` = `{0xFFFFFFFF, 0xFFFFFFFF}`
- `Loc::none()` = `{0xFFFFFFFF, 0}`

### Query Caching
- Queries computed once at first use
- Results cached in global `QueryState`
- Reference counting for memory management
- Thread-safe query creation via atomic operations

## Tensor Export Mapping System

The export mapping system creates a bridge between ECS component data and external tensor formats. The key insight is that this system establishes **persistent mappings** from integer slot IDs to specific component column memory addresses.

### 1. Registration Phase: Creating the Mapping

```cpp
// Application code during startup
registry.exportColumn<Agent, Action>((uint32_t)ExportID::Action);  // Slot 2
registry.exportColumn<Agent, Reward>((uint32_t)ExportID::Reward);  // Slot 3
registry.exportSingleton<WorldReset>((uint32_t)ExportID::Reset);   // Slot 0
```

#### Mapping Storage Data Structures

```cpp
// CPU Executor
struct ThreadPoolExecutor::Impl {
    StateManager stateMgr;
    HeapArray<void *> exportPtrs;  // ◄── THE MAPPING TABLE
    // exportPtrs[slot] = pointer to component data
};

// GPU Executor
struct MWCudaExecutor::EngineState {
    HeapArray<void *> exportedColumns;  // ◄── THE MAPPING TABLE
    // exportedColumns[slot] = GPU pointer to component data
};

// Registry receives pointer to the mapping table
class ECSRegistry {
    StateManager *state_mgr_;
    void **export_ptrs_;  // ◄── Points to executor's mapping table
};
```

#### Step-by-Step Mapping Creation

```
1. registry.exportColumn<Agent, Action>(ExportID::Action)
         │
         ▼
2. ECSRegistry::exportColumn(slot=2):
   export_ptrs_[2] = state_mgr_->exportColumn<Agent, Action>()
         │
         ▼
3. StateManager::exportColumn<Agent, Action>():
   return exportColumn(archetype_id=0, component_id=3)
         │
         ▼
4. StateManager::exportColumn(archetype_id=0, component_id=3):

   // Find the archetype
   auto &archetype = *archetype_stores_[0];  // Agent archetype

   // Map ComponentID → Column Index
   uint32_t col_idx = *archetype.columnLookup.lookup(3);  // Action is column 3

   // For Single-World: Return direct pointer
   return archetype.tblStorage.tbl.data(3);  // Direct column pointer

   // For Multi-World: Create ExportJob and return virtual buffer
   export_jobs_.push_back(ExportJob {
       .archetypeIdx = 0,      // ◄── REMEMBER: Agent archetype
       .columnIdx = 3,         // ◄── REMEMBER: Action column
       .numBytesPerRow = 16,   // ◄── REMEMBER: sizeof(Action)
       .mem = VirtualRegion(1GB)
   });
   return virtual_buffer_pointer;
         │
         ▼
5. Result stored in mapping table:
   export_ptrs_[2] = pointer_to_action_data
```

### 2. Mapping Lookup Data Structures

#### ComponentID → Column Index Mapping (Per Archetype)

```cpp
struct ArchetypeStore {
    ColumnMap columnLookup;  // ◄── ComponentID → Column Index
    // Example for Agent archetype:
    // columnLookup[Entity(0)] = 0
    // columnLookup[Position(1)] = 1
    // columnLookup[Velocity(2)] = 2
    // columnLookup[Action(3)] = 3
    // columnLookup[Reward(4)] = 4
};
```

#### ExportJob Mapping (Multi-World Only)

```cpp
// StateManager stores array of ExportJobs
DynArray<ExportJob> export_jobs_;

struct ExportJob {
    uint32_t archetypeIdx;    // Which archetype table (0=Agent, 1=Wall, etc.)
    uint32_t columnIdx;       // Which column in that table (0=Entity, 1=Position, etc.)
    uint32_t numBytesPerRow;  // Component size for memcpy calculations
    VirtualRegion mem;        // Large virtual buffer for concatenated data
};

// Example export_jobs_ contents after registration:
// [0]: {archetypeIdx=0, columnIdx=3, numBytesPerRow=16, mem=...}  // Action export
// [1]: {archetypeIdx=0, columnIdx=4, numBytesPerRow=4, mem=...}   // Reward export
```

### 3. Data Copy Operations: Using the Mapping

#### How copyOutExportedColumns Knows What to Read

```cpp
void StateManager::copyOutExportedColumns() {
    // Iterate through ALL registered export jobs
    for (ExportJob &export_job : export_jobs_) {

        // 1. LOOKUP: Use stored mapping to find source
        auto &archetype = *archetype_stores_[export_job.archetypeIdx];
        //              ▲ Stored during registration   ▲ Points to Agent table

        CountT cumulative_rows = 0;

        // 2. ITERATE: Copy from each world's table
        for (Table &tbl : archetype.tblStorage.tbls) {
            CountT num_rows = tbl.numRows();
            if (num_rows == 0) continue;

            // 3. READ: Use stored column index to find source data
            void *source = tbl.data(export_job.columnIdx);
            //                      ▲ Stored during registration (e.g., 3 for Action)

            // 4. WRITE: Use stored memory region as destination
            void *dest = (char*)export_job.mem.ptr() +
                        cumulative_rows * export_job.numBytesPerRow;
            //                   ▲ Stored virtual buffer    ▲ Stored component size

            // 5. COPY: Transfer the data
            memcpy(dest, source, export_job.numBytesPerRow * num_rows);

            cumulative_rows += num_rows;
        }
    }
}
```

#### Memory Layout During Copy

```
Agent Table (World 0):     Agent Table (World 1):     Agent Table (World 2):
┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
│ Entity │ Pos │ Action│    │ Entity │ Pos │ Action│    │ Entity │ Pos │ Action│
│   [0]  │ ... │{1,0,0}│    │   [0]  │ ... │{0,1,0}│    │   [0]  │ ... │{1,1,0}│
│   [1]  │ ... │{0,1,0}│    │   [1]  │ ... │{1,0,0}│    │   [1]  │ ... │{0,0,1}│
└─────────────────────┘    └─────────────────────┘    └─────────────────────┘
        Column 3                   Column 3                   Column 3
           │                          │                          │
           ▼                          ▼                          ▼
   ┌─────────────── Export Buffer (export_jobs_[0].mem) ─────────────────┐
   │ {1,0,0} {0,1,0}    {0,1,0} {1,0,0}    {1,1,0} {0,0,1}              │
   │ ◄─World 0 data──► ◄─World 1 data──► ◄─World 2 data──►              │
   │                                                                     │
   │ This buffer is at export_ptrs_[ExportID::Action]                   │
   └─────────────────────────────────────────────────────────────────────┘
```

### 4. Runtime Access: Retrieving the Mapping

```cpp
// Manager requests tensor
Tensor Manager::actionTensor() const {
    return impl_->exportTensor(ExportID::Action, TensorElementType::Int32, shape);
}

// Implementation looks up the mapping
void* CPUImpl::exportTensor(ExportID slot, ...) {
    void *dev_ptr = cpuExec.getExported((uint32_t)slot);  // ◄── LOOKUP
    return Tensor(dev_ptr, type, dims);
}

// getExported simply indexes into the mapping table
void* ThreadPoolExecutor::getExported(CountT slot) const {
    return impl_->exportPtrs[slot];  // ◄── DIRECT ARRAY ACCESS
    //     ▲ The mapping table created during registration
}
```

### 5. Complete Mapping Flow

```
Registration Time:
┌─────────────────────────────────────────────────────────────────┐
│ Code: registry.exportColumn<Agent,Action>(ExportID::Action)     │
│   ↓                                                             │
│ 1. Resolve types: Agent→archetype_id=0, Action→component_id=3   │
│ 2. Lookup column: archetype_stores_[0].columnLookup[3] = col_3  │
│ 3. Store mapping:                                               │
│    • export_ptrs_[ExportID::Action] = column_pointer            │
│    • export_jobs_[i] = {archetype=0, column=3, size=16}        │
└─────────────────────────────────────────────────────────────────┘

Runtime - Copy Out:
┌─────────────────────────────────────────────────────────────────┐
│ copyOutExportedColumns():                                       │
│   ↓                                                             │
│ 1. For each export_jobs_[i]:                                   │
│    • Read from: archetype_stores_[job.archetypeIdx]            │
│                   .tblStorage.tbls[world].data(job.columnIdx) │
│    • Write to: job.mem.ptr() + offset                          │
│ 2. Result: export_ptrs_[slot] points to concatenated data      │
└─────────────────────────────────────────────────────────────────┘

Runtime - Tensor Access:
┌─────────────────────────────────────────────────────────────────┐
│ manager.actionTensor():                                         │
│   ↓                                                             │
│ 1. getExported(ExportID::Action) returns export_ptrs_[slot]     │
│ 2. Wrap pointer in Tensor with shape [worlds, agents, 3]       │
│ 3. Python gets zero-copy access to ECS data                    │
└─────────────────────────────────────────────────────────────────┘
```

### Key Insights

1. **Slot-based indirection**: Export slots decouple external API from internal ECS layout
2. **Persistent mapping storage**: `export_ptrs_[]` and `export_jobs_[]` remember all the lookup information
3. **Multi-level lookup chain**: ComponentID → Column Index → Memory Address → Export Buffer
4. **Zero-copy when possible**: Single-world returns direct pointers, multi-world requires copying
5. **Batch copy optimization**: All exports copied together in `copyOutExportedColumns()`

The mapping system's genius is that all the complex lookups happen once during registration, then runtime access is just simple array indexing into pre-computed pointers.

This architecture achieves high performance through data locality, minimal indirection, and efficient batch processing while maintaining the flexibility of an ECS system.