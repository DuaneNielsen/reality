# [Feature Name] Sequence

## Overview
[Brief description of what this sequence does and why]

## Input

[What data enters this sequence and where it comes from]

### Input Sources
- **[Source 1]**: [Type of data and origin - e.g., command line args, config file, upstream process output]
- **[Source 2]**: [Type of data and origin]
- **[Source 3]**: [Type of data and origin]

### Input Data Format

#### Simple Values
- **[Parameter Name]** (`[type]`): [Purpose and valid range/values]
- **[Parameter Name]** (`[type]`): [Purpose and valid range/values]

#### Structured Data
**[Structure Name]**
```cpp
struct [DataStructureName] {
    [type] [field_name];              // [Purpose in processing]
    [type] [field_array][SIZE];       // [Purpose in processing]
    // ... additional fields
};
```

## Processing

[How the input is transformed into output]

### Processing Pipeline
```
Input → [Stage 1: Description] → [Stage 2: Description] → [Stage 3: Description] → Output
```

### Detailed Sequence

#### Phase 1: [Phase Name]
[When this phase occurs and its trigger]

#### Step 1: [Step Name]
**Function:** `[FunctionName]()`
**Location:** `src/[file].cpp:[line]`
**Purpose:** [What this step accomplishes]

**Details:**
- [Action or operation]
- [Resources affected]
- [Prerequisites or dependencies]

#### Step 2: [Step Name]
**Function:** `[FunctionName]()`
**Location:** `src/[file].cpp:[line]`
**Purpose:** [What this step accomplishes]

**Details:**
- [Action or operation]
- [Resources affected]
- [Prerequisites or dependencies]

### Phase 2: [Phase Name]
[When this phase occurs and its trigger]

#### Step 1: [Step Name]
**Function:** `[FunctionName]()`
**Location:** `src/[file].cpp:[line]`
**Purpose:** [What this step accomplishes]

**Details:**
- [Resources created]
- [State initialized]
- [Configuration applied]

### Phase 3: [Phase Name]
[When this phase occurs and its trigger]

#### Step 1: [Step Name]
**Function:** `[FunctionName]()`
**Location:** `src/[file].cpp:[line]`
**Purpose:** [What this step accomplishes]


## Output

[What this sequence produces]

### Output Data

#### Direct Outputs
Data explicitly produced by the sequence:

- **[Output Name]** (`[type]`): [Description and destination]
- **[Output Name]** (`[type]`): [Description and destination]

#### Side Effects
State changes or resources modified:

- **[State/Resource]**: [How it's modified]
- **[State/Resource]**: [How it's modified]

### Output Format
```cpp
// Example output structure if applicable
struct [OutputStructure] {
    [type] [field];  // [What this represents]
};
```
