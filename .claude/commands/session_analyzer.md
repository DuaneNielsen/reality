---
argument-hint: [goal achieved in current session]
description: analyze current conversation context and create a reusable command template
---
@.claude/include/substitutions.md
@docs/templates/template.md

# algorithm

session_goal = $ARGUMENT
context_steps = []
algorithm_pattern = ""
new_command_template = ""

1. analyze current context
2. extract performed steps
3. identify algorithm structure
4. generate command template
5. save new command file

## analyze current context

Review the current conversation context window to identify the work pattern used to achieve the specified goal:

**Sequential Pattern**: Steps executed once in order
- Look for: Linear progression of actions without repetition
- Examples: "First I read X, then analyzed Y, finally wrote Z"

**For Loop Pattern**: Same steps applied to multiple items
- Look for: Repeated actions across a collection
- Examples: Processing multiple files, iterating through a list

**While Loop Pattern**: Steps repeated until condition met
- Look for: Repetitive actions with a stopping condition
- Examples: "Keep fixing errors until build succeeds"

Scan the conversation for:
- Tool calls and their sequence
- User requests and responses
- Repeated patterns or iterations
- Decision points that led to loops or branches

## extract performed steps

From the current conversation context, identify:
1. **Actual tools used** - Read, Write, Edit, Bash, etc.
2. **Sequence of actions** - Order in which steps were performed
3. **Repeated patterns** - Any steps that were done multiple times
4. **Decision logic** - Conditions that determined next steps
5. **Input parameters** - What information was needed at each step
6. **Outputs produced** - Results or artifacts created

Document each step with:
- **Action verb** - What was done (read, analyze, create, fix)
- **Tools used** - Which Claude tools were employed
- **Inputs needed** - Required parameters or information
- **Expected results** - What should be produced
- **Conditions** - When to proceed or stop

## identify algorithm structure

Based on the extracted steps from the current context:

**If Sequential**: Map the linear sequence of tool calls and actions
**If For Loop**: Identify the collection being processed and repeated actions
**If While Loop**: Identify the continuation condition and repeated step sequence

Use the conversation history to determine:
- Which template pattern best matches the actual work performed
- What the core algorithmic structure should be
- How to generalize the specific steps for reuse

## generate command template

Using the master template patterns included in this file:

1. **Create metadata section** 
   - `argument-hint` based on what inputs were needed in the session
   - `description` summarizing the goal achieved

2. **Select appropriate pattern structure**
   - **Sequential Pattern** for linear workflows (analyze → test → report)
   - **For Loop Pattern** for collection processing (multiple files, items)
   - **While Loop Pattern** for repetitive tasks with conditions (fix until success)

3. **Fill in step definitions**
   - Convert observed tool calls into generalized steps
   - Document specific Claude tools to use (Read, Write, Edit, Bash, etc.)
   - Include error handling and decision logic found in context

4. **Add condition definitions** (if loops were used)
   - Extract the actual stopping conditions from the conversation
   - Document how to evaluate continuation criteria

5. **Specify output format**
   - Based on what was actually produced in the session
   - Match the format style used in the conversation

Replace template variables with context-derived information:
- `[ARGUMENT_DESCRIPTION]` ← What the command should take as input
- `[COMMAND_DESCRIPTION]` ← The session goal achieved
- `[STEP_N]` ← Action names derived from tool usage
- `[STEP_N_DESCRIPTION]` ← Specific instructions based on what was done
- `[CONDITION_NAME]` and `[CONDITION_DESCRIPTION]` ← Loop conditions if present
- `[OUTPUT_FORMAT_SPECIFICATION]` ← Format matching session results

## save new command file

Create the new command file in `.claude/commands/` directory:
- **Filename**: Convert session goal to lowercase with underscores (e.g., "create_session_analyzer" → `create_session_analyzer.md`)
- **Content**: Complete template with all variables replaced from context analysis
- **Structure**: Proper markdown formatting following existing command conventions

## output format

**Context Analysis Complete** \\n\\n

**Goal Achieved:** [SESSION_GOAL] \\n\\n

**Pattern Detected:** [PATTERN_TYPE] \\n\\n

**Steps Extracted from Context:**
[NUMBERED_LIST_OF_ACTUAL_STEPS] \\n\\n

**Generated Command:** `.claude/commands/[COMMAND_NAME].md` \\n\\n

**Command Structure:**
- **Arguments:** [ARGUMENT_SUMMARY]
- **Steps:** [STEP_COUNT] steps identified from conversation
- **Tools Used:** [TOOL_LIST] (Read, Write, Edit, etc.)
- **Pattern:** [LOOP_TYPE] (if applicable)
- **Output:** [OUTPUT_TYPE]

**Usage:**
```
claude [COMMAND_NAME] "[EXAMPLE_ARGUMENT]"
```

The new command template captures the workflow from this session and is ready for reuse.