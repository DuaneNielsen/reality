#pragma once

// C API buffer sizes and limits
// These constants match the values defined in src/consts.hpp::capi namespace

#define MER_MAX_NAME_LENGTH 64
#define MER_MAX_TILES 1024
#define MER_MAX_WORLDS 10000
#define MER_MAX_AGENTS_PER_WORLD 100
#define MER_MAX_GRID_SIZE 64
#define MER_MAX_SCALE 100.0f
#define MER_MAX_COORDINATE 1000.0f

// Action constants
#define MER_NUM_ACTION_COMPONENTS 3