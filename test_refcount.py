#!/usr/bin/env python3
"""Test if keeping all objects alive prevents the crash"""
import faulthandler
import sys
import gc

# Enable faulthandler
faulthandler.enable()

# Disable garbage collection entirely
gc.disable()
print("Garbage collection DISABLED")

# Global lists to keep EVERYTHING alive
ALL_MANAGERS = []
ALL_CONFIGS = []
ALL_LEVELS = []
ALL_C_CONFIGS = []
ALL_C_LEVELS = []
ALL_ARRAYS = []

print("Will keep ALL objects alive to test reference counting hypothesis")
print("=" * 60)

from madrona_escape_room import SimManager, ExecMode
from madrona_escape_room.default_level import create_default_level
from madrona_escape_room.generated_dataclasses import ManagerConfig

# Run the test MANY times
for i in range(500):
    if i % 10 == 0:
        print(f"\n=== Run {i+1} === (Objects in memory: {len(ALL_MANAGERS)})")
        sys.stdout.flush()
    
    try:
        # Create and KEEP all objects
        level = create_default_level()
        ALL_LEVELS.append(level)
        
        config = ManagerConfig()
        config.exec_mode = 0
        config.num_worlds = 1
        config.rand_seed = 42 + i
        config.auto_reset = True
        ALL_CONFIGS.append(config)
        
        # Keep the ctypes conversions too
        c_level = level.to_ctype()
        ALL_C_LEVELS.append(c_level)
        
        c_config = config.to_ctype()
        ALL_C_CONFIGS.append(c_config)
        
        # Create manager - it will also store c_config and levels_array internally
        mgr = SimManager(
            exec_mode=ExecMode.CPU,
            gpu_id=0,
            num_worlds=1,
            rand_seed=42 + i,
            auto_reset=True,
        )
        ALL_MANAGERS.append(mgr)  # Keep manager alive!
        
        # Run a few steps
        for step in range(3):
            mgr.step()
        
        # DO NOT delete - keep everything alive
        # del mgr
        
        if i % 10 != 0:
            print(".", end="", flush=True)
        
    except Exception as e:
        print(f"\nPython exception at iteration {i+1}: {e}")
        import traceback
        traceback.print_exc()
        break

print(f"\n\nCompleted with {len(ALL_MANAGERS)} managers in memory")
print("=" * 60)

# Now test if deleting them all at once causes issues
print("Now deleting all managers...")
ALL_MANAGERS.clear()
ALL_CONFIGS.clear()
ALL_LEVELS.clear()
ALL_C_CONFIGS.clear()
ALL_C_LEVELS.clear()
ALL_ARRAYS.clear()
print("All objects cleared")

# Re-enable GC and collect
gc.enable()
gc.collect()
print(f"Garbage collected: {gc.collect()} objects")