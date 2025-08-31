#!/usr/bin/env python3
"""Debug the intermittent crash with detailed logging"""
import faulthandler
import sys
import os
import traceback
import ctypes

# Enable faulthandler to dump Python stack on crash
faulthandler.enable()

# Also write to a file
crash_log = open('crash_debug.log', 'w')
faulthandler.enable(file=crash_log, all_threads=True)

print("Faulthandler enabled - will dump stack trace on crash")
print("=" * 60)

# Add debugging to see what's happening
import logging
logging.basicConfig(level=logging.DEBUG)

# Now run the test multiple times
for i in range(20):
    print(f"\n=== Run {i+1} ===")
    sys.stdout.flush()
    
    try:
        # Import fresh each time to reset state
        import importlib
        import madrona_escape_room
        importlib.reload(madrona_escape_room)
        
        from madrona_escape_room import SimManager, ExecMode
        from madrona_escape_room.default_level import create_default_level
        
        # Log details about what we're creating
        level = create_default_level()
        print(f"Level created: size={level.__class__.size()}, spawns={level.num_spawns}")
        print(f"Level spawn: ({level.spawn_x[0]}, {level.spawn_y[0]})")
        
        # Check the converted ctypes object
        c_level = level.to_ctype()
        print(f"C Level type: {type(c_level)}")
        print(f"C Level spawn: ({c_level.spawn_x[0]}, {c_level.spawn_y[0]})")
        
        # Create manager with detailed logging
        print("Creating SimManager...")
        mgr = SimManager(
            exec_mode=ExecMode.CPU,
            gpu_id=0,
            num_worlds=1,
            rand_seed=42 + i,  # Vary seed each run
            auto_reset=True,
        )
        print("Manager created successfully")
        
        # Run a few steps
        for step in range(3):
            mgr.step()
            print(f"  Step {step+1} completed")
        
        # Explicitly delete to trigger cleanup
        del mgr
        print("Manager deleted")
        
    except Exception as e:
        print(f"Python exception: {e}")
        traceback.print_exc()
        break
        
    except:
        print("Unknown exception occurred!")
        break

print("\n" + "=" * 60)
print("Test completed - check crash_debug.log if crash occurred")
crash_log.close()