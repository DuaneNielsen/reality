#!/usr/bin/env python3
"""Debug the intermittent crash with detailed logging - run until crash"""
import faulthandler
import sys
import os
import traceback

# Enable faulthandler to dump Python stack on crash
faulthandler.enable()

# Also write to a file
crash_log = open('crash_debug.log', 'w')
faulthandler.enable(file=crash_log, all_threads=True)

print("Faulthandler enabled - will dump stack trace on crash")
print("Running until crash occurs...")
print("=" * 60)

# Run the test MANY times until crash
for i in range(1000):
    if i % 10 == 0:
        print(f"\n=== Run {i+1} ===")
        sys.stdout.flush()
    
    try:
        # Import fresh each time to reset state
        import importlib
        import madrona_escape_room
        importlib.reload(madrona_escape_room)
        
        from madrona_escape_room import SimManager, ExecMode
        from madrona_escape_room.default_level import create_default_level
        
        # Create manager
        mgr = SimManager(
            exec_mode=ExecMode.CPU,
            gpu_id=0,
            num_worlds=1,
            rand_seed=42 + i,  # Vary seed each run
            auto_reset=True,
        )
        
        # Run a few steps
        for step in range(3):
            mgr.step()
        
        # Explicitly delete to trigger cleanup
        del mgr
        
        # Print progress dot
        if i % 10 != 0:
            print(".", end="", flush=True)
        
    except Exception as e:
        print(f"\nPython exception at iteration {i+1}: {e}")
        traceback.print_exc()
        break
        
    except:
        print(f"\nUnknown exception at iteration {i+1}!")
        break

print("\n" + "=" * 60)
print("Test completed - check crash_debug.log if crash occurred")
crash_log.close()