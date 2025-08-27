#!/usr/bin/env python3
"""Run a single test that triggers the issue"""
import sys
sys.path.insert(0, '/home/duane/madrona_escape_room')

from tests.python.test_ascii_level_compiler import TestManagerIntegration
test = TestManagerIntegration()

# This uses cpu_manager fixture, so we need to simulate it
from madrona_escape_room import SimManager, ExecMode
mgr = SimManager(
    exec_mode=ExecMode.CPU,
    gpu_id=0,
    num_worlds=1,
    rand_seed=42,
    auto_reset=True,
)
print("Manager created successfully")
mgr.step()
print("Step completed")