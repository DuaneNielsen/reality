#!/usr/bin/env python3

import re
import subprocess
import sys


def run_sweep():
    # Create the sweep
    print("Creating sweep...")
    result = subprocess.run(
        ["uv", "run", "wandb", "sweep", "sweep_config.yaml"], capture_output=True, text=True
    )

    # Combine stdout and stderr for parsing
    output = result.stdout + result.stderr
    print("Sweep output:", output)

    if result.returncode != 0:
        print(f"Failed to create sweep: {result.stderr}")
        sys.exit(1)

    # Extract sweep ID from output
    sweep_id_match = re.search(r"wandb agent '([^']+)'", output)
    if not sweep_id_match:
        sweep_id_match = re.search(r"wandb agent ([^\s]+)", output)

    if not sweep_id_match:
        print("Could not find sweep ID in combined output")
        sys.exit(1)

    sweep_id = sweep_id_match.group(1)
    print(f"Created sweep: {sweep_id}")

    # Run the agent
    print("Starting sweep agent...")
    subprocess.run(["uv", "run", "wandb", "agent", sweep_id])


if __name__ == "__main__":
    run_sweep()
