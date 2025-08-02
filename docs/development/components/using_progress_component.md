# Using the Progress Component

This example shows how to access and use the Progress component from Python to track agent performance.

```python
import madrona_escape_room
import numpy as np

# Create manager
mgr = madrona_escape_room.Manager(
    exec_mode=madrona_escape_room.ExecMode.CPU,
    gpu_id=0,
    num_worlds=10,
    rand_seed=42,
    auto_reset=True
)

# Run some steps
for i in range(100):
    mgr.step()

# Access the progress tensor
progress = mgr.progress_tensor()
# Shape: [num_worlds, num_agents, 1]

# Get progress for world 0, agent 0
agent_progress = progress[0, 0, 0]
print(f"Agent 0 progress (maxY): {agent_progress}")

# Get average progress across all agents
avg_progress = np.mean(progress)
print(f"Average progress: {avg_progress}")

# Find best performing agent
best_world, best_agent = np.unravel_index(
    np.argmax(progress[..., 0]), 
    progress[..., 0].shape
)
print(f"Best agent: World {best_world}, Agent {best_agent} with progress {progress[best_world, best_agent, 0]}")

# Track progress over time
progress_history = []
for i in range(50):
    mgr.step()
    current_progress = mgr.progress_tensor()
    avg = np.mean(current_progress)
    progress_history.append(avg)
    if i % 10 == 0:
        print(f"Step {i}: Average progress = {avg:.3f}")

# Check if any agent has made significant progress
high_performers = np.where(progress > 5.0)  # Agents who traveled more than 5 units
if len(high_performers[0]) > 0:
    print(f"\n{len(high_performers[0])} agents have made significant progress!")
    for w, a in zip(high_performers[0], high_performers[1]):
        print(f"  World {w}, Agent {a}: {progress[w, a, 0]:.2f}")
```

## Notes

- Progress tracks the maximum Y position reached by each agent during the current episode
- Progress resets to 0 at the start of each new episode
- The tensor shape is always [num_worlds, num_agents, 1]
- Progress is updated automatically by the reward system as agents move