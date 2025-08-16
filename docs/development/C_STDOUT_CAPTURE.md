# C Stdout Capture in Python Environments

## Problem

When using ctypes/cffi Python bindings with C functions that use `fprintf(stdout, ...)`, the output doesn't appear in Python environments (especially Jupyter/IPython) because C-level `fprintf` writes to the underlying file descriptor, not Python's `sys.stdout` wrapper.

### Root Cause

Python's `sys.stdout` and C's `stdout` are different objects that share the same underlying file descriptor. When Python environments replace `sys.stdout` (common in IDEs and notebooks), C-level output still goes to the original file descriptor, making it invisible to the Python user.

### Specific Issue in Madrona Escape Room

The trajectory logging functionality in `src/mgr.cpp` uses:
```cpp
FILE* output = impl_->trajectoryLogFile ? impl_->trajectoryLogFile : stdout;
fprintf(output, "Step %4u: World %d Agent %d: pos=(%.2f,%.2f,%.2f) rot=%.1f° progress=%.2f\n", ...);
```

When `trajectoryLogFile` is `nullptr`, it defaults to `stdout`, but this C-level `fprintf(stdout, ...)` output is not visible in Python REPL environments.

## Solution

Redirect stdout at the file descriptor level using `os.dup2()` to capture C-level output and display it in Python.

### Implementation

```python
import os
import sys
import tempfile
import io
import ctypes
from contextlib import contextmanager

class CStdoutCapture:
    """Utility class to capture C-level stdout output in Python environments"""
    
    @staticmethod
    @contextmanager
    def capture():
        """Context manager to capture and display C-level stdout output"""
        original_stdout_fd = None
        saved_stdout_fd = None
        temp_file = None
        
        try:
            # Get stdout file descriptor, handling replaced stdout
            try:
                original_stdout_fd = sys.stdout.fileno()
            except (io.UnsupportedOperation, AttributeError):
                original_stdout_fd = sys.__stdout__.fileno()
            
            saved_stdout_fd = os.dup(original_stdout_fd)
            
            # Create temporary file for capturing output
            temp_file = tempfile.NamedTemporaryFile(mode='w+', delete=False)
            
            # Redirect stdout to temp file
            os.dup2(temp_file.fileno(), original_stdout_fd)
            
            # Flush C-level buffers
            try:
                libc = ctypes.CDLL(None)
                libc.fflush(None)
            except:
                pass
            
            yield
            
            # Flush again before reading
            try:
                libc.fflush(None)  
            except:
                pass
            
            # Restore stdout
            os.dup2(saved_stdout_fd, original_stdout_fd)
            
            # Read and display captured output
            temp_file.seek(0)
            captured = temp_file.read().strip()
            if captured:
                for line in captured.split('\n'):
                    if line.strip():
                        print(line)
            
        except Exception as e:
            print(f"Error capturing C stdout: {e}")
        finally:
            # Cleanup
            if saved_stdout_fd is not None:
                try:
                    os.close(saved_stdout_fd)
                except:
                    pass
            if temp_file is not None:
                try:
                    temp_file.close()
                    os.unlink(temp_file.name)
                except:
                    pass
```

### Usage Example

```python
from madrona_escape_room import SimManager
import madrona

# Create SimManager
mgr = SimManager(
    exec_mode=madrona.ExecMode.CPU,
    gpu_id=-1, 
    num_worlds=2, 
    rand_seed=42, 
    auto_reset=True
)

# Enable trajectory logging with C stdout capture
with CStdoutCapture.capture():
    mgr.enable_trajectory_logging(world_idx=0, agent_idx=0)
    
    # Run simulation steps
    actions = mgr.action_tensor().to_numpy()
    for i in range(5):
        actions[0, :] = [1, i % 3, i % 2]  # vary movement
        mgr.step()
```

This will output trajectory information like:
```
Trajectory logging enabled for World 0, Agent 0
Step    0: World 0 Agent 0: pos=(0.01,0.06,0.00) rot=-14.6° progress=2.27
Step    1: World 0 Agent 0: pos=(0.02,0.06,0.00) rot=-19.0° progress=2.27
Step    2: World 0 Agent 0: pos=(0.03,0.04,0.00) rot=-27.7° progress=2.27
```

## Alternative Solutions

### 1. Wurlitzer Library (Recommended for Jupyter)

For Jupyter environments, the `wurlitzer` library provides a cleaner solution:

```bash
pip install wurlitzer
```

```python
import wurlitzer

with wurlitzer.pipes() as (out, err):
    mgr.enable_trajectory_logging(world_idx=0, agent_idx=0)
    mgr.step()

print(out.read())
```

### 2. Modify C Code (Long-term Solution)

The most robust long-term solution would be to modify the C code to use callback functions instead of direct `fprintf`:

```cpp
// Instead of fprintf(stdout, ...)
if (trajectory_callback) {
    trajectory_callback(formatted_message);
}
```

```python
# Python callback
@ctypes.CFUNCTYPE(None, ctypes.c_char_p)
def trajectory_callback(msg):
    print(msg.decode())
```

## Key Points

- This issue affects any C/C++ extension that uses `printf`, `fprintf(stdout, ...)`, or similar C standard library functions
- The problem is most common in Jupyter/IPython environments where `sys.stdout` is replaced
- File descriptor redirection is the most universal solution that works across different Python environments
- Always ensure proper cleanup of file descriptors to avoid resource leaks
- Consider using dedicated logging libraries or callback mechanisms for production code

## References

- [Stack Overflow: How to get printed output from ctypes C functions into Jupyter](https://stackoverflow.com/questions/35745541/)
- [Eli Bendersky: Redirecting all kinds of stdout in Python](https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/)
- [Wurlitzer Library Documentation](https://pypi.org/project/wurlitzer/)