"""Simple stopwatch-style timer for performance profiling."""

import time
from typing import Dict, List, Optional

import numpy as np


class StepTimer:
    """Stopwatch-style timer that tracks multiple named operations."""

    def __init__(self):
        self._timings: Dict[str, List[float]] = {}  # tag -> list of elapsed times
        self._active: Dict[str, float] = {}  # tag -> start time
        self._verbose = False

    def start(self, tag: str) -> None:
        """Start timing for a given tag"""
        self._active[tag] = time.perf_counter()

    def stop(self, tag: str) -> Optional[float]:
        """Stop timing for a given tag and record the elapsed time"""
        if tag in self._active:
            elapsed = (time.perf_counter() - self._active[tag]) * 1000  # Convert to ms
            if tag not in self._timings:
                self._timings[tag] = []
            self._timings[tag].append(elapsed)
            del self._active[tag]

            if self._verbose:
                print(f"{tag}: {elapsed:.3f}ms")

            return elapsed
        return None

    def set_verbose(self, verbose: bool) -> None:
        """Enable/disable verbose output on each stop()"""
        self._verbose = verbose

    def report(self, title: str = "Timing Report") -> None:
        """Print a summary report of all timings"""
        if not self._timings:
            print("No timings recorded")
            return

        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title=title)

        # Add columns
        table.add_column("Operation", style="cyan", no_wrap=True)
        table.add_column("Samples", justify="right", style="green")
        table.add_column("Average (ms)", justify="right", style="yellow")
        table.add_column("Min (ms)", justify="right", style="blue")
        table.add_column("Max (ms)", justify="right", style="red")
        table.add_column("Total (ms)", justify="right", style="magenta")

        # Sort tags for consistent output
        for tag in sorted(self._timings.keys()):
            times = self._timings[tag]
            n = len(times)
            if n > 0:
                avg = sum(times) / n
                min_t = min(times)
                max_t = max(times)
                total = sum(times)

                table.add_row(
                    tag,
                    str(n),
                    f"{avg:.3f}",
                    f"{min_t:.3f}",
                    f"{max_t:.3f}",
                    f"{total:.3f}",
                )

        console.print()
        console.print(table)

        # Check for any unclosed timers
        if self._active:
            console.print(
                f"\n[bold red]Warning:[/bold red] Unclosed timers: {list(self._active.keys())}"
            )

    def clear(self) -> None:
        """Clear all recorded timings"""
        self._timings.clear()
        self._active.clear()


class FPSCounter:
    """Tracks frame timing and calculates FPS statistics."""

    def __init__(self, num_worlds: int = 1):
        self._num_worlds = num_worlds
        self._frame_times: List[float] = []  # Time between frames in ms
        self._last_frame_time: Optional[float] = None
        self._start_time: Optional[float] = None

    def start(self) -> None:
        """Start tracking frames."""
        self._start_time = time.perf_counter()
        self._last_frame_time = self._start_time

    def frame(self) -> None:
        """Record that a frame was completed."""
        now = time.perf_counter()
        if self._last_frame_time is not None:
            frame_time = (now - self._last_frame_time) * 1000  # Convert to ms
            self._frame_times.append(frame_time)
        self._last_frame_time = now

    def report(self) -> None:
        """Print FPS statistics."""
        if not self._frame_times or self._start_time is None:
            print("No frames recorded")
            return

        from rich.console import Console
        from rich.table import Table

        console = Console()

        # Calculate statistics
        frame_times = np.array(self._frame_times)
        total_time = self._last_frame_time - self._start_time
        num_frames = len(self._frame_times)

        # FPS calculations
        fps_from_avg = 1000.0 / np.mean(frame_times)  # Based on average frame time
        fps_from_total = num_frames / total_time  # Based on total elapsed time
        world_fps = fps_from_total * self._num_worlds

        # Create table
        table = Table(title="Performance Statistics")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", justify="right", style="yellow")

        table.add_row("Frames", f"{num_frames:,}")
        table.add_row("Worlds", f"{self._num_worlds:,}")
        table.add_row("Total Time", f"{total_time:.3f}s")
        table.add_row("", "")
        table.add_row("Frame Time (avg)", f"{np.mean(frame_times):.3f}ms")
        table.add_row("Frame Time (min)", f"{np.min(frame_times):.3f}ms")
        table.add_row("Frame Time (max)", f"{np.max(frame_times):.3f}ms")
        table.add_row("Frame Time (std)", f"{np.std(frame_times):.3f}ms")
        table.add_row("", "")
        table.add_row("FPS", f"{fps_from_total:,.0f}")
        table.add_row("World FPS", f"{world_fps:,.0f}")

        console.print()
        console.print(table)
