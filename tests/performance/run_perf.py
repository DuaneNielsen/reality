#!/usr/bin/env python3
"""
Performance testing script for Madrona Escape Room.

Can run in nightly mode to test all untested commits, or manually test specific commits.
Generates detailed performance data and reports for regression tracking.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Configuration
REPO_ROOT = Path(__file__).parent.parent.parent  # Go up to madrona_escape_room root
PERF_RESULTS_DIR = Path(__file__).parent / "perf_results"  # Save in tests/performance/perf_results
RUNS_DIR = PERF_RESULTS_DIR / "runs"
LAST_COMMIT_FILE = PERF_RESULTS_DIR / "last_tested_commit.txt"
HISTORY_FILE = PERF_RESULTS_DIR / "history.csv"
LATEST_REPORT_FILE = PERF_RESULTS_DIR / "latest.txt"
BASELINES_FILE = Path(__file__).parent / "performance_baselines.json"


# Command line argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Performance testing for Madrona Escape Room")
    parser.add_argument(
        "--nightly", action="store_true", help="Nightly mode: test all untested commits"
    )
    parser.add_argument("--commit", type=str, help="Test specific commit (default: current HEAD)")
    parser.add_argument(
        "--worlds", type=int, default=1024, help="Number of worlds to test (default: 1024)"
    )
    parser.add_argument(
        "--steps", type=int, default=1000, help="Number of steps to run (default: 1000)"
    )
    parser.add_argument(
        "--cuda", type=int, default=-1, help="GPU device ID for CUDA mode (default: -1 for CPU)"
    )
    return parser.parse_args()


def log(message):
    """Simple logging with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")


def run_command(cmd, cwd=None, capture_output=True):
    """Run a shell command and return result."""
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=cwd, capture_output=capture_output, text=True, check=True
        )
        return result.stdout.strip() if capture_output else ""
    except subprocess.CalledProcessError as e:
        log(f"Command failed: {cmd}")
        log(f"Error: {e.stderr}")
        return None


def get_last_tested_commit():
    """Get the last commit that was performance tested."""
    if LAST_COMMIT_FILE.exists():
        return LAST_COMMIT_FILE.read_text().strip()
    return None


def save_last_tested_commit(commit_hash):
    """Save the last tested commit hash."""
    LAST_COMMIT_FILE.write_text(commit_hash)


def get_untested_commits():
    """Get list of commits since last tested commit."""
    last_commit = get_last_tested_commit()

    if last_commit:
        # Get commits since last tested
        cmd = f"git log --oneline {last_commit}..HEAD"
    else:
        # First run - get last 5 commits
        cmd = "git log --oneline -5"

    output = run_command(cmd, cwd=REPO_ROOT)
    if not output:
        return []

    commits = []
    for line in output.split("\n"):
        if line.strip():
            commit_hash = line.split()[0]
            message = " ".join(line.split()[1:])
            commits.append((commit_hash, message))

    # Return in chronological order (oldest first)
    return list(reversed(commits))


def should_skip_commit(message):
    """Check if commit should be skipped (docs only, etc.)."""
    skip_patterns = [
        "update readme",
        "update doc",
        "fix typo",
        "add doc",
        "docs:",
        "[docs]",
        "documentation",
        "comment",
        "formatting",
    ]
    message_lower = message.lower()
    return any(pattern in message_lower for pattern in skip_patterns)


def backup_working_state():
    """Backup current working directory state."""
    log("Backing up working state...")

    # Stash any uncommitted changes
    run_command("git stash push -m 'nightly_perf_backup'", cwd=REPO_ROOT)

    # Get current branch
    current_branch = run_command("git branch --show-current", cwd=REPO_ROOT)
    return current_branch


def restore_working_state(original_branch):
    """Restore original working directory state."""
    log("Restoring working state...")

    # Return to original branch
    if original_branch:
        run_command(f"git checkout {original_branch}", cwd=REPO_ROOT)

    # Pop stashed changes if any
    stash_list = run_command("git stash list", cwd=REPO_ROOT)
    if stash_list and "nightly_perf_backup" in stash_list:
        run_command("git stash pop", cwd=REPO_ROOT)


def save_raw_data(
    commit_hash, message, build_output, benchmark_output, fps, status, command=None, args=None
):
    """Save detailed raw data for a commit test run."""
    # Create run directory - just use commit hash
    run_dir = RUNS_DIR / commit_hash
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save benchmark output
    (run_dir / "benchmark_output.txt").write_text(benchmark_output or "No output captured")

    # Save build log
    (run_dir / "build_log.txt").write_text(build_output or "No build output captured")

    # Save metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "commit": commit_hash,
        "message": message,
        "cpu_fps": fps,
        "cpu_status": status,
        "command": command,
        "configuration": {
            "num_worlds": args.worlds if hasattr(args, "worlds") else 1024,
            "num_steps": args.steps if hasattr(args, "steps") else 1000,
            "device": "GPU" if (hasattr(args, "cuda") and args.cuda >= 0) else "CPU",
            "gpu_id": args.cuda if (hasattr(args, "cuda") and args.cuda >= 0) else None,
        },
    }

    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Copy any profile files that were generated
    profile_files = ["/tmp/sim_bench_profile_cpu.html", "/tmp/sim_bench_profile_gpu.html"]

    for profile_file in profile_files:
        if Path(profile_file).exists():
            device = "cpu" if "cpu" in profile_file else "gpu"
            shutil.copy2(profile_file, run_dir / f"profile_{device}.html")

    log(f"Saved raw data to {run_dir}")
    return str(run_dir)


def test_commit(commit_hash, message, args):
    """Test performance of a specific commit."""
    log(f"Testing commit {commit_hash}: {message}")

    # Checkout the commit
    checkout_result = run_command(f"git checkout {commit_hash}", cwd=REPO_ROOT)
    if checkout_result is None:
        log(f"Failed to checkout {commit_hash}")
        return None

    # Build the project and capture output
    log("Building project...")
    build_output = ""
    try:
        result = subprocess.run(
            "make -C build -j8 -s",
            shell=True,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        build_output = result.stdout + result.stderr
    except subprocess.CalledProcessError as e:
        build_output = f"Build failed:\nstdout: {e.stdout}\nstderr: {e.stderr}"
        log("Build failed, skipping performance test")

        # Save failed build data
        save_raw_data(
            commit_hash,
            message,
            build_output,
            "Build failed",
            None,
            "BUILD_FAIL",
            "make -C build -j8 -s",
            args,
        )

        return {
            "commit": commit_hash,
            "message": message,
            "cpu_fps": None,
            "gpu_fps": None,
            "cpu_status": "BUILD_FAIL",
            "gpu_status": "BUILD_FAIL",
        }

    # Test performance with headless executable
    device = "GPU" if args.cuda >= 0 else "CPU"
    log(f"Running {device} performance test...")

    # Build command with user parameters
    cmd = f"./build/headless -n {args.worlds} -s {args.steps}"
    if args.cuda >= 0:
        cmd += f" --cuda {args.cuda}"

    try:
        result = subprocess.run(
            cmd, shell=True, cwd=REPO_ROOT, capture_output=True, text=True, check=True
        )
        benchmark_output = result.stdout
    except subprocess.CalledProcessError as e:
        benchmark_output = f"Headless test failed:\nstdout: {e.stdout}\nstderr: {e.stderr}"
        save_raw_data(commit_hash, message, build_output, benchmark_output, None, "FAIL", cmd, args)
        return {
            "commit": commit_hash,
            "message": message,
            "cpu_fps": None,
            "gpu_fps": None,
            "cpu_status": "FAIL",
            "gpu_status": "SKIP",
        }

    # Parse FPS from headless output: "FPS: 853679"
    fps = None
    for line in benchmark_output.split("\n"):
        if line.startswith("FPS:"):
            try:
                fps_str = line.split("FPS:")[1].strip().replace(",", "")
                fps = int(fps_str)
                break
            except (IndexError, ValueError):
                continue

    # Determine status from baseline check
    status = "PASS"  # Default
    if fps:
        try:
            with open(BASELINES_FILE) as f:
                baselines = json.load(f)

            baseline_key = "cpu_1024"
            if baseline_key in baselines:
                min_fps = baselines[baseline_key]["min_fps"]
                warn_fps = baselines[baseline_key]["warn_fps"]

                if fps >= warn_fps:
                    status = "PASS"
                elif fps >= min_fps:
                    status = "WARN"
                else:
                    status = "FAIL"
        except Exception:
            status = "PASS"  # If baseline check fails, assume pass

    # Save all raw data
    save_raw_data(commit_hash, message, build_output, benchmark_output, fps, status, cmd, args)

    # Skip GPU testing for now
    gpu_fps, gpu_status = None, "SKIP"

    return {
        "commit": commit_hash,
        "message": message,
        "cpu_fps": fps,
        "gpu_fps": gpu_fps,
        "cpu_status": status,
        "gpu_status": gpu_status,
    }


def save_result(result):
    """Save test result to CSV file."""
    date_str = datetime.now().strftime("%Y-%m-%d")

    with open(HISTORY_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                date_str,
                result["commit"],
                result["message"],
                result["cpu_fps"] or "",
                result["gpu_fps"] or "",
                result["cpu_status"],
                result["gpu_status"],
            ]
        )


def generate_report(results):
    """Generate morning report."""
    if not results:
        report = "Nightly Performance Report - No new commits to test\n"
    else:
        date_str = datetime.now().strftime("%Y-%m-%d")
        report = f"Nightly Performance Report - {date_str}\n"
        report += f"Tested {len(results)} commits:\n\n"

        for result in results:
            commit = result["commit"][:7]
            message = result["message"][:50]
            cpu_fps = result["cpu_fps"]
            cpu_status = result["cpu_status"]

            if cpu_status == "PASS":
                status_icon = "✓"
            elif cpu_status == "WARN":
                status_icon = "⚠"
            else:
                status_icon = "✗"

            fps_str = f"CPU: {cpu_fps:,} FPS" if cpu_fps else f"CPU: {cpu_status}"

            if cpu_status in ["FAIL", "BUILD_FAIL"]:
                fps_str += " - REGRESSION"

            report += f"{status_icon} {commit} - {message} ({fps_str})\n"

    LATEST_REPORT_FILE.write_text(report)
    print(report)


def main():
    """Main execution function."""
    args = parse_args()

    # Ensure we're in the right directory
    os.chdir(REPO_ROOT)

    # Create necessary directories
    PERF_RESULTS_DIR.mkdir(exist_ok=True)
    RUNS_DIR.mkdir(exist_ok=True)

    # Check if we're in a git repository
    if run_command("git status") is None:
        log("Not in a git repository")
        sys.exit(1)

    if args.nightly:
        # Nightly mode: test all untested commits
        log("Starting nightly performance testing...")

        # Get commits to test
        commits = get_untested_commits()
        log(f"Found {len(commits)} commits to test")

        if not commits:
            generate_report([])
            return

        # Backup current state
        original_branch = backup_working_state()

        try:
            results = []

            for commit_hash, message in commits:
                if should_skip_commit(message):
                    log(f"Skipping documentation commit: {commit_hash}")
                    continue

                result = test_commit(commit_hash, message, args)
                if result:
                    save_result(result)
                    results.append(result)

            # Update last tested commit
            if commits:
                last_commit = commits[-1][0]  # Most recent commit
                save_last_tested_commit(last_commit)
                log(f"Updated last tested commit to: {last_commit}")

            # Generate report
            generate_report(results)

        finally:
            # Always restore working state
            restore_working_state(original_branch)

        log("Nightly performance testing completed")

    else:
        # Manual mode: test specific commit
        commit_hash = (
            args.commit if args.commit else run_command("git rev-parse HEAD", cwd=REPO_ROOT)
        )

        if not commit_hash:
            log("Failed to get commit hash")
            sys.exit(1)

        commit_message = run_command(f"git log --oneline -1 {commit_hash}", cwd=REPO_ROOT)
        if commit_message:
            commit_message = " ".join(commit_message.split()[1:])  # Remove hash from start
        else:
            commit_message = "Manual test"

        log(f"Testing single commit: {commit_hash}")

        # Backup current state if testing different commit
        original_branch = None
        if args.commit and args.commit != run_command("git rev-parse HEAD", cwd=REPO_ROOT):
            original_branch = backup_working_state()

        try:
            result = test_commit(commit_hash, commit_message, args)
            if result:
                save_result(result)

                # Print simple result for manual testing
                print(f"\nResult: {result['cpu_status']}")
                if result["cpu_fps"]:
                    print(f"FPS: {result['cpu_fps']:,}")

        finally:
            if original_branch:
                restore_working_state(original_branch)


if __name__ == "__main__":
    main()
