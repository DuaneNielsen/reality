#!/usr/bin/env python3
"""
Git post-commit test tracker - tracks individual test results across commits.
Usage: python test_tracker.py [--commit HASH] [--dry-run]
"""

import argparse
import csv
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class TestTracker:
    def __init__(self, base_dir: Path = None, full_tests: bool = False, stress_tests: bool = False):
        self.base_dir = base_dir or Path.cwd()
        self.tests_dir = self.base_dir / "tests"
        self.full_tests = full_tests
        self.stress_tests = stress_tests

        # CSV file for individual test tracking (only one we need)
        self.individual_log = self.tests_dir / "individual-test-history.csv"

        # Initialize CSV files with headers if they don't exist
        self._init_csv_files()

    def _init_csv_files(self):
        """Initialize CSV file with header if it doesn't exist."""
        if not self.individual_log.exists():
            with open(self.individual_log, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "timestamp",
                        "commit",
                        "short_commit",
                        "branch",
                        "test_name",
                        "status",
                        "test_type",
                        "author",
                    ]
                )

    def get_commit_info(self, commit_hash: Optional[str] = None) -> Dict[str, str]:
        """Get commit information."""
        if commit_hash:
            commit = commit_hash
            short_commit = commit_hash[:7]
        else:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            short_commit = (
                subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
            )

        commit_msg = (
            subprocess.check_output(["git", "log", "-1", "--pretty=format:%s", commit])
            .decode()
            .strip()
        )

        author = (
            subprocess.check_output(["git", "log", "-1", "--pretty=format:%an", commit])
            .decode()
            .strip()
        )

        branch = subprocess.check_output(["git", "branch", "--show-current"]).decode().strip()

        return {
            "commit": commit,
            "short_commit": short_commit,
            "message": commit_msg,
            "author": author,
            "branch": branch,
            "timestamp": datetime.now().isoformat(),
        }

    def run_cpp_tests(self) -> Tuple[str, List[Tuple[str, str]]]:
        """Run C++ tests and return status and individual test results."""
        if self.stress_tests:
            test_type = "stress"
        elif self.full_tests:
            test_type = "full"
        else:
            test_type = "standard"
        print(f"Running C++ tests ({test_type} suite)...")

        # List of test executables to run
        test_executables = ["./build/mad_escape_tests"]

        if self.full_tests:
            # Add GPU test executables for full test suite (no stress tests)
            test_executables.append("./build/mad_escape_gpu_tests")

        if self.stress_tests:
            # Add stress test executables for stress test suite
            test_executables.append("./build/mad_escape_gpu_stress_tests")

        all_individual_results = []
        overall_status = "PASS"

        for executable in test_executables:
            print(f"  Running {executable}...")

            # Set environment for GPU tests
            env = None
            if (self.full_tests or self.stress_tests) and "gpu" in executable:
                env = dict(subprocess.os.environ)
                env["ALLOW_GPU_TESTS_IN_SUITE"] = "1"
                # Enable kernel cache for faster startup (58MB cache, 20x speedup)
                cache_path = self.base_dir / "build" / "madrona_kernels.cache"
                env["MADRONA_MWGPU_KERNEL_CACHE"] = str(cache_path)
                env["MADRONA_MWGPU_FORCE_DEBUG"] = "1"  # Debug mode for faster compilation

            try:
                result = subprocess.run(
                    [executable],
                    capture_output=True,
                    text=True,
                    cwd=self.base_dir,
                    env=env,
                    errors="replace",  # Replace invalid UTF-8 with replacement character
                )

                # Parse individual test results for this executable
                output_text = result.stdout + result.stderr
                executable_name = executable.split("/")[-1]  # Get just the filename

                # Parse actual GoogleTest output for individual test names
                for line in output_text.split("\n"):
                    # Full mode format: "[ RUN      ] TestSuite.TestCase"
                    # then "[       OK ] TestSuite.TestCase"
                    if match := re.search(r"\[\s+OK\s+\]\s+([A-Za-z0-9_]+\.[A-Za-z0-9_]+)", line):
                        test_name = f"cpp::{executable_name}::{match.group(1)}"
                        all_individual_results.append((test_name, "PASS"))
                    elif match := re.search(
                        r"\[\s+FAILED\s+\]\s+([A-Za-z0-9_]+\.[A-Za-z0-9_]+)", line
                    ):
                        test_name = f"cpp::{executable_name}::{match.group(1)}"
                        all_individual_results.append((test_name, "FAIL"))

                # Check if tests passed regardless of exit code
                if "[  PASSED  ]" in output_text and "[  FAILED  ]" not in output_text:
                    print(f"    {executable} PASSED")
                else:
                    overall_status = "FAIL"
                    print(f"    {executable} FAILED")

            except FileNotFoundError:
                print(f"    {executable} not found, skipping...")
            except Exception as e:
                print(f"    Error running {executable}: {e}")
                overall_status = "FAIL"

        return overall_status, all_individual_results

    def run_python_tests(self) -> Tuple[str, List[Tuple[str, str]], Dict[str, int]]:
        """Run Python tests and return status, individual results, and counts."""
        if self.stress_tests:
            test_type = "stress"
        elif self.full_tests:
            test_type = "full"
        else:
            test_type = "standard"
        print(f"Running Python tests ({test_type} suite)...")

        # Build pytest command based on test mode
        cmd = [
            "uv",
            "run",
            "pytest",
            "tests/python",
            "--tb=line",
            "-q",
            "--failed-first",
        ]

        if self.full_tests:
            # Full tests: include slow tests and GPU tests, but exclude skipped tests
            cmd.extend(["--ignore-glob", "**/test_*skip*"])
            # Set environment variable for GPU tests
            env = dict(subprocess.os.environ)
            env["ALLOW_GPU_TESTS_IN_SUITE"] = "1"
            # Enable kernel cache for faster startup (58MB cache, 20x speedup)
            cache_path = self.base_dir / "build" / "madrona_kernels.cache"
            env["MADRONA_MWGPU_KERNEL_CACHE"] = str(cache_path)
            env["MADRONA_MWGPU_FORCE_DEBUG"] = "1"  # Debug mode for faster compilation
        elif self.stress_tests:
            # Stress tests: run only slow/stress tests
            cmd.extend(["-m", "slow"])
            # Set environment variable for GPU tests
            env = dict(subprocess.os.environ)
            env["ALLOW_GPU_TESTS_IN_SUITE"] = "1"
            # Enable kernel cache for faster startup (58MB cache, 20x speedup)
            cache_path = self.base_dir / "build" / "madrona_kernels.cache"
            env["MADRONA_MWGPU_KERNEL_CACHE"] = str(cache_path)
            env["MADRONA_MWGPU_FORCE_DEBUG"] = "1"  # Debug mode for faster compilation
        else:
            # Standard tests: exclude slow tests as before
            cmd.extend(["-m", "not slow"])
            env = None

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=self.base_dir,
                env=env,
            )

            # Parse individual test results
            individual_results = []
            seen_tests = set()  # Track to avoid duplicates

            for line in result.stdout.split("\n"):
                # Match format: tests/python/file.py::test_name PASSED/FAILED/ERROR [XX%]
                # OR: tests/python/file.py::Class::test_name PASSED/FAILED/ERROR [XX%]
                if match := re.search(
                    r"^(tests/python/[^:]+::[^\s]+)\s+(PASSED|FAILED|ERROR)", line
                ):
                    test_name = match.group(1)
                    status = "PASS" if match.group(2) == "PASSED" else "FAIL"

                    # Avoid duplicates (can happen with ERROR + FAILED for same test)
                    if test_name not in seen_tests:
                        individual_results.append((test_name, status))
                        seen_tests.add(test_name)

            # Parse summary counts from the last line
            counts = {"pass": 0, "fail": 0, "skip": 0}
            output_text = result.stdout + result.stderr

            # Look for summary line:
            # "33 failed, 167 passed, 7 skipped, 11 deselected, 6 errors in 8.26s"
            for line in output_text.split("\n"):
                if re.search(r"\d+.*passed.*in.*s", line):  # Final summary line
                    if match := re.search(r"(\d+)\s+passed", line):
                        counts["pass"] = int(match.group(1))
                    if match := re.search(r"(\d+)\s+failed", line):
                        counts["fail"] = int(match.group(1))
                    if match := re.search(r"(\d+)\s+skipped", line):
                        counts["skip"] = int(match.group(1))
                    break

            overall_status = "PASS" if result.returncode == 0 else "FAIL"
            return overall_status, individual_results, counts

        except Exception as e:
            print(f"Error running Python tests: {e}")
            return "ERROR", [], {"pass": 0, "fail": 0, "skip": 0}

    def analyze_newly_broken_tests(self, current_failed: List[str]) -> List[str]:
        """Find tests that were passing before but are failing now."""
        if not self.individual_log.exists():
            return []

        newly_broken = []

        # Read previous test results
        with open(self.individual_log, "r", newline="") as f:
            reader = csv.DictReader(f)
            test_history = {}
            for row in reader:
                test_name = row["test_name"]
                if test_name not in test_history:
                    test_history[test_name] = []
                test_history[test_name].append(row)

        # Check each currently failing test
        for test_name in current_failed:
            if test_name in test_history and len(test_history[test_name]) > 0:
                # Get the most recent previous result
                last_result = test_history[test_name][-1]
                if last_result["status"] == "PASS":
                    newly_broken.append(test_name)

        return newly_broken

    def log_results(
        self,
        cpp_status: str,
        cpp_tests: List[Tuple[str, str]],
        py_status: str,
        py_tests: List[Tuple[str, str]],
        py_counts: Dict[str, int],
    ):
        """Log individual test results to CSV file."""

        # Log individual test results
        with open(self.individual_log, "a", newline="") as f:
            writer = csv.writer(f)

            # Log C++ tests
            for test_name, status in cpp_tests:
                writer.writerow(
                    [
                        self.commit_info["timestamp"],
                        self.commit_info["commit"],
                        self.commit_info["short_commit"],
                        self.commit_info["branch"],
                        test_name,
                        status,
                        "cpp",
                        self.commit_info["author"],
                    ]
                )

            # Log Python tests
            for test_name, status in py_tests:
                writer.writerow(
                    [
                        self.commit_info["timestamp"],
                        self.commit_info["commit"],
                        self.commit_info["short_commit"],
                        self.commit_info["branch"],
                        test_name,
                        status,
                        "python",
                        self.commit_info["author"],
                    ]
                )

    def run(self, dry_run: bool = False):
        """Run the complete test tracking process."""
        self.commit_info = self.get_commit_info()

        print(
            f"🧪 Testing commit {self.commit_info['short_commit']}: {self.commit_info['message']}"
        )

        # Run tests
        cpp_status, cpp_tests = self.run_cpp_tests()
        py_status, py_tests, py_counts = self.run_python_tests()

        # Analyze newly broken tests
        current_failed = [name for name, status in py_tests if status == "FAIL"]
        newly_broken = self.analyze_newly_broken_tests(current_failed)

        # Log results
        if not dry_run:
            self.log_results(cpp_status, cpp_tests, py_status, py_tests, py_counts)

        # Display summary
        overall_status = "PASS" if cpp_status == "PASS" and py_status == "PASS" else "FAIL"
        status_emoji = "✅" if overall_status == "PASS" else "❌"

        print(f"\n{status_emoji} Test Results for commit {self.commit_info['short_commit']}:")
        print(
            f"   C++: {cpp_status} "
            f"({len([t for t in cpp_tests if t[1] == 'PASS'])} passed, "
            f"{len([t for t in cpp_tests if t[1] == 'FAIL'])} failed)"
        )
        print(
            f"   Python: {py_status} "
            f"({py_counts['pass']} passed, {py_counts['fail']} failed, "
            f"{py_counts['skip']} skipped)"
        )
        print(f"   Overall: {overall_status}")

        if not dry_run:
            print(f"   Logged to: {self.individual_log}")

        # Show failed tests
        if cpp_status == "FAIL":
            failed_cpp_tests = [name for name, status in cpp_tests if status == "FAIL"]
            print(f"\n📋 Failed C++ tests ({len(failed_cpp_tests)} total):")
            for test in failed_cpp_tests:  # Show ALL failed tests
                print(f"     - {test}")

        if py_status == "FAIL":
            failed_tests = [name for name, status in py_tests if status == "FAIL"]
            print(f"\n📋 Failed Python tests ({len(failed_tests)} total):")
            for test in failed_tests:  # Show ALL failed tests
                print(f"     - {test}")

        # Show newly broken tests
        if newly_broken:
            print("\n⚠️  Newly broken tests (were passing before):")
            for test in newly_broken:
                print(f"   - {test}")
            print(
                f"   🎯 {len(newly_broken)} test(s) newly broken by commit "
                f"{self.commit_info['short_commit']}"
            )
        elif current_failed:
            print("\n   ✅ No newly broken tests (all failures were pre-existing)")

        # Show usage hints
        if overall_status == "FAIL":
            print("\n💡 To find when a specific test started failing:")
            print(f"   grep 'test_name' {self.individual_log}")


def main():
    parser = argparse.ArgumentParser(description="Track test results across git commits")
    parser.add_argument("--commit", help="Specific commit hash to analyze")
    parser.add_argument("--dry-run", action="store_true", help="Run tests but don't log results")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full test suite including GPU tests and slow tests (excludes skipped tests)",
    )
    parser.add_argument(
        "--stress",
        action="store_true",
        help="Run stress test suite including GPU stress tests and slow Python tests only",
    )

    args = parser.parse_args()

    # Validate mutually exclusive options
    if args.full and args.stress:
        parser.error("--full and --stress are mutually exclusive")

    tracker = TestTracker(full_tests=args.full, stress_tests=args.stress)
    tracker.run(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
