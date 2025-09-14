#!/usr/bin/env python3
"""
Utility script for wandb operations used by shell scripts.
Provides subcommands for common wandb tasks.
"""

import argparse
import os
import sys
from pathlib import Path

import wandb


def lookup_run(identifier, project="madrona-escape-room-dev"):
    """Look up wandb run hash from name or identifier."""
    # Suppress wandb verbose output
    os.environ["WANDB_SILENT"] = "true"

    try:
        api = wandb.Api()
        runs = api.runs(project)

        # Find run by exact name match
        matching_runs = [run for run in runs if run.name == identifier]

        # If no exact match, try by run ID/hash
        if not matching_runs:
            matching_runs = [run for run in runs if run.id == identifier]

        # If still no match, try partial name match
        if not matching_runs:
            matching_runs = [run for run in runs if identifier.lower() in run.name.lower()]

        if not matching_runs:
            return "NOT_FOUND"

        if len(matching_runs) == 1:
            return matching_runs[0].id
        else:
            return "DUPLICATES"

    except Exception as e:
        return f"ERROR: {e}"


def get_run_object(identifier, project="madrona-escape-room-dev", show_available_on_error=True):
    """Get wandb run object from name or identifier, with detailed error handling."""
    # Suppress wandb verbose output
    os.environ["WANDB_SILENT"] = "true"

    try:
        api = wandb.Api()
        runs = api.runs(project)

        # Find run by exact name match
        matching_runs = [run for run in runs if run.name == identifier]

        # If no exact match, try by run ID/hash
        if not matching_runs:
            matching_runs = [run for run in runs if run.id == identifier]

        # If still no match, try partial name match
        if not matching_runs:
            matching_runs = [run for run in runs if identifier.lower() in run.name.lower()]

        if not matching_runs:
            error_msg = f"ERROR: No wandb run found matching '{identifier}'"
            if show_available_on_error:
                error_msg += "\nAvailable runs:"
                for run in runs[:10]:  # Show first 10 runs
                    error_msg += f"\n  {run.name} ({run.id})"
                if len(runs) > 10:
                    error_msg += f"\n  ... and {len(runs) - 10} more runs"
            return error_msg

        if len(matching_runs) > 1:
            warning_msg = f"WARNING: Multiple runs found matching '{identifier}', using first one:"
            for run in matching_runs:
                warning_msg += f"\n  {run.name} ({run.id})"
            # Print warning but return the first run
            print(warning_msg)

        return matching_runs[0]

    except Exception as e:
        return f"ERROR: {e}"


def find_checkpoint(wandb_run_path):
    """Find the latest checkpoint in the wandb run directory."""
    checkpoints_dir = Path(wandb_run_path) / "files" / "checkpoints"

    if not checkpoints_dir.exists():
        return f"ERROR: Checkpoints directory not found: {checkpoints_dir}"

    # Find all .pth files and sort by numerical value
    checkpoint_files = list(checkpoints_dir.glob("*.pth"))
    if not checkpoint_files:
        return f"ERROR: No checkpoint files found in {checkpoints_dir}"

    # Sort by numerical value (extract number from filename)
    try:
        checkpoint_files.sort(key=lambda x: int(x.stem))
        latest_checkpoint = checkpoint_files[-1]
        return str(latest_checkpoint)
    except ValueError as e:
        return f"ERROR: Invalid checkpoint filename format: {e}"


def find_recording(wandb_run_path, checkpoint_name=None):
    """Find train.rec file in the checkpoint directory."""
    checkpoints_dir = Path(wandb_run_path) / "files" / "checkpoints"

    if not checkpoints_dir.exists():
        return f"ERROR: Checkpoints directory not found: {checkpoints_dir}"

    # If no specific checkpoint name provided, look for train.rec
    if checkpoint_name is None:
        train_rec = checkpoints_dir / "train.rec"
        if train_rec.exists():
            return str(train_rec)
        else:
            return "NOT_FOUND"

    # Look for recording file matching the checkpoint name
    recording_file = checkpoints_dir / f"{checkpoint_name}.rec"
    if recording_file.exists():
        return str(recording_file)
    else:
        return "NOT_FOUND"


def find_wandb_run_dir(run_hash):
    """Find local wandb run directory by hash."""
    wandb_base = Path("wandb")
    if not wandb_base.exists():
        return "ERROR: No wandb directory found"

    wandb_run_dirs = list(wandb_base.glob(f"*-{run_hash}"))

    if not wandb_run_dirs:
        return "NOT_FOUND"

    return str(wandb_run_dirs[0])


def main():
    parser = argparse.ArgumentParser(description="Utility for wandb operations")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Lookup subcommand
    lookup_parser = subparsers.add_parser("lookup", help="Look up wandb run hash from name")
    lookup_parser.add_argument("identifier", help="Run name or identifier")
    lookup_parser.add_argument(
        "--project", default="madrona-escape-room-dev", help="Wandb project name"
    )

    # Get run object subcommand
    get_run_parser = subparsers.add_parser(
        "get_run", help="Get wandb run object info from name (returns name and id)"
    )
    get_run_parser.add_argument("identifier", help="Run name or identifier")
    get_run_parser.add_argument(
        "--project", default="madrona-escape-room-dev", help="Wandb project name"
    )
    get_run_parser.add_argument(
        "--no-list", action="store_true", help="Don't show available runs on error"
    )

    # Find checkpoint subcommand
    checkpoint_parser = subparsers.add_parser(
        "find_checkpoint", help="Find latest checkpoint in wandb run"
    )
    checkpoint_parser.add_argument("wandb_run_path", help="Path to wandb run directory")

    # Find recording subcommand
    recording_parser = subparsers.add_parser(
        "find_recording", help="Find recording file in checkpoint directory"
    )
    recording_parser.add_argument("wandb_run_path", help="Path to wandb run directory")
    recording_parser.add_argument(
        "--checkpoint-name", help="Specific checkpoint name (without extension)"
    )

    # Find wandb run directory subcommand
    rundir_parser = subparsers.add_parser(
        "find_run_dir", help="Find local wandb run directory by hash"
    )
    rundir_parser.add_argument("run_hash", help="Wandb run hash")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "lookup":
        result = lookup_run(args.identifier, args.project)
        print(result)

    elif args.command == "get_run":
        result = get_run_object(args.identifier, args.project, not args.no_list)
        if isinstance(result, str) and result.startswith("ERROR"):
            print(result)
        else:
            # Return run name and id in format: name|id
            print(f"{result.name}|{result.id}")

    elif args.command == "find_checkpoint":
        result = find_checkpoint(args.wandb_run_path)
        print(result)

    elif args.command == "find_recording":
        result = find_recording(args.wandb_run_path, args.checkpoint_name)
        print(result)

    elif args.command == "find_run_dir":
        result = find_wandb_run_dir(args.run_hash)
        print(result)


if __name__ == "__main__":
    main()
