# Plan: Add Sensor Config Support to Wandb Inference Scripts

## Required Reading

**IMPORTANT**: Read these files before starting implementation:

1. **scripts/inference_core.py** - InferenceConfig class, InferenceRunner.setup_simulation()
   - Lines 16-84: InferenceConfig parameters (note lidar_config on line 34)
   - Lines 100-112: setup_simulation() - how lidar_config is passed through

2. **scripts/inference_utils.py** - create_inference_config_from_wandb() function
   - Shows how wandb config is extracted (model_kwargs pattern)
   - Shows how overrides dict is processed
   - Template for adding sensor config extraction

3. **scripts/infer_from_wandb.py** - Main inference script
   - Lines 60-95: Current argument parsing pattern
   - Lines 96-167: Wandb config extraction and overrides building
   - Lines 150-161: Where to add sensor config to overrides

4. **madrona_escape_room/sensor_config.py** - LidarConfig dataclass
   - Lines 310-350: LidarConfig class definition and validation
   - Understand field names: lidar_num_samples, lidar_fov_degrees, etc.

5. **docs/specs/mgr.md** - Replay format v5 specification
   - Lines 515-535: ReplayMetadata with SensorConfig field
   - Understanding why sensor config matters for recordings

## Task List

- [ ] **Phase 0: Understand Current Code**
  - [ ] Read all files in Required Reading section
  - [ ] Trace how model_kwargs are extracted from wandb (inference_utils.py)
  - [ ] Understand how InferenceConfig receives and uses lidar_config

- [ ] **Phase 1: Update inference_utils.py**
  - [ ] Add sensor config extraction from wandb run.config
  - [ ] Create LidarConfig from wandb config values
  - [ ] Handle override from overrides['lidar_config'] if present
  - [ ] Pass lidar_config to InferenceConfig constructor
  - [ ] Add verbose logging for sensor config source

- [ ] **Phase 2: Add CLI Arguments to infer_from_wandb.py**
  - [ ] Add --lidar-beams argument
  - [ ] Add --lidar-fov argument
  - [ ] Add --lidar-noise-factor argument
  - [ ] Add --lidar-base-sigma argument
  - [ ] Update help text with examples

- [ ] **Phase 3: Build LidarConfig in infer_from_wandb.py**
  - [ ] Check if any sensor CLI args provided
  - [ ] Get wandb run config as fallback defaults
  - [ ] Create LidarConfig with merged values
  - [ ] Add lidar_config to overrides dict
  - [ ] Handle case where no config available (use None for defaults)

- [ ] **Phase 4: Add Verbose Output**
  - [ ] Print active sensor config after InferenceConfig creation
  - [ ] Show source: "from wandb", "from command line", or "using defaults"
  - [ ] Display beam count, FOV, noise parameters

- [ ] **Phase 5: Testing**
  - [ ] Test with wandb run that has sensor config saved
  - [ ] Test command-line override (--lidar-beams 64)
  - [ ] Test old wandb run without sensor config (default fallback)
  - [ ] Verify recording metadata includes correct sensor config
  - [ ] Test that lidar tensor shape matches config

- [ ] **Phase 6: Documentation**
  - [ ] Update scripts/infer_from_wandb.py docstring
  - [ ] Add sensor config examples to help text
  - [ ] Document in README or scripts/ documentation

## Problem

The `scripts/infer_from_wandb.py` script doesn't extract or use sensor configuration when loading models from wandb. This causes inference to use default sensor settings (128 beams, 120° FOV) instead of the configuration the model was trained with, leading to:

1. **Mismatched observation spaces** - Model expects different lidar dimensions
2. **Poor inference quality** - Model wasn't trained on the default config
3. **Incorrect recordings** - Recorded replays don't match training conditions

## Current State

### What Works
- `InferenceConfig` already has `lidar_config` parameter (scripts/inference_core.py:34)
- `InferenceRunner.setup_simulation()` properly passes lidar_config through (line 110)
- Level loading from wandb checkpoint directory works (lines 137-148)
- Model kwargs extraction from wandb config works (lines 111-121)

### What's Missing
- No sensor config extraction from wandb run config
- No command-line overrides for sensor config parameters
- `create_inference_config_from_wandb()` doesn't handle sensor config in overrides

## Solution: Extract and Use Sensor Config from Wandb

Extract sensor configuration from wandb run config (saved during training) and allow command-line overrides.

---

## Implementation Plan

### Phase 1: Update `create_inference_config_from_wandb()` in `scripts/inference_utils.py`

**Location:** Add after model_kwargs extraction (around line 115)

```python
# Extract sensor config from wandb if available
lidar_config = None
if hasattr(run, 'config'):
    # Check if sensor config was saved during training
    if 'lidar_num_samples' in run.config:
        from madrona_escape_room.sensor_config import LidarConfig

        lidar_config = LidarConfig(
            lidar_num_samples=run.config.get('lidar_num_samples', 128),
            lidar_fov_degrees=run.config.get('lidar_fov_degrees', 120.0),
            lidar_noise_factor=run.config.get('lidar_noise_factor', 0.0),
            lidar_base_sigma=run.config.get('lidar_base_sigma', 0.0)
        )

# Allow override from overrides dict
if overrides and 'lidar_config' in overrides:
    lidar_config = overrides['lidar_config']
```

**Then pass to InferenceConfig:**
```python
return InferenceConfig(
    ckpt_path=ckpt_path,
    compiled_levels=compiled_levels,
    model_kwargs=model_kwargs,
    num_worlds=num_worlds,
    num_steps=num_steps,
    exec_mode=exec_mode,
    gpu_id=gpu_id,
    sim_seed=sim_seed,
    fp16=fp16,
    level_file=level_file,
    lidar_config=lidar_config,  # ADD THIS LINE
    recording_path=None,
)
```

### Phase 2: Add Command-Line Arguments to `scripts/infer_from_wandb.py`

**Location:** After existing arguments (around line 58)

```python
parser.add_argument(
    "--lidar-beams",
    type=int,
    help="Override lidar beam count (default: from wandb config or 128)"
)
parser.add_argument(
    "--lidar-fov",
    type=float,
    help="Override lidar FOV in degrees (default: from wandb config or 120.0)"
)
parser.add_argument(
    "--lidar-noise-factor",
    type=float,
    help="Override lidar proportional noise (default: from wandb config or 0.0)"
)
parser.add_argument(
    "--lidar-base-sigma",
    type=float,
    help="Override lidar base noise floor (default: from wandb config or 0.0)"
)
```

### Phase 3: Build LidarConfig from Command-Line Args in `scripts/infer_from_wandb.py`

**Location:** After level file handling, before building overrides dict (around line 149)

```python
# Build lidar config from command-line args if any provided
lidar_config = None
if any([args.lidar_beams, args.lidar_fov, args.lidar_noise_factor, args.lidar_base_sigma]):
    from madrona_escape_room.sensor_config import LidarConfig

    # Get from wandb config as defaults, then override with command line
    run = get_run_object(args.wandb_run_identifier, args.project)

    lidar_beams = args.lidar_beams or run.config.get('lidar_num_samples', 128)
    lidar_fov = args.lidar_fov or run.config.get('lidar_fov_degrees', 120.0)
    noise_factor = args.lidar_noise_factor or run.config.get('lidar_noise_factor', 0.0)
    base_sigma = args.lidar_base_sigma or run.config.get('lidar_base_sigma', 0.0)

    lidar_config = LidarConfig(
        lidar_num_samples=lidar_beams,
        lidar_fov_degrees=lidar_fov,
        lidar_noise_factor=noise_factor,
        lidar_base_sigma=base_sigma
    )
```

**Add to overrides dict:**
```python
overrides = {
    "model_kwargs": model_kwargs,
    "level_file": level_file,
    "compiled_levels": compiled_levels,
    "num_worlds": num_worlds,
    "num_steps": args.num_steps,
    "exec_mode": "CUDA" if args.gpu_sim else "CPU",
    "gpu_id": args.gpu_id,
    "sim_seed": args.sim_seed,
    "fp16": args.fp16,
    "lidar_config": lidar_config,  # ADD THIS LINE
}
```

### Phase 4: Add Verbose Output

**Location:** After config creation (around line 180)

```python
if config.lidar_config:
    print(f"Lidar config: {config.lidar_config.lidar_num_samples} beams, "
          f"{config.lidar_config.lidar_fov_degrees}° FOV, "
          f"noise_factor={config.lidar_config.lidar_noise_factor}, "
          f"base_sigma={config.lidar_config.lidar_base_sigma}")
else:
    print("Using default lidar config (128 beams, 120° FOV, no noise)")
```

---

## Testing Plan

### Test 1: Verify Config Extraction from Wandb
```bash
# Find a run that was trained with custom sensor config
scripts/infer_from_wandb.py <run_hash> --num-steps 10

# Expected: Should print the sensor config used during training
# Expected: Lidar tensor shape should match training config
```

### Test 2: Command-Line Override
```bash
# Override with custom sensor config
scripts/infer_from_wandb.py <run_hash> --lidar-beams 64 --lidar-fov 180 --num-steps 10

# Expected: Should use 64 beams and 180° FOV
# Expected: Recording should preserve these settings in metadata
```

### Test 3: Replay Compatibility
```bash
# Create recording with custom config
scripts/infer_from_wandb.py <run_hash> --lidar-beams 64 --num-steps 100

# Load recording and verify sensor config
python -c "from madrona_escape_room import SimManager; \
           meta = SimManager.read_replay_metadata('<recording.rec>'); \
           print(f'Beams: {meta.sensor_config.lidar_num_samples}')"

# Expected: Should show 64 beams in metadata
```

### Test 4: Default Fallback
```bash
# Run on old wandb run without sensor config saved
scripts/infer_from_wandb.py <old_run_hash> --num-steps 10

# Expected: Should fall back to defaults (128 beams, 120° FOV)
# Expected: Should print warning about using defaults
```

---

## Files Modified

### Core Changes
- `scripts/inference_utils.py` - Extract sensor config from wandb, pass to InferenceConfig
- `scripts/infer_from_wandb.py` - Add CLI args, build LidarConfig, add to overrides

### No Changes Needed
- `scripts/inference_core.py` - Already supports lidar_config ✓
- `madrona_escape_room/sensor_config.py` - LidarConfig already exists ✓

---

## Backward Compatibility

### Old Wandb Runs (No Sensor Config Saved)
- Will use default sensor config (128 beams, 120° FOV, no noise)
- Will print: "Using default lidar config (no sensor config found in wandb)"
- Can still override with command-line args

### Existing Scripts
- No breaking changes - lidar_config is optional in InferenceConfig
- Scripts that don't use infer_from_wandb.py are unaffected

---

## Example Usage

```bash
# Use sensor config from training run
./scripts/infer_from_wandb.py abc123de --num-steps 1000

# Override beam count
./scripts/infer_from_wandb.py abc123de --lidar-beams 256 --num-steps 1000

# Full custom sensor config
./scripts/infer_from_wandb.py abc123de \
    --lidar-beams 64 \
    --lidar-fov 180 \
    --lidar-noise-factor 0.01 \
    --lidar-base-sigma 0.02 \
    --num-steps 1000

# GPU inference with custom config
./scripts/infer_from_wandb.py abc123de \
    --gpu-sim \
    --lidar-beams 128 \
    --num-worlds 64 \
    --num-steps 5000
```

---

## Success Criteria

✅ Sensor config extracted from wandb run config when available
✅ Command-line overrides work for all sensor parameters
✅ Default fallback works for old runs without sensor config
✅ Verbose output shows active sensor configuration
✅ Recordings preserve sensor config in metadata
✅ No breaking changes to existing scripts
✅ Backward compatible with old wandb runs

---

## Future Enhancements

1. **Auto-detect from checkpoint**: Extract sensor config from model architecture if wandb config missing
2. **Config validation**: Warn if overriding sensor config might cause dimension mismatch with model
3. **Replay-based inference**: Load sensor config from existing replay file metadata
4. **Sensor config presets**: Support --lidar-preset=wide/narrow/noisy shortcuts
