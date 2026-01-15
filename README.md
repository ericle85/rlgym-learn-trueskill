# TrueSkill Evaluation System for rlgym-learn

Pluggable TrueSkill rating system to track training progress.

## Installation

Copy this folder into your project (e.g., `utils/trueskill/`):
```bash
git clone https://github.com/ericle85/rlgym-learn-trueskill.git utils/trueskill
```

Install dependencies:
```bash
pip install trueskill
```
## Full Example

See `example_quick_start_trueskill.py` for complete integration with rlgym-learn training loop.

## Quick Start

```python
# Adjust import path based on where you placed the folder
from utils.trueskill import EvalRunner, TrueSkillTracker, ModelVersionManager

# 1. Define obs/action (MUST match your training env)
obs_builder = YourObsBuilder()
action_parser = YourActionParser()

# 2. Create components
version_manager = ModelVersionManager(
    versions_dir="model_versions",
    max_versions=20,
    save_every_ts=5_000_000,
)

trueskill_tracker = TrueSkillTracker(min_sigma=1.0)

eval_runner = EvalRunner(
    policy_class=YourPolicy,
    policy_kwargs={"n_actions": 126},
    obs_builder=obs_builder,      # Required
    action_parser=action_parser,  # Required
    device="cpu",
    deterministic=True,
)

# 3. During training - save versions
if version_manager.should_save_version(cumulative_ts):
    version = version_manager.save_version(
        actor_state_dict=policy.state_dict(),
        cumulative_timesteps=cumulative_ts,
    )
    trueskill_tracker.promote_current_to_version(version.version_id)

# 4. Run eval against past versions
version_state_dicts = {
    vid: version_manager.load_actor(vid, "cpu")
    for vid in version_manager.get_version_ids()[-3:]
}

results = eval_runner.run_eval_batch(
    current_state_dict=policy.state_dict(),
    version_state_dicts=version_state_dicts,
    gamemodes=["1v1"],
    trueskill_tracker=trueskill_tracker,
)

# 5. Save/load ratings
trueskill_tracker.save("trueskill_ratings.json")
trueskill_tracker.print_leaderboard()
```

## Components

| Component | Purpose |
|-----------|---------|
| `EvalRunner` | Runs eval games between model versions |
| `TrueSkillTracker` | Tracks TrueSkill ratings per gamemode |
| `ModelVersionManager` | Saves/loads versioned checkpoints |
| `OpponentPool` | Opponent selection for self-play |
| `GameOutcomeProvider` | SharedInfoProvider for game outcomes |

## EvalRunner Parameters

```python
EvalRunner(
    policy_class,           # Policy class to instantiate
    policy_kwargs,          # Kwargs for policy constructor
    obs_builder,            # ObsBuilder instance (MUST match training)
    action_parser,          # ActionParser instance (MUST match training)
    device="cpu",           # Device for inference
    deterministic=True,     # Deterministic action selection
    game_length_seconds=300.0,
    max_overtime_seconds=300.0,
    no_touch_timeout_seconds=30.0,
)
```

## Key: obs_builder and action_parser MUST match training config

If eval uses different obs/action than training, the policy will produce garbage outputs.
