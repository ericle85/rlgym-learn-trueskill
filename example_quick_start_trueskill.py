"""
Quick Start with TrueSkill Tracking

This example shows how to integrate TrueSkill tracking with rlgym-learn training:
1. Saves versioned checkpoints at intervals
2. Runs periodic evaluation games against past versions
3. Tracks TrueSkill ratings per selected gamemode (1v1, 2v2, 3v3)
4. Logs ratings to wandb

Key additions over basic quick_start.py:
- ModelVersionManager: Saves versioned checkpoints
- TrueSkillTracker: Tracks ratings from eval games (per gamemode)
- EvalRunner: Runs fair eval games with proper kickoff setup
- Periodic eval callback: Runs eval games during training
"""

import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# === REPLACE WITH YOUR ENV BUILDER ===
# from your_env import build_rlgym_v2_env
def build_rlgym_v2_env():
    """Replace with your environment builder."""
    raise NotImplementedError("Replace with your env builder")


if __name__ == "__main__":
    from typing import Tuple
    import numpy as np
    import threading

    from rlgym_learn_algos.logging import (
        WandbMetricsLogger,
        WandbMetricsLoggerConfigModel,
    )
    from rlgym_learn_algos.ppo import (
        ExperienceBufferConfigModel,
        GAETrajectoryProcessor,
        GAETrajectoryProcessorConfigModel,
        NumpyExperienceBuffer,
        PPOAgentController,
        PPOAgentControllerConfigModel,
        PPOLearnerConfigModel,
        PPOMetricsLogger,
    )

    from rlgym_learn import (
        BaseConfigModel,
        LearningCoordinator,
        LearningCoordinatorConfigModel,
        ProcessConfigModel,
        PyAnySerdeType,
        SerdeTypesModel,
        generate_config,
    )

    from rlgym.rocket_league.action_parsers import RepeatAction

    # TrueSkill components
    from utils.trueskill import ModelVersionManager, TrueSkillTracker, EvalRunner

    # ============== YOUR IMPORTS ==============
    # Replace these with your actual implementations:
    #
    # from your_policy import YourPolicy
    # from your_critic import YourCritic
    # from your_obs import YourObsBuilder
    # from your_action import YourActionParser

    # ============== CONFIGURATION ==============

    # TrueSkill settings
    VERSION_SAVE_INTERVAL = 5_000_000  # Save a version every 5M timesteps
    MAX_VERSIONS = 20  # Keep last 20 versions
    EVAL_INTERVAL = 10_000_000  # Run evals every 10M timesteps
    EVAL_GAMES_PER_VERSION = 2  # Games per matchup per gamemode during eval
    VERSIONS_TO_EVAL = 3  # Number of past versions to eval against

    # Gamemodes to evaluate (separate TrueSkill ratings per mode)
    EVAL_GAMEMODES = ["1v1"]  # Add "2v2", "3v3" as needed

    # Paths
    VERSIONS_DIR = "model_versions"
    TRUESKILL_PATH = "trueskill_ratings.json"

    # ============== SETUP ==============

    # === REPLACE WITH YOUR OBS/ACTION SPACE TYPES ===
    DefaultObsSpaceType = Tuple[str, int]
    DefaultActionSpaceType = Tuple[str, int]

    # === REPLACE WITH YOUR FACTORY FUNCTIONS ===
    def actor_factory(
        obs_space: DefaultObsSpaceType,
        action_space: DefaultActionSpaceType,
        device: str,
    ):
        # return YourPolicy(n_actions=YOUR_N_ACTIONS, device=device)
        raise NotImplementedError("Replace with your policy factory")

    def critic_factory(obs_space: DefaultObsSpaceType, device: str):
        # return YourCritic(obs_space[1], hidden_layers, device)
        raise NotImplementedError("Replace with your critic factory")

    # === REPLACE WITH YOUR GAMMA CALCULATION ===
    gamma = 0.99  # Or use: horizon_to_gamma(horizon_seconds, tick_skip)

    # Initialize TrueSkill components
    version_manager = ModelVersionManager(
        versions_dir=VERSIONS_DIR,
        max_versions=MAX_VERSIONS,
        save_every_ts=VERSION_SAVE_INTERVAL,
    )

    trueskill_tracker = TrueSkillTracker(min_sigma=1.0)
    if os.path.exists(TRUESKILL_PATH):
        trueskill_tracker.load(TRUESKILL_PATH)
        print(f"Loaded TrueSkill ratings from {TRUESKILL_PATH}")
        print(f"  Total versions tracked: {len(trueskill_tracker.version_ratings)}")
        print(f"  Eval games played: {trueskill_tracker.total_eval_games}")

    # === REPLACE WITH YOUR OBS/ACTION BUILDERS ===
    # These MUST match your training env configuration
    # obs_builder = YourObsBuilder()
    # action_parser = RepeatAction(YourActionParser(...), repeats=YOUR_TICK_SKIP)
    obs_builder = None  # REPLACE
    action_parser = None  # REPLACE

    # === REPLACE WITH YOUR POLICY CLASS AND KWARGS ===
    POLICY_CLASS = None  # YourPolicy
    POLICY_KWARGS = {"n_actions": 126}  # Adjust to your action space size

    # Eval runners - one for each mode to compare deterministic vs stochastic
    eval_runner_deterministic = EvalRunner(
        policy_class=POLICY_CLASS,
        policy_kwargs=POLICY_KWARGS,
        obs_builder=obs_builder,
        action_parser=action_parser,
        device="cpu",  # Run eval on CPU to not interfere with training
        deterministic=True,
        game_length_seconds=300.0,  # 5 minute games
        max_overtime_seconds=300.0,  # 5 minute max overtime
        no_touch_timeout_seconds=30.0,
    )

    eval_runner_stochastic = EvalRunner(
        policy_class=POLICY_CLASS,
        policy_kwargs=POLICY_KWARGS,
        obs_builder=obs_builder,
        action_parser=action_parser,
        device="cpu",
        deterministic=False,
        game_length_seconds=300.0,
        max_overtime_seconds=300.0,
        no_touch_timeout_seconds=30.0,
    )

    # ============== CUSTOM METRICS LOGGER ==============

    class TrueSkillMetricsLogger(PPOMetricsLogger):
        """Extended metrics logger that includes TrueSkill metrics."""

        def __init__(self, ts_tracker: TrueSkillTracker):
            super().__init__()
            self.ts_tracker = ts_tracker

        def get_metrics(self):
            base_metrics = super().get_metrics()
            ts_metrics = self.ts_tracker.get_metrics()
            return {**base_metrics, **ts_metrics}

    # ============== TRAINING CONFIG ==============

    config = LearningCoordinatorConfigModel(
        base_config=BaseConfigModel(
            serde_types=SerdeTypesModel(
                agent_id_serde_type=PyAnySerdeType.STRING(),
                action_serde_type=PyAnySerdeType.NUMPY(np.int64),
                obs_serde_type=PyAnySerdeType.NUMPY(np.float64),
                reward_serde_type=PyAnySerdeType.FLOAT(),
                obs_space_serde_type=PyAnySerdeType.TUPLE(
                    (PyAnySerdeType.STRING(), PyAnySerdeType.INT())
                ),
                action_space_serde_type=PyAnySerdeType.TUPLE(
                    (PyAnySerdeType.STRING(), PyAnySerdeType.INT())
                ),
            ),
            timestep_limit=100_000_000_000,
        ),
        process_config=ProcessConfigModel(
            n_proc=100,
            render=False,
            render_delay=1/30,
        ),
        agent_controllers_config={
            "PPO": PPOAgentControllerConfigModel(
                learner_config=PPOLearnerConfigModel(
                    batch_size=100_000,
                    n_minibatches=5,
                    ent_coef=0.01,
                    actor_lr=3e-4,
                    critic_lr=3e-4,
                ),
                experience_buffer_config=ExperienceBufferConfigModel(
                    max_size=300_000,
                    trajectory_processor_config=GAETrajectoryProcessorConfigModel(gamma=gamma),
                ),
                metrics_logger_config=WandbMetricsLoggerConfigModel(
                    group="rlgym-learn-trueskill"
                ),
                timesteps_per_iteration=100_000,
                # checkpoint_load_folder="path/to/checkpoint",  # Optional
            )
        },
        agent_controllers_save_folder="agent_controllers_checkpoints",
    )

    generate_config(
        learning_coordinator_config=config,
        config_location="config.json",
        force_overwrite=True,
    )

    # ============== EVALUATION CALLBACK ==============

    last_eval_ts = 0
    eval_lock = threading.Lock()
    eval_running = False

    def run_eval_background(actor_state_dict: dict, versions_to_eval: list):
        """Run evaluation in background thread."""
        global eval_running
        try:
            print(f"\n[Eval] Starting evaluation against {len(versions_to_eval)} versions...")
            print(f"[Eval] Gamemodes: {EVAL_GAMEMODES}")

            # Load all version state dicts
            version_state_dicts = {}
            for vid in versions_to_eval:
                state_dict = version_manager.load_actor(vid, "cpu")
                if state_dict is not None:
                    version_state_dicts[vid] = state_dict

            if not version_state_dicts:
                print("[Eval] No valid versions to evaluate against")
                return

            # Run deterministic evaluation
            print("[Eval] Running deterministic evaluation...")
            eval_runner_deterministic.run_eval_batch(
                current_state_dict=actor_state_dict,
                version_state_dicts=version_state_dicts,
                gamemodes=EVAL_GAMEMODES,
                games_per_version=EVAL_GAMES_PER_VERSION,
                trueskill_tracker=trueskill_tracker,
            )

            # Run stochastic evaluation
            print("[Eval] Running stochastic evaluation...")
            eval_runner_stochastic.run_eval_batch(
                current_state_dict=actor_state_dict,
                version_state_dicts=version_state_dicts,
                gamemodes=EVAL_GAMEMODES,
                games_per_version=EVAL_GAMES_PER_VERSION,
                trueskill_tracker=trueskill_tracker,
            )

            # Save updated ratings
            trueskill_tracker.save(TRUESKILL_PATH)
            trueskill_tracker.print_leaderboard(deterministic=True)
            trueskill_tracker.print_leaderboard(deterministic=False)

        except Exception as e:
            print(f"[Eval] Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
        finally:
            with eval_lock:
                eval_running = False

    def on_iteration_end(agent_controller):
        """Called after each training iteration."""
        global last_eval_ts, eval_running

        cumulative_ts = agent_controller.cumulative_timesteps

        # Save version checkpoint if needed
        if version_manager.should_save_version(cumulative_ts):
            actor_state_dict = agent_controller.learner.actor.state_dict()
            version = version_manager.save_version(
                actor_state_dict=actor_state_dict,
                cumulative_timesteps=cumulative_ts,
            )
            trueskill_tracker.promote_current_to_version(version.version_id)
            trueskill_tracker.save(TRUESKILL_PATH)

        # Run evaluation if needed (in background)
        if cumulative_ts - last_eval_ts >= EVAL_INTERVAL:
            with eval_lock:
                if eval_running:
                    return  # Skip if eval already running
                eval_running = True

            # Get versions to evaluate against
            all_versions = version_manager.get_version_ids()
            if len(all_versions) >= 1:
                # Select most recent versions
                versions_to_eval = all_versions[-VERSIONS_TO_EVAL:]

                # Get current model state dict (clone to CPU for background thread)
                actor_state_dict = {
                    k: v.cpu().clone()
                    for k, v in agent_controller.learner.actor.state_dict().items()
                }

                # Start eval in background
                eval_thread = threading.Thread(
                    target=run_eval_background,
                    args=(actor_state_dict, versions_to_eval),
                    daemon=True
                )
                eval_thread.start()

                last_eval_ts = cumulative_ts
            else:
                with eval_lock:
                    eval_running = False

    # ============== CREATE LEARNING COORDINATOR ==============

    # Custom agent controller that calls our callback
    class TrueSkillAgentController(PPOAgentController):
        def _learn(self):
            result = super()._learn()
            on_iteration_end(self)
            return result

    learning_coordinator = LearningCoordinator(
        build_rlgym_v2_env,
        agent_controllers={
            "PPO": TrueSkillAgentController(
                actor_factory=actor_factory,
                critic_factory=critic_factory,
                experience_buffer=NumpyExperienceBuffer(GAETrajectoryProcessor()),
                metrics_logger=WandbMetricsLogger(TrueSkillMetricsLogger(trueskill_tracker)),
                obs_standardizer=None,
            )
        },
        config=config,
    )

    print("\n" + "="*60)
    print("TrueSkill Training Started")
    print("="*60)
    print(f"Version save interval: {VERSION_SAVE_INTERVAL:,} timesteps")
    print(f"Eval interval: {EVAL_INTERVAL:,} timesteps")
    print(f"Eval gamemodes: {EVAL_GAMEMODES}")
    print(f"Games per version per mode: {EVAL_GAMES_PER_VERSION}")
    print(f"Max versions kept: {MAX_VERSIONS}")
    print(f"Versions directory: {VERSIONS_DIR}")
    print(f"Ratings file: {TRUESKILL_PATH}")
    print("="*60 + "\n")

    learning_coordinator.start()
