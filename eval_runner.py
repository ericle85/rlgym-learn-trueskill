"""
Evaluation Runner for running games between model versions.

Runs eval games to generate TrueSkill ratings for model versions.
Supports multiple gamemodes (1v1, 2v2, 3v3) with proper kickoff setup.
"""

import torch
import numpy as np
from typing import Callable, Dict, List, Optional
from dataclasses import dataclass

from .trueskill_tracker import TrueSkillTracker


@dataclass
class EvalResult:
    """Result of an evaluation match."""
    blue_version: int
    orange_version: int
    blue_goals: int
    orange_goals: int
    game_length_steps: int
    gamemode: str = "1v1"

    @property
    def result(self) -> int:
        """1 = blue win, -1 = orange win, 0 = draw"""
        if self.blue_goals > self.orange_goals:
            return 1
        elif self.orange_goals > self.blue_goals:
            return -1
        return 0


def _gamemode_to_team_size(gamemode: str) -> int:
    """Convert gamemode string to team size."""
    return int(gamemode[0])


class EvalRunner:
    """
    Runs evaluation games between model versions.

    Features:
    - Supports multiple gamemodes (1v1, 2v2, 3v3)
    - Uses proper kickoff setup for fair evaluation
    - Swaps out training env settings for eval-specific settings
    """

    def __init__(
        self,
        policy_class: type,
        policy_kwargs: dict,
        obs_builder,
        action_parser,
        device: str = "cpu",
        deterministic: bool = True,
        game_length_seconds: float = 300.0,
        max_overtime_seconds: float = 300.0,
        no_touch_timeout_seconds: float = 30.0,
    ):
        """
        Args:
            policy_class: Policy class to instantiate for each model
            policy_kwargs: Kwargs to pass to policy constructor
            obs_builder: Observation builder (must match training)
            action_parser: Action parser (must match training)
            device: Device to run models on
            deterministic: Whether to use deterministic action selection
            game_length_seconds: Length of eval games (default 5 min)
            max_overtime_seconds: Max overtime before ending (default 5 min)
            no_touch_timeout_seconds: End if no touch for this long
        """
        self.policy_class = policy_class
        self.policy_kwargs = policy_kwargs
        self.obs_builder = obs_builder
        self.action_parser = action_parser
        self.device = device
        self.deterministic = deterministic
        self.game_length_seconds = game_length_seconds
        self.max_overtime_seconds = max_overtime_seconds
        self.no_touch_timeout_seconds = no_touch_timeout_seconds

    def _create_eval_env(self, team_size: int):
        """Create an environment configured for evaluation."""
        from rlgym.api import RLGym
        from rlgym.rocket_league.sim import RocketSimEngine
        from rlgym.rocket_league.reward_functions import GoalReward
        from rlgym.rocket_league.done_conditions import NoTouchTimeoutCondition, AnyCondition
        from rlgym.rocket_league.state_mutators import MutatorSequence, FixedTeamSizeMutator

        from rlgym_tools.rocket_league.state_mutators.game_mutator import GameMutator
        from rlgym_tools.rocket_league.shared_info_providers.scoreboard_provider import ScoreboardProvider

        from .eval_game_utils import (
            KickoffOnGoalEngine,
            GameCondition,
            ScoreboardInitMutator,
        )

        # Use injected obs_builder and action_parser (must match training)
        obs_builder = self.obs_builder
        action_parser = self.action_parser

        # Eval-specific components
        eval_engine = KickoffOnGoalEngine(RocketSimEngine())

        eval_state_mutator = MutatorSequence(
            FixedTeamSizeMutator(blue_size=team_size, orange_size=team_size),
            ScoreboardInitMutator(game_length_seconds=self.game_length_seconds),
            GameMutator(),
        )

        eval_termination = AnyCondition(
            GameCondition(
                seconds_per_goal_forfeit=10 * team_size * 2,
                max_overtime_seconds=self.max_overtime_seconds
            ),
        )

        eval_truncation = NoTouchTimeoutCondition(
            timeout_seconds=self.no_touch_timeout_seconds
        )

        return RLGym(
            state_mutator=eval_state_mutator,
            obs_builder=obs_builder,
            action_parser=action_parser,
            reward_fn=GoalReward(), # Placeholder reward function
            termination_cond=eval_termination,
            truncation_cond=eval_truncation,
            transition_engine=eval_engine,
            shared_info_provider=ScoreboardProvider(),
        )

    def load_policy(self, state_dict: dict):
        """Create and load a policy from state dict."""
        policy = self.policy_class(**self.policy_kwargs, device=self.device)
        policy.load_state_dict(state_dict)
        policy.to(self.device)
        policy.eval()
        return policy

    def run_eval_game(
        self,
        blue_state_dict: dict,
        orange_state_dict: dict,
        gamemode: str = "1v1",
        blue_version_id: int = -1,
        orange_version_id: int = -1,
        max_steps: int = 54000,  # 15 minutes at 60 fps (with overtime)
    ) -> EvalResult:
        """
        Run a single evaluation game between two models.

        Args:
            blue_state_dict: State dict for blue team model
            orange_state_dict: State dict for orange team model
            gamemode: Game mode ("1v1", "2v2", "3v3")
            blue_version_id: Version ID for blue (for logging)
            orange_version_id: Version ID for orange (for logging)
            max_steps: Maximum game steps (safety limit)

        Returns:
            EvalResult with game outcome
        """
        from utils.wrapper import RLGymV2GymWrapper

        team_size = _gamemode_to_team_size(gamemode)

        # Load policies
        blue_policy = self.load_policy(blue_state_dict)
        orange_policy = self.load_policy(orange_state_dict)

        # Create eval environment with proper kickoff setup
        eval_env = self._create_eval_env(team_size)
        env = RLGymV2GymWrapper(eval_env)

        blue_goals = 0
        orange_goals = 0
        total_steps = 0

        try:
            with torch.no_grad():
                obs = env.reset()
                state = env.rlgym_env.state
                done = False

                while not done and total_steps < max_steps:
                    action_indices = []

                    # Get actions for each agent
                    for i in range(2 * team_size):
                        obs_list = [obs[i]]
                        agent_id_list = [i]

                        # Blue team (first half) vs Orange team (second half)
                        if i < team_size:
                            policy = blue_policy
                        else:
                            policy = orange_policy

                        actions, _ = policy.get_action(
                            agent_id_list, obs_list, deterministic=self.deterministic
                        )
                        action_indices.append(actions[0])

                    actions_array = np.array(action_indices, dtype=np.int32).reshape(-1, 1)
                    obs, _, done, truncated, _ = env.step(actions_array)
                    done = done or truncated
                    state = env.rlgym_env.state
                    total_steps += 1

                    if state.goal_scored:
                        if state.scoring_team == 0:
                            blue_goals += 1
                        else:
                            orange_goals += 1

        finally:
            env.close()

        return EvalResult(
            blue_version=blue_version_id,
            orange_version=orange_version_id,
            blue_goals=blue_goals,
            orange_goals=orange_goals,
            game_length_steps=total_steps,
            gamemode=gamemode,
        )

    def run_eval_batch(
        self,
        current_state_dict: dict,
        version_state_dicts: Dict[int, dict],
        gamemodes: List[str] = None,
        games_per_version: int = 2,
        trueskill_tracker: Optional[TrueSkillTracker] = None,
    ) -> List[EvalResult]:
        """
        Run evaluation games of current model against multiple past versions.

        Args:
            current_state_dict: Current model state dict
            version_state_dicts: Dict mapping version_id -> state_dict
            gamemodes: List of gamemodes to eval (e.g., ["1v1", "2v2"])
            games_per_version: Number of games per version per gamemode
            trueskill_tracker: Optional tracker to update ratings

        Returns:
            List of EvalResults
        """
        if gamemodes is None:
            gamemodes = ["1v1"]

        results = []
        current_version = -1  # -1 represents current (latest) model

        for gamemode in gamemodes:
            print(f"\n--- Evaluating {gamemode} ---")

            for version_id, past_state_dict in version_state_dicts.items():
                for game_idx in range(games_per_version):
                    # Alternate which side plays which model
                    if game_idx % 2 == 0:
                        result = self.run_eval_game(
                            blue_state_dict=current_state_dict,
                            orange_state_dict=past_state_dict,
                            gamemode=gamemode,
                            blue_version_id=current_version,
                            orange_version_id=version_id,
                        )
                    else:
                        result = self.run_eval_game(
                            blue_state_dict=past_state_dict,
                            orange_state_dict=current_state_dict,
                            gamemode=gamemode,
                            blue_version_id=version_id,
                            orange_version_id=current_version,
                        )

                    results.append(result)

                    # Update TrueSkill if tracker provided
                    if trueskill_tracker is not None:
                        blue_ver = f"v{result.blue_version}" if result.blue_version >= 0 else "current"
                        orange_ver = f"v{result.orange_version}" if result.orange_version >= 0 else "current"
                        trueskill_tracker.update_ratings(
                            blue_ver, orange_ver, result.result,
                            gamemode=gamemode, deterministic=self.deterministic
                        )

                    winner = "Blue" if result.result == 1 else "Orange" if result.result == -1 else "Draw"
                    print(f"[{gamemode}] v{result.blue_version} vs v{result.orange_version}: "
                          f"{result.blue_goals}-{result.orange_goals} ({winner})")

        return results


def run_eval_process(
    policy_class: type,
    policy_kwargs: dict,
    obs_builder,
    action_parser,
    current_state_dict: dict,
    version_state_dicts: Dict[int, dict],
    gamemodes: List[str] = None,
    device: str = "cpu",
    games_per_version: int = 2,
    deterministic: bool = True,
) -> List[EvalResult]:
    """
    Standalone function to run eval in a separate process.

    This can be called via multiprocessing to not block training.
    """
    runner = EvalRunner(
        policy_class=policy_class,
        policy_kwargs=policy_kwargs,
        obs_builder=obs_builder,
        action_parser=action_parser,
        device=device,
        deterministic=deterministic,
    )

    return runner.run_eval_batch(
        current_state_dict=current_state_dict,
        version_state_dicts=version_state_dicts,
        gamemodes=gamemodes,
        games_per_version=games_per_version,
    )
