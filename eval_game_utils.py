"""
Utilities for running fair evaluation games.

Provides components to set up proper game conditions for evaluation:
- KickoffOnGoalEngine: Resets to kickoff after goals
- GameCondition: Standard game termination (time, overtime, forfeit)
- ScoreboardInitMutator: Initializes scoreboard for game tracking
"""

from typing import Any, Dict, List
import numpy as np
from rlgym.api.config.transition_engine import TransitionEngine
from rlgym.api import StateMutator, DoneCondition, AgentID
from rlgym.rocket_league.state_mutators import KickoffMutator
from rlgym.rocket_league.api import GameState
from rlgym_tools.rocket_league.shared_info_providers.scoreboard_provider import ScoreboardInfo
from rlgym.rocket_league.common_values import TICKS_PER_SECOND


class KickoffOnGoalEngine(TransitionEngine[AgentID, GameState, np.ndarray]):
    """
    Wraps a transition engine to automatically reset to kickoff after goals.

    This enables running full games with proper kickoff behavior after each goal.
    """

    def __init__(self, engine: TransitionEngine):
        self.engine = engine
        self.kickoff_mutator = KickoffMutator()

    @property
    def agents(self) -> List[AgentID]:
        return self.engine.agents

    @property
    def max_num_agents(self) -> int:
        return self.engine.max_num_agents

    @property
    def state(self) -> GameState:
        return self.engine.state

    @property
    def config(self) -> Dict[str, Any]:
        return self.engine.config

    @config.setter
    def config(self, value: Dict[str, Any]):
        self.engine.config = value

    def step(self, actions: Dict[AgentID, np.ndarray], shared_info: Dict[str, Any]) -> GameState:
        new_state = self.engine.step(actions, shared_info)
        scoreboard: ScoreboardInfo = shared_info["scoreboard"]

        # Check if we need to trigger kickoff after a goal
        if scoreboard.go_to_kickoff:
            # Apply kickoff mutator to reset positions
            self.kickoff_mutator.apply(new_state, shared_info)
            scoreboard.kickoff_timer_seconds = 5
            scoreboard.go_to_kickoff = False
            return self.engine.set_state(new_state, shared_info)

        return new_state

    def set_state(self, desired_state: GameState, shared_info: Dict[str, Any]) -> GameState:
        return self.engine.set_state(desired_state, shared_info)

    def get_state(self) -> GameState:
        return self.engine.get_state() if hasattr(self.engine, 'get_state') else self.engine.state

    def create_base_state(self) -> GameState:
        return self.engine.create_base_state()

    def close(self) -> None:
        self.engine.close()


class GameCondition(DoneCondition[AgentID, GameState]):
    """
    Standard Rocket League game termination condition.

    Ends the game when:
    - Time runs out and a team is leading
    - Overtime exceeds maximum allowed time
    - Score difference is insurmountable (forfeit)
    """

    def __init__(
        self,
        seconds_per_goal_forfeit: float = None,
        max_overtime_seconds: float = None
    ):
        """
        Args:
            seconds_per_goal_forfeit: If set, forfeit when score diff >= 3 and
                                     time_remaining / goal_diff < this value
            max_overtime_seconds: Maximum overtime duration before ending
        """
        self.seconds_per_goal_forfeit = seconds_per_goal_forfeit
        self.max_overtime_seconds = max_overtime_seconds
        self.overtime_duration = 0
        self.last_ticks = None

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.overtime_duration = 0
        self.last_ticks = initial_state.tick_count

    def is_done(self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]) -> Dict[AgentID, bool]:
        scoreboard: ScoreboardInfo = shared_info["scoreboard"]

        done = False
        if scoreboard.is_over:
            done = True
        elif scoreboard.is_overtime:
            self.overtime_duration += (state.tick_count - self.last_ticks) / TICKS_PER_SECOND
            if self.max_overtime_seconds is not None and self.overtime_duration >= self.max_overtime_seconds:
                scoreboard.is_over = True
                done = True
        else:
            # Check for forfeit condition
            if self.seconds_per_goal_forfeit is not None:
                goal_diff = abs(scoreboard.blue_score - scoreboard.orange_score)
                if goal_diff >= 3:
                    seconds_per_goal = scoreboard.game_timer_seconds / goal_diff
                    if seconds_per_goal < self.seconds_per_goal_forfeit:
                        scoreboard.is_over = True
                        done = True

        self.last_ticks = state.tick_count
        return {agent: done for agent in agents}


class ScoreboardInitMutator(StateMutator[GameState]):
    """
    Initializes the scoreboard for a new game.

    Sets up game timer, kickoff timer, and resets scores.
    """

    def __init__(self, game_length_seconds: float = 300.0, kickoff_timer_seconds: float = 5.0):
        """
        Args:
            game_length_seconds: Total game time (default 5 minutes)
            kickoff_timer_seconds: Kickoff countdown time
        """
        self.game_length_seconds = game_length_seconds
        self.kickoff_timer_seconds = kickoff_timer_seconds

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        shared_info["scoreboard"] = ScoreboardInfo(
            game_timer_seconds=self.game_length_seconds,
            kickoff_timer_seconds=self.kickoff_timer_seconds,
            blue_score=0,
            orange_score=0,
            go_to_kickoff=True,
            is_over=False
        )
