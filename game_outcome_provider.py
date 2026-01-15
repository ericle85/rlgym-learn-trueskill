"""
SharedInfoProvider that tracks game outcomes for TrueSkill/Elo tracking.
"""

from typing import Any, Dict, List
from rlgym.api import AgentID, SharedInfoProvider
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_TEAM


class GameOutcomeProvider(SharedInfoProvider[AgentID, GameState]):
    """
    Tracks game outcomes including goals scored and final results.

    Provides the following shared_info keys:
    - 'game_outcome': Dict with keys:
        - 'blue_goals': int
        - 'orange_goals': int
        - 'blue_agents': List[str]
        - 'orange_agents': List[str]
        - 'goal_scored_this_step': bool
        - 'scoring_team': int or None
    """

    def __init__(self):
        self.blue_goals = 0
        self.orange_goals = 0
        self.blue_agents: List[str] = []
        self.orange_agents: List[str] = []

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        """Reset tracking at the start of each episode."""
        self.blue_goals = 0
        self.orange_goals = 0

        # Identify which agents are on which team
        self.blue_agents = []
        self.orange_agents = []
        for agent_id in agents:
            if agent_id in initial_state.cars:
                car = initial_state.cars[agent_id]
                if car.team_num == BLUE_TEAM:
                    self.blue_agents.append(str(agent_id))
                else:
                    self.orange_agents.append(str(agent_id))

        shared_info['game_outcome'] = self._get_outcome_dict(
            goal_scored_this_step=False,
            scoring_team=None
        )

    def set_shared_info(
        self,
        agents: List[AgentID], #noqa: ARG002
        state: GameState,
        shared_info: Dict[str, Any]
    ) -> None:
        """Update shared_info with current game state."""
        goal_scored_this_step = state.goal_scored
        scoring_team = None

        if goal_scored_this_step:
            scoring_team = state.scoring_team
            if scoring_team == BLUE_TEAM:
                self.blue_goals += 1
            elif scoring_team == ORANGE_TEAM:
                self.orange_goals += 1

        shared_info['game_outcome'] = self._get_outcome_dict(
            goal_scored_this_step=goal_scored_this_step,
            scoring_team=scoring_team
        )

    def _get_outcome_dict(self, goal_scored_this_step: bool, scoring_team: Any) -> Dict[str, Any]:
        """Create the outcome dictionary."""
        return {
            'blue_goals': self.blue_goals,
            'orange_goals': self.orange_goals,
            'blue_agents': self.blue_agents.copy(),
            'orange_agents': self.orange_agents.copy(),
            'goal_scored_this_step': goal_scored_this_step,
            'scoring_team': scoring_team,
        }
