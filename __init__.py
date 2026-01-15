"""
TrueSkill tracking and evaluation system for rlgym-learn.

Components:
- TrueSkillTracker: Tracks TrueSkill ratings for model versions
- ModelVersionManager: Saves and manages versioned checkpoints
- EvalRunner: Runs evaluation games between model versions
- OpponentPool: Manages opponent selection for training
- GameOutcomeProvider: SharedInfoProvider for tracking game outcomes
"""

from .trueskill_tracker import (
    TrueSkillTracker,
    ModelRating,
    VersionRatings,
    GameOutcome,
    TRUESKILL_AVAILABLE,
)
from .model_version_manager import (
    ModelVersionManager,
    ModelVersion,
)
from .eval_runner import (
    EvalRunner,
    EvalResult,
    run_eval_process,
)
from .eval_game_utils import (
    KickoffOnGoalEngine,
    GameCondition,
    ScoreboardInitMutator,
)
from .opponent_pool import (
    OpponentPool,
    Opponent,
    MixedOpponentAgentController,
)
from .game_outcome_provider import GameOutcomeProvider

__all__ = [
    "TrueSkillTracker",
    "ModelRating",
    "VersionRatings",
    "GameOutcome",
    "TRUESKILL_AVAILABLE",
    "ModelVersionManager",
    "ModelVersion",
    "EvalRunner",
    "EvalResult",
    "run_eval_process",
    "KickoffOnGoalEngine",
    "GameCondition",
    "ScoreboardInitMutator",
    "OpponentPool",
    "Opponent",
    "MixedOpponentAgentController",
    "GameOutcomeProvider",
]
